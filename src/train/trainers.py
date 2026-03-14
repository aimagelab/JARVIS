from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    EvalPrediction,
    DataCollator,
    TrainerCallback
)
from typing import Any, Callable, List, Optional, Union
import src.custom_utils as utils
from torch.utils.data import DataLoader
import webdataset as wds
from src.train.args import DataArguments
from transformers.utils.deprecation import deprecate_kwarg
import torch.nn as nn
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import Sampler
from collections import defaultdict
from transformers.trainer_utils import speed_metrics, SaveStrategy
import torch.distributed as dist
from peft import PeftModel

logger = utils.get_logger()


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lens: Optional[List[int]],
        generator=None
    ):
        self.batch_size = batch_size
        self.world_size = world_size
        self.lens = lens
        self.generator = generator

    def __len__(self):
        return len(self.lens)

    def __iter__(self):
        indices = get_modality_length_grouped_indices(self.lens, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
    

class NTPTrainer(Trainer):

    @deprecate_kwarg("tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, None] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        data_args: Optional[DataArguments] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.data_args = data_args
        self.train_metrics = defaultdict(float)

        # required to fix proper loss logging with grad acc.
        # Reference: https://github.com/huggingface/transformers/issues/40564#issuecomment-3240114783
        self.model_accepts_loss_kwargs = not args.compute_loss_return_outputs

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        self.processing_class.save_pretrained(output_dir)
        if isinstance(self.model, PeftModel):
            self.model.config.save_pretrained(output_dir)
        
    def get_train_dataloader(self) -> DataLoader:        
        if isinstance(self.train_dataset, wds.DataPipeline):
            logger.info("Using WebDataset for training")
            return wds.WebLoader(
                self.train_dataset,
                batch_size=None,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False, # assuming shuffling is handled by WebDataset,
                collate_fn=self.data_collator,
                persistent_workers=self.args.dataloader_persistent_workers,
            )
        else:
            return super().get_train_dataloader()    
    
    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_mod_lens:
            lens = self.train_dataset.mod_lens
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size,
                world_size=utils.get_world_size() * self.args.gradient_accumulation_steps,
                lens=lens,
            )
        else:
            return super()._get_train_sampler(train_dataset)
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen))

        if len(self.train_metrics) and self.state.global_step < self.state.max_steps:
            metric_tensors = torch.tensor(list(self.train_metrics.values()), device=utils.get_local_rank())
            dist.reduce(metric_tensors, dst=0, op=dist.ReduceOp.SUM)
            if utils.is_main_process():
                metric_tensors /= (self.args.logging_steps * utils.get_world_size() * self.args.gradient_accumulation_steps)
                for i, k in enumerate(self.train_metrics):
                    logs[k] = metric_tensors[i].item()
            self.train_metrics = defaultdict(float)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


class NTPJEPATrainer(NTPTrainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):  
        do_jepa = True
        if self.args.jepa_lambda > 0.0:
            do_jepa = torch.tensor(True, dtype=torch.bool, device=inputs['input_ids'].device)
            if torch.rand(1).item() < self.args.jepa_lambda:
                do_jepa.fill_(False)
            do_jepa = do_jepa.item()
            inputs['forward_ntp_only'] = not do_jepa

        loss, outputs = super(NTPTrainer, self).compute_loss(
            model=model, 
            inputs=inputs, 
            return_outputs=self.args.compute_loss_return_outputs, 
            num_items_in_batch=num_items_in_batch
        )
        if not do_jepa:
            outputs['loss_ntp'] = outputs.loss.item()
        outputs = (loss, outputs)

        if self.args.compute_loss_return_outputs:
            for k, v in outputs[1].items():
                if 'loss_' in k and v is not None:
                    self.train_metrics[k] += v

        if not return_outputs and isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs        
