from typing import Optional
import torch
import json
from pathlib import Path
from src.models.conversations import CONV_TEMPLATES, CONV_MAPPING, DEFAULT_IMAGE_TOKEN
import src.custom_utils as utils
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor
)
from src.train.args import CustomTrainingArguments, DataArguments, ModelArguments, postprocess_args
from src.train.trainers import NTPTrainer, NTPJEPATrainer
from argparse import Namespace
from src.models import MLLMConfig, MLLMForCausalLM, MLLMJEPAConfig, MLLMJEPA
from src.models.model_utils import build_vision_encoder
from src.train.collators import MaskCollator, SupervisedCollator
from PIL import Image

logger = utils.get_logger()


class LlavaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        args: DataArguments
    ):
        super().__init__()
        self.data_path = data_path
        self.image_folder = Path(image_folder)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.conv_template = CONV_TEMPLATES(args.conv_template)
        self.img_placeholder = Image.new('RGB', (400, 400))

        n_samples = utils.is_debug_n_dataset_samples()
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
                if n_samples:
                    self.data = self.data[:n_samples]
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                self.data = [json.loads(line.strip()) for line in f.readlines()]
        else:
            raise ValueError(
                f"Unsupported data file format: {data_path}, only .json and .jsonl are supported.")
        logger.info(f'Loaded {len(self.data)} samples from {data_path}')
        if n_samples:
            logger.info(
                f"Using only {len(self.data)} dataset samples for debugging.")

    def __len__(self):
        return len(self.data)

    @property
    def mod_lens(self):
        lens = []
        for sample in self.data:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            lens.append(cur_len)
        return lens

    def __getitem__(self, idx):
        sample = self.data[idx]

        img_err = False
        if sample.get('image', None):
            try:
                image_path = self.image_folder.joinpath(sample['image'])
                image = Image.open(image_path).convert('RGB')
                image_mask = 1
                if self.args.image_aspect_ratio == 'pad':
                    image = utils.expand2square(image, tuple(int(x*255)
                                                             for x in self.image_processor.image_mean))
            except FileNotFoundError as e:
                logger.warning(e)
                image = self.img_placeholder
                image_mask = 0
                img_err = True
        else:
            image = self.img_placeholder
            image_mask = 0

        input_ids = None
        attention_mask = None
        labels = None
        tok_kwargs = {}

        if self.args.prompt_max_length is not None:
            tok_kwargs['max_length'] = self.args.prompt_max_length
            tok_kwargs['truncation'] = True

        if self.conv_template == CONV_TEMPLATES.PLAIN:
            text = DEFAULT_IMAGE_TOKEN + \
                sample['conversations'][1]['value']
            inputs_text = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False, **tok_kwargs)
            input_ids = inputs_text.input_ids[0]
            attention_mask = inputs_text.attention_mask[0]
            image_token_id = self.tokenizer.convert_tokens_to_ids(
                DEFAULT_IMAGE_TOKEN)
            labels = input_ids.clone()
            labels[labels == image_token_id] = -100
        else:
            conv = CONV_MAPPING[self.conv_template].new_empty()
            conv.add_message(dict(role='system'))
            for turn in sample['conversations']:
                if turn['from'] == 'human':
                    role = 'user'
                    if DEFAULT_IMAGE_TOKEN in turn['value']:
                        turn['value'] = turn['value'].replace(
                            DEFAULT_IMAGE_TOKEN, '').strip()
                        turn['value'] = DEFAULT_IMAGE_TOKEN + \
                            '\n' + turn['value']
                        turn['value'] = turn['value'].strip()
                elif turn['from'] == 'gpt':
                    role = 'assistant'
                conv.add_message(dict(role=role, content=turn['value']))

            text = conv.get_prompt()
            input_ids = self.tokenizer(
                text, return_tensors="pt", **tok_kwargs).input_ids[0]
            attention_mask = torch.ones_like(
                input_ids, dtype=input_ids.dtype)

            if img_err:
                labels = torch.full((len(input_ids),),
                                    fill_value=-100, dtype=input_ids.dtype)
            else:
                tokenized_text = self.tokenizer.convert_ids_to_tokens(
                    input_ids)
                labels = conv.get_labels(input_ids, tokenized_text)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image=image,
            image_mask=image_mask
        )


def build_model_commons(args: ModelArguments) -> Namespace:
    if args.jepa_loss:
        config_cls = MLLMJEPAConfig
        model_cls = MLLMJEPA
    else:
        config_cls = MLLMConfig
        model_cls = MLLMForCausalLM

    if args.model_name:
        config = None
        if utils.DEBUG_LOAD_FIRST_LLM_LAYERS:
            logger.info(
                f"Using a {utils.DEBUG_LOAD_FIRST_LLM_LAYERS}-layer language model for debugging")
            config = config_cls.from_pretrained(args.model_name)
            config.text_config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_LLM_LAYERS
        lm_kwargs = {}
        if args.attn_implementation:
            lm_kwargs['attn_implementation'] = args.attn_implementation
        model = model_cls.from_pretrained(
            args.model_name, config=config, lm_kwargs=lm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        text_config = AutoConfig.from_pretrained(args.language_model_name)
        if utils.DEBUG_LOAD_FIRST_LLM_LAYERS:
            text_config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_LLM_LAYERS
            logger.info(
                f"Using a {utils.DEBUG_LOAD_FIRST_LLM_LAYERS}-layer language model for debugging")
        logger.info(f"Attention implementation: {args.attn_implementation}")
        language_model = AutoModelForCausalLM.from_pretrained(
            args.language_model_name,
            config=text_config,
            attn_implementation=args.attn_implementation
        )

        vision_model = build_vision_encoder(args.vision_model_name)
        model_kwargs = {}
        config_kwargs = {}
        if args.jepa_loss:
            model_kwargs['tgt_vision_model'] = build_vision_encoder(
                args.tgt_vision_model_name)
            config_kwargs['tgt_vision_model_name_or_path'] = model_kwargs['tgt_vision_model'].config.name_or_path
            config_kwargs['tgt_skip_left_visual_tokens'] = args.tgt_skip_left_visual_tokens
            config_kwargs['jepa_loss_fn'] = args.jepa_loss_fn
            config_kwargs['jepa_loss_weight'] = args.jepa_loss_weight
            config_kwargs['tgt_vision_layer_idx'] = args.tgt_vision_layer_idx
            config_kwargs['tgt_proj_type'] = args.tgt_proj_type
            config_kwargs['tgt_proj_output_size'] = args.tgt_proj_output_size
            config_kwargs['tgt_proj_intermediate_size'] = args.tgt_proj_intermediate_size
            config_kwargs['jepa_llm_layer_idx'] = args.jepa_llm_layer_idx
            config_kwargs['tgt_img_size'] = args.tgt_img_size       

        config = config_cls(
            text_config=text_config,
            vision_config=vision_model.config,
            vision_layer_idx=args.vision_layer_idx,
            skip_left_visual_tokens=args.skip_left_visual_tokens,
            projector_type=args.projector_type,
            projector_bias=args.projector_bias,
            bidir_visual_attn=args.bidir_visual_attn,
            img_size=args.img_size,
            **config_kwargs
        )

        # set_seed just before initializing the model, so to ensure that projector weights are initialized from the same seed
        # That is not the case in the original LLaVA codebase
        set_seed(args.seed)
        model = model_cls(config=config, language_model=language_model,
                          vision_model=vision_model, **model_kwargs)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_config.name_or_path)

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            ret = tokenizer.add_special_tokens(
                dict(pad_token=tokenizer.unk_token))
        else:
            tokenizer.pad_token = tokenizer.eos_token
            ret = tokenizer.add_special_tokens(
                dict(pad_token=tokenizer.eos_token))
        assert ret == 0, "Failed to add pad token to tokenizer"

    image_processor = AutoImageProcessor.from_pretrained(
        model.config.vision_config.name_or_path)
    model.init_vision_tokenizer(tokenizer)
    model.init_image_processor(image_processor)
    tgt_image_processor = image_processor
    if args.jepa_loss:
        tgt_image_processor = AutoImageProcessor.from_pretrained(
            model.tgt_vision_model.config.name_or_path)
        model.init_image_processor(
            tgt_image_processor, img_size=args.tgt_img_size)

    if args.override_bidir_visual_attn:
        model.config.bidir_visual_attn = args.bidir_visual_attn
        logger.info(
            f"`model.config.bidir_visual_attn` set to {model.config.bidir_visual_attn}")

    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.contiguous()

    return Namespace(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        tgt_image_processor=tgt_image_processor
    )


def build_data_commons(
    args: DataArguments,
    image_processor,
    tokenizer,
    vision_config,
    training_args: Optional[CustomTrainingArguments] = None,
    tgt_image_processor=None
) -> Namespace:
    train_dataset = LlavaDataset(args.train_data_path, args.train_image_folder,
                                     tokenizer=tokenizer, image_processor=image_processor, args=args)

    if args.jepa_loss:
        def get_input_size():
            if hasattr(image_processor, 'crop_size'):
                # CLIP, DiNO
                return image_processor.crop_size['height'], image_processor.crop_size['width']
            elif hasattr(image_processor, 'size'):
                # I-JEPA, SigLIP
                return image_processor.size['height'], image_processor.size['width']

        data_collator = MaskCollator(
            image_processor=image_processor,
            tgt_image_processor=tgt_image_processor,
            tokenizer=tokenizer,
            input_size=get_input_size(),
            patch_size=vision_config.patch_size,
            enc_mask_scale=args.jepa_enc_mask_scale,
            pred_mask_scale=args.jepa_pred_mask_scale,
            aspect_ratio=args.jepa_aspect_ratio,
            nenc=args.jepa_num_enc_masks,
            npred=args.jepa_num_pred_masks,
            min_keep=args.jepa_min_keep,
            allow_overlap=False,
            allow_overlap_tgt=args.jepa_allow_overlap_tgt,
            tgt_img_size=args.tgt_img_size
        )

    else:
        data_collator = SupervisedCollator(image_processor, tokenizer)

    return Namespace(
        train_dataset=train_dataset,
        data_collator=data_collator
    )


if __name__ == '__main__':
    parser = HfArgumentParser(
        (CustomTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = postprocess_args(
        *parser.parse_args_into_dataclasses())
    set_seed(training_args.seed)

    local_rank = utils.get_local_rank()
    torch.cuda.set_device(local_rank)
    print(f"Rank [{utils.get_rank()}]: set device ID {local_rank}")

    model_commons = build_model_commons(model_args)

    if training_args.gradient_checkpointing:
        model_commons.model.gradient_checkpointing_enable()

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model_commons.model.get_input_embeddings(
        ).register_forward_hook(make_inputs_require_grad)

    data_commons = build_data_commons(
        data_args,
        image_processor=model_commons.image_processor,
        tokenizer=model_commons.tokenizer,
        vision_config=model_commons.model.config.vision_config,
        training_args=training_args,
        tgt_image_processor=model_commons.tgt_image_processor
    )

    model_commons.model.get_vision_model().requires_grad_(False)
    if isinstance(model_commons.model, MLLMJEPA):
        model_commons.model.tgt_vision_model.requires_grad_(False)
    if training_args.train_proj_only:
        model_commons.model.get_language_model().requires_grad_(False)

    trainer_cls = NTPTrainer
    if model_args.jepa_loss:
        trainer_cls = NTPJEPATrainer
    logger.info(f"Using trainer class `{trainer_cls}`")

    trainer = trainer_cls(
        model=model_commons.model,
        args=training_args,
        train_dataset=data_commons.train_dataset,
        data_collator=data_commons.data_collator,
        processing_class=model_commons.tokenizer,
        data_args=data_args
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    logger.info('Training completed')
    logger.info(f"Training finished. Checkpoint saved.")