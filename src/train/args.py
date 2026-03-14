from dataclasses import dataclass
from typing import Optional
from src.models.conversations import CONV_TEMPLATES
from transformers import TrainingArguments
from src.models import ProjectorType
import src.custom_utils as utils
from transformers.trainer_utils import get_last_checkpoint

logger = utils.get_logger()

@dataclass
class CustomTrainingArguments(TrainingArguments):
    resume_from_last_checkpoint: bool = False
    jepa_lambda: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.resume_from_last_checkpoint:
            try:
                check = get_last_checkpoint(self.output_dir)
            except FileNotFoundError:
                check = False
            if not check:
                self.resume_from_last_checkpoint = False
                logger.warning(f"No valid checkpoint found in {self.output_dir}. Setting `resume_from_last_checkpoint` to False")
        if self.resume_from_last_checkpoint and self.resume_from_checkpoint is None:
            self.resume_from_checkpoint = True
            logger.info(f"Setting `resume_from_checkpoint` to True because `resume_from_last_checkpoint` is set to True and no checkpoint was specified in `resume_from_checkpoint`.")


@dataclass
class DataArguments:
    train_data_path: Optional[str] = None
    train_image_folder: Optional[str] = None
    conv_template: str = CONV_TEMPLATES.PLAIN
    prompt_max_length: Optional[int] = None
    image_aspect_ratio: Optional[str] = None
    group_by_mod_lens: bool = False

    jepa_aspect_ratio: str = '0.75 1.5'
    jepa_enc_mask_scale: str = '0.85 1.0'
    jepa_min_keep: int = 10
    jepa_num_enc_masks: int = 1
    jepa_num_pred_masks: int = 4
    jepa_pred_mask_scale: str = '0.15 0.2'  
    jepa_allow_overlap_tgt: bool = True

    def __post_init__(self):
        self.jepa_aspect_ratio = [float(x) for x in self.jepa_aspect_ratio.strip().split()]
        self.jepa_enc_mask_scale = [float(x) for x in self.jepa_enc_mask_scale.strip().split()]
        self.jepa_pred_mask_scale = [float(x) for x in self.jepa_pred_mask_scale.strip().split()]


@dataclass
class ModelArguments:
    train_proj_only: bool = False
    attn_implementation: str = 'sdpa'

    # from scratch
    language_model_name: Optional[str] = None
    vision_model_name: Optional[str] = None
    vision_layer_idx: int = -2
    skip_left_visual_tokens: int = 1
    bidir_visual_attn: Optional[bool] = None
    img_size: Optional[int] = None
    projector_type: ProjectorType = ProjectorType.LINEAR
    projector_bias: bool = True
    projector_input_size: Optional[int] = None
    projector_output_size: Optional[int] = None
    projector_tie_weights: Optional[bool] = True
    tgt_vision_model_name: Optional[str] = None
    tgt_skip_left_visual_tokens: Optional[int] = None
    tgt_vision_layer_idx: int = -1
    tgt_img_size: Optional[int] = None
    tgt_proj_type: ProjectorType = ProjectorType.LINEAR
    tgt_proj_intermediate_size: int = 0
    tgt_proj_output_size: int = 0
    jepa_loss: bool = False
    jepa_loss_fn: str = 'cos_sim'
    jepa_loss_weight: float = 1.0    
    jepa_llm_layer_idx: int = -1

    # from checkpoint
    model_name: Optional[str] = None
    load_from_checkpoint: Optional[str] = None
    override_bidir_visual_attn: bool = False


def postprocess_args(training_args: TrainingArguments, model_args: ModelArguments, data_args: DataArguments):
    model_args.seed = training_args.seed
    
    training_args.compute_loss_return_outputs = model_args.jepa_loss
    training_args.jepa_loss = model_args.jepa_loss
    training_args.train_proj_only = model_args.train_proj_only
    training_args.group_by_mod_lens = data_args.group_by_mod_lens

    data_args.jepa_loss = model_args.jepa_loss
    data_args.tgt_img_size = model_args.tgt_img_size

    return training_args, model_args, data_args