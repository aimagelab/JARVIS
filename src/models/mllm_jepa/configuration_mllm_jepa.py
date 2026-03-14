from typing import Optional
from src.models.model_utils import ProjectorType
from src.models.mllm import MLLMConfig


class MLLMJEPAConfig(MLLMConfig):
    model_type = 'mllm_jepa'

    def __init__(
        self,
        tgt_vision_model_name_or_path: Optional[str] = None,
        tgt_proj_type: ProjectorType = ProjectorType.LINEAR,
        tgt_proj_intermediate_size: int = 0,
        tgt_proj_output_size: int = 0,
        tgt_skip_left_visual_tokens: Optional[int] = None,
        tgt_vision_layer_idx: int = -1,
        tgt_img_size: Optional[int] = None,
        jepa_loss_fn: str = 'cos_sim',
        jepa_loss_weight: float = 1.0,
        jepa_llm_layer_idx: int = -1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tgt_vision_model_name_or_path = tgt_vision_model_name_or_path
        self.tgt_proj_type = tgt_proj_type
        self.tgt_proj_intermediate_size = tgt_proj_intermediate_size
        self.tgt_proj_output_size = tgt_proj_output_size
        self.tgt_skip_left_visual_tokens = self.skip_left_visual_tokens if tgt_skip_left_visual_tokens is None else tgt_skip_left_visual_tokens
        self.tgt_vision_layer_idx = tgt_vision_layer_idx
        self.tgt_img_size = tgt_img_size
        self.jepa_loss_fn = jepa_loss_fn
        self.jepa_loss_weight = jepa_loss_weight
        self.jepa_llm_layer_idx = jepa_llm_layer_idx


__all__ = [
    "MLLMJEPAConfig"
]