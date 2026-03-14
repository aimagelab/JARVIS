from typing import Dict, Optional, Union
from transformers import PretrainedConfig, AutoConfig
from ..conversations import DEFAULT_IMAGE_TOKEN
from src.models.model_utils import ProjectorType


class MLLMConfig(PretrainedConfig):
    model_type = 'mllm'
    is_composition = True
    image_token = DEFAULT_IMAGE_TOKEN

    def __init__(
        self,
        text_config: Optional[Union[PretrainedConfig, Dict]] = None,
        vision_config: Optional[Union[PretrainedConfig, Dict]] = None,
        projector_type: Optional[ProjectorType] = ProjectorType.LINEAR,
        projector_input_size: Optional[int] = None,
        projector_output_size: Optional[int] = None,
        projector_bias: Optional[bool] = False,
        image_token_id: Optional[int] = None,
        bidir_visual_attn: bool = True,
        vision_layer_idx: int = -2,
        skip_left_visual_tokens: int = 1,
        img_size: Optional[int] = None,
        **kwargs
    ):
        assert bidir_visual_attn is not None
        super().__init__(**kwargs)

        if isinstance(text_config, PretrainedConfig) or text_config is None:
            self.text_config = text_config
        else:
            self.text_config = AutoConfig.for_model(text_config.pop('model_type'), **text_config)

        if isinstance(vision_config, PretrainedConfig) or vision_config is None:
            self.vision_config = vision_config
        else:
            self.vision_config = AutoConfig.for_model(vision_config.pop('model_type'), **vision_config)
        
        self.projector_type = projector_type
        self.projector_input_size = projector_input_size
        self.projector_output_size = projector_output_size
        self.projector_bias = projector_bias
        self.image_token_id = image_token_id
        self.bidir_visual_attn = bidir_visual_attn
        self.vision_layer_idx = vision_layer_idx

        # 1 skips the CLS token. Can be used to skip e.g. register tokens
        self.skip_left_visual_tokens = skip_left_visual_tokens

        self.img_size = img_size

        self.mm_use_im_start_end = False

        self.maybe_init_defaults()

    def maybe_init_defaults(self):
        if self.projector_input_size is None and self.vision_config is not None:
            self.projector_input_size = self.vision_config.hidden_size
        if self.projector_output_size is None and self.text_config is not None:
            self.projector_output_size = self.text_config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size


__all__ = [
    "ProjectorType",
    "MLLMConfig"
]