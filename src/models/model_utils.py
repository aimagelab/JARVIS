from transformers import (
    AutoModel,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    SiglipVisionModel
)
from enum import StrEnum
import torch.nn as nn
from transformers.integrations import is_deepspeed_zero3_enabled
import deepspeed
from contextlib import nullcontext


class ProjectorType(StrEnum):
    LINEAR = 'linear'
    MULTI_LINEAR = 'multi_linear'
    MLP = 'mlp'


def build_projector(projector_type, **kwargs):
    if projector_type == ProjectorType.LINEAR:
        return nn.Linear(
            in_features=kwargs['input_size'],
            out_features=kwargs['output_size'],
            bias=kwargs['bias']
        )
    elif projector_type == ProjectorType.MULTI_LINEAR:
        n_layers = kwargs['n_layers']
        return nn.ModuleList(
            nn.Linear(kwargs['input_size'],
                      kwargs['output_size'], bias=kwargs['bias']) for _ in range(n_layers)
        )
    elif projector_type == ProjectorType.MLP:
        intermediate_size = kwargs.get('intermediate_size', kwargs['output_size'])
        return nn.Sequential(
            nn.Linear(kwargs['input_size'],
                      intermediate_size, bias=kwargs['bias']),
            nn.GELU(),
            nn.Linear(intermediate_size,
                      kwargs['output_size'], bias=kwargs['bias'])
        )
    else:
        raise ValueError(f'Projector type {projector_type} not supported')


def build_vision_encoder(name_or_path, clip_with_proj: bool = False, **kwargs):
    vision_cls = AutoModel
    if 'openai/clip' in name_or_path:
        vision_cls = CLIPVisionModel
        if clip_with_proj:
            vision_cls = CLIPVisionModelWithProjection
    elif 'siglip' in name_or_path.lower():
        vision_cls = SiglipVisionModel
    return vision_cls.from_pretrained(name_or_path)


def init_projector(projector: nn.Module):
    ctx = nullcontext()
    if is_deepspeed_zero3_enabled():
        ctx = deepspeed.zero.GatheredParameters(projector.parameters(), modifier_rank=0)
    with ctx:
        if isinstance(projector, (nn.Sequential, nn.ModuleList)):
            for mod in projector:
                if len(list(mod.parameters())):
                    mod.reset_parameters()
        else:
            projector.reset_parameters()