from .configuration_mllm_jepa import *
from .modeling_mllm_jepa import *
from transformers import AutoConfig, AutoModel

AutoConfig.register(MLLMJEPAConfig.model_type, MLLMJEPAConfig)
AutoModel.register(MLLMJEPAConfig, MLLMJEPA)
