from .configuration_mllm import *
from .modeling_mllm import *
from transformers import AutoConfig, AutoModel

AutoConfig.register(MLLMConfig.model_type, MLLMConfig)
AutoModel.register(MLLMConfig, MLLMPreTrainedModel)

AutoConfig.register(MLLMConfig.model_type, MLLMConfig)
AutoModel.register(MLLMConfig, MLLMForCausalLM)