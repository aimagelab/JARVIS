import torch
from src.models.mllm import MLLMForCausalLM
from transformers import AutoTokenizer, AutoImageProcessor, AutoConfig
import src.custom_utils as utils
from peft import PeftConfig, PeftModel

logger = utils.get_logger()

def process_images(images, image_processor, image_aspect_ratio='pad'):
    if image_aspect_ratio == 'pad':
        if isinstance(images, list):
            images = [utils.expand2square(image, tuple(int(x*255) for x in image_processor.image_mean)) for image in images]
        else:
            images = utils.expand2square(images, tuple(int(x*255) for x in image_processor.image_mean))
    
    if not isinstance(images, list):
        images = [images]

    return image_processor(images, return_tensors='pt').pixel_values


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    kwargs['lm_kwargs'] = dict(
        torch_dtype=torch.float16,
        attn_implementation=kwargs.get('attn_implementation', 'sdpa')
    )

    model_cls = MLLMForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    peft_config = None
    if utils.DEBUG_LOAD_FIRST_LLM_LAYERS:
        if utils.is_peft_checkpoint(model_path):
            peft_config = PeftConfig.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
        config.text_config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_LLM_LAYERS

    if utils.is_peft_checkpoint(model_path):
        peft_config = PeftConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(peft_config.base_model_name_or_path, config=config, **kwargs)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload().cuda()
        logger.info(f"Loaded PEFT model from {model_path} based on {peft_config.base_model_name_or_path}")
    else:
        model = model_cls.from_pretrained(model_path, config=config, **kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.init_vision_tokenizer(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(model.config.vision_config.name_or_path)
    model.init_image_processor(image_processor)

    context_len = 2048

    return tokenizer, model, image_processor, context_len
