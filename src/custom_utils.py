from transformers import logging
import os
import torch.distributed as dist
from pathlib import Path, PosixPath
from PIL import Image

logging.set_verbosity_info()
logging.enable_explicit_format()

DEBUG_LOAD_FIRST_LLM_LAYERS = int(os.getenv('DEBUG_LOAD_FIRST_LLM_LAYERS', 0))

def get_logger():
    return logging.get_logger("transformers")

logger = get_logger()


def is_debug():
    return int(os.getenv('DEBUG', 0))


def is_debug_skip_pca_train():
    return int(os.getenv('DEBUG_SKIP_PCA_TRAIN', 0))


def is_debug_n_dataset_samples():
    return int(os.getenv('DEBUG_N_DATASET_SAMPLES', 0))


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    return int(os.getenv('LOCAL_RANK'))


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_image_size_from_image_processor(image_processor):
    if hasattr(image_processor, 'crop_size'):
        # CLIP, DiNO
        return image_processor.crop_size['height'], image_processor.crop_size['width']
    elif hasattr(image_processor, 'size'):
        # I-JEPA, SigLIP
        return image_processor.size['height'], image_processor.size['width']
    

def is_peft_checkpoint(path: str | PosixPath):
    return Path(path).joinpath('adapter_config.json').exists()  


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
