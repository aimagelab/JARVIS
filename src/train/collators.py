# refernce for I-JEPA multi-block masking: https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py

import math

from multiprocessing import Value
from typing import List

from src.custom_utils import get_logger, get_rank

import torch
import torch.nn.functional as F
import numpy as np
from transformers import DINOv3ViTImageProcessorFast


logger = get_logger()


def pad_token_sequence(tokens: List[torch.Tensor], padding_value):
    return torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=padding_value)


# We use frozen visual encoders that have been trained with fixed-sized images.
# So, rather than feeding only the context patches to the context encoder,
# we feed the entire image to the context encoder,
# and we mask out all but the context patches
def rescale_and_normalize(x, img_proc):
    x = img_proc.rescale(x, scale=img_proc.rescale_factor)
    if isinstance(img_proc, DINOv3ViTImageProcessorFast) and not isinstance(x, torch.Tensor):
        # DINOv3 image processor expects images in [0, 1] range
        x = torch.from_numpy(x).to(torch.float32)
    x = img_proc.normalize(
        x,
        mean=img_proc.image_mean,
        std=img_proc.image_std
    )
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return x
    

class MaskCollator(object):

    def __init__(
        self,
        image_processor,
        tgt_image_processor,
        tokenizer,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        allow_overlap_tgt=True,
        tgt_img_size=None
    ):
        super(MaskCollator, self).__init__()
        self.image_processor = image_processor
        self.tgt_image_processor = tgt_image_processor
        self.tokenizer = tokenizer
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        # minimum number of patches to keep
        self.min_keep = min_keep
        # whether to allow overlap b/w enc and pred masks
        self.allow_overlap = allow_overlap
        self.allow_overlap_tgt = allow_overlap_tgt
        # collator is shared across worker processes
        self._itr_counter = Value('i', -1 + 1_000_000 * get_rank())
        logger.info(
            f"[Rank {get_rank()}] Initialized MaskCollator with seed {self._itr_counter.value}")
        self.tgt_img_size = tgt_img_size

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones(
            (self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for i in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(
                    p_size, acceptable_regions=None if self.allow_overlap_tgt else masks_C)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = torch.stack(
            [torch.stack([y[:min_keep_pred] for y in x]) for x in collated_masks_pred])
        assert self.nenc == 1, "Not implemented for nenc > 1"
        collated_masks_enc = torch.stack(
            [x[0][:min_keep_enc] for x in collated_masks_enc])

        pixel_values_enc = []
        pixel_values_pred = []
        texts = []
        input_ids = []
        attention_mask = []
        labels = []
        image_mask = []
        for i, sample in enumerate(batch):
            img = sample['image']

            pixel_values_pred_unnorm = self.image_processor(
                img, do_rescale=False, do_normalize=False).pixel_values[0]
            
            if self.tgt_img_size is not None and pixel_values_pred_unnorm.shape[-1] != self.tgt_img_size:
                pixel_values_pred_unnorm = F.interpolate(
                    torch.from_numpy(pixel_values_pred_unnorm).unsqueeze(0),
                    size=(self.tgt_img_size, self.tgt_img_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).numpy()
        
            pixel_values_enc.append(
                rescale_and_normalize(pixel_values_pred_unnorm, self.image_processor))
            pixel_values_pred.append(
                rescale_and_normalize(pixel_values_pred_unnorm, self.tgt_image_processor))
            input_ids.append(sample.get('input_ids'))
            attention_mask.append(sample.get('attention_mask'))
            labels.append(sample.get('labels'))
            image_mask.append(sample['image_mask'])

        pixel_values_enc = torch.from_numpy(np.stack(pixel_values_enc))
        pixel_values_pred = torch.from_numpy(np.stack(pixel_values_pred))
    
        text_inputs = dict(
            input_ids=pad_token_sequence(
                input_ids, self.tokenizer.pad_token_id),
            attention_mask=pad_token_sequence(
                attention_mask, padding_value=0),
            labels=pad_token_sequence(labels, -100)
        )

        image_mask = torch.tensor(image_mask, dtype=torch.bool)

        return dict(
            **text_inputs,
            pixel_values_enc=pixel_values_enc,
            pixel_values_pred=pixel_values_pred,
            mask_idxs_enc=collated_masks_enc,
            mask_idxs_pred=collated_masks_pred,
            image_mask=image_mask
        )

class SupervisedCollator:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []

        for sample in batch:
            input_ids.append(sample['input_ids'])
            attention_mask.append(sample['attention_mask'])
            labels.append(sample['labels'])
            pixel_values.append(sample['image'])

        input_ids = pad_token_sequence(input_ids, self.tokenizer.pad_token_id)
        attention_mask = pad_token_sequence(attention_mask, padding_value=0)
        labels = pad_token_sequence(labels, -100)

        pixel_values = self.image_processor(
            pixel_values, return_tensors='pt').pixel_values

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
        )
