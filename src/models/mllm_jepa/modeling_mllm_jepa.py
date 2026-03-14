from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from .configuration_mllm_jepa import MLLMJEPAConfig
from src.models.mllm import MLLMPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.custom_utils as utils
from torch import Tensor
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from src.models.model_utils import build_vision_encoder, build_projector, init_projector

logger = utils.get_logger()


@dataclass
class MLLMJEPAOutput(ModelOutput):
    loss: Optional[Tensor] = None
    loss_ntp: Optional[float] = None
    loss_jepa: Optional[float] = None


class MLLMJEPA(MLLMPreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = MLLMJEPAConfig

    def __init__(
        self,
        config: MLLMJEPAConfig,
        language_model: Optional[AutoModelForCausalLM] = None,
        vision_model: Optional[PreTrainedModel] = None,
        tgt_vision_model: Optional[PreTrainedModel] = None,
        projector: Optional[nn.Module] = None,
        lm_kwargs: Dict = {},
        **kwargs
    ):
        super().__init__(config, language_model=language_model,
                         vision_model=vision_model, projector=projector, lm_kwargs=lm_kwargs, **kwargs)

        if tgt_vision_model is None:
            tgt_vision_model = build_vision_encoder(
                config.jepa_tgt_vision_config.name_or_path)
            tgt_vision_model.requires_grad_(False)
            tgt_vision_model.eval()
        self._tgt_vision_model = tgt_vision_model

        self.z = nn.Embedding(1, self.config.text_config.hidden_size)
        tgt_proj_in_size = self.config.text_config.hidden_size 
        tgt_proj_out_size = self.config.tgt_proj_output_size
        tgt_proj_intermediate_size = self.config.tgt_proj_intermediate_size
        if not tgt_proj_out_size:
            tgt_proj_out_size = self.tgt_vision_model.config.hidden_size
        if not tgt_proj_intermediate_size:
            tgt_proj_intermediate_size = tgt_proj_out_size
        self.tgt_proj = build_projector(
            self.config.tgt_proj_type,
            input_size=tgt_proj_in_size,
            intermediate_size=tgt_proj_intermediate_size,
            output_size=tgt_proj_out_size,
            bias=False
        )
        self.init_z_and_tgt_proj()

    def init_z_and_tgt_proj(self):
        nn.init.normal_(
            self.z.weight, mean=0, std=self.config.text_config.initializer_range)
        init_projector(self.tgt_proj)

    @property
    def tgt_vision_model(self):
        if self._tgt_vision_model is None:
            return self.vision_model
        else:
            return self._tgt_vision_model

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values_enc: Optional[Tensor] = None,
        pixel_values_pred: Optional[Tensor] = None,
        mask_idxs_enc: Optional[Tensor] = None,
        mask_idxs_pred: Optional[Tensor] = None,
        image_mask: Optional[Tensor] = None,
        forward_ntp_only: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ) -> Union[MLLMJEPAOutput, Tuple, CausalLMOutputWithPast]:
        if forward_ntp_only:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values_pred if pixel_values_enc is None else pixel_values_enc, num_items_in_batch=num_items_in_batch)

        bsz = input_ids.size(
            0) if input_ids is not None else pixel_values_pred.size(0)
        n_pred_masks = mask_idxs_pred.size(1)
        lm = self.get_language_model()

        vision_model = self.get_vision_model()
        tgt_vision_model = self.tgt_vision_model

        if self.config.tgt_vision_layer_idx == -1:
            vision_embeds_pred = tgt_vision_model(
                pixel_values_pred).last_hidden_state[:, self.config.tgt_skip_left_visual_tokens:]
        else:
            vision_embeds_pred = tgt_vision_model(pixel_values_pred,
                                                  output_hidden_states=True).hidden_states[self.config.tgt_vision_layer_idx][:, self.config.tgt_skip_left_visual_tokens:]

        if vision_model is not tgt_vision_model or self.config.vision_layer_idx != self.config.tgt_vision_layer_idx:
            vision_embeds_enc = self.get_visual_embeds(pixel_values_enc)
        else:
            vision_embeds_enc = self.projector(vision_embeds_pred)
        enc_mask = torch.zeros(bsz, vision_embeds_enc.size(
            1), device=self.device, dtype=vision_embeds_enc.dtype)
        enc_mask.scatter_(1, mask_idxs_enc, 1)

        if vision_embeds_enc.size(1) != vision_embeds_pred.size(1):
            with torch.no_grad():
                ps = int(vision_embeds_enc.size(1) ** 0.5)
                tps = int(vision_embeds_pred.size(1) ** 0.5)
                th = vision_embeds_pred.size(2)
                vision_embeds_pred = F.interpolate(
                    vision_embeds_pred.transpose(1, 2).reshape(bsz, th, tps, tps),
                    size=(ps, ps),
                    mode="bilinear",
                    align_corners=False
                ).flatten(2, 3).transpose(1, 2)
        assert vision_embeds_enc.size(1) == vision_embeds_pred.size(1)

        # add latent variable
        H = self.config.hidden_size
        z = self.z.weight[None].expand(bsz, mask_idxs_pred.size(2), H).to(vision_embeds_enc.dtype)
        for i in range(n_pred_masks):
                    vision_embeds_enc.scatter_(1, mask_idxs_pred[:, i, :, None].expand(bsz, -1, H), z)

        # token -> embeddings
        inputs_embeds, attention_mask, labels, _, visual_position_ids = self.embed_tokens(
            input_ids, attention_mask, labels, vision_embeds_enc)
        # assuming there is at most 1 image per sample
        visual_position_ids = torch.tensor(
            visual_position_ids, dtype=torch.long, device=self.device).squeeze(1)
        
        L = attention_mask.size(2)
        for i, vis_pos in enumerate(visual_position_ids):
            # --- view attention map with `PIL Preview VSCode extension` ---
            # AAA = attention_mask[i].squeeze().clone()
            # AAA[AAA == 0] = 1
            # AAA[AAA == float('-inf')] = 0
            # AAA = ((AAA - AAA.min(dim=-1).values) / (AAA.max(dim=1).values - AAA.min(dim=1).values) * 255).to(torch.uint8).cpu().numpy()
            # from PIL import Image
            # AAA = Image.fromarray(AAA)
            if image_mask[i]:
                # 1. clear attention on visual tokens
                attention_mask[i, :, :, vis_pos[0]:vis_pos[1]] = float('-inf')
                
                # 2. any (future) token can attend to context block
                attention_mask[i, :, vis_pos[0]:, vis_pos[0]:] = attention_mask[i, :, vis_pos[0]:, vis_pos[0]:].scatter_(2, mask_idxs_enc[i:i + 1, None, :].expand(1, L - vis_pos[0], -1), 0.0)

                # 3. each target block can only attend to itself, other than the context block
                for mask in mask_idxs_pred[i]:
                    attention_mask[i, :, vis_pos[0] + mask, vis_pos[0] + mask[:, None]] = 0.0

                # 4. any future, textual token can attend to all target blocks
                attention_mask[i, :, vis_pos[1]:, vis_pos[0] + mask_idxs_pred[i].flatten()] = 0.0

        loss_ntp = 0.0
        loss_jepa = 0.0
        lm_inputs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=self.config.jepa_llm_layer_idx != -1
        )
        lm_outputs = lm.model(**lm_inputs)
        last_hidden_state = lm_outputs.last_hidden_state

        if self.config.jepa_llm_layer_idx != -1:
            # jepa_llm_layer_idx is the n-th layer of the LLM whose activation is used to compute the jepa loss
            # jepa_llm_layer_idx is not 0-index.
            # That is, jepa_llm_layer_idx=8 means that we keep the first 8 layers, i.e. layers[:8].
            # Since hidden_states includes the embeddings before layer 0, i.e. there are N_Layers + 1 hidden_states,
            # lm_outputs.hidden_states[8] correctly returns the activation of the 8-th layer whose layer index is 7
            jepa_hidden_state = lm_outputs.hidden_states[self.config.jepa_llm_layer_idx]
        else:
            jepa_hidden_state = last_hidden_state = lm_outputs.last_hidden_state
        del lm_outputs

        if labels is not None:
            logits = lm.get_output_embeddings()(last_hidden_state[:, :-1])
            labels = labels[:, 1:]
            loss_ntp = F.cross_entropy(logits.flatten(
                0, 1), labels.flatten(), ignore_index=-100)

        img_bsz = 0 if image_mask is None else image_mask.sum().item()
        force_jepa = False
        if not img_bsz:
            img_bsz = 1
            force_jepa = True
            image_mask[0] = True
            jepa_idxs = jepa_idxs.clone()
            image_mask[0] = True
        if img_bsz:
            vision_embeds_pred = vision_embeds_pred[image_mask]
            mask_idxs_pred = mask_idxs_pred[image_mask]
            visual_position_ids = visual_position_ids[image_mask]
            jepa_hidden_state = jepa_hidden_state[image_mask]
            H = jepa_hidden_state.size(-1)

            for i in range(n_pred_masks):
                preds = jepa_hidden_state.gather(1, (mask_idxs_pred[:, i] + visual_position_ids[:, 0:1]).unsqueeze(-1).expand(img_bsz, -1, H))
                preds = self.tgt_proj(preds)
                targets = vision_embeds_pred.gather(1, mask_idxs_pred[:, i].unsqueeze(-1).expand(img_bsz, -1, vision_embeds_pred.size(-1)))
                preds = preds.flatten(0, 1)
                targets = targets.flatten(0, 1)
                if self.config.jepa_loss_fn == 'smooth_l1':
                    targets = F.layer_norm(targets, (targets.size(-1),))
                    loss_jepa = loss_jepa + F.smooth_l1_loss(preds, targets)
                elif self.config.jepa_loss_fn == 'cos_sim':
                    loss_jepa = loss_jepa - F.cosine_similarity(preds, targets, dim=-1).mean()
                else:
                    raise ValueError(f"Unsupported jepa_loss_fn: {self.config.jepa_loss_fn}")
            loss_jepa = loss_jepa / n_pred_masks
            if force_jepa:
                logger.warning(f"[Rank {utils.get_rank()}] No image in the batch, will zero out the JEPA loss.")
                loss_jepa *= 0.0

        return MLLMJEPAOutput(
            loss=loss_jepa * self.config.jepa_loss_weight + loss_ntp,
            loss_jepa=loss_jepa.item() if isinstance(loss_jepa, Tensor) else loss_jepa,
            loss_ntp=loss_ntp.item() if isinstance(loss_ntp, Tensor) else loss_ntp
        )


class MLLMForCausalLM(MLLMPreTrainedModel, GenerationMixin):
    supports_gradient_checkpointing = True

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        putback_att_mask = False
        attention_mask = model_kwargs.pop('attention_mask', None)
        if attention_mask is None or attention_mask.ndim == 2:
            model_kwargs['attention_mask'] = attention_mask
        elif attention_mask.ndim == 4:
            putback_att_mask = True
            N, _, L = attention_mask.shape[:3]
            dtype, device = attention_mask.dtype, attention_mask.device
            attention_mask = torch.ones((N, L + 1), dtype=dtype, device=device)

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens)

        if putback_att_mask:
            model_kwargs['attention_mask'] = attention_mask
        return model_kwargs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if attention_mask is None:
            assert input_ids.size(0) == 1
            attention_mask = torch.ones_like(input_ids)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError(
                "`inputs_embeds` is not supported in generation")
        if 'position_ids' in kwargs:
            raise NotImplementedError(
                "`position_ids` is not supported in generation")

        inputs_embeds, attention_mask, _, position_ids, __ = self.embed_tokens_with_pixel_values(
            batch_input_ids=input_ids, batch_attention_mask=attention_mask, batch_pixel_values=pixel_values)

        return super().generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **kwargs
        )


__all__ = [
    "MLLMJEPAOutput",
    "MLLMJEPA"
]