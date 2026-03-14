from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoModel
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from .configuration_mllm import MLLMConfig
import torch
import torch.nn as nn
import src.custom_utils as utils
from torch import Tensor
from src.models.conversations import VICUNA_V1_JINJA_TEMPLATE
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from src.models.model_utils import build_vision_encoder, build_projector

logger = utils.get_logger()


class MLLMPreTrainedModel(PreTrainedModel):
    config_class = MLLMConfig
    base_model_prefix = 'mllm'
    _tied_weights_keys = ["language_model.lm_head.weight"]

    def __init__(
        self,
        config: MLLMConfig,
        language_model: Optional[AutoModelForCausalLM] = None,
        vision_model: Optional[PreTrainedModel] = None,
        projector: Optional[nn.Module] = None,
        lm_kwargs: Dict = {},
        **kwargs
    ):
        if language_model is None:
            attn_implementation = lm_kwargs.pop('attn_implementation', None)
            if attn_implementation is None:
                attn_implementation = config.text_config._attn_implementation if config.text_config._attn_implementation else 'eager'
            if utils.DEBUG_LOAD_FIRST_LLM_LAYERS:
                config.text_config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_LLM_LAYERS
                logger.debug(
                    f"Debug mode: forcing text_config.num_hidden_layers to {config.text_config.num_hidden_layers}")
            lm_kwargs = dict(config=config.text_config, attn_implementation=attn_implementation, **lm_kwargs)
            if config.text_config.name_or_path:
                language_model = AutoModelForCausalLM.from_pretrained(
                    config.text_config.name_or_path, **lm_kwargs)
            else:
                language_model = AutoModelForCausalLM.from_config(**lm_kwargs)
            logger.info(f"attn_implementation: {attn_implementation}")
        config.text_config = language_model.config

        if vision_model is None:
            if config.vision_config.name_or_path:
                vision_model = build_vision_encoder(
                    config.vision_config.name_or_path)
            else:
                vision_model = AutoModel.from_config(config.vision_config)
        else:
            config.vision_config = vision_model.config

        super().__init__(config, **kwargs)

        self.language_model = language_model
        self.vision_model = vision_model

        if projector is None:
            projector = build_projector(
                config.projector_type,
                input_size=config.projector_input_size,
                output_size=config.text_config.hidden_size,
                bias=config.projector_bias
            )
        self.projector = projector.to(self.language_model.dtype)

        self.tie_weights()

    def get_projector(self):
        return self.projector

    def get_vision_model(self):
        return self.vision_model

    def get_language_model(self) -> AutoModelForCausalLM:
        return self.language_model

    def get_lm_head(self):
        return self.language_model.lm_head

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def init_vision_tokenizer(self, tokenizer, resize_embeds: bool = False):
        ret = tokenizer.add_tokens(
            [self.config.image_token], special_tokens=True)
        assert ret in [0, 1]
        if ret:
            image_token_id = tokenizer.convert_tokens_to_ids(
                self.config.image_token)
            assert self.config.image_token_id in [None, image_token_id]
            self.config.image_token_id = image_token_id
            logger.info(
                f'Added special token {self.config.image_token} to the tokenizer')
        else:
            logger.info(
                f'Special token {self.config.image_token} already exists in the tokenizer')

        if tokenizer.chat_template is None:
            tokenizer.chat_template = VICUNA_V1_JINJA_TEMPLATE
            test_prompt = tokenizer.apply_chat_template([
                dict(role='user', content=f"{self.config.image_token}Briefly describe the image."),
                dict(role='assistant', content='The image shows a ...'),
                dict(role='user', content='Please repeat it in Italian.')
            ], add_generation_prompt=True, tokenize=False)
            logger.info(
                f"No chat_template found in the tokenizer. Defaulting to `VICUNA_V1_JINJA_TEMPLATE`. "
                "\n***** CHAT EXAMPLE *****\n"
                f"{test_prompt}"
                "\n************************\n"
            )

        if resize_embeds:
            lm = self.get_language_model()
            if lm.get_input_embeddings().weight.size(0) < len(tokenizer):
                mean_resizing = False
                lm.resize_token_embeddings(
                    len(tokenizer), pad_to_multiple_of=64, mean_resizing=mean_resizing)
                logger.info('Resized the language model token embeddings')

    def init_image_processor(self, image_processor, img_size: Optional[int] = None):
        if img_size is None:
            img_size = self.config.img_size
        if img_size:
            if hasattr(image_processor, 'size'):
                if isinstance(image_processor.size, dict):
                    if 'shortest_edge' in image_processor.size:
                        image_processor.size['shortest_edge'] = img_size
                elif isinstance(image_processor.size, int):
                    image_processor.size = img_size

            if hasattr(image_processor, 'crop_size'):
                if isinstance(image_processor.crop_size, dict):
                    if 'height' in image_processor.crop_size and 'width' in image_processor.crop_size:
                        image_processor.crop_size['height'] = img_size
                        image_processor.crop_size['width'] = img_size
                elif isinstance(image_processor.crop_size, int):
                    image_processor.crop_size = img_size

            logger.info(f"Modified image processor:\n{image_processor}")

    def visual_model_forward(self, pixel_values):
        # TODO: use different layers from the visual encoder, keep the CLS, etc.
        if self.config.vision_layer_idx == -1:
            ret = self.vision_model(
                pixel_values=pixel_values).last_hidden_state[:, self.config.skip_left_visual_tokens:]
        else:
            ret = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True).hidden_states[self.config.vision_layer_idx][:, self.config.skip_left_visual_tokens:]
        return ret.to(self.dtype)

    def get_visual_embeds(self, pixel_values):
        return self.projector(self.visual_model_forward(pixel_values))

    def prepare_causal_mask_with_full_mask_on_image_tokens(
        self,
        attention_mask_2d=None,
        visual_position_ids=None,
        has_image=None
    ) -> Union[Tensor, None]:
        attn_mask_4d = None
        if has_image:
            N, L = attention_mask_2d.shape
            dtype, device = self.dtype, self.device

            # start with a causal mask
            attn_mask_4d = torch.tril(torch.ones(
                (L, L), dtype=dtype, device=device)).unsqueeze(0).expand(N, -1, -1).clone()

            # add full mask among visual tokens
            for i, imgs_pos in enumerate(visual_position_ids):
                for img_pos in imgs_pos:
                    attn_mask_4d[i, img_pos[0]:img_pos[1],
                                 img_pos[0]:img_pos[1]] = 1

            # add padding and make 4d
            # Note that we only mask out the padding columns, i.e. we prevent tokens from attending to padding tokens,
            # while we do not mask out the padding rows, i.e. padding tokens can still attend to non-padding tokens,
            # because the loss should not be computed on padding tokens anyway.
            attn_mask_4d = (
                attn_mask_4d * attention_mask_2d.unsqueeze(1)).unsqueeze(1)

            attn_mask_4d = torch.where(attn_mask_4d == 1.0, 0.0, float('-inf')).to(dtype)
            # from PIL import Image
            # from pathlib import Path
            # attn_mask_4d[attn_mask_4d == float('-inf')] = 1
            # for i in range(attn_mask_4d.size(0)):
            #     show_inv_attn_mask = ((1 - attn_mask_4d[i, 0]) * 255).to(torch.uint8).cpu().numpy()
            #     show_inv_attn_mask = Image.fromarray(show_inv_attn_mask)
            #     p = Path('.vscode').joinpath('attn_mask_check')
            #     p.mkdir(parents=True, exist_ok=True)
            #     show_inv_attn_mask.save(p.joinpath(f"attn_mask_{i}.png"))
        else:
            attn_mask_4d = attention_mask_2d
        return attn_mask_4d
    
    def embed_tokens(
        self,
        batch_input_ids: Tensor,
        batch_attention_mask: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
        batch_visual_embeds: Optional[Tensor] = None,
        batch_position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, List[List[List[int]]]]:
        E = self.get_input_embeddings()
        if batch_attention_mask is None:
            batch_attention_mask = torch.ones_like(batch_input_ids)
        if batch_visual_embeds is None:
            return (
                E(batch_input_ids),
                batch_attention_mask,
                batch_labels,
                batch_position_ids,
                [[] for _ in batch_input_ids]
            )

        IMG_TOK = self.config.image_token_id
        has_labels = batch_labels is not None
        all_input_ids = []
        all_attention_mask = []
        all_labels = [] if has_labels else None
        all_visual_position_ids = []
        toks_dtype, device = batch_input_ids.dtype, batch_input_ids.device

        if batch_input_ids.size(0) < batch_visual_embeds.size(0):
            batch_visual_embeds = batch_visual_embeds.view(
                batch_input_ids.size(0), -1, *batch_visual_embeds.shape[1:])
        if batch_visual_embeds.ndim == 3:
            # (N, C, L) -> (N, ImgsPerSample, C, L)
            batch_visual_embeds = batch_visual_embeds.unsqueeze(1)
        n_vembeds = batch_visual_embeds.size(2)

        # first pass - get visual token positions and replace image tokens with fuzzy ones, e.g. 0
        for i, input_ids in enumerate(batch_input_ids):
            curr_toks = []
            curr_vis_pos = []
            for j, tok in enumerate(input_ids):
                if tok == IMG_TOK:
                    curr_vis_pos.append([j, -1])
                    curr_toks.append(torch.full(
                        (1,), fill_value=0, dtype=toks_dtype, device=device))
                else:
                    curr_toks.append(input_ids[j:j + 1])
            all_input_ids.append(torch.cat(curr_toks))
            all_visual_position_ids.append(curr_vis_pos)

        # embed txt tokens
        all_input_ids = torch.stack(all_input_ids)
        all_inputs_embeds = E(all_input_ids)

        # second pass -> add visual embeds, handle mask and labels, add last position of each image sequence
        all_mm_inputs_embeds = []
        all_attention_mask = []
        all_labels = []

        for i, vis_pos in enumerate(all_visual_position_ids):
            chunked_embeds = []
            chunked_attn_mask = []
            chunked_labels = []
            txt_start = 0
            n_img = 0
            for vp in vis_pos:
                chunked_embeds.extend([
                    all_inputs_embeds[i, txt_start:vp[0]],
                    batch_visual_embeds[i, n_img],
                ])
                chunked_attn_mask.extend([
                    batch_attention_mask[i, txt_start:vp[0]],
                    torch.ones((n_vembeds,), dtype=toks_dtype, device=device)
                ])
                if has_labels:
                    chunked_labels.extend([
                        batch_labels[i, txt_start:vp[0]],
                        torch.full((n_vembeds,), fill_value=-100,
                                   dtype=toks_dtype, device=device)
                    ])
                txt_start = vp[0] + 1
                vp[0] += n_img * n_vembeds
                vp[1] = vp[0] + n_vembeds
                n_img += 1
            if n_img == 0:
                vis_pos.append([0, 0])
            if txt_start < len(batch_attention_mask[i]):
                last_no_pad_mask = batch_attention_mask[i, txt_start:].nonzero(
                )
                # if training with jepa with text conditioning, there is nothing after the visual embeds
                # and what comes before has already been added
                if last_no_pad_mask.numel():
                    txt_end = txt_start + last_no_pad_mask[-1].item() + 1
                    chunked_embeds.append(
                        all_inputs_embeds[i, txt_start:txt_end])
                    chunked_attn_mask.append(
                        batch_attention_mask[i, txt_start:txt_end])
                    if has_labels:
                        chunked_labels.append(
                            batch_labels[i, txt_start:txt_end])
                    if not n_img and self.training:
                        # ensure that fuzzy batch_visual_embeds are accessed by this rank
                        # to avoid NCCL mismatch collective errors and hangs
                        chunked_embeds.append(batch_visual_embeds[i, 0])
                        chunked_attn_mask.append(torch.zeros(
                            (n_vembeds,), dtype=toks_dtype, device=device))
                        if has_labels:
                            chunked_labels.append(torch.full(
                                (n_vembeds,), fill_value=-100, dtype=toks_dtype, device=device))

            all_mm_inputs_embeds.append(torch.cat(chunked_embeds))
            all_attention_mask.append(torch.cat(chunked_attn_mask))
            if has_labels:
                all_labels.append(torch.cat(chunked_labels))

        all_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            all_mm_inputs_embeds, True, 0.)
        del all_mm_inputs_embeds
        all_attention_mask = torch.nn.utils.rnn.pad_sequence(
            all_attention_mask, True, 0)
        if has_labels:
            all_labels = torch.nn.utils.rnn.pad_sequence(
                all_labels, True, -100)
        else:
            all_labels = None

        # TODO: check if position_ids are needed

        if self.config.bidir_visual_attn:
            all_attention_mask = self.prepare_causal_mask_with_full_mask_on_image_tokens(
                all_attention_mask, all_visual_position_ids, has_image=True)

        return all_inputs_embeds, all_attention_mask, all_labels, batch_position_ids, all_visual_position_ids

    def embed_tokens_with_pixel_values(
        self,
        batch_input_ids: Tensor,
        batch_attention_mask: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
        batch_pixel_values: Optional[Tensor] = None,
        batch_position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[List[List[int]]]]:
        batch_visual_embeds = None
        if batch_pixel_values is not None:
            batch_visual_embeds = self.get_visual_embeds(batch_pixel_values)
        return self.embed_tokens(batch_input_ids, batch_attention_mask, batch_labels, batch_visual_embeds, batch_position_ids)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache,
                                        List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds, attention_mask, labels, position_ids, visual_position_ids = self.embed_tokens_with_pixel_values(
                input_ids, attention_mask, labels, pixel_values, position_ids)
            input_ids = None
        else:
            visual_position_ids = None

        # assuming the LLM uses RoPE, we do not need, `position_ids` for most cases.
        # If we do not set it to None, then there will be problem in `prepare_inputs_for_generation` if using full attention mask on image tokens
        position_ids = None
        
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )

        if return_dict:
            outputs['visual_position_ids'] = visual_position_ids

        return outputs


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
    "MLLMPreTrainedModel",
    "MLLMForCausalLM"
]