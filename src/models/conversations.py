from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List
from torch import Tensor
import torch
import src.custom_utils as utils

logger = utils.get_logger()


DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IM_START_TOKEN = '<img_start>'
DEFAULT_IM_END_TOKEN = '<img_end>'

VICUNA_V1_JINJA_TEMPLATE = """
    {{- "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. " -}}
    {% for message in messages %}
        {% if message['role'] == 'user' %}
            {{- 'USER: '+ message['content'] -}}
        {% elif message['role'] == 'assistant' %}
            {{- ' ASSISTANT: ' + message['content'] + eos_token -}}
        {% elif message['role'] == 'system' %}
            {{- message['content'] -}}
        {% else %}
            {% set error_msg = "Unknown role " + message['role'] + " in the conversation." %}
            {{ error_msg | error }}
        {% endif %}
    {% endfor %}
    {% if add_generation_prompt %}
        {{- ' ASSISTANT: ' -}}
    {% endif %}
"""


class CONV_TEMPLATES(StrEnum):
    PLAIN = 'plain'
    VICUNA_V1 = 'vicuna_v1'
    LLAMA3_2 = 'llama3_2'
    GEMMA2 = 'gemma2'
    QWEN2 = 'qwen2'
    MISTRAL = 'mistral'


@dataclass
class Conversation:
    '''
    - A `role` is defined as `[<role_start>]ROLE[<role_end>]`.
    - A `sep` is defined as the string after a role + message: `[<role_start>]ROLE[<role_end>]Here is the message<sep>`. 
     Each role has its own separator.
    '''
    roles: Dict[str, str]
    seps: Dict[str, str] = field(default_factory=lambda: defaultdict(str))
    messages: List[str] = field(default_factory=list)
    sys_prompt: str = ''
    img_tok: str = DEFAULT_IMAGE_TOKEN
    bos_tok: str = ''
    eos_tok: str = ''
    eot_tok: str | None = None
    assistant_toks: List[str] = field(default_factory=list)
    add_bos: bool = False

    def __post_init__(self):
        if self.eot_tok is None:
            self.eot_tok = self.seps['assistant'].rstrip()

    def new_empty(self):
        return type(self)(
            roles=self.roles,
            seps=self.seps,
            messages=[],
            sys_prompt=self.sys_prompt,
            img_tok=self.img_tok,
            bos_tok=self.bos_tok,
            eos_tok=self.eos_tok,
            eot_tok=self.eot_tok,
            assistant_toks=self.assistant_toks,
            add_bos=self.add_bos
        )
    
    def reset(self):
        self.messages = []

    def add_message(self, msg: Dict[str, str]):
        role = msg['role']
        ret = self.roles[role]
        content = msg.get('content')
        if content is None and role == 'system':
            content = self.sys_prompt
        if content:
            ret += content
            ret += self.seps[role]
        self.messages.append(ret)

    def add_sys_prompt(self):
        self.add_message(dict(role='system'))

    def get_prompt(self, add_generation_prompt=False, add_bos=None):
        add_bos = self.add_bos if add_bos is None else add_bos
        bos_tok = self.bos_tok if add_bos else ''
        if add_generation_prompt:
            self.add_message(dict(role='assistant'))
            ret = bos_tok + ''.join(self.messages)
        else:
            ret = bos_tok + ''.join(self.messages) + self.eos_tok
        return ret
    
    def get_labels(self, input_ids: Tensor, tokenized_text: List[str], ignore_index: int = -100):
        labels = [ignore_index] * len(input_ids)
        assistant_len = len(self.assistant_toks)
        i = 0
        maxlen = len(tokenized_text)
        while i < maxlen:
            tok = tokenized_text[i]
            if tok == self.assistant_toks[0]:
                if tokenized_text[i:i + assistant_len] == self.assistant_toks:
                    i += assistant_len
                    while i < maxlen and tokenized_text[i] != self.eot_tok:
                        labels[i] = input_ids[i]
                        i += 1
                    if i < maxlen:
                        labels[i] = input_ids[i]
            if i < maxlen:
                i += 1
        if i != len(input_ids):
            logger.warning(f"Tokenization mismatch: expected {len(input_ids)} tokens, but found {i}. Setting all labels to `ignore_index` {ignore_index}.")
            labels = [ignore_index] * len(input_ids)
        return torch.tensor(labels, dtype=input_ids.dtype)


@dataclass
class VicunaConversation(Conversation):
    def add_message(self, msg: Dict[str, str]):
        role = msg['role']
        ret = self.roles[role]
        content = msg.get('content')
        if content is None and role == 'system':
            content = self.sys_prompt
        elif content and role == 'user':
            content = content.replace(self.img_tok, self.img_tok + '▁')
        if content:
            ret += content
            ret += self.seps[role]
        self.messages.append(ret)

    def get_prompt(self, add_generation_prompt=False, add_bos=None):
        ret = super().get_prompt(add_generation_prompt, add_bos)
        if add_generation_prompt:
            ret = ret.rstrip()
        return ret


CONV_MAPPING = {
    CONV_TEMPLATES.VICUNA_V1: VicunaConversation(
        roles={
            'system': '',
            'user': 'USER: ',
            'assistant': ' ASSISTANT: '
        },
        seps={
            'system': ' ',
            'user': '',
            'assistant': '</s>'
        },
        sys_prompt='A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.',
        img_tok=DEFAULT_IMAGE_TOKEN,
        eos_tok='',
        assistant_toks=['▁A', 'SS', 'IST', 'ANT', ':']
    ),

    CONV_TEMPLATES.LLAMA3_2: Conversation(
        roles={
            'system': '<|start_header_id|>system<|end_header_id|>\n\n',
            'user': '<|start_header_id|>user<|end_header_id|>\n\n',
            'assistant': '<|start_header_id|>assistant<|end_header_id|>\n\n'
        },
        seps=defaultdict(lambda: '<|eot_id|>'),
        sys_prompt='Cutting Knowledge Date: December 2023\nToday Date: 22 Jul 2025\n\n',
        img_tok=DEFAULT_IMAGE_TOKEN,
        bos_tok='<|begin_of_text|>',
        eos_tok='<|end_of_text|>',
        assistant_toks=['<|start_header_id|>', 'assistant', '<|end_header_id|>', 'ĊĊ'],
        add_bos=False
    ),    

    CONV_TEMPLATES.GEMMA2: Conversation(
        roles={
            'system': '',
            'user': '<start_of_turn>user\n',
            'assistant': '<start_of_turn>model\n'
        },
        seps={
            'system': '',
            'user': '<end_of_turn>\n',
            'assistant': '<end_of_turn>\n'
        },
        sys_prompt='',
        img_tok=DEFAULT_IMAGE_TOKEN,
        bos_tok='<bos>',
        eos_tok='',
        eot_tok='<end_of_turn>',
        assistant_toks=['model', '\n'],
        add_bos=False
    ),

    CONV_TEMPLATES.QWEN2: Conversation(
        roles={
            'system': '<|im_start|>system\n',
            'user': '<|im_start|>user\n',
            'assistant': '<|im_start|>assistant\n'
        },
        seps={
            'system': '<|im_end|>\n',
            'user': '<|im_end|>\n',
            'assistant': '<|im_end|>\n'
        },
        sys_prompt='You are a helpful assistant.',
        img_tok=DEFAULT_IMAGE_TOKEN,
        bos_tok='',
        eos_tok='',
        assistant_toks=['<|im_start|>', 'assistant', chr(266)]
    ),      

    CONV_TEMPLATES.MISTRAL: Conversation(
        roles={
            'system': '',
            'user': '[INST]',
            'assistant': ''
        },
        seps={
            'system': '',
            'user': '[/INST]',
            'assistant': '</s>'
        },
        sys_prompt='',
        img_tok=DEFAULT_IMAGE_TOKEN,
        bos_tok='',
        eos_tok='',
        assistant_toks=['[/INST]']
    ),

}


if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    MODEL_NAME_OR_PATHS = (
        'google/gemma-2-2b-it',
        'meta-llama/Llama-3.2-3B-Instruct',
        'lmsys/vicuna-7b-v1.5',
        'mistralai/Ministral-8B-Instruct-2410',
        'Qwen/Qwen2-7B-Instruct'
    )
    conv_modes = (
        'gemma2',
        'llama3_2',
        'vicuna_v1'
        'mistral',
        'qwen2',
    )

    for model_name_or_path, conv_mode in zip(MODEL_NAME_OR_PATHS, conv_modes):

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        if tokenizer.chat_template is None:
            tokenizer.chat_template = VICUNA_V1_JINJA_TEMPLATE

        conv = CONV_MAPPING[conv_mode].new_empty()
        conv.add_message(dict(role='system'))
        
        msgs = [
            # dict(role='system', content='sysprompt test.'),
            dict(role='user', content='<image>Please describe the image.'),
            dict(role='assistant', content='The image depicts a dog.'),
            dict(role='user', content='What is the color of the dog?'),
            dict(role='assistant', content='Brown.')
        ]

        for msg in msgs:
            conv.add_message(msg)

        prompt = conv.get_prompt()
        gt_prompt = tokenizer.apply_chat_template([msgs], add_generation_prompt=False, tokenize=False)[0]

        print(prompt)

        input_ids = tokenizer([prompt], return_tensors='pt').input_ids[0]
        gt_input_ids = tokenizer.apply_chat_template([msgs], add_generation_prompt=False, tokenize=True, return_tensors='pt')[0]

        labels = conv.get_labels(input_ids, tokenizer.convert_ids_to_tokens(input_ids))
        gt_labels = conv.get_labels(gt_input_ids, tokenizer.convert_ids_to_tokens(gt_input_ids))

        print(tokenizer.batch_decode(labels[labels != -100]))
        print(tokenizer.batch_decode(gt_labels[gt_labels != -100]))

        gen_prompt = conv.get_prompt(add_generation_prompt=True)
        gen_gt_prompt = tokenizer.apply_chat_template([msgs], add_generation_prompt=True, tokenize=False)[0]        
        
        gen_input_ids = tokenizer([gen_prompt], return_tensors='pt').input_ids[0]
        gen_gt_input_ids = tokenizer.apply_chat_template([msgs], add_generation_prompt=True, tokenize=True, return_tensors='pt')[0]
        
        ...
        # assert prompt == gt_prompt
        # assert (tokenizer([prompt], return_tensors='pt').input_ids == tokenizer([gt_prompt], return_tensors='pt').input_ids).all().item()

    
