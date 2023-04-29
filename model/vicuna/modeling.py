import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from model.vicuna.compression import compress_module
from model.vicuna.conversation import conv_templates
from model.vicuna.conversation import SeparatorStyle

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from typing import List


def load_model(model_name, device, num_gpus,  load_8bit=False, debug=False, cache_dir=None):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            load_in_8bit=load_8bit,
            cache_dir=cache_dir,
            **kwargs
        )

    # if load_8bit:
    #     compress_module(model, device)

    # if (device == "cuda" and num_gpus == 1) or device == "mps":
    #     model.to(device)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_output(model, tokenizer, params, device,
                    context_len=2048):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    # output_ids = list(input_ids)
    output_ids = []

    max_src_len = context_len - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        pos = output.rfind(stop_str)
        if pos != -1:
            output = output[:pos]
            stopped = True

        if stopped:
            break

    del past_key_values
    # output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output


class Vicuna:
    def __init__(
            self,
            model_name,
            max_new_tokens=512,
            temperature=0.7,
            device='cuda',
            num_gpus='auto',
            load_8bit=True,
            debug=False,
            cache_dir='../vicuna/cache'
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        model, self.tokenizer = load_model(
            model_name,
            device,
            num_gpus,
            load_8bit,
            debug,
            cache_dir
        )

        # if load_8bit:
        #     model = prepare_model_for_int8_training(model)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(model, config)

        conv_template = 'v1'
        self.conv = conv_templates[conv_template].copy()

    def chat(self, input_text, print_prompt=False):


        self.conv.append_message(self.conv.roles[0], input_text)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        if print_prompt:
            print(prompt)
        skip_echo_len = len(prompt) + 1

        params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "stop": self.conv.sep if self.conv.sep_style == SeparatorStyle.SINGLE else self.conv.sep2,
        }

        output = generate_output(self.model, self.tokenizer, params, self.device)
        self.conv.messages.append(("Assistant", output))

        return output

    def clear_history(self):
        conv_template = 'v1'
        self.conv = conv_templates[conv_template].copy()

    def generate(self, input_text, print_prompt=False):

        conv_template = 'v1'
        conv = conv_templates[conv_template].copy()

        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if print_prompt:
            print(prompt)
        skip_echo_len = len(prompt) + 1

        params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        output = generate_output(self.model, self.tokenizer, params, self.device)

        return output
