import copy
import transformers
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM


class CausalLMModel:
    def __init__(
        self,
        model_name,
        max_new_tokens=512,
        temperature=0.7,
        device="cuda",
        load_8bit=True,
        cache_dir=None
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_8bit,
            device_map="auto",
            cache_dir=cache_dir
        )

        if load_8bit:
            model = prepare_model_for_int8_training(model)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, config)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.accumulation_steps = 32
        self.current_step = 0
        self.losses = []

    def generate(self, input_text, print_prompt=False):
        prompt = input_text
        inputs = self.tokenizer(prompt, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                                      temperature=self.temperature)
        inputs_token_length = len(inputs.input_ids[0])
        new_tokens = outputs[0][inputs_token_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return text

    def learn(self, instruction: str, output: str, print_prompt=False):
        inputs_token_length = len(self.tokenizer(instruction, return_tensors='pt').input_ids[0])
        prompt = instruction + output
        inputs = self.tokenizer(prompt, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        labels = copy.deepcopy(inputs.input_ids)
        labels[0][:inputs_token_length] = -100  # Ignore index
        if print_prompt:
            print(labels)

        outputs = self.model(**inputs, labels=labels)
        self.current_step += 1
        self.losses.append(outputs.loss.item())
        outputs.loss.backward()
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return outputs