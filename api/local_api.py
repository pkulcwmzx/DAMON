import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .baseapi import BaseAPI

# need to set your local model paths here
local_model_dict = {
    "llama3-8b-instruct": "/data2/pretrain/meta-llama-new/Meta-Llama-3.1-8B-Instruct",
    "llama2-13b-chat": "/data2/pretrain/meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-chat": "/data2/pretrain/meta-llama/Llama-2-7b-chat-hf", 
    "llama3-8b": "/data2/pretrain/meta-llama-new/Meta-Llama-3.1-8B",
    "Qwen-2.5-7b-instruct": "/data2/pretrain/Qwen/Qwen2.5-7B-Instruct"
}

class LlamaAPI(BaseAPI):
    def __init__(self, model_path, device, generation_config={}):
        super().__init__(generation_config)
        self.model_path = model_path
        self.device = torch.device(device)
        print('loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "USER: {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}"
                "ASSISTANT: {{ message['content'] }}\n"
                "{% endif %}"
                "{% endfor %}"
                "ASSISTANT:"
            )
        print('finish loading')

    def generate_response(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **self.generation_config)
        return self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
if __name__ == '__main__':
    model_path = "/data2/pretrain/meta-llama-new/Meta-Llama-3.1-8B-Instruct"
    local_model_generation_config = {
        "do_sample": False,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "use_cache": True,
        "max_new_tokens": 2048,
    }
    
    test_input = "hello world!"
    messages = [{"role": "user", "content": test_input}]
    model = LlamaAPI(model_path, device="cuda", generation_config=local_model_generation_config)
    output = model.generate_response(messages)
    print(output)