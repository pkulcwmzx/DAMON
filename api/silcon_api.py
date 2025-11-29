import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .baseapi import BaseAPI
import requests

model_dict = {
    "Qwen": "Qwen/Qwen2.5-72B-Instruct", 
    "Qwen-32B": "Qwen/Qwen2.5-32B-Instruct", 
    "Qwen-14B": "Qwen/Qwen2.5-14B-Instruct",
    "Yi-34B": "01-ai/Yi-1.5-34B-Chat-16K", 
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct", 
    "Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B-Instruct", 
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    "Deepseek": "deepseek-ai/DeepSeek-V2.5", 
    "chatglm": "THUDM/chatglm3-6b",
    "glm4": "THUDM/glm-4-9b-chat", 
    "deepseek-v3": "Pro/deepseek-ai/DeepSeek-V3"
}

class SilconAPI(BaseAPI):
    def __init__(self, model_name, api_key, generation_config={}):
        super().__init__(generation_config)
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer " + api_key
        }
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.payload = {
            "model": model_dict[model_name],
            "messages": [
                {
                    "role": "user",
                    "content": ""
                }
            ],
            **self.generation_config
        }
        
    def generate_response(self, messages):
        self.payload["messages"] = messages
        try:
            response = requests.post(self.url, json=self.payload, headers=self.headers)
            data = response.json()
            response_string = data["choices"][0]["message"]["content"]
            if response_string != "error":
                return response_string

        except:
            print("Error in get LLM API!")
            return "error"

if __name__ == '__main__':
    pass