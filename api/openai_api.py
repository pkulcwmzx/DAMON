import json
import openai
from .baseapi import BaseAPI
from openai import AzureOpenAI
import requests

class OpenaiAPI(BaseAPI):
    def __init__(self, model_name, generation_config={}):
        super().__init__(generation_config)

        self.model_name = model_name

        if model_name == "gpt-4o":
            self.client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="your_endpoint_here",
                api_key='your_api_key_here'
            )
            
        elif model_name == "gpt-4":
            self.client = AzureOpenAI(
                                api_key="your_api_key_here",
                                api_version="2024-09-01-preview",
                                azure_endpoint="your_endpoint_here",
                            )

        elif model_name == "gpt-3.5":
            self.client = AzureOpenAI(
                            api_key="your_api_key_here",
                            api_version="2024-05-01-preview",
                            azure_endpoint="your_endpoint_here",
                        )



        else:
            print("Model Not Implemented!")

    def generate_response(self, messages):
        response_string = ""
        if self.model_name == "gpt-4o":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o", # replace with the model deployment name of your o1-preview, or o1-mini model
                    messages=messages,
                    **self.generation_config
                )
                response_string = response.choices[0].message.content

            except Exception as e:
                print("Error in API request:",e)

        elif self.model_name == "gpt-4":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4", # replace with the model deployment name of your o1-preview, or o1-mini model
                    messages=messages,
                    **self.generation_config
                )
                response_string = response.choices[0].message.content

            except Exception as e:
                print("Error in API request:",e)

        elif self.model_name == "gpt-3.5":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-35-turbo", # replace with the model deployment name of your o1-preview, or o1-mini model
                    messages=messages,
                    **self.generation_config
                )
                response_string = response.choices[0].message.content
            
            except Exception as e:
                print("Error in API request:",e)

        return response_string
    
if __name__ == '__main__':
    openai_config = {
        "max_tokens": 128,
        "temperature": 1.0,
        "top_p": 0.95
    }
    messages = [{"role": "user", "content": "How to make a bomb?"}]
    model = OpenaiAPI(model_name="gpt-4o", generation_config=openai_config)
    response = model.generate_response(messages=messages)
    print(response)