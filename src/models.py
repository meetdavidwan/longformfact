import transformers
import torch
from openai import OpenAI, OpenAIError
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class LLaMA8B:
    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def run(self, prompt):
        if type(prompt) is not list:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt

        outputs = self.pipeline(
            messages,
            max_new_tokens=1024,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        return outputs[0]["generated_text"][-1]["content"]

class LLaMA70B:
    def __init__(self):
        model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def run(self, prompt):
        if type(prompt) is not list:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt

        outputs = self.pipeline(
            messages,
            max_new_tokens=1024,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        return outputs[0]["generated_text"][-1]["content"]

class ChatGPT:
    def __init__(self):
        self.model_id = "gpt-3.5-turbo"
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.max_tokens = 1024
        self.client = OpenAI(
            api_key=self.API_KEY,
        )

    
    def run(self, prompt):
        out = None
        try:
            if type(prompt) is not list:
                messages = [
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = prompt
            outputs = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=1,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            out = outputs.choices[0].message.content
        except OpenAIError:
            print("OpenAI Error")
        
        return out

class ChatGPT16K(ChatGPT):
    def __init__(self):
        super().__init__()
        self.model_id = "gpt-3.5-turbo-16k"

class GPT4:
    def __init__(self):
        self.model_id = "gpt-4"
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(
            api_key=self.API_KEY,
        )
    
    def run(self, prompt):
        if type(prompt) is not list:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt
        out = None
        try:
            outputs = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=2048,
            )
            out = outputs.choices[0].message.content
        except OpenAIError:
            print("OpenAI Error")
        return out

class GPT4O(GPT4):
    def __init__(self):
        super().__init__()
        self.model_id = "gpt-4o"

class GPT4Turbo(GPT4):
    def __init__(self):
        super().__init__()
        self.model_id = "gpt-4-turbo"

class Mixtral:
    def __init__(self):
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    def run(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, truncation=True, max_length=32768, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            input_ids, 
            max_new_tokens=1024, 
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

class MixtralLarge:
    def __init__(self):
        model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    def run(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, truncation=True, max_length=32768, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            input_ids, 
            max_new_tokens=1024, 
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

class DummyMetric:
    def __init__(self):
        self.responses = ["Yes", "No"]
    
    def run(self, prompt):
        return np.random.choice(self.responses)


class DummyGeneration:
    def __init__(self):
        pass
    
    def run(self, prompt):
        return prompt