from models import *
import json
from tqdm import tqdm
from openai import OpenAIError
from collections import defaultdict
import sys
import argparse


import nltk
try:
    nltk.download('punkt')
except:
    pass

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, choices=["chatgpt16k","gpt4o","mixtral","mixtral_large","llama8b","llama70b", "dummy"])
parser.add_argument("--data_file", type=str)
parser.add_argument("--prompt_file", type=str)
parser.add_argument("--original_prompt_file", type=str)
parser.add_argument("--output_file", type=str)

args = parser.parse_args()

# get model name
model_name = args.model_name
if model_name == "chatgpt16k":
    model = ChatGPT16K()
elif model_name == "gpt4o":
    model = GPT4O()
elif model_name == "mixtral":
    model = Mixtral()
elif model_name == "mixtral_large":
    model = MixtralLarge()
elif model_name == "llama8b":
    model = LLaMA8B()
elif model_name == "llama70b":
    model = LLaMA70B()
elif model_name == "dummy":
    model = DummyGeneration()
else:
    raise ValueError("Invalid model name")

data = [json.loads(line) for line in open(args.data_file)]

generation_prompt = open(args.prompt_file).read()
generation_prompt_original = open(args.original_prompt_file).read()

out_prefix = sys.argv[4]

# generating the original summary

document_splitter = "\n====\n"

for i, dat in tqdm(enumerate(data), total=len(data)):
    doc = dat["document"]

    gen_summary = None
    generated_summaries = []
    for j, d in enumerate(doc):
        if j == 0:
            prompt = generation_prompt_original.replace("[ARTICLES]", d)
        else:
            prompt = generation_prompt.replace("[ARTICLES]", d).replace("[SUMMARY]", gen_summary)
        summ = model.run(prompt)

        if summ is not None:
            if "Summary:" in summ:
                summ = summ.replace("Summary:", "")
            if '"""' in summ:
                summ = summ.replace('"""', "")
            summ = summ.strip()
            gen_summary = summ
        generated_summaries.append(gen_summary)
    
    dat["generated_summaries"] = generated_summaries
    dat["generated_summary"] = gen_summary

with open(args.output_file, "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")