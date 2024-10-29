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

out_prefix = sys.argv[4]

# generating the original summary

document_splitter = "\n====\n"

for i, dat in tqdm(enumerate(data), total=len(data)):
    doc = dat["document"]
    document_str = document_splitter.join(doc).strip()
    prompt = generation_prompt.replace("[ARTICLES]", document_str)
    gen_summary = model.run(prompt)
    dat["generated_summary"] = gen_summary

with open(args.output_file, "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")