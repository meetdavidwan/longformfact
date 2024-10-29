import json
import sys
from tqdm import tqdm
from models import *
import numpy as np
import argparse

import nltk
try:
    nltk.download('punkt')
except:
    pass


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, choices=["chatgpt16k","gpt4o","mixtral","mixtral_large","llama8b","llama70b", "dummy"])
parser.add_argument("--data_file", type=str)
parser.add_argument("--document_merge_type", type=str, default="max")
parser.add_argument("--summary_merge_type", type=str, default="mean")
parser.add_argument("--output_file", type=str)

args = parser.parse_args()

# get model name
model_name = args.model_name
if model_name == "chatgpt16k":
    model = ChatGPT16K()
elif model_name == "gpt4o":
    model = GPT4O()
elif model_name == "mixtral_large":
    model = MixtralLarge()
elif model_name == "mixtral":
    model = Mixtral()
elif model_name == "llama8b":
    model = LLaMA8B()
elif model_name == "llama70b":
    model = LLaMA70B()
elif model_name == "dummy":
    model = DummyMetric()
else:
    raise ValueError("Invalid model name")

verification_prompt = """Document:
[ARTICLE]

Sentence:
[SUMMARY]

Determine if the sentence is factually consistent with the document provided above. A sentence is factually consistent if it can be entailed (either stated or implied) by the document. Please start your answer with “Yes.” or “No.” Please briefly explain the reason within 50 words."""


def parse_score(s):
    if s is None:
        return np.nan
    else:
        if s.lower().strip().startswith("yes"):
            score = 1
        elif s.lower().strip().startswith("no"):
            score = 0
        else:
            if "yes" in s.lower():
                score = 1
            elif "no" in s.lower():
                score = 0
            else:
                score = np.nan
        return score

merge_fn = {
    "full": np.nanmean,
    "max": np.nanmax,
    "mean": np.nanmean,
    "min": np.nanmin,
}

data = [json.loads(line) for line in open(args.data_file)]

output_prediction = []
output_raw = []

for dat in tqdm(data):
    if args.document_merge_type == "full":
        documents = [dat["document"]]
    else:
        documents = dat["documents"]

    if args.summary_merge_type == "full":
        summaries = [dat["summary"]]
    else:
        summary = dat["summary"]
        summaries = []
        if summary is not None:
            summary = summary.strip()
            summaries = nltk.sent_tokenize(summary)
            summaries = [summ for summ in summaries if len(summ) > 0]

    responses_raw = []
    responses = []
    for doc in documents:
        responses_doc = []
        responses_doc_raw = []
        for summ in summaries:
            response = model.run(verification_prompt.replace("[ARTICLE]", doc).replace("[SUMMARY]", summ))
            responses_doc_raw.append(response)
            responses_doc.append(parse_score(response))
        responses.append(responses_doc)
        responses_raw.append(responses_doc_raw)
    
    # output always in the shape of (num_doc, num_summ)
    doc_merge_fn = merge_fn[args.document_merge_type]
    summ_merge_fn = merge_fn[args.summary_merge_type]
    score = summ_merge_fn(doc_merge_fn(np.array(responses), axis=0))
    output_prediction.append(score)
    output_raw.append(responses_raw)

with open(args.output_file, "w") as f:
    json.dump(output_prediction, f, indent=4)

    if "json" in args.output_file and "jsonl" not in args.output_file:
        with open(args.output_file.replace("json","raw"), "w") as f:
            json.dump(output_raw, f, indent=4)
