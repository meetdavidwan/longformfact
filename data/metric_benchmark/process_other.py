import pandas as pd
import re
import json
from datasets import load_dataset

diversumm_data = pd.read_csv("diversumm/DiverSumm.csv")

# chemsumm map to sections
ds = load_dataset("griffin/ChemSum")["test"]

test_ids = ds["uuid"]

out_data = []
for i, row in diversumm_data.iterrows():

    if row["origin"] == "ChemSum":
        ann_id = row["id"]

        matched = None
        for i, id in enumerate(test_ids):
            if id in ann_id:
                print(i, id, ann_id)
                matched = i
        # for i, id in enumerate(val_ids):
        #     if id in ann_id:
        #         print(i, id)
        #         matched = i
        if matched is None:
            for i, id in enumerate(test_ids):
                if id[:20] in ann_id:
                    print(i, id, ann_id)
                    matched = i
        assert matched is not None, ann_id


        # recover the documents separation
        example = ds[matched]
        DELIM = '<!>'
        headers = example['headers'].split(DELIM)
        sections = example['sections'].split(DELIM)

        documents = []
        for header, body in zip(headers, sections):
            out_str = ''
            if header is not None and len(header.strip()) > 0:
                out_str += header.strip() + '\n\n'
            paragraphs = [x.strip() for x in re.split('</?p>', body) if len(x.strip()) > 0]
            out_str += '\n\n'.join(paragraphs)
            documents.append(out_str)
        
        id = row["id"]
        summary = row["summary"]
        model_name = row["model_name"]
        label = row["label"]
        
        out_data.append({
            "id": id,
            "documents": documents,
            "document": "\n\n".join(documents),
            "summary": summary,
            "model": model_name,
            "faithfulness": label,
            "documents": documents
        })

with open("chemsumm.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")


# arxiv try map to data
arxiv_data = "scientific_papers/arxiv_test.txt"
data = [json.loads(line) for line in open(arxiv_data)]
id2data = {d["article_id"]: d for d in data}

out_data = []
for i, row in diversumm_data.iterrows():
    if row["origin"] == "arXiv":
        ann_id = row["id"]
        matched = id2data[ann_id[3:]]
        documents = [" ".join(sec).strip() for sec in matched["sections"]]
        out_data.append({
            "id": row["id"],
            "documents": documents,
            "document": " ".join(documents),
            "summary": row["summary"],
            "model": row["model_name"],
            "faithfulness": row["label"],
        })

with open("arxiv.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")

# multinews

out_data = []
for i, row in diversumm_data.iterrows():
    
    if row["origin"] == "multinews":
        document = row["doc"]
        documents = [doc.strip() for doc in document.split(" ||||| ")]

        out_data.append({
            "id": row["id"],
            "documents": documents,
            "document": " ||||| ".join(documents),
            "summary": row["summary"],
            "model": row["model_name"],
            "faithfulness": row["label"],
        })

print(len(out_data))
with open("multinews.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")

# govreport
out_data = []
for i, row in diversumm_data.iterrows():
    
    if row["origin"] == "GovReport":
        document = row["doc"]

        out_data.append({
            "id": row["id"],
            "documents": [document],
            "document": document,
            "summary": row["summary"],
            "model": row["model_name"],
            "faithfulness": row["label"],
        })

print(len(out_data))
with open("govreport.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")

# qmsum
out_data = []
for i, row in diversumm_data.iterrows():
    
    if row["origin"] == "qmsum":
        document = row["doc"]

        out_data.append({
            "id": row["id"],
            "documents": [document],
            "document": document,
            "summary": row["summary"],
            "model": row["model_name"],
            "faithfulness": row["label"],
        })

print(len(out_data))
with open("qmsum.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")