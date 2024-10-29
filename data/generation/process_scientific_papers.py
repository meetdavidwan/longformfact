import json
import numpy as np
np.random.seed(42)

data = [json.loads(line) for line in open("scientific_papers/arxiv_test.txt")]

print(len(data))

data = [d for d in data if len(d["sections"]) == 5]

print(len(data))

np.random.shuffle(data)

data = data[:100]


out_data = []

# dict_keys(['article_id', 'article_text', 'abstract_text', 'labels', 'section_names', 'sections'])
for dat in data:
    out_data.append({
        "id": dat["article_id"],
        "section_names": dat["section_names"],
        "document": [" ".join(sec).strip() for sec in dat["sections"]],
        "summary": " ".join([t.replace("<S>", "").replace("</S>", "").strip() for t in dat["abstract_text"]]).strip(),
    })

with open("arxiv.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")

"""
Pubmed
"""

np.random.seed(42)

data = [json.loads(line) for line in open("scientific_papers/pubmed_test.txt")]

print(len(data))

data = [d for d in data if len(d["sections"]) == 5]

print(len(data))

np.random.shuffle(data)

data = data[:100]

out_data = []

# dict_keys(['article_id', 'article_text', 'abstract_text', 'labels', 'section_names', 'sections'])
for dat in data:
    out_data.append({
        "id": dat["article_id"],
        "section_names": dat["section_names"],
        "document": [" ".join(sec).strip() for sec in dat["sections"]],
        "summary": " ".join([t.replace("<S>", "").replace("</S>", "").strip() for t in dat["abstract_text"]]).strip(),
    })

with open("pubmed.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")