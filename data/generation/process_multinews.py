from datasets import load_dataset
import json
from collections import defaultdict

dataset = load_dataset("alexfabbri/multi_news")["test"]

data = []
data_total = defaultdict(list)

for i in range(len(dataset)):
    dat = dataset[i]

    doc = dat["document"].split(" ||||| ")
    doc = [d for d in doc if d.strip()]
    summary = dat["summary"]

    data_total[len(doc)].append({
        "document": doc,
        "summary": summary
    })


for k in sorted(data_total.keys()):
    print(k, len(data_total[k]))

"""
0 1
1 71
2 3022
3 1540
4 609
5 219
6 96
7 40
8 15
9 8
10 1
"""

with open("multinews_5_docs.jsonl", "w") as f:
    for d in data_total[5]:
        f.write(json.dumps(d) + "\n")