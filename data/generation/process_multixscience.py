from datasets import load_dataset
import json
from collections import defaultdict
import numpy as np

dataset = load_dataset("yaolu/multi_x_science_sum")["test"]


"""
{abstract: text of paper abstract
aid: arxiv id
mid: microsoft academic graph id
ref_abstract:
{
abstract: text of reference paper (cite_N) abstract
cite_N: special cite symbol,
mid: reference paper's (cite_N) microsoft academic graph id
},
related_work: text of paper related work
}
"""

data = []

data_total = defaultdict(list)

for i in range(len(dataset)):
    dat = dataset[i]

    id = dat["aid"]
    summary = dat["related_work"]
    abstract = dat["abstract"]

    related_documents = dat["ref_abstract"]
    cite_N = related_documents["cite_N"]
    abstracts = related_documents["abstract"]
    assert len(cite_N) == len(abstracts)

    # remove empty instances
    cite_N, abstracts = zip(*[(c, a) for c, a in zip(cite_N, abstracts) if a.strip()])
    
    document_ids = cite_N
    document = abstracts

    data_total[len(document)].append({
        "id": id,
        "document_ids": document_ids,
        "document": document,
        "summary": summary,
        "abstract": abstract
    })


for k in sorted(data_total.keys()):
    print(k, len(data_total[k]))

test_data = data_total[4]
np.random.seed(42)
np.random.shuffle(test_data)

for doc in test_data[0]["document"]:
    print(doc)
    print("---")

aeuae

"""
1 1308
2 1125
3 726
4 581
5 530
6 280
7 162
8 136
9 84
10 53
11 39
12 17
13 20
14 11
15 8
16 7
17 5
18 1
"""

with open("multixscience_5_docs.jsonl", "w") as f:
    # abstract of the original paper also counts as one
    for d in data_total[4]:
        f.write(json.dumps(d) + "\n")