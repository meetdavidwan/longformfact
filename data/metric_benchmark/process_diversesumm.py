import json
import sys

# original diversesumm data
data = [json.loads(line) for line in open(sys.argv[1])]

for dat in data:
    document = dat.pop("document")
    documents = [doc["content"] for doc in document]
    document = "\n ==== \n".join(documents)
    dat["document"] = document
    dat["documents"] = documents

print(len(data))

print(data[0].keys())

with open("diversesumm.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")