import json
import numpy as np

# load diversesumm
# data_file: DiverseSumm/data/diverse_summ.json
# ann_file: DiverseSumm/data/faithfulness_eval_results.json
data_file = "DiverseSumm/data/diverse_summ.json"
annotation_file = "DiverseSumm/data/faithfulness_eval_results.json"
data = json.load(open(data_file))
annotations = json.load(open(annotation_file))

id2data = {d["eid"]: d for d in data}

out_data = []
seen_ids = set()

for ann in annotations:
    doc = id2data[ann["event_id"]]

    doc = [d["content"] for d in doc["articles"]]
    
    if ann["summary_sentences"]:
        if ann["event_id"] in seen_ids:
            continue
        
        out_data.append({
                    "id": ann["event_id"],
                    "document": doc,
                })
        seen_ids.add(ann["event_id"])

with open("diversesumm.jsonl", "w") as f:
    for d in out_data:
        f.write(json.dumps(d) + "\n")