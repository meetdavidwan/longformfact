import argparse
import tqdm
import json
import random
import numpy as np

# from longeval.metrics.plot_violin_utils import plot_data
from results_utils import read_coarse_data, compute_fine_scores, get_story_list, read_mturk_data, read_squality_coarse_data, get_squality_data_stats
from metric_utils import get_metric, get_metrics_list, get_correlation_intervals
from preprocessing_utils import jsonl_read

parser = argparse.ArgumentParser()
parser.add_argument('--fine_src_files', default="pubmed_annotations/pubmed*fine/*", type=str)
parser.add_argument('--coarse_src_files', default="pubmed_annotations/pubmed*coarse/*", type=str)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--num_keep', default=1000, type=int,
                    help=("Number of FINE units to keep for each summary."
                          "A large number like 1000 will include all units"))
parser.add_argument('--corr_type', default="pearson", type=str)
parser.add_argument('--skip_model', default="human", type=str,
                    help=("Skip a model (e.g. human) when computing the correlation."
                          "Set to human by default since reference-based metrics are being used."
                          "Since PubMed has just one reference, reference-based metrics would give"
                          "a score of 1.0 for the human model."))
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--output_dir', default="outputs/metric_corrs", type=str)
parser.add_argument('--compare_to', default="references", type=str)
parser.add_argument('--pubmed_file', default="data/scientific_papers/pubmed_test.txt")
args = parser.parse_args()

def main(args):
    # Read in the FINE annotations, which has the AMT format
    fine_annotation_data = read_mturk_data(args.fine_src_files)
    story_list, story_key_fn = get_story_list(fine_annotation_data)

    # Read in the COARSE annotations and compute COARSE scores
    all_coarse_scores, summary_src_doc_data = read_coarse_data(
        args.coarse_src_files, story_list, args.skip_model, verbose=args.verbose
    )

    summaries = []
    for story in story_list:
        summary = summary_src_doc_data[story]['summary']
        if '"""' in summary:
            summary = summary.replace('"""', "")
        summary = summary.strip()
        summaries.append(summary)

    # read pubmed so that we can try to recover document sections
    pubmed_data = [json.loads(line) for line in open(args.pubmed_file)]
    abs2dat = {" ".join(x["article_text"]): x for x in pubmed_data}
    id2dat = {x["article_id"]: x for x in pubmed_data}

    # try to map the summaries to the pubmed data
    document2sections = dict()

    for story in story_list:
        document = summary_src_doc_data[story]['source_doc']

        doc = document[:75]
        
        candidates = []
        for k in abs2dat:
            if doc in k:
                candidates.append(k)
        if len(candidates) == 0:
            # manual check
            id = "PMC3469068"
        else:
            id = abs2dat[candidates[0]]["article_id"]
        
        document2sections[document] = id2dat[id]["sections"]


    # compute the FINE scores
    all_fine_scores = compute_fine_scores(
        fine_annotation_data, story_list, story_key_fn, args.skip_model, args.num_keep, summaries=summaries, verbose=args.verbose
    )
    
    all_data = []

    for story in tqdm.tqdm(story_list):
        if args.skip_model and args.skip_model in story:
            print("SKIP", story)
            continue
        document = summary_src_doc_data[story]['source_doc']
        # document = ' '.join(summary_src_doc_data[story]['source_doc'].split('\n'))
        summary = summary_src_doc_data[story]['summary']

        sections = document2sections[document]

        sections = [" ".join(sec).strip().replace("\n", " ") for sec in sections]

        document_new = " ".join(sections).strip()
        

        all_data.append(
            {
                "documents": sections,
                "document": document_new,
                "summary": summary,
            }
        )
    
    assert len(all_data) == len(all_fine_scores)
    for dat, score in zip(all_data, all_fine_scores):
        dat['faithfulness'] = score
    
    
    with open("longeval_pubmed.jsonl", "w") as f:
        for data in all_data:
            f.write(json.dumps(data) + "\n")


    # squality
    test_set_inputs = jsonl_read("squality_annotations/squality_coarse/test.jsonl")
    test_set_inputs = {x['metadata']['passage_id']: x for x in test_set_inputs}

    fine_annotation_data = read_mturk_data("squality_annotations/squality_fine/*")
    story_list, story_key_fn = get_story_list(fine_annotation_data)
    all_coarse_scores, raw_data = read_squality_coarse_data(
        "squality_annotations/squality_coarse/all-human-responses.jsonl",
        story_list,
        args.skip_model,
        verbose=args.verbose
    )
    summaries = []
    for story in story_list:
        orig_human_ratings = get_squality_data_stats(raw_data, story)
        test_set_instance = test_set_inputs[orig_human_ratings["passage_id"]]
        summary = orig_human_ratings["response"]
        if '"""' in summary:
            summary = summary.replace('"""', "")
        summary = summary.strip()
        summaries.append(summary)

    all_fine_scores = compute_fine_scores(
        fine_annotation_data, story_list, story_key_fn, args.skip_model, args.num_keep, summaries=summaries, verbose=args.verbose
    )

    all_data = []
    for story in tqdm.tqdm(story_list):
        if args.skip_model and args.skip_model in story:
            print("SKIP", story)
            continue
        orig_human_ratings = get_squality_data_stats(raw_data, story)
        test_set_instance = test_set_inputs[orig_human_ratings["passage_id"]]
        summary = orig_human_ratings["response"]
        document = test_set_instance['document']
        # document = ' '.join(test_set_instance['document'].split('\n'))
        all_data.append(
            {
                "documents": [document],
                "document": document,
                "summary": summary,
            }
        )
    
    assert len(all_data) == len(all_fine_scores)
    for dat, score in zip(all_data, all_fine_scores):
        dat['faithfulness'] = score
    
    with open("longeval_squality.jsonl", "w") as f:
        for data in all_data:
            f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    main(args)
