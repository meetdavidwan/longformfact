
# Code for "On Positional Bias of Faithfulness for Long-form Summarization"

by  [David Wan](https://meetdavidwan.github.io/),
[Jesse Vig](https://jessevig.com/),
[Mohit Bansal](https://www.cs.unc.edu/~mbansal/),
[Shafiq Joty](https://raihanjoty.github.io/)

## 1. Structure
- `data/generation`: Code for replicating data for generating summaries
- `data/metric_benchmark`: Code for replicating data for meta-evaluation
- `src`: Code for running the metric and different generation methods
- `prompts`: All prompts used for generation

## 1.1 Data Processing
This section describes how to create the meta-evaluation benchmark and the data for generation.

### 1.2 Metric Meta-Evaluation Benchmark
Each `jsonl` file contains `id`, `document`, `documents`, `summary`, and `faithfulness` for each line, where:
- `documents` is a list of documents
- `document` is the original document
- `summary` is the system summary to be evaluated
- `faithfulness` is the binary faithfulness label.

### 1.2.1 DiverseSumm

Please download the data from the [original authors](https://github.com/salesforce/DiverseSumm), and run
```
python data/metric_benchmark/process_diversesumm.py ${path_to_diversesumm_annotation_jsonl}
```


### 1.2.2 LongEval

We adapt from the [author's code](https://github.com/martiansideofthemoon/longeval-summarization) to obtain the sentence-level faithfulness annotations.

Ensure the following files are in `data/metric_benchmark/longeval`:
- `pubmed_annotations` and `squality_annotations`: The actual annotations contained in the actual repo. You may need to change line 13 and 14, 116, 119, and 122 in `process_longeval.py`.
- 'pubmed_test.txt': From [Cohan et al., 2018](https://github.com/armancohan/long-summarization). Specify it with the `--pubmed_file` argument or directly change it in line 28 of `process_longeval.py`.


Finally run:
```
python data/longeval/process_longeval.py
```


### 1.2.3 Other Benchmarks
For ChemSumm, ArXiv, MultiNews, GovReport, and QmSum, please download the annotations compiled by [Infuse](https://github.com/HJZnlp/infuse). After downloading their annotations.

Place the following files in `data/metric_benchmark`:
- `diversumm/DiverSumm.csv`: The annotation compiled by the Infuse paper. You may need to change the path in line 6 of `process_other.py`.
- `scientific_papers/arxiv_test.txt`: The original annotations by [Cohan et al., 2018](https://github.com/armancohan/long-summarization). Please download both ArXiv and PubMed as both will be needed later. You may need to edit the path in line 72 of `process_other.py`.

Finally, run:
```
python data/metric_benchmark/process_other.py
````

### 2. Generation
Each `jsonl` file contains `id` and `document` for each line, where `document` is a list of documents.

For Diversesumm, wuse the data from the original authors and run: 
```
python data/generation/process_diversesumm
```

For other datasets, we use `datasets`. Please refer to the respective files for more details.

##  2.1 Code

### 2.1.1 Preliminary
The code can be found in `src`. We abstract the models in `src/model.py`, and for running GPT-4o, please supply the key with `OPENAI_API_KEY`.

### 2.1.2 Metric Meta-Evaluation Benchmark
`src/evaluate_summaries.py` contains the code for running the LLM-based metric. Example:

```
python src/evaluate_summaries.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --output_file test.json \
  --document_merge_type max
```

### 2.1.3 Generation
All prompts can be found in `prompts`.

**For regular generation & Focus prompts:**
```
python src/generate_summaries.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --prompt_file prompts/arxiv.txt \
  --output_file test.json
```
This can be similarly used for the focus prompt by swapping with the respective prompts, i.e. `prompts/arxiv_prompt_top.txt`

**For incremental updating:**
```
python src/generate_summaries_incremental.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --prompt_file prompts/arxiv_incremental.txt \
  --original_prompt_file prompts/arxiv.txt \
  --output_file test.json 
```

**For hierarchical merging:**
First generate summaries for each individual document
```
python src/generate_summaries_individual.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --prompt_file prompts/arxiv.txt \
  --output_file test_initial.json
```

Then merge the generated summaries:
```
python src/generate_summaries_merge.py \
  --model_name gpt4o \
  --prompt_file prompts/arxiv_merge.txt \
  --data_file test_initial.json \
  --output_file test_hierarchical_merge.json

```

**For calibration:**
Similarly, generate summaries fear ocha document intiially:
```
python src/generate_summaries_calibration_initial.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.json/ \
  --prompt_file prompts/arxiv.txt \
  --output_file test_individual.json
```

Then, merge:
```
python src/generate_summaries_merge.py \
  --model_name gpt4o \
  --prompt_file prompts/arxiv_merge.txt \
  --data_file test_individual.json \
  --output_file test_calibration_merge.json
```

## 3. Citation

If you find our project useful in your research, please cite the following paper:

```bibtex
@misc{Wan2024PositionalBias,
  author    = {David Wan and Jesse Vig and Mohit Bansal and Shafiq Joty},
  title     = {On Positional Bias of Faithfulness for Long-form Summarization},
  year      = {2024},
}
```