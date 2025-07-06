# Code and Data for "On Positional Bias of Faithfulness for Long-form Summarization"

By [David Wan](https://meetdavidwan.github.io/), [Jesse Vig](https://jessevig.com/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), and [Shafiq Joty](https://raihanjoty.github.io/).

## 1. Structure
- `data/generation`: Data and code for replicating data for generating summaries
- `data/metric_benchmark`: Data and code for replicating data for meta-evaluation
- `src`: Code for running the metric and different generation methods
- `prompts`: All prompts used for generation


## 2. Data Replication

This section describes how to replicate the datasets used for both the metric meta-evaluation and the summary generation tasks. All processed data and processing scripts are located in the `data/` directory.

### 2.1 Metric Meta-Evaluation Benchmark

The benchmark data is used to evaluate the performance of faithfulness metrics. Each line in the `jsonl` files has the following structure:

  - `id`: A unique identifier.
  - `documents`: A list of source documents.
  - `document`: The concatenated original document.
  - `summary`: The system-generated summary to be evaluated.
  - `faithfulness`: A binary label indicating faithfulness (1 for faithful, 0 for not).
  - `ranking`: A list of document indices, ordered from most to least important.

#### **DiverseSumm**

1.  Download the original annotations from the [DiverseSumm repository](https://github.com/salesforce/DiverseSumm).
2.  Run the processing script:
    ```bash
    python data/metric_benchmark/process_diversesumm.py ${path_to_diversesumm_annotation_jsonl}
    ```

#### **LongEval**

1.  Clone the [LongEval repository](https://github.com/martiansideofthemoon/longeval-summarization) to obtain the sentence-level faithfulness annotations.
2.  Place the `pubmed_annotations` and `squality_annotations` directories into `data/metric_benchmark/longeval/`.
3.  Download the `pubmed_test.txt` file from [Cohan et al., 2018](https://github.com/armancohan/long-summarization) and place it in the same directory.
4.  **Note:** The script `process_longeval.py` contains hardcoded paths. You may need to adjust the paths on lines 13, 14, 28, 116, 119, and 122 to match your file locations.
5.  Run the processing script:
    ```bash
    python data/metric_benchmark/process_longeval.py
    ```

#### **Other Benchmarks (ChemSumm, ArXiv, etc.)**

1.  Download the compiled annotations from [Infuse](https://github.com/HJZnlp/infuse).
2.  From the Infuse data, place `DiverSumm.csv` into `data/metric_benchmark/diversumm/`.
3.  Download the ArXiv and PubMed test sets from [Cohan et al., 2018](https://github.com/armancohan/long-summarization) and place `arxiv_test.txt` into `data/metric_benchmark/scientific_papers/`.
4.  **Note:** You may need to update the hardcoded paths in `process_other.py` (e.g., lines 6 and 72) if your file structure differs.
5.  Run the processing script:
    ```bash
    python data/metric_benchmark/process_other.py
    ```

### 2.2 Generation Data

This data is used as input for the various summary generation methods. Each `jsonl` file contains lines with `id` and `document` (a list of documents).

  - **DiverseSumm**: Use the data from the original authors and run:
    ```bash
    python data/generation/process_diversesumm.py
    ```
  - **Other Datasets**: We use the `datasets` library. Please refer to the corresponding scripts in `data/generation/` for more details.

## 3. Running the Code

All source code is located in the `src/` directory. Prompts for all models and tasks are in `prompts/`.

### 3.1 Metric Meta-Evaluation

To evaluate generated summaries using an LLM-based metric (e.g., GPT-4o), use `src/evaluate_summaries.py`.

**Example:**

```bash
python src/evaluate_summaries.py \
  --model_name gpt4o \
  --data_file data/metric_benchmark/arxiv.jsonl \
  --output_file results/arxiv_eval.json \
  --document_merge_type max
```

### 3.2 Summary Generation

All prompts used for generation can be found in the `prompts/` directory.

#### **Standard & "Focus" Generation**

Use `src/generate_summaries.py`. The "Focus" method uses a modified prompt (`prompts/arxiv_prompt_top.txt`) to guide the model.

```bash
python src/generate_summaries.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --prompt_file prompts/arxiv.txt \
  --output_file results/summaries_standard.json
```

#### **Incremental Updating**

```bash
python src/generate_summaries_incremental.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --prompt_file prompts/arxiv_incremental.txt \
  --original_prompt_file prompts/arxiv.txt \
  --output_file results/summaries_incremental.json
```

#### **Hierarchical Merging**

This is a two-step process.

**Step 1: Generate summaries for each document individually.**

```bash
python src/generate_summaries_individual.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --prompt_file prompts/arxiv.txt \
  --output_file results/summaries_individual.json
```

**Step 2: Merge the individual summaries.**

```bash
python src/generate_summaries_merge.py \
  --model_name gpt4o \
  --prompt_file prompts/arxiv_merge.txt \
  --data_file results/summaries_individual.json \
  --output_file results/summaries_hierarchical.json
```

#### **Calibration**

This process is similar to hierarchical merging but uses a different initial generation script.

**Step 1: Generate summaries for each document with calibration.**

```bash
python src/generate_summaries_calibration_initial.py \
  --model_name gpt4o \
  --data_file data/generation/arxiv.jsonl \
  --prompt_file prompts/arxiv.txt \
  --output_file results/summaries_calibration_individual.json
```

**Step 2: Merge the individual summaries.**

```bash
python src/generate_summaries_merge.py \
  --model_name gpt4o \
  --prompt_file prompts/arxiv_merge.txt \
  --data_file results/summaries_calibration_individual.json \
  --output_file results/summaries_calibration.json
```

## 4. Pre-computed Outputs

We provide all our generated outputs and metric scores in the following Google Drive folder:

[**Link to Pre-computed Outputs**](https://drive.google.com/drive/folders/1Mv9GhGqhUtMsW51FJ8O49M0IBPlJer0i?usp=sharing)

  - **Metric Scores**: These are JSON files containing sentence-level faithfulness scores. The structure is a list of lists: `[num_examples, num_documents, num_sentences]`.
      - Scores from `MiniCheck` are floats; scores from `GPT-4o` are strings.
      - The `_split` variant contains scores for each document separately. The `_full` variant contains scores for the concatenated document.
  - **Generated Summaries**: These are JSON files that mirror the input data format but include an additional `generated_summary` field. We also include the `MiniCheck` scores for all generated outputs.

## 5. Citation

If you find our work useful in your research, please cite our paper:

```bibtex
@inproceedings{wan-etal-2025-positional,
    title = "On Positional Bias of Faithfulness for Long-form Summarization",
    author = "Wan, David  and Vig, Jesse  and Bansal, Mohit  and Joty, Shafiq",
    editor = "Chiruzzo, Luis  and Ritter, Alan  and Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.442/",
    doi = "10.18653/v1/2025.naacl-long.442",
    pages = "8791--8810",
    ISBN = "979-8-89176-189-6",
}
```