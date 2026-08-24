# DataCrossBench

DataCrossBench is a cross-source data analysis benchmark with 200 `flag` tasks and a four-dimensional evaluator. Each flag contains a `meta-info.json` with the analysis goal and reference insights, plus an `output/` directory containing the data files that participants are expected to analyze.

## Repository Layout

```text
DataCrossBench/
├── DataCrossBench/
│   ├── cleaned_insights_200.json
│   └── flag-*/
│       ├── meta-info.json
│       └── output/
└── evaluation/
    ├── eval_with_stratification.py
    ├── metrics_custom.py
    ├── utils.py
    ├── prompts/
    ├── flag_classification.json
    ├── validate_benchmark.py
    ├── AUDIT.md
    └── requirements.txt
```

The intermediate directories `output_csv_origin/` and `output/processed/` have been removed from every flag. The release does not include evaluation results, checkpoints, caches, or unrelated experiment directories.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_URL="https://api.openai.com/v1"  # optional
export EVAL_MODEL="gpt-4o"                            # optional
```

Evaluation requires an OpenAI-compatible Chat Completions endpoint supporting `logprobs` and `top_logprobs`, as well as an Embeddings endpoint.

## Participant Output Format

Participants only need to submit one JSON file containing their predicted insights. They do not need to copy the ground truth:

```json
{
  "flag-1": {
    "pred_insights": [
      "Trend: ...",
      "Comparison: ...",
      "Extreme: ...",
      "Attribution: ..."
    ]
  },
  "flag-2": {"pred_insights": ["..."]}
}
```

The evaluator also accepts `{"flag-1": ["insight"]}` and a list of records containing `flag_id` and `pred_insights`. `pred_insights` must be a list of strings. Ground-truth insights and the analysis goal are loaded automatically from `DataCrossBench/flag-*/meta-info.json`. A formal evaluation requires predictions for all 200 flags; use `--allow-subset` only when debugging.

## Run Evaluation

```bash
python3 evaluation/eval_with_stratification.py \
  --predictions /path/to/predictions.json \
  --bench ./DataCrossBench
```

Results are written to `evaluation/results/` at runtime and should not be committed. Use `--resume` to continue from a checkpoint. When changing the prediction file or model, remove the old checkpoint or specify a separate output directory.

To validate the benchmark without making API calls:

```bash
python3 evaluation/validate_benchmark.py \
  ./DataCrossBench \
  --classification evaluation/flag_classification.json
```

## Scoring and Fixed Processing

The four dimensions use these weights: Factuality 0.30, Completeness 0.25, Logic 0.20, and Insightfulness 0.25. Scores are designed to fall in the range 0-1. Factuality combines a numeric hard metric and an LLM score with equal weights. Completeness is the average one-way maximum embedding similarity from ground-truth insights to predicted insights; cosine values are not clipped by the current implementation.

The evaluator removes the case-sensitive prefixes `Trend:`, `Comparison:`, `Extreme:`, and `Attribution:` from the beginning of each insight. Numeric matching uses a default relative tolerance of 10%; years and identifiers are also treated as numbers. If the ground truth contains no numbers, the hard factuality score is 1. If an LLM response does not contain `<rating>1-10</rating>`, the evaluator falls back to the first standalone number from 1 to 10, or returns 0.5 if no rating can be found; a final API failure returns 0. The easy/hard split is based only on whether `output/` contains image files. The evaluator does not reread those data files to recompute the ground truth.

## Hosting Notes

The release is approximately 1.0 GB and includes a SQLite file of about 253 MB and a CSV file of about 122 MB, exceeding GitHub's regular 100 MB per-file limit. Git LFS or release attachments are required for GitHub hosting. Anonymous 4open.science is suitable for an anonymous read-only copy. Confirm that the data, images, and third-party sources are permitted for redistribution before publication.
