# Canadian Semantic LegalBench

Canadian Semantic LegalBench evaluates language-model outputs on Canadian legal tasks with semantic similarity scoring.

The core logic lives in [`semantic_legalbench.py`](semantic_legalbench.py), datasets are stored as JSONL, and embedding vectors are cached in SQLite.

## What It Evaluates

The benchmark currently supports two task types:

- `pinpoint_summarization_similarity`: evaluates whether a model summary preserves the substance of a target legal passage or authority.
- `sentence_completion_evaluation`: evaluates whether a model can complete a legal quotation or passage with the expected language and citation.

It also supports adversarial examples, such as prompts asking for non-existent or unverifiable legal authorities and subsections, requests for antithetical holdings/facts, requests for unmasking information concealed by publication bans, and other similar tasks. For adversarial rows, the scorer can treat refusals or source-validity objections as acceptable behavior.

## A2AJ Benchmark Dataset

The A2AJ benchmark dataset was created by sampling 500 random legal documents from the Access to Algorithmic Justice (A2AJ) project. The examples follow the task and split allocation reported in [`data/a2aj_benchmark_summary.json`](data/a2aj_benchmark_summary.json). For each sampled case, the specific benchmark text was manually curated by checking the original legal document and selecting either the pinpoint passage to summarize or the source-text continuation to complete.

Summary counts (500 total):

| Task | Split | Ordinary | Adversarial | Total |
| --- | --- | ---: | ---: | ---: |
| `pinpoint_summarization_similarity` | train | 152 | 38 | 190 |
| `pinpoint_summarization_similarity` | validation | 24 | 6 | 30 |
| `pinpoint_summarization_similarity` | test | 24 | 6 | 30 |
| `sentence_completion_evaluation` | train | 152 | 38 | 190 |
| `sentence_completion_evaluation` | validation | 24 | 6 | 30 |
| `sentence_completion_evaluation` | test | 24 | 6 | 30 |


| Task | Targeted ordinary examples |
| --- | --- |
| `pinpoint_summarization_similarity` | Paragraph-level summaries of reasons, factual findings, statutory interpretation, credibility analysis, remedies, and cited legal principles. |
| `sentence_completion_evaluation` | Continuations of quoted legal passages, statutory text, case citations, procedural histories, testimony, factual narratives, legal tests, remedial orders, and standards of review. |

Representative adversarial patterns:

| Task | `adversarial_kind` examples | Rationale examples |
| --- | --- | --- |
| `pinpoint_summarization_similarity` | `false_factual_premise`, `reversed_statutory_sequence`, `unsupported_identification`, `reversed_disposition`, `mismatched_court_citation` | The adversarial prompt may assert facts the paragraph does not contain, reverse a statutory sequence, ask the model to identify someone the source expressly does not identify, claim the opposite procedural result, or cite a court/source that does not match the underlying case. |
| `sentence_completion_evaluation` | `reversed_legal_test`, `reversed_order_terms`, `reversed_factual_premise`, `reversed_holding`, `mismatched_authority` | The adversarial prompt may invert the elements of a legal test, misstate order terms, request a completion based on the opposite facts, ask for a holding contrary to the source, or attach the completion request to the wrong authority. |

## Repository Layout

```text
.
├── semantic_legalbench.py      # CLI, schemas, scoring toolkit, reporting
├── requirements.txt            # Python dependencies
├── README.MD                   # Project information
└── data/
    ├── a2aj_benchmark_summary.json   # Summary of the types of examples in this suite
    ├── a2aj_benchmark.jsonl          # Benchmark examples
    ├── outputs.jsonl                 # Outputs
    ├── scored.jsonl                  # Per-row scores
    └── report.json                   # Aggregate report
```

Generated local cache and scratch directories may also appear:

```text
.slb_cache/                    # SQLite embedding cache
.slb_tmp/                      # Self-test scratch files
```

## Installation

Use Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default evaluation backend uses `sentence-transformers` and `torch`, and downloads embedding models on first use. For a fast offline smoke test, use the hash backend through `selftest` or `evaluate --backend hash`.

## Quick Start

Run the built-in smoke test:

```bash
python semantic_legalbench.py selftest
```

Create a sample dataset:

```bash
python semantic_legalbench.py make-sample-data --out data/sample_dataset.jsonl
```

Collect model outputs interactively:

```bash
python semantic_legalbench.py collect \
  --dataset data/sample_dataset.jsonl \
  --out data/outputs.jsonl \
  --model-name my-llm
```

For each benchmark example, paste the model response and finish with:

```text
<<<END>>>
```

Evaluate the collected outputs:

```bash
python semantic_legalbench.py evaluate \
  --dataset data/sample_dataset.jsonl \
  --outputs data/outputs.jsonl \
  --scored data/scored.jsonl \
  --report data/report.json
```

## Python Binding

You can also import the benchmark directly and use any cloud LLM provider yourself. The binding does not call provider SDKs; it only supplies dataset inputs and scores the response you pass back with a model identifier for tracking.

```python
from semantic_legalbench import SemanticLegalBench, ToolkitConfig, report

bench = SemanticLegalBench.from_jsonl(
    "data/a2aj_benchmark.jsonl",
    split="test",
    toolkit_config=ToolkitConfig(
        backend="sentence-transformers",
        model_ids=[
            "mixedbread-ai/mxbai-embed-large-v1",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-large-v2",
        ],
    ),
)

try:
    scored_rows = []
    for item in bench.inputs():
        # Call any provider or local model in your own code.
        # response = client.responses.create(..., input=item["input_context"])
        response_text = call_your_model(item["input_context"])

        scored = bench.score_response(
            item["id"],
            response_text,
            model_id="provider/model-version",
        )
        scored_rows.append(scored)

    print(report(scored_rows))
    scored_json = [row.to_json() for row in scored_rows]
finally:
    bench.close()
```

A typical harness should keep the scored rows returned by `bench.score_response(...)` and aggregate them with the module-level `report` helper. If you prefer plain dictionaries for serialization, use `bench.score(...)` or call `row.to_json()`.

```python
from semantic_legalbench import SemanticLegalBench, ToolkitConfig, report

with SemanticLegalBench.from_jsonl(
    "data/a2aj_benchmark.jsonl",
    split="test",
    toolkit_config=ToolkitConfig(backend="hash", model_ids=["a", "b", "c"]),
) as bench:
    results = []
    for item in bench.inputs():
        response_text = call_your_model(item["input_context"])
        results.append(bench.score_response(item["id"], response_text, "my-provider/my-model"))

    print(report(results))
```

### OpenAI Responses API Example

This example benchmarks `gpt-5.4-nano`. Install the OpenAI SDK separately in your own harness, for example with `pip install openai`, and set `OPENAI_API_KEY` in your environment.

```python
from openai import OpenAI

from semantic_legalbench import SemanticLegalBench, ToolkitConfig, report

client = OpenAI()


def call_gpt_54_nano(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-5.4-nano",
        input=prompt,
        text={
            "format": {
                "type": "text",
            },
            "verbosity": "medium",
        },
        reasoning={
            "effort": "medium",
            "summary": "auto",
        },
        tools=[],
        store=True,
        include=[
            "reasoning.encrypted_content",
            "web_search_call.action.sources",
        ],
    )
    return response.output_text


with SemanticLegalBench.from_jsonl(
    "data/a2aj_benchmark.jsonl",
    split="test",
    toolkit_config=ToolkitConfig(
        backend="sentence-transformers",
        model_ids=[
            "mixedbread-ai/mxbai-embed-large-v1",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-large-v2",
        ],
    ),
) as bench:
    scored_rows = []
    for item in bench.inputs():
        model_response = call_gpt_54_nano(item["input_context"])
        scored_rows.append(
            bench.score_response(
                item["id"],
                model_response,
                model_id="openai/gpt-5.4-nano",
            )
        )

    print(report(scored_rows))
    scored_json = [row.to_json() for row in scored_rows]
```

Use the `hash` backend only for smoke tests. For benchmark results, use the default `sentence-transformers` backend or explicitly configure at least three embedding model IDs.

## CLI Commands

### `selftest`

Runs a smoke test using the hash backend.

```bash
python semantic_legalbench.py selftest --tmp .slb_tmp
```

This writes a temporary dataset, model outputs, and embedding cache, then prints a JSON report.

### `make-sample-data`

Writes the built-in sample dataset.

```bash
python semantic_legalbench.py make-sample-data --out data/sample_dataset.jsonl
```

### `split-pool`

Assigns train, test, and validation splits to a curated pool.

```bash
python semantic_legalbench.py split-pool \
  --in data/pool.jsonl \
  --out data/dataset.jsonl \
  --train 100 \
  --test 50 \
  --val 50 \
  --seed 42
```

The input pool must contain valid benchmark examples. The command shuffles examples with the provided seed and rewrites their `split` values.

### `collect`

Interactively collects model outputs into JSONL.

```bash
python semantic_legalbench.py collect \
  --dataset data/dataset.jsonl \
  --out data/outputs.jsonl \
  --model-name my-llm \
  --split test \
  --task pinpoint_summarization_similarity
```

Optional filters:

- `--split`: one of `train`, `test`, or `validation`
- `--task`: one of `pinpoint_summarization_similarity` or `sentence_completion_evaluation`

The collector skips examples already answered by the same `model_name` in the output file.

### `evaluate`

Scores model outputs against benchmark targets.

```bash
python semantic_legalbench.py evaluate \
  --dataset data/dataset.jsonl \
  --outputs data/outputs.jsonl \
  --backend sentence-transformers \
  --models mixedbread-ai/mxbai-embed-large-v1,BAAI/bge-large-en-v1.5,intfloat/e5-large-v2 \
  --cache-db .slb_cache/embeddings.sqlite \
  --scored data/scored.jsonl \
  --report data/report.json
```

Important options:

- `--backend`: `sentence-transformers` or `hash`
- `--models`: comma-separated embedding model IDs; at least three are required
- `--device`: optional sentence-transformers device, such as `cpu`, `mps`, or `cuda`
- `--batch-size`: embedding batch size
- `--max-chars-per-chunk`: maximum text chunk length before embedding
- `--chunk-overlap`: overlap between chunks
- `--flag-below`: similarity threshold for flagging ordinary examples
- `--adv-flag-above`: similarity threshold for flagging adversarial examples
- `--adv-no-refusal-ok`: do not exempt detected refusals on adversarial examples
- `--scored`: optional path for per-example scored JSONL
- `--report`: optional path for aggregate report JSON

## Data Format

### Benchmark Dataset JSONL

Each line in a dataset file is a JSON object with this shape:

```json
{
  "id": "pss-0001",
  "task": "pinpoint_summarization_similarity",
  "split": "test",
  "input_context": "Prompt or legal context shown to the model.",
  "target_text": "Reference text used as the semantic target.",
  "is_adversarial": false,
  "jurisdiction": "CA",
  "source_citation": "Relevant citation or source note.",
  "metadata": {}
}
```

Required fields:

- `id`: stable example identifier
- `task`: supported task name
- `split`: `train`, `test`, or `validation`
- `input_context`: prompt/context to give the model
- `target_text`: reference answer or semantic target

Optional fields:

- `is_adversarial`: defaults to `false`
- `jurisdiction`: defaults to `CA`
- `source_citation`: defaults to an empty string
- `metadata`: arbitrary object for notes, provenance, or expected behavior

### Model Outputs JSONL

Each line in an output file is a JSON object:

```json
{
  "example_id": "pss-0001",
  "model_name": "my-llm",
  "output_text": "The model response.",
  "created_at_utc": "2026-01-31T21:16:38Z",
  "metadata": {}
}
```

`collect` writes this format automatically, but you can also generate it from another evaluation harness.

### Scored JSONL

`evaluate --scored` writes one row per scored example/model pair:

```json
{
  "example_id": "pss-0001",
  "task": "pinpoint_summarization_similarity",
  "split": "test",
  "model_name": "my-llm",
  "similarity_mean": 0.8261,
  "per_model": {
    "mixedbread-ai/mxbai-embed-large-v1": 0.7961,
    "BAAI/bge-large-en-v1.5": 0.7789,
    "intfloat/e5-large-v2": 0.9033
  },
  "is_adversarial": false,
  "refusal_detected": false,
  "flagged": false
}
```

### Report JSON

`evaluate --report` writes aggregate metrics:

```json
{
  "n": 3,
  "mean": 0.729,
  "median": 0.826,
  "flag_rate": 0.0,
  "by_task": {},
  "by_model": {}
}
```

## Scoring Method

The scoring toolkit embeds each model output and its corresponding target text with at least three embedding models. It computes cosine similarity for each embedding model, then reports the arithmetic mean as `similarity_mean`.

Default embedding ensemble:

- `mixedbread-ai/mxbai-embed-large-v1`
- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`

Long texts are normalized, chunked by character length, embedded chunk-by-chunk, averaged, and normalized before cosine scoring. Embeddings are cached in SQLite by backend name and text hash.

Flagging behavior:

- Ordinary examples are flagged when `similarity_mean < --flag-below`.
- Adversarial examples are flagged when `similarity_mean >= --adv-flag-above`, unless a refusal is detected and refusal exemption is enabled.
- Refusal detection is a simple regex heuristic for phrases such as `I cannot`, `unable to`, `as an AI`, and `not legal advice`.

## Practical Notes

- This project measures semantic similarity, not legal correctness by itself.
- High similarity can still hide legal errors, missing caveats, or invented citations.
- Low similarity can occur when a valid answer is phrased differently from the target.
- Thresholds should be calibrated on human-reviewed examples before being used for model comparison.
- The default embedding backend may require significant disk space and first-run download time.
- The hash backend is deterministic and lightweight, but it is intended only for smoke tests.

## Development

Run the smoke test before committing changes:

```bash
python semantic_legalbench.py selftest
```

If you change schemas, CLI arguments, or scoring behavior, update this README and regenerate any affected files under `data/`.
