# Embedding ensembles

Presets are selected from the user-supplied [MTEB Law leaderboard archive](../MTEB%20Leaderboard%20-%20a%20Hugging%20Face%20Space%20by%20mteb.mhtml). Selection updated 2026-09-08. Snapshot ranks and scores are historical, not live rankings.

| Preset | Model card | Parameters (snapshot) | Snapshot Mean(Task) | Licence |
| --- | --- | --- | --- | --- |
| small (default) | [Hanno-Labs/dinghy-law-0.6b-v1](https://huggingface.co/Hanno-Labs/dinghy-law-0.6b-v1) | 597M | 65.83 | Apache-2.0 |
| small | [codefuse-ai/F2LLM-v2-1.7B](https://huggingface.co/codefuse-ai/F2LLM-v2-1.7B) | 1.7B | 60.32 | Apache-2.0 |
| small | [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | 568M | 58.96 | Apache-2.0 |
| top | [litillabs/octen-law-8b-v1](https://huggingface.co/litillabs/octen-law-8b-v1) | 7.6B | 76.77 | Apache-2.0 |
| top | [Hanno-Labs/dinghy-law-4b-v1](https://huggingface.co/Hanno-Labs/dinghy-law-4b-v1) | 4.0B | 71.22 | Apache-2.0 |
| top | [Mira190/Euler-Legal-Embedding-V1](https://huggingface.co/Mira190/Euler-Legal-Embedding-V1) | 7.6B | 70.37 | Apache-2.0 |

The small preset selects strong sub-2B models compatible with the native loader. Higher-listed GreenLeaf and Inf Retriever require custom remote code and are omitted. F2LLM's Hub metadata confirms 1,720,574,976 parameters, below 2B despite the rounded “2B” badge. Model cards and small-preset loading configurations were checked on the selection date.

## Usage and protocol

Use Python 3.10+ and `pip install -U -r requirements.txt`. Run with no size flag for `small`, or add `--embedding-preset top` to the evaluation CLI or OpenAI runner. In Python use `ToolkitConfig(embedding_preset="top")`. Explicit model IDs override the preset. The shared preset definitions live in `semantic_legalbench.py`.

Both presets use automatic checkpoint dtype and native model implementations, without remote code execution or mandatory Flash Attention. Euler retains its 1536-token cap. Small models encode both passages with an explicit empty prompt to disable query defaults; the top preset retains its configured document prompt. Chunking, normalization, and averaging of the three cosine similarities are unchanged. Retrieval leaderboard results motivate selection but do not validate Canadian legal correctness or this symmetric similarity protocol.

Models load one at a time, only when an embedding is missing from the SQLite cache. Batch evaluation scores all responses with one model, releases its weights, runs garbage collection, and clears unused CUDA/MPS allocations before loading the next. This also releases the active model if scoring fails. First use still downloads weights; memory must accommodate the largest individual model plus inference overhead.

No new flags are needed: `./scripts/run_openai.sh --limit 3 --score-only` and the evaluation CLI use this behavior automatically. In Python, use `bench.score_outputs(outputs)` for a batch of `ModelOutput` records, or `toolkit.score_many([(response, target), ...])` for text pairs. Single-response methods also release each model, but repeated calls may reload weights for each response. For an offline smoke test use `python semantic_legalbench.py selftest`; lifecycle regression tests run with `python -m unittest discover -s tests` without downloading weights.

Existing scored files and reports are historical results from the previous ensemble and split. Collect missing test outputs and write fresh scores and reports; do not relabel historical scores. README numbers are illustrative. Full weight loading and real inference with this ensemble have not yet been validated in this checkout.
