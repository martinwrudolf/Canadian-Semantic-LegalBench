#!/usr/bin/env python3
"""Run Semantic LegalBench against the OpenAI Responses API."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_legalbench import (  # noqa: E402
    DEFAULT_EMBEDDING_MODELS as TOOLKIT_EMBEDDING_MODELS,
    EMBEDDING_PRESETS,
    EvalConfig,
    ModelOutput,
    SemanticLegalBench,
    ToolkitConfig,
    append_jsonl,
    load_examples,
    load_outputs,
    report,
    write_jsonl,
)

DEFAULT_EMBEDDING_MODELS = ",".join(TOOLKIT_EMBEDDING_MODELS)
DEFAULT_INCLUDE = "reasoning.encrypted_content,web_search_call.action.sources"


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def response_metadata(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif usage is not None and not isinstance(usage, (dict, list, str, int, float, bool)):
        usage = str(usage)

    return {
        "provider": "openai",
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "created_at": getattr(response, "created_at", None),
        "usage": usage,
    }


def create_response(client: Any, args: argparse.Namespace, prompt: str) -> Any:
    request: Dict[str, Any] = {
        "model": args.model,
        "input": prompt,
        "text": {
            "format": {"type": "text"},
            "verbosity": args.text_verbosity,
        },
        "reasoning": {
            "effort": args.reasoning_effort,
            "summary": args.reasoning_summary,
        },
        "tools": [],
        "store": args.store,
    }
    include = parse_csv(args.include)
    if include:
        request["include"] = include
    return client.responses.create(**request)


def collect_outputs(args: argparse.Namespace, model_id: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    examples = load_examples(args.dataset, split=args.split, task=args.task)
    if args.limit is not None:
        examples = examples[: args.limit]

    existing = set()
    output_path = Path(args.outputs)
    if output_path.exists():
        for row in load_outputs(output_path):
            if row.model_name == model_id:
                existing.add(row.example_id)

    client = OpenAI(max_retries=args.max_retries, timeout=args.timeout)
    total = len(examples)
    for index, example in enumerate(examples, 1):
        if example.id in existing:
            print(f"[{index}/{total}] skip existing {example.id}", flush=True)
            continue

        print(f"[{index}/{total}] calling {args.model} for {example.id}", flush=True)
        response = create_response(client, args, example.input_context)
        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise RuntimeError(f"OpenAI response for {example.id} did not include output_text")

        append_jsonl(
            output_path,
            [
                ModelOutput(
                    example_id=example.id,
                    model_name=model_id,
                    output_text=output_text,
                    metadata=response_metadata(response),
                ).to_json()
            ],
        )
        existing.add(example.id)

        if args.sleep:
            time.sleep(args.sleep)


def score_outputs(args: argparse.Namespace, model_id: str) -> Dict[str, Any]:
    model_ids = (parse_csv(args.embedding_models) if args.embedding_models is not None
                 else list(EMBEDDING_PRESETS[args.embedding_preset]))
    if len(model_ids) < 3:
        raise SystemExit("--embedding-models must list at least 3 model IDs")

    examples = load_examples(args.dataset, split=args.split, task=args.task)
    if args.limit is not None:
        examples = examples[: args.limit]
    example_ids = {example.id for example in examples}

    all_outputs = load_outputs(args.outputs)
    outputs_by_id: Dict[str, ModelOutput] = {}
    for output in all_outputs:
        if output.model_name == model_id and output.example_id in example_ids:
            outputs_by_id[output.example_id] = output

    missing = [example.id for example in examples if example.id not in outputs_by_id]
    if missing:
        print(f"Warning: {len(missing)} examples have no output for {model_id}.", file=sys.stderr)

    with SemanticLegalBench(
        examples,
        toolkit_config=ToolkitConfig(
            backend=args.backend,
            model_ids=model_ids,
            cache_db=Path(args.cache_db),
            device=args.device,
            batch_size=args.batch_size,
            max_chars_per_chunk=args.max_chars_per_chunk,
            chunk_overlap=args.chunk_overlap,
        ),
        eval_config=EvalConfig(
            flag_below=args.flag_below,
            adv_flag_above=args.adv_flag_above,
            adv_refusal_ok=not args.adv_no_refusal_ok,
        ),
    ) as bench:
        # Process all responses with one embedding model before loading the next.
        scored = bench.score_outputs(list(outputs_by_id.values()))

    scored_path = Path(args.scored)
    report_path = Path(args.report)
    write_jsonl(scored_path, [row.to_json() for row in scored])
    rep = report(scored)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Semantic LegalBench against the OpenAI Responses API."
    )
    parser.add_argument("--dataset", default="data/a2aj_benchmark.jsonl")
    parser.add_argument("--split", choices=["train", "test", "validation"], default="test")
    parser.add_argument(
        "--task",
        choices=["pinpoint_summarization_similarity", "sentence_completion_evaluation"],
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N matched examples.")

    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--model-id", default=None, help="Result label. Defaults to openai/<model>.")
    parser.add_argument("--text-verbosity", default="medium")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--reasoning-summary", default="auto")
    parser.add_argument("--include", default=DEFAULT_INCLUDE, help="Comma-separated Responses include paths.")
    parser.add_argument("--store", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--score-only", action="store_true", help="Skip API calls and score existing outputs.")

    parser.add_argument("--outputs", default="data/openai_outputs.jsonl")
    parser.add_argument("--scored", default="data/openai_scored.jsonl")
    parser.add_argument("--report", default="data/openai_report.json")

    parser.add_argument("--backend", choices=["sentence-transformers", "hash"], default="sentence-transformers")
    parser.add_argument("--embedding-preset", choices=list(EMBEDDING_PRESETS), default="small",
                        help="small: models under 2B (default); top: saved MTEB Law leaders, any size")
    parser.add_argument("--embedding-models", default=None,
                        help="Comma-separated model IDs; overrides --embedding-preset")
    parser.add_argument("--cache-db", default=".slb_cache/embeddings.sqlite")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--flag-below", type=float, default=0.45)
    parser.add_argument("--adv-flag-above", type=float, default=0.35)
    parser.add_argument("--adv-no-refusal-ok", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    model_id = args.model_id or f"openai/{args.model}"

    if not args.score_only:
        collect_outputs(args, model_id)

    rep = score_outputs(args, model_id)
    print(json.dumps(rep, indent=2))
    print(f"Wrote outputs: {args.outputs}")
    print(f"Wrote scored rows: {args.scored}")
    print(f"Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
