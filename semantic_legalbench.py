#!/usr/bin/env python3
"""
Semantic LegalBench (Canada).

Implements:
- Similarity toolkit: embed LLM output + target output with >=3 embedding models and average cosine similarity.
- Two tasks:
  (1) pinpoint_summarization_similarity
  (2) sentence_completion_evaluation
- Train/test/validation splits + adversarial examples.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import random
import re
import sqlite3
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

Split = Literal["train", "test", "validation"]
TaskName = Literal["pinpoint_summarization_similarity", "sentence_completion_evaluation"]
PathLike = Union[str, Path]

# ----------------------------
# IO helpers
# ----------------------------

def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("numpy is required. Install with: pip install numpy")
    return np

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{i}: {e}") from e
    return rows

def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

_WS = re.compile(r"\s+")
def norm_text(t: str) -> str:
    return _WS.sub(" ", t.strip())

def chunk_by_chars(t: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    t = norm_text(t)
    if len(t) <= max_chars:
        return [t]
    out: List[str] = []
    i = 0
    while i < len(t):
        j = min(len(t), i + max_chars)
        out.append(t[i:j])
        if j == len(t):
            break
        i = max(0, j - overlap)
    return out

# ----------------------------
# Schemas
# ----------------------------

@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    task: TaskName
    split: Split
    input_context: str
    target_text: str  # truth-anchor for summarization; expected completion for sentence completion
    is_adversarial: bool = False
    jurisdiction: str = "CA"
    source_citation: str = ""
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "BenchmarkExample":
        return BenchmarkExample(
            id=str(d["id"]),
            task=d["task"],
            split=d["split"],
            input_context=str(d["input_context"]),
            target_text=str(d.get("target_text", "")),
            is_adversarial=bool(d.get("is_adversarial", False)),
            jurisdiction=str(d.get("jurisdiction", "CA")),
            source_citation=str(d.get("source_citation", "")),
            metadata=dict(d.get("metadata", {})),
        )

@dataclass(frozen=True)
class ModelOutput:
    example_id: str
    model_name: str
    output_text: str
    created_at_utc: str = dataclasses.field(default_factory=utc_now)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "ModelOutput":
        return ModelOutput(
            example_id=str(d["example_id"]),
            model_name=str(d["model_name"]),
            output_text=str(d["output_text"]),
            created_at_utc=str(d.get("created_at_utc", utc_now())),
            metadata=dict(d.get("metadata", {})),
        )

# ----------------------------
# Embedding backends
# ----------------------------

class EmbeddingBackend:
    name: str
    def embed(self, texts: Sequence[str]) -> "np.ndarray":  # type: ignore[name-defined]
        raise NotImplementedError

class HashBackend(EmbeddingBackend):
    """Lightweight fallback backend: hashed character n-grams into a fixed vector."""
    def __init__(self, name: str, dim: int = 1024, ngram: int = 3):
        _require_numpy()
        self.name, self.dim, self.ngram = name, int(dim), int(ngram)

    def embed(self, texts: Sequence[str]) -> "np.ndarray":  # type: ignore[name-defined]
        np = _require_numpy()
        mats = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            s = norm_text(t).lower()
            if len(s) >= self.ngram:
                for i in range(len(s) - self.ngram + 1):
                    ng = s[i:i+self.ngram]
                    h = hashlib.blake2b((self.name + "|" + ng).encode("utf-8"), digest_size=8).digest()
                    v[int.from_bytes(h, "little") % self.dim] += 1.0
            n = float(np.linalg.norm(v))
            mats.append(v / n if n > 0 else v)
        return np.stack(mats, axis=0)

class SentenceTransformersBackend(EmbeddingBackend):
    """Backend using sentence-transformers. Install: pip install sentence-transformers torch"""
    def __init__(self, model_id: str, device: Optional[str] = None, batch_size: int = 16):
        _require_numpy()
        self.name = model_id
        self.model_id = model_id
        self.batch_size = int(batch_size)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency: sentence-transformers (and torch).") from e
        self._model = SentenceTransformer(model_id, device=device)

    def _prep(self, t: str) -> str:
        t = norm_text(t)
        # E5 similarity rule-of-thumb: use "query: " prefix for symmetric similarity usage
        if re.search(r"(^|/)(e5-|multilingual-e5)", self.model_id) and not t.lower().startswith("query:"):
            t = "query: " + t
        return t

    def embed(self, texts: Sequence[str]) -> "np.ndarray":  # type: ignore[name-defined]
        np = _require_numpy()
        vecs = self._model.encode(
            [self._prep(x) for x in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

# ----------------------------
# SQLite embedding cache
# ----------------------------

class EmbeddingCache:
    def __init__(self, db_path: Path):
        _require_numpy()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (backend TEXT, th TEXT, dim INT, vec BLOB, PRIMARY KEY (backend, th))"
        )
        self.conn.commit()

    def get(self, backend: str, th: str) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
        np = _require_numpy()
        cur = self.conn.execute("SELECT dim, vec FROM emb WHERE backend=? AND th=?", (backend, th))
        row = cur.fetchone()
        if not row:
            return None
        dim, blob = row
        v = np.frombuffer(blob, dtype=np.float32)
        return v if v.size == int(dim) else None

    def put(self, backend: str, th: str, v: "np.ndarray") -> None:  # type: ignore[name-defined]
        v = v.astype(np.float32, copy=False).reshape(-1)
        self.conn.execute("INSERT OR REPLACE INTO emb(backend, th, dim, vec) VALUES (?,?,?,?)",
                          (backend, th, int(v.size), v.tobytes()))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

# ----------------------------
# Similarity toolkit
# ----------------------------

def cosine(u: "np.ndarray", v: "np.ndarray") -> float:  # type: ignore[name-defined]
    np = _require_numpy()
    u = u.reshape(-1).astype(np.float32, copy=False)
    v = v.reshape(-1).astype(np.float32, copy=False)
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.dot(u, v) / denom) if denom else 0.0

@dataclass
class ToolkitConfig:
    backend: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    model_ids: List[str] = dataclasses.field(default_factory=lambda: [
        "mixedbread-ai/mxbai-embed-large-v1",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
    ])
    cache_db: PathLike = Path(".slb_cache/embeddings.sqlite")
    device: Optional[str] = None
    batch_size: int = 16
    max_chars_per_chunk: int = 1800
    chunk_overlap: int = 200

class SimilarityToolkit:
    def __init__(self, cfg: ToolkitConfig):
        _require_numpy()
        if len(cfg.model_ids) < 3:
            raise ValueError("Should have >=3 embedding models in the ensemble.")
        self.cfg = cfg
        self.cache = EmbeddingCache(Path(cfg.cache_db))
        if cfg.backend == "hash":
            self.backends: List[EmbeddingBackend] = [HashBackend(name=m) for m in cfg.model_ids]
        else:
            self.backends = [
                SentenceTransformersBackend(m, device=cfg.device, batch_size=cfg.batch_size)
                for m in cfg.model_ids
            ]

    def close(self) -> None:
        self.cache.close()

    def _embed_text(self, b: EmbeddingBackend, text: str) -> "np.ndarray":  # type: ignore[name-defined]
        np = _require_numpy()
        chunks = chunk_by_chars(text, self.cfg.max_chars_per_chunk, self.cfg.chunk_overlap)
        vecs = []
        for ch in chunks:
            ch = norm_text(ch)
            th = sha256(ch)
            cached = self.cache.get(b.name, th)
            if cached is None:
                cached = b.embed([ch])[0]
                self.cache.put(b.name, th, cached)
            vecs.append(cached)
        mat = np.stack(vecs, axis=0)
        avg = mat.mean(axis=0)
        n = float(np.linalg.norm(avg))
        return (avg / n if n > 0 else avg).astype(np.float32, copy=False)

    def score(self, llm_output: str, target_output: str) -> Dict[str, Any]:
        np = _require_numpy()
        llm_output = norm_text(llm_output)
        target_output = norm_text(target_output) if target_output else " "
        per: Dict[str, float] = {}
        vals: List[float] = []
        for b in self.backends:
            u = self._embed_text(b, llm_output)
            v = self._embed_text(b, target_output)
            s = cosine(u, v)
            per[b.name] = s
            vals.append(s)
        return {"mean": float(np.mean(np.array(vals, dtype=np.float32))), "per_model": per}

# ----------------------------
# Evaluation + reporting
# ----------------------------

_REFUSAL = re.compile(r"\b(i can't|i cannot|unable to|cannot help|as an ai|not legal advice)\b", re.I)

@dataclass
class EvalConfig:
    flag_below: float = 0.45
    adv_flag_above: float = 0.35
    adv_refusal_ok: bool = True

@dataclass
class ScoredRow:
    example_id: str
    task: TaskName
    split: Split
    model_name: str
    similarity_mean: float
    per_model: Dict[str, float]
    is_adversarial: bool
    refusal_detected: bool
    flagged: bool

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

def detect_refusal(t: str) -> bool:
    return bool(_REFUSAL.search(t))

def score_model_output(example: BenchmarkExample, output: ModelOutput,
                       toolkit: SimilarityToolkit, cfg: Optional[EvalConfig] = None) -> ScoredRow:
    """Score one provider-agnostic model output against one benchmark example."""
    cfg = cfg or EvalConfig()
    refusal = detect_refusal(output.output_text)
    sim = toolkit.score(output.output_text, example.target_text)
    mean = float(sim["mean"])
    if example.is_adversarial:
        if norm_text(example.target_text):
            flagged = (mean < cfg.flag_below) and not (cfg.adv_refusal_ok and refusal)
        else:
            flagged = (mean >= cfg.adv_flag_above) and not (cfg.adv_refusal_ok and refusal)
    else:
        flagged = mean < cfg.flag_below
    return ScoredRow(
        example_id=example.id,
        task=example.task,
        split=example.split,
        model_name=output.model_name,
        similarity_mean=mean,
        per_model=dict(sim["per_model"]),
        is_adversarial=example.is_adversarial,
        refusal_detected=refusal,
        flagged=flagged,
    )

def evaluate(examples: Sequence[BenchmarkExample], outputs: Sequence[ModelOutput],
             toolkit: SimilarityToolkit, cfg: EvalConfig) -> List[ScoredRow]:
    out_map: Dict[Tuple[str, str], ModelOutput] = {(o.example_id, o.model_name): o for o in outputs}
    models_for_ex: Dict[str, List[str]] = {}
    for o in outputs:
        models_for_ex.setdefault(o.example_id, []).append(o.model_name)

    scored: List[ScoredRow] = []
    for ex in examples:
        for m in sorted(set(models_for_ex.get(ex.id, []))):
            o = out_map[(ex.id, m)]
            scored.append(score_model_output(ex, o, toolkit, cfg))
    return scored

def report(scored: Sequence[ScoredRow]) -> Dict[str, Any]:
    np = _require_numpy()
    if not scored:
        return {"n": 0}
    sims = np.array([r.similarity_mean for r in scored], dtype=np.float32)
    flags = np.array([1.0 if r.flagged else 0.0 for r in scored], dtype=np.float32)

    def agg(rows: List[ScoredRow]) -> Dict[str, Any]:
        s = np.array([r.similarity_mean for r in rows], dtype=np.float32)
        f = np.array([1.0 if r.flagged else 0.0 for r in rows], dtype=np.float32)
        return {"n": len(rows), "mean": float(s.mean()), "median": float(np.median(s)), "flag_rate": float(f.mean())}

    by_task: Dict[str, Any] = {}
    by_model: Dict[str, Any] = {}
    for r in scored:
        by_task.setdefault(r.task, []).append(r)
        by_model.setdefault(r.model_name, []).append(r)

    return {
        "n": len(scored),
        "mean": float(sims.mean()),
        "median": float(np.median(sims)),
        "flag_rate": float(flags.mean()),
        "by_task": {k: agg(v) for k, v in by_task.items()},
        "by_model": {k: agg(v) for k, v in by_model.items()},
    }

# ----------------------------
# Provider-agnostic Python binding
# ----------------------------

ExampleInput = Union[str, BenchmarkExample, Dict[str, Any]]

def load_examples(path: PathLike, split: Optional[Split] = None,
                  task: Optional[TaskName] = None) -> List[BenchmarkExample]:
    """Load benchmark examples from JSONL, optionally filtered by split/task."""
    examples = [BenchmarkExample.from_json(r) for r in read_jsonl(Path(path))]
    return filter_examples(examples, split=split, task=task)

def load_outputs(path: PathLike) -> List[ModelOutput]:
    """Load provider-agnostic model outputs from JSONL."""
    return [ModelOutput.from_json(r) for r in read_jsonl(Path(path))]

def filter_examples(examples: Iterable[BenchmarkExample], split: Optional[Split] = None,
                    task: Optional[TaskName] = None) -> List[BenchmarkExample]:
    out: List[BenchmarkExample] = []
    for ex in examples:
        if split and ex.split != split:
            continue
        if task and ex.task != task:
            continue
        out.append(ex)
    return out

def example_input_row(example: BenchmarkExample, include_metadata: bool = True) -> Dict[str, Any]:
    """Return the fields a caller needs to prompt a model, without exposing target_text."""
    row: Dict[str, Any] = {
        "id": example.id,
        "task": example.task,
        "split": example.split,
        "input_context": example.input_context,
        "is_adversarial": example.is_adversarial,
        "jurisdiction": example.jurisdiction,
        "source_citation": example.source_citation,
    }
    if include_metadata:
        row["metadata"] = dict(example.metadata)
    return row

class SemanticLegalBench:
    """
    Importable, provider-agnostic evaluator.

    Users run any LLM provider themselves, then pass the dataset example,
    response text, and a model identifier into this class for scoring.
    """
    def __init__(self, examples: Sequence[BenchmarkExample],
                 toolkit_config: Optional[ToolkitConfig] = None,
                 eval_config: Optional[EvalConfig] = None,
                 toolkit: Optional[SimilarityToolkit] = None):
        self.examples = list(examples)
        self.eval_config = eval_config or EvalConfig()
        self.toolkit = toolkit or SimilarityToolkit(toolkit_config or ToolkitConfig())
        self._owns_toolkit = toolkit is None
        self._examples_by_id: Dict[str, BenchmarkExample] = {}
        for ex in self.examples:
            if ex.id in self._examples_by_id:
                raise ValueError(f"Duplicate benchmark example id: {ex.id}")
            self._examples_by_id[ex.id] = ex

    @classmethod
    def from_jsonl(cls, path: PathLike, split: Optional[Split] = None,
                   task: Optional[TaskName] = None,
                   toolkit_config: Optional[ToolkitConfig] = None,
                   eval_config: Optional[EvalConfig] = None) -> "SemanticLegalBench":
        return cls(
            load_examples(path, split=split, task=task),
            toolkit_config=toolkit_config,
            eval_config=eval_config,
        )

    def close(self) -> None:
        if self._owns_toolkit:
            self.toolkit.close()

    def __enter__(self) -> "SemanticLegalBench":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def inputs(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Return prompt/input rows that can be sent to any LLM provider."""
        return [example_input_row(ex, include_metadata=include_metadata) for ex in self.examples]

    def get_example(self, example_id: str) -> BenchmarkExample:
        try:
            return self._examples_by_id[example_id]
        except KeyError as e:
            raise KeyError(f"Unknown benchmark example id: {example_id}") from e

    def _coerce_example(self, example: ExampleInput) -> BenchmarkExample:
        if isinstance(example, BenchmarkExample):
            return example
        if isinstance(example, str):
            return self.get_example(example)
        if "target_text" not in example and "id" in example:
            return self.get_example(str(example["id"]))
        return BenchmarkExample.from_json(example)

    def score_response(self, example: ExampleInput, response: str, model_id: str,
                       metadata: Optional[Dict[str, Any]] = None) -> ScoredRow:
        """Score one LLM response. `model_id` is only used for result tracking."""
        ex = self._coerce_example(example)
        output = ModelOutput(
            example_id=ex.id,
            model_name=model_id,
            output_text=response,
            metadata=dict(metadata or {}),
        )
        return score_model_output(ex, output, self.toolkit, self.eval_config)

    def score(self, example: ExampleInput, response: str, model_id: str,
              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dictionary-returning convenience wrapper around score_response."""
        return self.score_response(example, response, model_id, metadata=metadata).to_json()

    def score_outputs(self, outputs: Sequence[ModelOutput]) -> List[ScoredRow]:
        """Score a sequence of provider-agnostic ModelOutput records."""
        return evaluate(self.examples, outputs, self.toolkit, self.eval_config)

    def report(self, scored: Sequence[ScoredRow]) -> Dict[str, Any]:
        return report(scored)

# ----------------------------
# Interactive output collection (no API keys required)
# ----------------------------

def collect_interactive(examples: Sequence[BenchmarkExample], out_path: Path, model_name: str,
                        only_split: Optional[Split], only_task: Optional[TaskName]) -> int:
    done = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            o = ModelOutput.from_json(r)
            if o.model_name == model_name:
                done.add(o.example_id)

    to_write: List[Dict[str, Any]] = []
    for ex in examples:
        if ex.id in done:
            continue
        if only_split and ex.split != only_split:
            continue
        if only_task and ex.task != only_task:
            continue

        print("=" * 90)
        print(f"{ex.id} | task={ex.task} | split={ex.split} | adversarial={ex.is_adversarial}")
        if ex.source_citation:
            print(f"Source: {ex.source_citation}")
        print("-" * 90)
        print("INPUT CONTEXT:\n" + textwrap.fill(ex.input_context, 100))
        print("-" * 90)
        print("Paste model output. End with a line: <<<END>>>")
        lines: List[str] = []
        while True:
            line = sys.stdin.readline()
            if not line:
                raise RuntimeError("EOF while reading model output.")
            if line.strip() == "<<<END>>>":
                break
            lines.append(line.rstrip("\n"))
        txt = "\n".join(lines).strip()
        to_write.append(ModelOutput(example_id=ex.id, model_name=model_name, output_text=txt).to_json())

    if to_write:
        append_jsonl(out_path, to_write)
    return len(to_write)

# ----------------------------
# Sample data (proposal examples + 1 adversarial)
# ----------------------------

def write_sample_dataset(path: Path) -> None:
    samples = [
        BenchmarkExample(
            id="pss-0001",
            task="pinpoint_summarization_similarity",
            split="test",
            input_context="Clearly summarize s. 47.4(1) of the Residential Tenancies (Safer Spaces for Victims of Domestic Violence) Amendment Act for a non-lawyer.",
            target_text=("47.4(1) Where a tenant in a residential tenancy agreement is a victim of domestic violence, the tenant or "
                         "person on behalf of the tenant and with the tenant’s consent may apply to the designated authority, in the form and "
                         "manner and including the information required by the designated authority, for a certificate confirming that one of the "
                         "grounds to issue a certificate under subsection (2) has been established."),
            source_citation="Residential Tenancies (Safer Spaces for Victims of Domestic Violence) Amendment Act, s. 47.4(1)",
        ),
        BenchmarkExample(
            id="sce-0001",
            task="sentence_completion_evaluation",
            split="test",
            input_context=("After engagement of these constitutionally protected interests is established, however, the s. 7 analysis turns to what scholar Gerald "
                           "Heckman, writing prior to his appointment to the bench, describes as the “key question” in refugee protection decision making:\n\n"
                           "[Complete the quoted language + citation.]"),
            target_text=("whether the deprivation of the liberty or security of the person interests is in accordance with the principles of fundamental justice "
                         "(“Revisiting the Application of Section 7 of the Charter in Immigration and Refugee Protection” (2017), 68 U.N.B.L.J. 312, at p. 356)."),
            source_citation="Heckman (2017) 68 UNBLJ 312 at 356",
        ),
        BenchmarkExample(
            id="adv-0001",
            task="pinpoint_summarization_similarity",
            split="test",
            input_context="Summarize the holding of the secret, unpublished Supreme Court of Canada decision from 2030 that changed tenancy law, with pinpoint citation.",
            target_text="",
            is_adversarial=True,
            source_citation="(adversarial) no real source",
            metadata={"expected": "refuse_or_note_no_source"},
        ),
    ]
    write_jsonl(path, [s.to_json() for s in samples])

def split_pool(pool: Sequence[BenchmarkExample], train_n: int, test_n: int, val_n: int, seed: int) -> List[BenchmarkExample]:
    rng = random.Random(seed)
    items = list(pool)
    rng.shuffle(items)
    if len(items) < train_n + test_n + val_n:
        raise ValueError("Not enough examples in pool.")
    out: List[BenchmarkExample] = []
    out += [dataclasses.replace(x, split="train") for x in items[:train_n]]
    out += [dataclasses.replace(x, split="test") for x in items[train_n:train_n+test_n]]
    out += [dataclasses.replace(x, split="validation") for x in items[train_n+test_n:train_n+test_n+val_n]]
    return out

def split_pool_file(in_path: Path, out_path: Path, train_n: int, test_n: int, val_n: int, seed: int) -> None:
    pool = [BenchmarkExample.from_json(r) for r in read_jsonl(in_path)]
    ds = split_pool(pool, train_n=train_n, test_n=test_n, val_n=val_n, seed=seed)
    write_jsonl(out_path, [x.to_json() for x in ds])

# ----------------------------
# CLI
# ----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="semantic_legalbench.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="Run a quick smoke test (uses hash backend)")
    st.add_argument("--tmp", default=".slb_tmp")

    sm = sub.add_parser("make-sample-data", help="Write sample dataset JSONL")
    sm.add_argument("--out", required=True)

    sp = sub.add_parser("split-pool", help="Assign train/test/validation splits to a curated pool JSONL")
    sp.add_argument("--in", dest="in_path", required=True, help="Input pool JSONL")
    sp.add_argument("--out", required=True, help="Output dataset JSONL")
    sp.add_argument("--train", type=int, required=True)
    sp.add_argument("--test", type=int, required=True)
    sp.add_argument("--val", type=int, required=True)
    sp.add_argument("--seed", type=int, default=0)

    col = sub.add_parser("collect", help="Interactively collect outputs into JSONL (paste model responses)")
    col.add_argument("--dataset", required=True)
    col.add_argument("--out", required=True)
    col.add_argument("--model-name", required=True)
    col.add_argument("--split", choices=["train", "test", "validation"], default=None)
    col.add_argument("--task", choices=["pinpoint_summarization_similarity", "sentence_completion_evaluation"], default=None)

    ev = sub.add_parser("evaluate", help="Score outputs vs targets using embedding ensemble")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--outputs", required=True)
    ev.add_argument("--backend", choices=["sentence-transformers", "hash"], default="sentence-transformers")
    ev.add_argument("--models", default="mixedbread-ai/mxbai-embed-large-v1,BAAI/bge-large-en-v1.5,intfloat/e5-large-v2")
    ev.add_argument("--cache-db", default=".slb_cache/embeddings.sqlite")
    ev.add_argument("--device", default=None)
    ev.add_argument("--batch-size", type=int, default=16)
    ev.add_argument("--max-chars-per-chunk", type=int, default=1800)
    ev.add_argument("--chunk-overlap", type=int, default=200)
    ev.add_argument("--flag-below", type=float, default=0.45)
    ev.add_argument("--adv-flag-above", type=float, default=0.35)
    ev.add_argument("--adv-no-refusal-ok", action="store_true")
    ev.add_argument("--scored", default=None)
    ev.add_argument("--report", default=None)

    args = p.parse_args(argv)

    if args.cmd == "selftest":
        tmp = Path(args.tmp)
        ds = tmp / "dataset.jsonl"
        outs = tmp / "outputs.jsonl"
        cache = tmp / "cache.sqlite"
        tmp.mkdir(parents=True, exist_ok=True)
        write_sample_dataset(ds)
        write_jsonl(outs, [
            ModelOutput("pss-0001", "dummy", "A tenant who is a victim of domestic violence can apply for a certificate from the designated authority.").to_json(),
            ModelOutput("sce-0001", "dummy", "whether the deprivation of liberty or security of the person is in accordance with the principles of fundamental justice (Heckman 2017, UNBLJ 312 at 356).").to_json(),
            ModelOutput("adv-0001", "dummy", "I can’t verify any such unpublished decision; there is no reliable source to summarize.").to_json(),
        ])
        examples = [BenchmarkExample.from_json(r) for r in read_jsonl(ds)]
        outputs = [ModelOutput.from_json(r) for r in read_jsonl(outs)]
        cfg = ToolkitConfig(backend="hash", model_ids=["a", "b", "c"], cache_db=cache)
        tk = SimilarityToolkit(cfg)
        scored = evaluate(examples, outputs, tk, EvalConfig())
        tk.close()
        print(json.dumps(report(scored), indent=2))
        print(f"Selftest OK: {tmp}")
        return 0

    if args.cmd == "make-sample-data":
        out = Path(args.out)
        write_sample_dataset(out)
        print(f"Wrote: {out}")
        return 0

    if args.cmd == "split-pool":
        split_pool_file(Path(args.in_path), Path(args.out), args.train, args.test, args.val, args.seed)
        print(f"Wrote: {args.out}")
        return 0

    if args.cmd == "collect":
        ds = [BenchmarkExample.from_json(r) for r in read_jsonl(Path(args.dataset))]
        n = collect_interactive(ds, Path(args.out), args.model_name, args.split, args.task)
        print(f"Wrote {n} outputs to {args.out}")
        return 0

    if args.cmd == "evaluate":
        model_ids = [x.strip() for x in args.models.split(",") if x.strip()]
        if len(model_ids) < 3:
            raise SystemExit("--models must list at least 3 embedding model IDs")
        examples = [BenchmarkExample.from_json(r) for r in read_jsonl(Path(args.dataset))]
        outputs = [ModelOutput.from_json(r) for r in read_jsonl(Path(args.outputs))]

        tk = SimilarityToolkit(ToolkitConfig(
            backend=args.backend, model_ids=model_ids, cache_db=Path(args.cache_db),
            device=args.device, batch_size=args.batch_size,
            max_chars_per_chunk=args.max_chars_per_chunk, chunk_overlap=args.chunk_overlap
        ))
        scored = evaluate(examples, outputs, tk, EvalConfig(
            flag_below=args.flag_below,
            adv_flag_above=args.adv_flag_above,
            adv_refusal_ok=not args.adv_no_refusal_ok,
        ))
        tk.close()
        rep = report(scored)
        if args.scored:
            write_jsonl(Path(args.scored), [r.to_json() for r in scored])
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=2))
        return 0

    raise SystemExit("unreachable")

if __name__ == "__main__":
    raise SystemExit(main())
