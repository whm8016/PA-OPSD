#!/usr/bin/env python3
"""Offline Eedi cosine candidates only; never labels or student runtime inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/eedi/misconceptions-graph/data/misconception-map.csv"
DEFAULT_MODEL = "BAAI/bge-m3"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def retrieval_query(seed):
    """Use only the question stem and explicitly allowlisted knowledge concepts."""
    question = seed["student_input"]["current_question"]
    concepts = seed["teacher_privileged"]["knowledge_concepts"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("current_question must be a nonempty string")
    if not isinstance(concepts, list) or any(not isinstance(c, str) for c in concepts):
        raise ValueError("knowledge_concepts must be a list of strings")
    question = re.sub(r"^\s*\[\d+\]\s*", "", question)
    # Native question_info appends knowledge, timestamp and student outcome.
    question = re.split(
        r"(?:^|\n)\s*(?:知识点|时间|结果|答案|正确答案|标准答案|解析|分析|诊断|"
        r"answer|standard_answer|analysis|diagnosis)\s*[:：]",
        question, maxsplit=1, flags=re.IGNORECASE,
    )[0].strip()
    if not question:
        raise ValueError("empty question stem")
    parts = [question]
    cleaned = list(dict.fromkeys(c.strip() for c in concepts if c.strip()))
    if cleaned:
        parts.append("知识点: " + ", ".join(cleaned))
    return "\n".join(parts)


def validate_revision(revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a fixed 40-character model commit SHA (not main or a branch)")


def load_index(path):
    """Read the real graph schema without importing the pandas-based seed builder."""
    path = Path(path)
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    reader = csv.DictReader(io.StringIO(text))
    required = {"MisconceptionId", "MisconceptionName", "ConstructName", "SubTopicName", "TopicName", "SubjectName"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("missing Eedi misconception-map columns")
    source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "encoding": encoding}
    grouped = {}
    for line_number, row in enumerate(reader, 2):
        identifier = (row["MisconceptionId"] or "").strip()
        name = (row["MisconceptionName"] or "").strip()
        if not identifier or not name:
            raise ValueError(f"empty misconception id/text at CSV line {line_number}")
        entry = grouped.setdefault(identifier, {"id": identifier, "text": name, "contexts": set(), "rows": []})
        if entry["text"] != name:
            raise ValueError(f"conflicting misconception text for {identifier}")
        context = " | ".join((row[k] or "").strip() for k in ("SubjectName", "TopicName", "SubTopicName", "ConstructName"))
        entry["contexts"].add(context)
        entry["rows"].append(line_number)
    if not grouped:
        raise ValueError("empty misconception index")
    index = []
    for identifier in sorted(grouped):
        entry = grouped[identifier]
        index.append({
            "id": identifier, "text": entry["text"],
            "retrieval_text": entry["text"] + "\n" + "\n".join(sorted(entry["contexts"])),
            "source": {**source, "csv_lines": entry["rows"]},
        })
    return index


def normalized_vectors(vectors, expected_count):
    vectors = list(vectors)
    if len(vectors) != expected_count:
        raise ValueError("embedding count mismatch")
    normalized = []
    dimension = None
    for vector in vectors:
        values = [float(v) for v in vector]
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError("embeddings must be finite nonempty vectors")
        if dimension is None:
            dimension = len(values)
        if len(values) != dimension:
            raise ValueError("embedding dimension mismatch")
        scale = max(abs(v) for v in values)
        if not scale:
            raise ValueError("zero embedding")
        values = [v / scale for v in values]
        norm = math.sqrt(sum(v * v for v in values))
        normalized.append([v / norm for v in values])
    return normalized


def candidate_rows(seeds, index, encode, embedding, top_k=5, threshold=0.0):
    """encode is injectable for deterministic small fixtures, not lexical fallback."""
    validate_revision(embedding.get("revision"))
    if not embedding.get("model") or not re.fullmatch(r"[0-9a-f]{64}", embedding.get("source_sha256", "")):
        raise ValueError("embedding model and model source_sha256 are required")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be positive")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be a cosine score in [-1, 1]")
    if not index:
        raise ValueError("empty misconception index")
    seeds = list(seeds)
    seen = set()
    queries = []
    for seed in seeds:
        identifier = seed.get("sample_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("sample_id must be a unique nonempty string")
        seen.add(identifier)
        queries.append(retrieval_query(seed))
    documents = normalized_vectors(encode([item["retrieval_text"] for item in index]), len(index))
    for seed, query in zip(seeds, queries):
        vector = normalized_vectors(encode([query]), 1)[0]
        if len(vector) != len(documents[0]):
            raise ValueError("query/document embedding dimension mismatch")
        scored = []
        for item, document in zip(index, documents):
            score = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, document))))
            if score >= threshold:
                scored.append({k: item[k] for k in ("id", "text", "source")} | {"score": score})
        scored.sort(key=lambda item: (-item["score"], item["id"]))
        yield {
            "sample_id": seed["sample_id"], "candidates": scored[:top_k],
            "embedding": dict(embedding), "query_fingerprint": fingerprint(query),
        }


def model_source_hash(directory):
    """Content-address the exact local model snapshot, including weights/config."""
    digest = hashlib.sha256()
    files = sorted(p for p in Path(directory).rglob("*") if p.is_file())
    if not files:
        raise ValueError("empty model snapshot")
    for path in files:
        content = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content.update(chunk)
        digest.update(json.dumps([path.relative_to(directory).as_posix(), content.hexdigest()], ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_encoder(model, revision, batch_size=16, device="cpu", local_files_only=True):
    validate_revision(revision)
    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence_transformers is unavailable; no dependency installation or lexical fallback is performed") from exc
    snapshot = snapshot_download(repo_id=model, revision=revision, local_files_only=local_files_only)
    source_hash = model_source_hash(Path(snapshot))
    encoder = SentenceTransformer(snapshot, device=device, local_files_only=True, trust_remote_code=False)

    def encode(texts):
        return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

    return encode, {"model": model, "revision": revision, "source_sha256": source_hash, "similarity": "cosine", "normalized": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True, help="fixed model commit SHA, never main")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0, help="cosine cutoff, not a probability")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true", help="allow fetching the pinned model; default is cache-only")
    args = parser.parse_args(argv)
    try:
        validate_revision(args.revision)
        if args.output.resolve() in (args.seed.resolve(), args.source.resolve()):
            raise ValueError("output must be a separate candidate sidecar")
        if args.output.exists():
            raise ValueError("output already exists; refusing to overwrite")
        if args.batch_size < 1 or args.top_k < 1 or not math.isfinite(args.threshold) or not -1 <= args.threshold <= 1:
            raise ValueError("invalid batch-size, top-k or cosine threshold")
        with args.seed.open(encoding="utf-8") as stream:
            seeds = [json.loads(line) for line in stream if line.strip()]
        index = load_index(args.source)
        encode, metadata = load_encoder(args.model, args.revision, args.batch_size, args.device, not args.allow_download)
        rows = list(candidate_rows(seeds, index, encode, metadata, args.top_k, args.threshold))
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
