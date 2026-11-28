"""Measure named historical exercise coverage under equal Qwen record-token budgets."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from tokenizers import Tokenizer

from .prepare import BUNDLE, EXPLICIT_ID, ROOT, SPLIT

MEMORY = ROOT / "outputs/pa_opsd/training_free_memory/gold140_20261125_verified"
BUDGETS = (128, 256, 512, 1024, 2048)
HEADER = re.compile(r"^\[(\d+)\]")
WORDS = re.compile(r"[\u4e00-\u9fff]+|[0-9]+|[a-z]+", re.IGNORECASE)


def terms(text):
    words = []
    for word in WORDS.findall(text.lower()):
        if "\u4e00" <= word[0] <= "\u9fff":
            words.extend(word)
            words.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            words.append(word)
    return Counter(words)


def bm25(history, query):
    documents = [terms(record) for record in history]
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(history)
    frequency = Counter(term for doc in documents for term in doc)
    scores = [0.0] * len(history)
    for term in terms(query):
        idf = math.log1p(
            (len(history) - frequency[term] + 0.5) / (frequency[term] + 0.5)
        )
        for i, doc in enumerate(documents):
            tf = doc[term]
            scores[i] += (
                idf * 2.2 * tf / (tf + 1.2 * (0.25 + 0.75 * lengths[i] / average))
            )
    return scores


def load_index(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result = {row["_key"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample keys in {path}")
    return result


def run(output, tokenizer_path):
    paths = [
        BUNDLE / "inputs.jsonl",
        MEMORY / "queries_only.jsonl",
        SPLIT,
        tokenizer_path,
    ]
    samples, questions = map(load_index, paths[:2])
    ids = json.loads(SPLIT.read_text(encoding="utf-8"))["ids"]["validation"]
    if len(ids) != 140 or set(samples) != set(questions) or set(samples) != set(ids):
        raise ValueError("Expected exactly the frozen 140 development cases")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    groups = defaultdict(list)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for key in ids:
            sample = samples[key]
            history = sample["history_info"]
            qids = [HEADER.match(record)[1] for record in history]
            current = HEADER.match(sample["question_info"])[1]
            queries = [item["query"] for item in questions[key]["memory"]]
            named = [
                {a or b for a, b in EXPLICIT_ID.findall(q)} - {current} for q in queries
            ]
            dates = [re.search(r"时间[:：]\s*(\S+)", r)[1] for r in history]
            sizes = [
                len(e.ids)
                for e in tokenizer.encode_batch(
                    [r + "\n\n" for r in history], add_special_tokens=False
                )
            ]
            lexical = bm25(history, "\n".join(queries))
            mentioned = set.union(*named)
            order = range(len(history))
            ranks = {
                "recent": sorted(order, key=lambda i: (dates[i], i), reverse=True),
                "id_recent": sorted(
                    order,
                    key=lambda i: (qids[i] in mentioned, dates[i], i),
                    reverse=True,
                ),
                "bm25": sorted(
                    order, key=lambda i: (lexical[i], dates[i], i), reverse=True
                ),
                "id_bm25": sorted(
                    order,
                    key=lambda i: (qids[i] in mentioned, lexical[i], dates[i], i),
                    reverse=True,
                ),
            }
            for arm, ranking in ranks.items():
                for limit in BUDGETS:
                    selected, used = [], 0
                    for pos in ranking:
                        if used + sizes[pos] <= limit:
                            selected.append(pos)
                            used += sizes[pos]
                    retained = {qids[i] for i in selected}
                    hits = [
                        ids_in_query <= retained if ids_in_query else None
                        for ids_in_query in named
                    ]
                    row = {
                        "key": key,
                        "current": current,
                        "arm": arm,
                        "budget": limit,
                        "tokens": used,
                        "named": [sorted(items) for items in named],
                        "hit": hits,
                        "positions": [p + 1 for p in sorted(selected)],
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    groups[arm, limit].append(row)
    summary = []
    for (arm, limit), rows in groups.items():
        by_q = list(zip(*(r["hit"] for r in rows)))
        summary.append(
            {
                "arm": arm,
                "budget": limit,
                "median_tokens": median(r["tokens"] for r in rows),
                "correct": [v.count(True) for v in by_q],
                "eligible": [len(v) - v.count(None) for v in by_q],
            }
        )
    for name, obj in [
        ("summary", summary),
        (
            "manifest",
            {
                "samples": ids,
                "budgets": BUDGETS,
                "unit": "each whole record plus two newlines independently tokenized",
                "input_sha256": {
                    str(p.relative_to(ROOT))
                    if p.is_relative_to(ROOT)
                    else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths
                },
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
        ),
    ]:
        (output / f"{name}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"{len(ids)} cases, {len(groups)} conditions, {sum(map(len, groups.values()))} selections"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / ".codex_tmp/opd_qwen_teacher_tokenizer/tokenizer.json",
    )
    args = parser.parse_args()
    run(args.output, args.tokenizer.resolve())
