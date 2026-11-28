"""Blind the public and no-retrieval candidate hints on the three independent confirmation panels."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.experiments.action_distillation.analyze import rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("command", choices=("prepare", "score"))
    a = p.parse_args()
    folder = a.bundle / "blind_confirmation"
    if a.command == "prepare":
        folder.mkdir(exist_ok=False)
        data = {r["id"]: r for r in rows(a.bundle / "dev.jsonl")}
        cases = []
        key = {}
        rng = random.Random(19984153)
        for rd in range(1, 4):
            cycle = a.bundle / "cycles" / str(rd)
            v = [
                json.loads((cycle / name).read_text(encoding="utf-8"))
                for name in ("confirm_public.json", "confirm_candidate_off.json")
            ]
            for pair in zip(*v, strict=True):
                k = pair[0]["id"]
                assert k == pair[1]["id"]
                r = data[k]
                case = {
                    "id": k,
                    "current": r["public"].rpartition("\n当前题\n")[2],
                    "history": r["public"].rpartition("\n当前题\n")[0],
                    "teacher_reference": r["solution"],
                    "round": rd,
                }
                texts = list(dict.fromkeys(x["text"] for x in pair))
                rng.shuffle(texts)
                case["options"] = {}
                for text in texts:
                    code = f"{rng.getrandbits(48):012x}"
                    assert code not in key
                    case["options"][code] = text
                    key[code] = {
                        "id": k,
                        "round": rd,
                        "models": [
                            name
                            for x, name in zip(
                                pair, ("incumbent", "candidate"), strict=True
                            )
                            if x["text"] == text
                        ],
                    }
                cases.append(case)
        rng.shuffle(cases)
        (folder / "blind.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases),
            encoding="utf-8",
        )
        (folder / "key.json").write_text(
            json.dumps(key, ensure_ascii=False), encoding="utf-8"
        )
        return
    raw = (folder / "ratings.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    lock = folder / "ratings.sha256"
    if lock.exists() and lock.read_text() != digest:
        raise ValueError("Changed ratings after unblinding")
    ratings = json.loads(raw)
    key = json.loads((folder / "key.json").read_text())
    assert set(ratings) == set(key)
    allowed = {"U", "M", "H", "L", "V"}
    if not all(
        isinstance(v, str) and v and set(v) <= allowed and ("U" not in v or v == "U")
        for v in ratings.values()
    ):
        raise ValueError(
            "Use U usable, M math/task error, H unsupported history, L leakage, V vague/off-task"
        )
    lock.write_text(digest)
    result = {}
    for code, record in key.items():
        for model in record["models"]:
            branch = result.setdefault(str(record["round"]), {}).setdefault(
                model, {"total": 0, "U": 0, "M": 0, "H": 0, "L": 0, "V": 0}
            )
            branch["total"] += 1
            for flag in set(ratings[code]):
                branch[flag] += 1
    (folder / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
