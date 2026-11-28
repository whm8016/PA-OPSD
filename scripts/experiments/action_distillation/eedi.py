"""Auditable topic-to-construct crosswalk for the three-role teacher; never a student diagnosis."""

import argparse
import csv
import json
from pathlib import Path

MAPPING = (
    (
        ("带余除法", "除法中四量关系"),
        "14241",
        "Interpret remainders from written division problems",
    ),
    (
        ("多边形的内角和",),
        "14154",
        "Work out the sum of the interior angles in polygons",
    ),
    (
        ("路程速度时间", "单人变速问题"),
        "15517",
        "Calculate speed where distance and time are in the appropriate units",
    ),
    (
        ("混合运算", "四则运算"),
        "13790",
        "Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division",
    ),
    (
        ("奇数与偶数", "奇偶性"),
        "15353",
        "Prove statements involving odd and even numbers",
    ),
    (("平均数",), "13813", "Solve missing number problems involving the mean"),
    (
        ("周长",),
        "13803",
        "Calculate the perimeter of a rectilinear shape by counting squares",
    ),
    (
        ("长方形的面积",),
        "14019",
        "Given the area of a rectangle, calculate a missing dimension",
    ),
    (("质因数", "质数"), "13946", "Recognise a prime factor"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("map", type=Path)
    a = p.parse_args()
    with a.map.open(encoding="cp1252") as f:
        bank = {
            (r["MisconceptionId"], r["ConstructName"]): r["MisconceptionName"]
            for r in csv.DictReader(f)
        }
    out = {}
    for split in ("train", "dev"):
        for l in (
            (a.bundle / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
        ):
            r = json.loads(l)
            concept = r["public"].rpartition("知识点:")[2].splitlines()[0]
            z = next(
                (
                    (idx, c, bank[idx, c])
                    for words, idx, c in MAPPING
                    if any(w in concept for w in words)
                ),
                None,
            )
            if z:
                out[r["id"]] = {"id": z[0], "construct": z[1], "candidate": z[2]}
    (a.bundle / "eedi_candidates.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
