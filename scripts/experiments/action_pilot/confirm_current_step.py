"""Freeze an unseen-student, unseen-question development panel for the first-hint pilot."""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.build_pa_opsd_gold_split import question_group
from scripts.experiments.action_pilot import classifier_teacher as c
from scripts.experiments.action_pilot import evidence_probe as e
from scripts.experiments.action_pilot import run as g
from scripts.experiments.training_free_memory import prepare as temporal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source
    out = args.output
    cfg = g.read_json(source / "manifest.json")
    split = g.read_json(g.SPLIT / "split_manifest.json")
    dev = set(split["ids"]["validation"])
    train = set(split["ids"]["train"])
    assert g.digest(g.SPLIT / "split_manifest.json") == cfg["split_sha256"]
    refs = {
        r["sample_id"]: r["target"]["diagnosis"]
        for r in g.read_rows(
            g.ROOT
            / "outputs/pa_opsd/gold_evidence_first/inputs_v1/dev_references.jsonl"
        )
    }
    assert set(refs) == dev
    hist = {
        g.history_key(r): r
        for r in g.read_rows(g.REPRO / "gold_inputs.jsonl")
        if g.history_key(r) in dev | train
    }
    bank = g.indexed(g.read_rows(g.BANK), "id")
    identities = {
        k: (r["uid"], *question_group(r["question_info"])) for k, r in hist.items()
    }
    excluded = (set(), set(), set())
    for k in train:
        c.remember(identities[k], excluded)
    rows = []
    audit = []
    repeats = []
    smoke = []
    for label in g.LABELS:
        eligible = []
        for k in sorted(
            dev,
            key=lambda k: g.sha256(
                ("proposal-confirm-20261127|" + k).encode()
            ).digest(),
        ):
            r = hist[k]
            identity = identities[k]
            sol = bank[identity[1]]["info"]
            if (
                refs[k] != label
                or c.overlaps(identity, excluded)
                or g.IMAGE.search(r["question_info"])
                or not sol.get("answer")
                or not str(sol.get("analysis") or "").strip()
            ):
                continue
            c.remember(identity, excluded)
            eligible.append(k)
            _, route = temporal.sample_inputs(dict(r, _key=k), {"memory": []})
            pos = route["temporal_positions"]
            rows.append(
                {
                    "sample_id": k,
                    "current": r["question_info"],
                    "history": [(p, r["history_info"][p - 1]) for p in pos],
                    "facts": e.facts(r["question_info"], r["history_info"], pos),
                    "solution": {
                        "standard_answer": sol["answer"],
                        "analysis": sol["analysis"],
                    },
                }
            )
            audit.append(
                {
                    "sample_id": k,
                    "gold_diagnosis_for_stratification_only": label,
                    "uid": identity[0],
                    "qid": identity[1],
                    "stem_sha256": identity[2],
                }
            )
            if len(eligible) == 6:
                break
        if not eligible:
            raise ValueError(f"Empty class {label}")
        repeats += eligible[:2]
        smoke += eligible[:1]
    out.mkdir(parents=True, exist_ok=False)
    g.write_rows(out / "inputs.jsonl", rows)
    g.write_rows(out / "local_strata.jsonl", audit)
    ids = [r["sample_id"] for r in rows]
    cfg.update(
        ids=ids,
        smoke_ids=smoke,
        repeat_ids=repeats,
        groups={"regression": [], "fresh": ids},
        cohort_labels={
            "regression": {},
            "fresh": dict(
                Counter(r["gold_diagnosis_for_stratification_only"] for r in audit)
            ),
        },
        inputs_sha256=g.digest(out / "inputs.jsonl"),
        confirmation="Frozen development subset. All training students, question IDs and normalized stems excluded. Official development outcomes were previously explored. No prior teaching-hint outputs from these conditions used in selection. No sealed examples.",
        primary="Two contrasts. Paired safe step: step_solution minus original; same-rule privilege: step_solution minus step_public.",
    )
    g.write_json(out / "manifest.json", cfg)
    with (
        ZipFile(source / "inference.zip") as z,
        ZipFile(out / "inference.zip", "x", ZIP_DEFLATED) as dest,
    ):
        files = {
            n: z.read(n)
            for n in z.namelist()
            if n not in ("manifest.json", "inputs.jsonl", "SHA256SUMS")
        }
        for n in ("inputs.jsonl", "manifest.json"):
            files[n] = (out / n).read_bytes()
        for n, b in files.items():
            dest.writestr(n, b)
        dest.writestr(
            "SHA256SUMS",
            "".join(
                g.sha256(v).hexdigest() + "  " + k + "\n" for k, v in files.items()
            ),
        )
    print(cfg["cohort_labels"])


if __name__ == "__main__":
    main()
