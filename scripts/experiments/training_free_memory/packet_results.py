"""Analyze the frozen packet study, separating generated answers, direct execution and blind ratings."""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .budget_curve import BUNDLE, HEADER, MEMORY, ROOT, load_index
from .evidence_executor import FOLLOW
from .packet_followup import STUDY
from .paired_report import norm, read, summarize
from .prepare import digest, load_upstream, prompt_tokens, write_json, write_rows

DEV = ("bm25", "id_recent", "id_bm25", "packet_control", "packet")
TRANSFER = ("transfer_control", "transfer_packet")
MECHANISM = ("repeat_control", "repeat_packet", "masked", "altered")
SCORE_SEED = 2711512


def measure_prompts(run):
    from transformers import AutoTokenizer

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Set PYTHONHASHSEED=0 to match the official generator")
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / ".codex_tmp/opd_qwen_teacher_tokenizer", local_files_only=True
    )
    upstream = load_upstream(BUNDLE)
    dev_queries = load_index(MEMORY / "queries_only.jsonl")
    extra_queries = load_index(run / "followup/queries.jsonl")
    inputs = {
        name: (STUDY / "gpu_evidence512" / f"full.{name}.jsonl", dev_queries)
        for name in DEV
    }
    inputs.update(
        full_history=(BUNDLE / "inputs.jsonl", dev_queries),
        temporal=(FOLLOW / "full.temporal.jsonl", dev_queries),
    )
    inputs.update(
        {
            name: (run / "followup" / f"{name}.jsonl", extra_queries)
            for name in (*TRANSFER, *MECHANISM)
        }
    )
    if (run / "trace/design.json").exists():
        from .trace_memory import ARMS

        inputs.update({a: (run / "trace" / f"{a}.jsonl", extra_queries) for a in ARMS})
    if (run / "adaptive/design.json").exists():
        inputs["adaptive"] = (run / "adaptive/adaptive.jsonl", dev_queries)
    if (run / "transfer_temporal/design.json").exists():
        all_queries = load_index(run / "transfer_temporal/queries.jsonl")
        inputs["transfer_temporal"] = (
            run / "transfer_temporal/temporal.jsonl",
            all_queries,
        )
    observed = []
    for arm, (path, queries) in inputs.items():
        for key, row in load_index(path).items():
            observed.append(
                {
                    "arm": arm,
                    "key": key,
                    "prompt_tokens": prompt_tokens(
                        tokenizer, upstream, row, queries[key]
                    ),
                }
            )
    write_rows(run / "prompt_costs.jsonl", observed)
    write_json(
        run / "prompt_cost_summary.json",
        {
            arm: float(
                np.median([r["prompt_tokens"] for r in observed if r["arm"] == arm])
            )
            for arm in inputs
        },
    )


def answer(row, task):
    items = row.get("memory") or []
    return items[task - 1].get("answer", "") if len(items) >= task else ""


def check_outputs(directory, arms, keys):
    outputs = {arm: load_index(directory / f"{arm}.jsonl") for arm in arms}
    for arm, by_key in outputs.items():
        if set(by_key) != set(keys):
            raise ValueError(f"Unexpected sample keys for {arm}")
        for sample in by_key.values():
            reason = sample.get("reason", "")
            if reason.startswith("ERROR:") and not any(
                s in reason
                for s in ("invalid_output_after_retries", "UnboundLocalError")
            ):
                raise ValueError(f"{arm}: inference error: {reason}")
    return outputs


def teaching(run):
    """Review the student-facing first step on every packet-active dev case."""
    gold = load_index(BUNDLE / "gold.jsonl")
    matched_input = load_index(STUDY / "gpu_evidence512/full.packet_control.jsonl")
    generated = check_outputs(
        run / "results/runs/full", ("packet_control", "packet"), gold
    )
    cases = read(run / "followup/interventions.jsonl")
    rng = random.Random(63481)
    blinded, roles = [], []
    for i, case in enumerate(cases):
        key = case["key"]
        order = ["packet_control", "packet"]
        rng.shuffle(order)
        ident = f"T{i + 1:03}"
        blinded.append(
            {
                "id": ident,
                "current": matched_input[key]["question_info"],
                "visible_records": matched_input[key]["history_info"],
                "reference_teaching": gold[key]["content"],
                "answers": [generated[a][key]["content"] for a in order],
            }
        )
        roles.append(
            {
                "id": ident,
                "key": key,
                "qid": HEADER.match(matched_input[key]["question_info"])[1],
                "order": order,
            }
        )
    out = run / "teaching"
    out.mkdir(exist_ok=False)
    write_rows(out / "key.jsonl", roles)
    for i in range(0, len(blinded), 10):
        write_rows(out / f"blind_{i // 10 + 1}.jsonl", blinded[i : i + 10])
    write_json(
        out / "rubric.json",
        {
            "actionable": "Binary. The response gives a concrete mathematically appropriate next action or question the pupil can carry out. Exclude empty, generic, internally contradictory, or materially mistaken guidance.",
            "past_claim_error": "Binary. Explicitly contradicts a supplied past fact or attributes a specific unobserved mathematical method or mistake to the pupil as established fact. An unseen record alone is insufficient; flag uncertainty in the note.",
            "scope": "Exploratory blinded CodeX model review of 30 dev pairs, not a LongTutor official teaching score.",
        },
    )


def replicate(run):
    cases = {
        r["key"]: r
        for r in read(STUDY / "transfer560/rows.jsonl")
        if r["task"] == 2 and r["full"]
    }
    generated = check_outputs(run / "results/followup/runs", TRANSFER, cases)
    rng = random.Random(89534)
    keys = list(cases)
    rng.shuffle(keys)
    blinded, identity = [], []
    for i, key in enumerate(keys, 1):
        order = list(TRANSFER)
        rng.shuffle(order)
        rid = f"R{i:03}"
        blinded.append(
            {
                "id": rid,
                "question": cases[key]["question"],
                "reference": cases[key]["reference"],
                "answers": [answer(generated[a][key], 2) for a in order],
            }
        )
        identity.append({"id": rid, "key": key, "order": order})
    out = run / "replication"
    out.mkdir(exist_ok=False)
    write_rows(out / "identity.jsonl", identity)
    for i in range(0, len(blinded), 45):
        write_rows(out / f"blind_{i // 45 + 1}.jsonl", blinded[i : i + 45])


def paired(pairs, qids):
    clusters, group = np.unique(qids, return_inverse=True)
    rng = np.random.default_rng(SCORE_SEED)
    weights = rng.multinomial(
        len(clusters), np.repeat(1 / len(clusters), len(clusters)), size=10000
    )
    data = summarize(pairs, group, weights)
    data["clusters"] = len(clusters)
    return data


def prepare(run, early=False):
    gold = load_index(BUNDLE / "gold.jsonl")
    qid = {
        k: HEADER.match(r["question_info"])[1]
        for k, r in load_index(BUNDLE / "inputs.jsonl").items()
    }
    trigger = {r["key"]: r for r in read(run / "followup/interventions.jsonl")}
    direct = {
        (r["key"], r["task"]): r["full"]["answer"]
        for r in read(STUDY / "four_rankers/executor_fixed.jsonl")
        if r["full"]
    }
    development = check_outputs(run / "results/runs/full", DEV, gold)

    full_history = load_index(BUNDLE / "base.jsonl")
    temporal = load_index(
        FOLLOW / "autodl_4090_49862_20261126/runs/followup_full/temporal.jsonl"
    )
    if set(full_history) != set(gold) or set(temporal) != set(gold):
        raise ValueError("Historical comparison has different development IDs")
    development.update(full_history=full_history, temporal=temporal)
    mechanisms = (
        {}
        if early
        else check_outputs(run / "results/followup/runs", MECHANISM, trigger)
    )
    transfer_rows = {
        r["key"]: r
        for r in read(STUDY / "transfer560/rows.jsonl")
        if r["task"] == 2 and r["full"]
    }
    if early:
        transfer_rows = {}
    transfer = (
        {}
        if early
        else check_outputs(run / "results/followup/runs", TRANSFER, transfer_rows)
    )
    transfer_qids = {
        k: HEADER.match(s["question_info"])[1]
        for k, s in load_index(run / "followup/transfer_control.jsonl").items()
    }
    rng = random.Random(SCORE_SEED)
    blinded, identity, exact_rows = [], [], []

    def blind(pop, key, task, question, reference, variants, current):
        unique = list(dict.fromkeys(variants.values()))
        rng.shuffle(unique)
        ident = f"{pop[:1]}{len(blinded) + 1:03d}"
        blinded.append(
            {
                "id": ident,
                "question": question,
                "reference": reference,
                "answers": unique,
            }
        )
        identity.append(
            {
                "id": ident,
                "population": pop,
                "key": key,
                "qid": current,
                "task": task,
                "roles": {a: unique.index(v) for a, v in variants.items()},
            }
        )

    for key, label in gold.items():
        for arm, records in development.items():
            row = records[key]
            exact_rows.append(
                {
                    "key": key,
                    "qid": qid[key],
                    "population": "dev",
                    "arm": arm,
                    "format_failure": not bool(row["diagnosis"]),
                    "diagnosis": norm(label["diagnosis"]) == norm(row["diagnosis"]),
                    **{
                        f"Q{t}": norm(answer(row, t))
                        == norm(label["memory"][t - 1]["answer"])
                        for t in (1, 2, 3)
                    },
                }
            )
        for task in (1, 2):
            if not early and (key not in trigger or task != trigger[key]["task"]):
                continue
            item = label["memory"][task - 1]
            variants = {arm: answer(development[arm][key], task) for arm in development}
            if not early:
                variants.update(
                    {arm: answer(mechanisms[arm][key], task) for arm in MECHANISM}
                )
                variants["direct"] = direct[key, task]
                # Rate the altered claim too: intervention can accidentally repair a wrong original.
                variants["altered_claim"] = trigger[key]["replacement"]
            blind(
                "baseline" if early else "dev",
                key,
                task,
                item["query"],
                item["answer"],
                variants,
                qid[key],
            )
    # Full reference rows are loaded locally; no annotation is sent to generation.
    for key, row in transfer_rows.items():
        variants = {arm: answer(records[key], 2) for arm, records in transfer.items()}
        variants["direct"] = row["full"]["answer"]
        blind(
            "transfer",
            key,
            2,
            row["question"],
            row["reference"],
            variants,
            transfer_qids[key],
        )
    out = run / ("dev_review" if early else "analysis")
    out.mkdir(exist_ok=False)
    write_rows(out / "identity.jsonl", identity)
    write_rows(out / "exact.jsonl", exact_rows)
    if not early:
        echo, keyrows = [], []
        for i, (key, intervention) in enumerate(trigger.items(), 1):
            task = intervention["task"]
            variants = {a: answer(mechanisms[a][key], task) for a in MECHANISM}
            choices = list(dict.fromkeys(variants.values()))
            rng.shuffle(choices)
            ident = f"C{i:03}"
            echo.append(
                {
                    "id": ident,
                    "question": gold[key]["memory"][task - 1]["query"],
                    "comparison_value": intervention["replacement"],
                    "answers": choices,
                }
            )
            keyrows.append(
                {
                    "id": ident,
                    "key": key,
                    "qid": qid[key],
                    "operator": intervention["operator"],
                    "roles": {a: choices.index(s) for a, s in variants.items()},
                }
            )
        write_rows(out / "echo_probe.jsonl", echo)
        write_rows(out / "echo_key.jsonl", keyrows)
    for pop in ("baseline", "dev", "transfer"):
        ids = {r["id"] for r in identity if r["population"] == pop}
        selected = [r for r in blinded if r["id"] in ids]
        for i in range(0, len(selected), 40):
            write_rows(out / f"blind_{pop}_{i // 40 + 1}.jsonl", selected[i : i + 40])
    exact_summary = {
        arm: {
            metric: sum(row[metric] for row in exact_rows if row["arm"] == arm)
            for metric in ("Q1", "Q2", "Q3", "diagnosis", "format_failure")
        }
        for arm in development
    }
    write_json(out / "exact_summary.json", exact_summary)
    write_json(
        out / "protocol.json",
        {
            "seed": SCORE_SEED,
            "rows": len(blinded),
            "main": "transfer Q2 on all 131 executable cases",
            "dev_primary": "queried field for the 30 active cases; Q2 also on all 140",
            "rating": "single-family Codex anonymous semantic review; separate from official DK",
            "software_sha256": digest(Path(__file__)),
        },
    )
    print(exact_summary)


def reviewed_rows(directories):
    """Join one blind verdict to each case and validate every role position."""
    identity = [r for d in directories for r in read(d / "identity.jsonl")]
    public_rows = [
        r for d in directories for p in d.glob("blind_*.jsonl") for r in read(p)
    ]
    verdict_rows = [
        r for d in directories for p in d.glob("verdict_*.jsonl") for r in read(p)
    ]
    public = {r["id"]: r["answers"] for r in public_rows}
    verdict = {r["id"]: r["scores"] for r in verdict_rows}
    ids = {r["id"] for r in identity}
    if (
        not ids
        or ids != public.keys()
        or ids != verdict.keys()
        or any(len(rows) != len(ids) for rows in (public_rows, verdict_rows, identity))
    ):
        raise ValueError("Missing or duplicate blind cases")
    detail = []
    for row in identity:
        rid = row["id"]
        scores, options, roles = verdict[rid], public[rid], row["roles"]
        if (
            not options
            or len(scores) != len(options)
            or any(type(s) is not int or s not in (0, 1) for s in scores)
            or any(type(i) is not int for i in roles.values())
            or set(roles.values()) != set(range(len(options)))
        ):
            raise ValueError(f"Invalid blind verdict or role positions: {rid}")
        detail.append({**row, "scores": {arm: scores[i] for arm, i in roles.items()}})
    return detail


def aggregate(run):
    out = run / "analysis"
    detailed = reviewed_rows((out, run / "dev_review"))
    groups = defaultdict(list)
    for entry in detailed:
        row = entry
        if row["population"] == "transfer":
            groups["transfer"].append(entry)
        elif row["population"] == "baseline":
            if row["task"] == 1:
                groups["all_dev_Q1"].append(entry)
            if row["task"] == 2:
                groups["all_dev_Q2"].append(entry)
        else:
            groups["active_dev"].append(entry)
    summary = {}
    for name, rows in groups.items():
        arms = set.intersection(*(set(r["scores"]) for r in rows))
        base = "transfer_control" if name == "transfer" else "packet_control"
        summary[name] = {
            arm: paired(
                [(r["scores"][base], r["scores"][arm]) for r in rows],
                [r["qid"] for r in rows],
            )
            for arm in sorted(arms)
        }
    stratum = {}
    for correct in (0, 1):
        chosen = [r for r in groups["transfer"] if r["scores"]["direct"] == correct]
        stratum[str(correct)] = {
            a: sum(r["scores"][a] for r in chosen) for a in TRANSFER
        }
        stratum[str(correct)]["n"] = len(chosen)
    summary["transfer_by_direct_correct"] = stratum
    transfer_inputs = load_index(run / "followup/transfer_packet.jsonl")
    single = [
        r
        for r in groups["transfer"]
        if "\nQ1 " not in transfer_inputs[r["key"]]["history_info"][-1]
    ]
    summary["single_Q2_packet_only"] = paired(
        [
            (r["scores"]["transfer_control"], r["scores"]["transfer_packet"])
            for r in single
        ],
        [r["qid"] for r in single],
    )
    decisions = {r["key"]: r for r in read(run / "policy/decisions.jsonl")}
    policy_results = {}
    for name in ("transfer", "active_dev"):
        rows = [r for r in groups[name] if r["task"] == 2]
        base, packet = TRANSFER if name == "transfer" else ("packet_control", "packet")
        table = {}
        for rule in ("scope_agrees", "0.5", "0.75", "0.9"):
            selected = [
                decisions[r["key"]]["scope_agrees"]
                if rule == "scope_agrees"
                else decisions[r["key"]]["probability"] >= float(rule)
                for r in rows
            ]
            pair = [
                (r["scores"][base], r["scores"][packet if yes else base])
                for r, yes in zip(rows, selected, strict=True)
            ]
            table[rule] = {
                "selected": sum(selected),
                "versus_control": paired(pair, [r["qid"] for r in rows]),
                "versus_always_packet": paired(
                    [
                        (r["scores"][packet], v[1])
                        for r, v in zip(rows, pair, strict=True)
                    ],
                    [r["qid"] for r in rows],
                ),
            }
        policy_results[name] = table
    write_json(out / "policy_summary.json", policy_results)
    write_rows(out / "semantic_by_id.jsonl", detailed)
    write_json(out / "semantic_summary.json", summary)
    print(
        {
            k: (v.get("packet", v.get("transfer_packet", {})).get("difference"))
            for k, v in summary.items()
        }
    )


def supplement(run):
    out = run / "analysis"
    details = read(out / "semantic_by_id.jsonl")
    primary = {r["key"]: r for r in details if r["population"] == "transfer"}
    active = {(r["key"], r["task"]): r for r in details if r["population"] == "dev"}

    def lookup(directory, pattern):
        rows = [r for p in directory.glob(pattern) for r in read(p)]
        lookup = {r["id"]: r for r in rows}
        if len(lookup) != len(rows):
            raise ValueError("Duplicate review IDs")
        return lookup

    repeat = lookup(run / "replication", "verdict_*.jsonl")
    pairs, qids = [], []
    matches = {a: 0 for a in TRANSFER}
    for r in read(run / "replication/identity.jsonl"):
        vals = dict(zip(r["order"], repeat[r["id"]]["scores"], strict=True))
        original = primary[r["key"]]
        pairs.append(tuple(vals[a] for a in TRANSFER))
        qids.append(original["qid"])
        for a in TRANSFER:
            matches[a] += int(vals[a] == original["scores"][a])
    summary = {
        "replication": paired(pairs, qids),
        "replication_label_agreement": matches,
    }
    judges = lookup(run / "teaching", "verdict_*.jsonl")
    teaching_key = read(run / "teaching/key.jsonl")
    for m in ("actionable", "past_claim_error"):
        values = []
        for r in teaching_key:
            slots = dict(zip(r["order"], judges[r["id"]][m], strict=True))
            values.append((slots["packet_control"], slots["packet"]))
        summary[m] = paired(values, [r["qid"] for r in teaching_key])
    echo = lookup(out, "echo_verdict.jsonl")
    summary["altered_value_agreement"] = {
        a: sum(
            echo[r["id"]]["scores"][r["roles"][a]] for r in read(out / "echo_key.jsonl")
        )
        for a in MECHANISM
    }
    scope = lookup(run / "scope", "verdict.jsonl")
    summary["changed_scope_correct"] = {
        a: sum(
            scope[r["id"]]["scores"][r["order"].index(a)]
            for r in read(run / "scope/identity.jsonl")
        )
        for a in ("full", "related")
    }
    temporal = load_index(
        STUDY.parent / "proposal_evidence_20261127/paired_official.rows.jsonl"
    )
    full = load_index(BUNDLE.parent / "base.metrics.jsonl")
    dev = [r for r in details if r["population"] == "baseline"]
    summary["archived_DK_agreement"] = {
        a: [
            sum(
                r["scores"][a]
                == (
                    full[r["key"]]["memory_correct"][t - 1]
                    if a == "full_history"
                    else temporal[r["key"]][f"Q{t}"][1]
                )
                for r in dev
                if r["task"] == t
            )
            for t in (1, 2)
        ]
        for a in ("full_history", "temporal")
    }
    reviews = lookup(run / "dev_review", "blind_*.jsonl")
    votes = lookup(run / "dev_review", "verdict_*.jsonl")
    system = load_index(run / "system/dev/predictions.jsonl")
    summary["dev_system_vs_control"] = {}
    for t in (1, 2):
        rows = [r for r in dev if r["task"] == t]
        pairs = []
        for r in rows:
            k = r["key"]
            candidate = answer(system[k], t)
            answers = reviews[r["id"]]["answers"]
            score = (
                votes[r["id"]]["scores"][answers.index(candidate)]
                if candidate in answers
                else active[k, t]["scores"]["direct"]
            )
            pairs.append((r["scores"]["packet_control"], score))
        summary["dev_system_vs_control"][t] = paired(pairs, [r["qid"] for r in rows])
    write_json(out / "checks_and_system.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "aggregate",
            "tokens",
            "teaching",
            "dev",
            "replicate",
            "supplement",
        ),
    )
    parser.add_argument("--run", type=Path, default=STUDY / "gpu_run1")
    args = parser.parse_args()
    {
        "prepare": prepare,
        "aggregate": aggregate,
        "tokens": measure_prompts,
        "teaching": teaching,
        "replicate": replicate,
        "supplement": supplement,
        "dev": lambda run: prepare(run, early=True),
    }[args.stage](args.run)
