"""Verify the copied GPU archives, trained adapters and fixed dev-generation receipts."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from scripts.experiments.action_distillation.official import HASHES, PIN
from scripts.experiments.action_distillation.provenance import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    d = parser.parse_args().bundle
    result = {
        "archives": {},
        "weights": {},
        "structured": {},
        "external": {},
        "official": {},
    }
    for name in (
        "phase1_archive",
        "results_fast",
        "weights_and_training",
        "scale_structured",
        "final_receipts",
        "history_mechanism",
    ):
        tar = d / (name + ".tgz")
        actual = digest(tar)
        verify(actual == (d / (name + ".sha256")).read_text().split()[0], name)
        result["archives"][name] = {"bytes": tar.stat().st_size, "sha256": actual}
        if name not in ("phase1_archive", "weights_and_training"):
            continue
        with tarfile.open(tar, "r|gz") as reader:
            for member in reader:
                path = Path(member.name)
                if path.name != "adapter_model.safetensors":
                    continue
                group = path.parent.parent
                sha = hashlib.file_digest(
                    reader.extractfile(member), "sha256"
                ).hexdigest()
                if group.parts[0] == "cycles":
                    expected = read(d / group / "result.json")["model_sha"]
                else:
                    expected = read(d / group / "receipt.json")["final"]
                verify(sha == expected, str(group))
                result["weights"][group.as_posix()] = sha
    verify(
        len(result["weights"]) == 24,
        "22 candidate checkpoints plus two smoke checkpoints expected",
    )
    ref = {
        json.loads(l)["id"]
        for l in (d / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    }
    for folder in [d, d / "scale"]:
        for f in (folder / "evaluation").glob("*.proof.json"):
            arm = f.name.split(".")[0]
            name = (folder / arm).relative_to(d).as_posix()
            proof = read(f)
            weight_key = (
                "cycles/" + arm.split("_")[1] if arm.startswith("cycle_") else name
            )
            weight = result["weights"].get(weight_key)
            if arm == "gold_sft":
                weight = (
                    "40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b"
                )
            verify(arm == "base" or weight is not None, name)
            verify(
                proof["start"]["adapter"] == weight
                and proof["start"]["dev"] == digest(folder / "dev.jsonl"),
                name,
            )
            for field, suffix, group in [
                ("structured_sha256", ".jsonl", "structured"),
                ("external_sha256", ".external.jsonl", "external"),
            ]:
                if not proof.get(field):
                    continue
                pred = f.parent / (arm + suffix)
                verify(digest(pred) == proof[field], str(pred))
                lines = pred.read_text(encoding="utf-8").splitlines()
                verify(
                    len(lines) == (140 if group == "structured" else 4800), str(pred)
                )
                if group == "structured":
                    verify({json.loads(l)["id"] for l in lines} == ref, str(pred))
                result[group][name] = len(lines)
        for f in (folder / "official").glob("*.receipt.json"):
            info = read(f)
            key = (folder / info["arm"]).relative_to(d).as_posix()
            pred = f.parent / (info["arm"] + ".full.jsonl")
            rows = [
                json.loads(l) for l in pred.read_text(encoding="utf-8").splitlines()
            ]
            expected_weight = result["weights"].get(
                key,
                "40be07306e9070de6b47d9dfdd372ae7a6183b81aad1d728ae27141f4f37313b"
                if key == "gold_sft"
                else None,
            )
            verify(
                info["upstream"] == PIN
                and info["pred"] == digest(pred)
                and info["weights"] == expected_weight,
                key,
            )
            verify(
                info["input"] == HASHES["full.temporal.jsonl"]
                and len(rows) == 140
                and {r["_key"] for r in rows} == ref,
                key,
            )
            result["official"][key] = len(rows)
    for group, expected in [("structured", 24), ("external", 4), ("official", 9)]:
        verify(len(result[group]) == expected, group)
    probe = d / "history_probe"
    binding = read(probe / "binding.json")
    verify(
        binding["code"] == digest(d / "history_probe.py")
        and binding["dev_sha"] == digest(d / "dev.jsonl")
        and binding["preregistered_local_protocol"] == digest(probe / "design.json"),
        "history probe binding",
    )
    before = [
        d / "evaluation/gold_sft.proof.json",
        d / "evaluation/v3_pa.proof.json",
        d / "evaluation/v3_wrong.proof.json",
        d / "scale/evaluation/v3_pa.proof.json",
    ]
    configs = (probe / "configs.sha256").read_text().splitlines()
    logged = (probe / "probe.log").read_text(encoding="utf-8", errors="replace")
    result["history_probe"] = {}
    for group, proof, conf in zip(
        ["gold_sft", "pa72", "wrong72", "pa373"], before, configs, strict=True
    ):
        start = read(proof)["start"]
        verify(
            start["config"] == conf.split()[0]
            and start["adapter"] == binding["weights"][group],
            group,
        )
        raw = probe / (group + ".jsonl")
        items = [json.loads(l) for l in raw.read_text().splitlines()]
        verify(
            len(items) == 280
            and {(r["id"], r["view"]) for r in items}
            == {(k, v) for k in ref for v in ("real", "flip")},
            group,
        )
        verify(group + " 280 " + digest(raw) in logged, group)
        result["history_probe"][group] = digest(raw)
    (d / "复现核验.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
