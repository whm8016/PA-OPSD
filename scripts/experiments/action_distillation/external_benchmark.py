"""Prepare original MathTutorBench prompts and replay model results through its original evaluator."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PIN = "6faed173ec2bef55cb899b2a3e0f93982f9cb176"
TASKS = (
    "pedagogy_following",
    "pedagogy_following_hard",
    "mistake_location",
    "socratic_questioning",
)


def bind_local_data(repo):
    from dataloaders import base

    data = repo.parents[1]
    gsm = data / "gsm8k/grade-school-math/grade_school_math/data"
    sources = {
        ("eth-nlped/stepverify", "default", "train"): (
            repo.parent / "stepverify_c80445d/stepverify.json",
            "335351285ae4b58176a671e6590d5ca29448df98f6b59c2ab8e5598cc8106cbf",
        ),
        ("gsm8k", "socratic", "train"): (
            gsm / "train_socratic.jsonl",
            "6ec8eb18aa64f61a19886e8b80f70b021909bcdd73d0e9df865d3ef16ec1e274",
        ),
        ("gsm8k", "socratic", "test"): (
            gsm / "test_socratic.jsonl",
            "36b26f6790a864442b0db000af07f924faafd4499286ab834f2f9e6d89069a6d",
        ),
    }

    def local(dataset, name, split):
        path, digest = sources[dataset, name, split]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Unrecognized external dataset revision")
        values = (
            json.loads(raw)
            if path.suffix == ".json"
            else [json.loads(l) for l in raw.splitlines()]
        )
        return values

    base.load_dataset = local


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "score"))
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pred", type=Path)
    p.add_argument("--model")
    a = p.parse_args()
    repo = a.repo.resolve()
    out = a.out.resolve()
    pred = a.pred.resolve() if a.pred else None
    if (
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        != PIN
    ):
        raise ValueError("Wrong upstream version")
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", "."], check=True
    )
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import main as upstream
    import yaml
    from registry import TaskRegistry
    from tasks.base import TaskConfig

    bind_local_data(repo)

    out.mkdir(exist_ok=True, parents=True)
    if a.mode == "prepare":
        rows = []
        for name in TASKS:
            cfg = TaskConfig(
                **yaml.safe_load(Path("configs", name + ".yaml").read_text())
            )
            task = TaskRegistry.get_task(name)(cfg)
            for i, ex in enumerate(task.get_test_examples()):
                ex["shots"] = cfg.few_shot_samples
                rows.append(
                    {
                        "task": name,
                        "index": i,
                        "prompt": task.get_system_prompt(ex),
                        "stop": cfg.stop,
                    }
                )
        path = out / "external_inputs.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (out / "external_protocol.json").write_text(
            json.dumps(
                {
                    "pin": PIN,
                    "counts": {n: sum(r["task"] == n for r in rows) for n in TASKS},
                    "prompt_as_user": True,
                    "temperature": 0,
                    "max_tokens": 128,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        )
        return
    preds = [json.loads(l) for l in pred.read_text(encoding="utf-8").splitlines()]
    proof = json.loads(
        (pred.parent / (a.model + ".proof.json")).read_text(encoding="utf-8")
    )
    if (
        proof["start"]["model"] != a.model
        or proof["external_sha256"] != hashlib.sha256(pred.read_bytes()).hexdigest()
        or proof["start"]["external_commit"] != PIN
    ):
        raise ValueError("External predictions are not tied to the inference record")
    for name in TASKS:
        items = [r for r in preds if r["task"] == name]

        class Replay:
            pos = 0
            records = items

            def generate(self, messages, system_prompt, stop):
                r = self.records[self.pos]
                if (
                    messages
                    or r["model"] != a.model
                    or r["index"] != self.pos
                    or r["prompt"] != system_prompt
                    or r["stop"] != stop
                ):
                    raise ValueError(
                        "Saved generation does not match the original request"
                    )
                self.pos += 1
                return r["text"]

        replay = Replay()
        upstream.create_llm_model = lambda _cfg, replay=replay: replay
        final = out / name
        signature = hashlib.sha256(pred.read_bytes()).hexdigest()
        if final.exists():
            if (final / "pred.sha256").read_text() != signature:
                raise ValueError(
                    "Destination already holds another prediction artifact"
                )
            continue
        run_dir = Path(tempfile.mkdtemp(prefix=name + "-", dir=out))
        sys.argv = [
            "main.py",
            "--tasks",
            name + ".yaml",
            "--provider",
            "completion_api",
            "--model_args",
            "model="
            + a.model
            + ",temperature=0,max_tokens=128,is_chat=true,prompt_as_user=true",
            "--output",
            str(run_dir),
        ]
        upstream.main()
        if replay.pos != len(items):
            raise ValueError("Incomplete replay")
        (run_dir / "pred.sha256").write_text(signature)
        run_dir.rename(final)


if __name__ == "__main__":
    main()
