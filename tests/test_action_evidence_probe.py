import json
from types import SimpleNamespace

import pytest

from scripts.experiments.action_pilot import evidence_probe as probe
from scripts.experiments.action_pilot import run as gold
from scripts.experiments.action_pilot import score_evidence_probe as scoring


def test_probe_resumes_at_the_missing_arm(tmp_path, monkeypatch):
    import openai

    key = "student||id"
    sample = {"sample_id": key, "current": "当前题", "history": [[3, "原始记录"]],
              "solution": {"analysis": "标准解"}, "facts": [{"原始位置": 3}]}
    gold.write_rows(tmp_path / "inputs.jsonl", [sample])
    manifest = {"ids": [key], "smoke_ids": [key], "repeat_ids": [], "model_alias": probe.MODEL,
                "inputs_sha256": gold.digest(tmp_path / "inputs.jsonl"),
                "source_sha256": gold.digest(probe.Path(probe.__file__)), "adapter_sha256": {},
                "base_revision": "test"}
    gold.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(probe, "verify_service", lambda *args: {})
    monkeypatch.setattr(gold, "checkpoint_root", lambda *args: tmp_path)
    seen = []
    raw = json.dumps({"action": "C", "first_move": "先说出规则。", "history_positions": [3]})

    class Client:
        def __init__(self, **options):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.completion))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def completion(self, **request):
            seen.append(request["messages"])
            if len(seen) == 2:
                raise TimeoutError("simulate transport failure")
            choice = SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None)

    monkeypatch.setattr(openai, "OpenAI", Client)
    out = tmp_path / "smoke.jsonl"
    with pytest.raises(TimeoutError, match="transport"):
        probe.probe(tmp_path, out, "smoke", 8808)
    assert not out.exists()
    progress = list(gold.read_rows(out.with_suffix(".progress.jsonl")))
    assert len(progress) == 1 and len(progress[0]) == 2
    probe.probe(tmp_path, out, "smoke", 8808)
    assert seen[1] == seen[2] and seen[0] not in seen[2:]
    assert set(next(gold.read_rows(out))) == {"sample_id", *probe.CONDITIONS}


def test_blind_scores_are_packet_bound_and_expand_identical_responses(tmp_path):
    ids = [str(i) for i in range(64)]
    repeats = ids[:16]
    inputs = [{"sample_id": key, "current": "一道题", "history": [[1, "原记录"]],
               "facts": [1], "solution": {"analysis": "私有解答"}} for key in ids]
    gold.write_rows(tmp_path / "inputs.jsonl", inputs)
    manifest = {"ids": ids, "repeat_ids": repeats, "scope": "gold_train_teacher_first_move",
                "inputs_sha256": gold.digest(tmp_path / "inputs.jsonl"), "gate": scoring.GATE,
                "base_revision": "test", "temperature": 0, "sampling_seed": 42,
                "max_completion_tokens": 384, "adapter_sha256": {}}
    gold.write_json(tmp_path / "manifest.json", manifest)
    out = tmp_path / "full.jsonl"
    gold.write_json(out.with_suffix(".settings.json"), {
        **manifest, "phase": "full", "manifest_sha256": gold.digest(tmp_path / "manifest.json"),
    })

    def answer(text):
        decision = {"action": "C", "first_move": text, "history_positions": []}
        return {"raw": json.dumps(decision), "finish_reason": "stop", "decision": decision}

    good, bad = answer("先看哪两个数加起来不变？"), answer("直接作答。")
    gold.write_rows(out, ({"sample_id": key, "public": bad, "solution": bad, "solution_evidence": good,
                           **({probe.REPEAT: good} if key in repeats else {})} for key in ids))
    blind = tmp_path / "blind"
    scoring.make_blind(tmp_path, out, blind)
    ratings = []
    for case in gold.read_rows(blind / "blind.jsonl"):
        assert len(case["options"]) == 2
        for option, text in case["options"].items():
            ratings.append({"case_option": f"{case['packet']}-{case['case']}{option}", "observed_action": "C",
                            "usable": text != bad["decision"]["first_move"], "unsupported_history": False,
                            "math_or_task_error": False, "direct_answer": False, "reason": "fixture"})
    ratings_file = tmp_path / "ratings.jsonl"
    gold.write_rows(ratings_file, ratings)
    gate_path = tmp_path / "gate.json"
    scoring.score(tmp_path, blind, ratings_file, gate_path)
    report = gold.read_json(gate_path)
    assert report["gate_passed"] and report["repeat_same_action"] == 16
    other = tmp_path / "other"
    scoring.make_blind(tmp_path, out, other)
    with pytest.raises(ValueError, match="every blinded reply"):
        scoring.score(tmp_path, other, ratings_file, tmp_path / "wrong_packet.json")


def test_action_a_and_citations_are_checked():
    assert probe.parse_response('{"action":"A","first_move":"","history_positions":[]}', set())
    assert not probe.parse_response('{"action":"A","first_move":"想一想","history_positions":[]}', set())
    assert not probe.parse_response('{"action":"C","first_move":"想一想","history_positions":[true]}', {1})
