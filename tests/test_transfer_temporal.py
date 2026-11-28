import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.experiments.training_free_memory import transfer_temporal as study


class TransferTemporalTest(unittest.TestCase):
    def test_interim_delta_needs_only_distinct_pairs(self):
        mapping = [
            {"id": "a", "qid": "17", "compact": True, "roles": {"temporal": 1, "adaptive": 0}},
            {"id": "b", "qid": "17", "compact": False, "roles": {"temporal": 0, "adaptive": 0}},
            {"id": "c", "qid": "25", "compact": False, "roles": {"temporal": 0, "adaptive": 0}},
        ]
        public = {r["id"]: {"answers": ["x", "y"] if r["compact"] else ["x"]} for r in mapping}
        _, pending = study.count_scores(mapping, public, {})
        self.assertEqual(pending["ungraded_distinct"], 1)
        self.assertNotIn("paired_difference", pending)
        _, interim = study.count_scores(mapping, public, {"a": [1, 0]})
        self.assertEqual(interim["paired_difference"]["n"], 3)
        self.assertEqual(interim["paired_difference"]["clusters"], 2)
        self.assertEqual(interim["paired_difference"]["difference"], 1 / 3)
        self.assertEqual(interim["compact"]["adaptive_correct"], 1)
        self.assertNotIn("all", interim)
        self.assertNotIn("before", interim["paired_difference"])
        _, final = study.count_scores(mapping, public, {"a": [1, 0], "b": [1], "c": [0]})
        self.assertEqual(final["all"]["temporal_correct"], 1)
        self.assertEqual(final["all"]["adaptive_correct"], 2)
        self.assertEqual(interim["paired_difference"], final["paired_difference"])
        with self.assertRaises(ValueError):
            study.count_scores(mapping, public, {"a": [1]})

    def test_duplicate_verdict_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ("verdict_1", "verdict_2"):
                study.write_rows(path / (name + ".jsonl"), [{"id": "a", "scores": [1]}])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                study.review_index(path, "verdict_*.jsonl")

    def test_frozen_routing_preserves_format_failures_and_shared_q2(self):
        # Exercise the official positional memory schema using synthetic generations only.
        original = study.DEFAULT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / original.name
            out.mkdir()
            for name in ("design.json", "queries.jsonl", "temporal.jsonl"):
                shutil.copyfile(original / name, out / name)
            (root / "followup").mkdir()
            compact_file = root / "followup/transfer_control.jsonl"
            shutil.copyfile(original.parent / "followup" / compact_file.name, compact_file)
            keys = study.load_index(out / "temporal.jsonl")
            active = study.load_index(compact_file)
            failure = next(iter(active))

            def synth(k, src):
                if k == failure and src == "compact":
                    return {"_key": k, "diagnosis": "", "reason": "ERROR: invalid_output_after_retries", "memory": []}
                return {"_key": k, "diagnosis": "OK", "reason": src,
                        "memory": [{"answer": "synthetic"} for _ in range(3)]}

            baseline, compact = root / "temporal.jsonl", root / "compact.jsonl"
            study.write_rows(baseline, (synth(k, "temporal") for k in keys))
            study.write_rows(compact, (synth(k, "compact") for k in active))
            with self.assertRaisesRegex(ValueError, "archived transfer_control"):
                study.assemble(out, baseline, compact)
            with patch.object(study, "COMPACT_PREDICTION_SHA256", study.digest(compact)):
                study.assemble(out, baseline, compact)
            result = study.load_index(out / "system/predictions.jsonl")
            self.assertEqual(result[failure], synth(failure, "compact"))
            self.assertEqual(result.keys(), keys.keys())
            evidence = study.read(out / "system/evidence.jsonl")
            self.assertTrue(any(not r["compact"] and r["evidence"] for r in evidence))
            for key in keys.keys() - active.keys():
                self.assertEqual(study.answer(result[key], 2), "synthetic")
            study.blind(out, baseline)
            mapping = study.read(out / "review/identity.jsonl")
            self.assertEqual({r["key"] for r in mapping if r["compact"]}, active.keys())
            self.assertEqual({r["qid"] for r in mapping},
                             {study.HEADER.match(r["question_info"])[1] for r in keys.values()})
            public = study.review_index(out / "review", "blind_*.jsonl")
            records = [{"id": m["id"], "scores": [1] * len(public[m["id"]]["answers"])}
                       for m in mapping if m["compact"]]
            study.write_rows(out / "review/verdict_compact.jsonl", records)
            study.aggregate(out)
            summary = json.loads((out / "review/semantic_summary_interim.json").read_text())
            self.assertEqual(summary["reviewed"], 131)
            self.assertEqual(summary["paired_difference"]["n"], 560)
            self.assertEqual(summary["ungraded_distinct"], 0)
            self.assertNotIn("all", summary)
            path = next((out / "review").glob("blind_compact_*.jsonl"))
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed since dispatch"):
                study.aggregate(out)
