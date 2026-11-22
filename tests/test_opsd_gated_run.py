import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.experiments.opsd_repair import run_gated


class GatedRunTest(unittest.TestCase):
    def test_formal_uses_only_rollout_filter_with_legacy_settings(self):
        args = run_gated.training_args("formal", run_gated.ROOT / "example")
        self.assertIn("--validate-rollouts", args)
        self.assertNotIn("--strict-output-prompt", args)
        self.assertEqual(args[args.index("--max-completion-length") + 1], "1024")
        self.assertEqual(args[args.index("--max-steps") + 1], "35")
        self.assertEqual(args[args.index("--gradient-accumulation-steps") + 1], "4")

    def test_trace_audit_validates_retries_and_counts_kept_tokens(self):
        rows = [
            {"sample_id": "a", "step": 0, "attempt": 1, "status": "valid", "completion_tokens": 9},
            {"sample_id": "b", "step": 0, "attempt": 1, "status": "invalid_json", "completion_tokens": 15},
            {"sample_id": "b", "step": 0, "attempt": 2, "status": "valid", "completion_tokens": 11},
            {"sample_id": "c", "step": 1, "attempt": 1, "status": "missing_eos", "completion_tokens": 1024},
            {"sample_id": "c", "step": 1, "attempt": 2, "status": "invalid_native_schema", "completion_tokens": 14},
        ]
        result = run_gated.rollout_totals(rows, {"a", "b", "c"})
        self.assertEqual(result, {"first_pass": 1, "saved_by_retry": 1, "skipped": 1, "accepted_tokens": 20})
        with self.assertRaisesRegex(ValueError, "retry"):
            run_gated.rollout_totals(rows[:-1], {"a", "b", "c"})
        with self.assertRaisesRegex(ValueError, "frozen"):
            run_gated.rollout_totals(rows, {"a", "b"})

    def test_run_audit_checks_the_sample_count_in_each_window(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = root / "generations"
            traces.mkdir()
            sample_ids = [str(index) for index in range(8)]
            (root / "continuation.jsonl").write_text(
                "".join(f'{{"sample_id": "{sample_id}"}}\n' for sample_id in sample_ids), encoding="utf-8",
            )
            run_gated.write_json(root / "training_summary.json", {"rollout_validation": {
                "optimizer_windows": 2, "effective_optimizer_steps": 2, "counts": {"attempt_1_valid": 8},
                "skipped_by_gold_diagnosis": {},
            }})
            rows = [{"sample_id": sample_id, "step": index // 4, "attempt": 1,
                     "status": "valid", "completion_tokens": 3} for index, sample_id in enumerate(sample_ids)]
            trace = traces / "generations_step_2.json"
            run_gated.write_json(trace, {"generations": rows})
            with patch.object(run_gated, "ROOT", root):
                run_gated.audit_run("smoke", root)
                rows[3]["step"] = 1
                run_gated.write_json(trace, {"generations": rows})
                with self.assertRaisesRegex(ValueError, "sample budget"):
                    run_gated.audit_run("smoke", root)
