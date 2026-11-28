import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from copy import deepcopy
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.experiments.action_distillation import cycles, v3
from scripts.experiments.action_distillation.provenance import bind


class V3Tests(unittest.TestCase):
    def test_skill_training_rejects_historical_dynamic_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            adapter = bundle / "v3_pa" / "adapter"
            adapter.mkdir(parents=True)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"fixture weights")
            final_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
            receipt = bundle / "v3_pa" / "receipt.json"
            # The original 72- and 373-example receipts predate teacher_mode.
            receipt.write_text(json.dumps({"arm": "pa", "final": final_hash}))
            with self.assertRaisesRegex(ValueError, "fixed-teacher V3"):
                cycles.execute(Namespace(bundle=bundle))
            receipt.write_text(
                json.dumps(
                    {
                        "arm": "pa",
                        "teacher_mode": "fixed",
                        "final": final_hash,
                        "initial": "gold_sft",
                        "teacher_snapshot_adapter_sha256": "gold_sft",
                    }
                )
            )
            self.assertEqual(cycles.verified_cycle_seed(bundle), adapter)
            weights.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "do not match"):
                cycles.verified_cycle_seed(bundle)

    def test_v3_rejects_dynamic_teacher_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "not_present")
            argv = ["v3.py", "--arm", "pa", "--teacher-mode", "dynamic"]
            for option in ("bundle", "model", "adapter", "out"):
                argv.extend(("--" + option, missing))
            with (
                patch("sys.argv", argv),
                redirect_stderr(StringIO()) as stderr,
                self.assertRaises(SystemExit) as error,
            ):
                v3.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("invalid choice: 'dynamic'", stderr.getvalue())

    @unittest.skipUnless(
        find_spec("peft"), "PEFT is required for the two-adapter check"
    )
    def test_fixed_teacher_stays_at_gold_sft_while_student_updates(self):
        from peft import LoraConfig, PeftModel, get_peft_model

        torch.manual_seed(31)
        base = torch.nn.Sequential(torch.nn.Linear(4, 3))
        model = get_peft_model(
            deepcopy(base),
            LoraConfig(target_modules=["0"], r=2, lora_alpha=2),
        ).eval()
        x = torch.ones(1, 4)
        # Simulate a nonzero Gold SFT checkpoint before making its teacher copy.
        initial_opt = torch.optim.SGD(
            (p for p in model.parameters() if p.requires_grad), lr=0.1
        )
        model(x).sum().backward()
        initial_opt.step()
        initial_opt.zero_grad(set_to_none=True)
        initial = model(x).detach().clone()
        with model.disable_adapter():
            self.assertFalse(torch.equal(initial, model(x)))

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            v3.load_fixed_teacher(model, Path(directory))
            student_parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(student_parameters, lr=0.1)
            teacher_weights = {
                name: value.detach().clone()
                for name, value in model.named_parameters()
                if ".fixed_teacher." in name
            }
            self.assertTrue(teacher_weights)
            for _ in range(2):
                for _ in range(2):
                    previous_gradients = [p.grad for p in student_parameters]
                    with v3.fixed_teacher_forward(model):
                        target = model(x)
                        torch.testing.assert_close(target, initial, rtol=0, atol=0)
                        self.assertFalse(target.requires_grad)
                        self.assertTrue(
                            all(not p.requires_grad for p in model.parameters())
                        )
                    self.assertEqual(model.active_adapter, "default")
                    for parameter, gradient in zip(
                        student_parameters, previous_gradients
                    ):
                        self.assertIs(parameter.grad, gradient)
                    (model(x).sum() / 2).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            updated_student = model(x).detach()
            self.assertFalse(torch.equal(updated_student, initial))
            for name, value in model.named_parameters():
                if name in teacher_weights:
                    torch.testing.assert_close(
                        value, teacher_weights[name], rtol=0, atol=0
                    )
                    self.assertIsNone(value.grad)
            with v3.fixed_teacher_forward(model):
                torch.testing.assert_close(model(x), initial, rtol=0, atol=0)
            with (
                self.assertRaisesRegex(RuntimeError, "interrupted teacher"),
                v3.fixed_teacher_forward(model),
            ):
                raise RuntimeError("interrupted teacher")
            self.assertEqual(model.active_adapter, "default")
            self.assertTrue(model(x).requires_grad)
            output = Path(directory) / "updated"
            model.save_pretrained(output, selected_adapters=["default"])
            self.assertFalse((output / "fixed_teacher").exists())
            reloaded = PeftModel.from_pretrained(base, output).eval()
            torch.testing.assert_close(reloaded(x), updated_student, rtol=0, atol=0)

    def test_history_intervention_changes_only_historical_results(self):
        from scripts.experiments.action_distillation.history_probe import flip

        current = "\n当前题\n题目说：正确还是错误？\n结果: 错误"
        history = "知识点: 正确\n结果: 正确\n结果: 错误"
        self.assertEqual(
            flip(history + current, 2), "知识点: 正确\n结果: 错误\n结果: 正确" + current
        )
        with self.assertRaises(ValueError):
            flip(history + current, 3)

    def test_nested_relative_bundle_finds_frozen_base_in_parent(self):
        import json
        from contextlib import chdir

        from scripts.experiments.action_distillation.provenance import (
            digest,
            verified_base,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scale = root / "scale"
            scale.mkdir()
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            receipt = {
                "base_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "verified": {"config.json": digest(model / "config.json")},
            }
            (root / "base_verification.json").write_text(json.dumps(receipt))
            with chdir(scale):
                self.assertEqual(verified_base(Path("."), model), receipt)

    def test_cache_tariff_uses_ceiling_and_keeps_unconfirmed_calls_reserved(self):
        from scripts.experiments.action_distillation.judge import CacheAwareMeter

        with tempfile.TemporaryDirectory() as tmp:
            meter = CacheAwareMeter(Path(tmp) / "usage.jsonl")
            for key in [1, 2]:
                meter.apply({"id": key, "type": "reserve", "microyuan": 450})
            meter.apply(
                {
                    "id": 1,
                    "type": "settle",
                    "prompt": 100,
                    "cache_hit": 26,
                    "completion": 3,
                    "upper_microyuan": 224,
                }
            )
            self.assertEqual(meter.spent, 174)
            self.assertEqual(meter.pending, {2: 450})

    def test_unbound_outputs_and_changed_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pred, marker = folder / "pred.jsonl", folder / "start.json"
            pred.write_text("old generation")
            with self.assertRaises(ValueError):
                bind(marker, {"weight": "a"}, [pred])
            pred.unlink()
            bind(marker, {"weight": "a"}, [pred])
            pred.write_text("record")
            bind(marker, {"weight": "a"}, [pred])
            with self.assertRaises(ValueError):
                bind(marker, {"weight": "b"}, [pred])

    def test_disjoint_gradient_terms_and_detached_teacher(self):
        torch.manual_seed(31)
        student = torch.randn(9, 17, requires_grad=True)
        teacher = torch.randn(9, 17, requires_grad=True)
        seq = list(range(9))
        decisions = [(1, [1, 2], "1"), (3, [4, 5], "C"), (5, [6, 7], "F")]
        with patch.object(v3.core, "tensor", lambda x: torch.tensor([x])):
            a, t, s = v3.losses(student, teacher, seq, decisions, 7)
        expected = torch.nn.functional.kl_div(
            student[5, [6, 7]].log_softmax(-1),
            teacher[5, [6, 7]].detach().softmax(-1),
            reduction="sum",
        )
        torch.testing.assert_close(a, expected)
        for value, positions in [(a, [5]), (t, [7, 8]), (s, [0, 2, 4, 6])]:
            grad = torch.autograd.grad(value, student, retain_graph=True)[0]
            self.assertEqual(grad.abs().sum(1).nonzero().flatten().tolist(), positions)
        (a + t + s).backward()
        self.assertIsNone(teacher.grad)

    def test_different_prompt_lengths_align_to_independent_next_token(self):
        class Stub:
            def __call__(self, x, use_cache, logits_to_keep):
                from types import SimpleNamespace

                return SimpleNamespace(
                    logits=x.cumsum(1)[:, -logits_to_keep:, None].float()
                )

        seq = [7, 2, 13]
        with patch.object(v3.core, "tensor", lambda x: torch.tensor([x])):
            for p in [[2, 4], [5, 7, 2, 1]]:
                actual = v3.core.aligned(Stub(), p, seq).flatten().tolist()
                self.assertEqual(actual, [sum(p + seq[:j]) for j in range(len(seq))])

    def test_real_frozen_cohorts_are_mutually_exclusive(self):
        import json

        d = (
            Path(__file__).resolve().parents[1]
            / "outputs/pa_opsd/v3_full_20261127_run1"
        )
        if not (d / "cycles_protocol.json").exists():
            self.skipTest("Frozen experiment bundle unavailable")
        master = {
            r["id"]: r
            for n in ("train", "dev")
            for r in cycles.read(d / (n + ".jsonl"))
        }
        ids = json.loads((d / "protocol.json").read_text(encoding="utf-8"))["train_ids"]
        panels = [ids] + [
            v
            for it in json.loads(
                (d / "cycles_protocol.json").read_text(encoding="utf-8")
            )["rounds"]
            for v in it.values()
        ]
        for field in ("uid", "qid", "stem"):
            prior = set()
            for panel in panels:
                keys = {master[k][field] for k in panel}
                self.assertFalse(prior & keys, field)
                prior.update(keys)


if __name__ == "__main__":
    unittest.main()
