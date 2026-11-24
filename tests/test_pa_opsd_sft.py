"""Dependency-free contract/masking checks; optional local random-model gradient check."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import pa_opsd_contract
from scripts.train_pa_opsd_sft import (
    DEFAULT_MODEL,
    OverlengthError,
    encode_row,
    inference_rows,
    main,
    pad_examples,
    prepare_rows,
    read_rows,
    validate_against_seed_manifest,
    validate_native_output,
    validate_row,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>\n{m['content']}<end>\n" for m in messages)
        return text + ("<assistant>\n" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return {"input_ids": [ord(char) + 1 for char in text]}


def fixture():
    target = {
        "memory": [{"id": f"Q{i}", "answer": "not available"} for i in range(1, 4)],
        "diagnosis": "Recall Failure", "reason": "Retrieval failed.",
        "strategy": "Retrieval Practice", "content": "Can you recall the rule?",
    }
    return {
        "sample_id": "sample-1", "split": "train", "status": "accepted", "dry_run": False,
        "messages": [{"role": "system", "content": "Tutor"},
                     {"role": "user", "content": "Help me"},
                     {"role": "assistant", "content": json.dumps(target)}],
        "provenance": {"secret_marker": "SIDECAR_MUST_NOT_ENTER_TOKENS"},
    }


class SFTTests(unittest.TestCase):
    def setUp(self):
        self.row = fixture()
        self.tokenizer = FakeTokenizer()
        # Execute the real upstream pure validator without importing its API clients.
        source = (pa_opsd_contract.OFFICIAL / "eval_ai_tutor.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_validate_output")
        constants = ast.parse((pa_opsd_contract.OFFICIAL / "gpt_memory_diagnose.py").read_text(encoding="utf-8"))
        namespace = {"Any": object, "List": list, "Optional": __import__("typing").Optional}
        for node in constants.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"DIAGNOSES", "strategy_MAP"}:
                        namespace[target.id] = ast.literal_eval(node.value)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream-validator", "exec"), namespace)  # noqa: S102 - trusted repository function only
        patcher = patch.object(pa_opsd_contract, "official_module", return_value=SimpleNamespace(
            _validate_output=namespace["_validate_output"]))
        patcher.start()
        self.addCleanup(patcher.stop)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "requires torch and transformers")
    def test_repetition_filter_only_sees_completion_tokens(self):
        import torch

        from scripts.train_pa_opsd_sft import completion_ngram_processors

        processors = completion_ngram_processors(prompt_length=4, ngram_size=2)
        prompt = [1, 2, 3, 4]
        scores = torch.zeros(1, 8)
        first = processors(torch.tensor([prompt + [1]]), scores.clone())
        repeated = processors(torch.tensor([prompt + [1, 2, 1]]), scores.clone())
        self.assertTrue(torch.isfinite(first).all().item())
        self.assertTrue(torch.isneginf(repeated[0, 2]).item())

    def test_prompt_and_padding_are_masked(self):
        encoded = encode_row(self.row, self.tokenizer, 4096)
        prompt = self.tokenizer.apply_chat_template(self.row["messages"][:2], add_generation_prompt=True)
        boundary = len(prompt)
        self.assertEqual(encoded["labels"][:boundary], [-100] * boundary)
        self.assertEqual(encoded["labels"][boundary:], encoded["input_ids"][boundary:])
        text = "".join(chr(token - 1) for token in encoded["labels"] if token != -100)
        self.assertTrue(text.startswith(self.row["messages"][-1]["content"]))
        self.assertNotIn("SIDECAR", "".join(chr(token - 1) for token in encoded["input_ids"]))
        longer = copy.deepcopy(self.row)
        longer["messages"][1]["content"] += " extra prompt"
        other = encode_row(longer, self.tokenizer, 4096)
        padded = pad_examples([encoded, other], self.tokenizer.pad_token_id)
        difference = len(other["input_ids"]) - len(encoded["input_ids"])
        self.assertEqual(padded["labels"][0][-difference:], [-100] * difference)
        self.assertEqual(padded["attention_mask"][0][-difference:], [0] * difference)
        self.assertEqual(padded["input_ids"][0][-difference:], [0] * difference)

    def test_sidecars_inside_messages_are_not_rendered(self):
        for message in self.row["messages"]:
            message["provenance"] = "do not train on me"
        clean = validate_row(self.row)
        self.assertTrue(all(set(m) == {"role", "content"} for m in clean))
        self.assertEqual(encode_row(self.row, self.tokenizer, 4096),
                         encode_row(fixture(), self.tokenizer, 4096))

    def test_refuse_dry_run_and_unaccepted(self):
        for changes in ({"dry_run": True}, {"dry_run": 0}, {"dry_run": None},
                        {"status": "pending"}, {"status": "rejected"}):
            with self.subTest(changes=changes):
                row = {**self.row, **changes}
                with self.assertRaisesRegex(ValueError, "accepted"):
                    encode_row(row, self.tokenizer, 4096)
        del self.row["dry_run"]
        with self.assertRaises(ValueError):
            validate_row(self.row)

    def test_exact_roles_identity_and_output_required(self):
        for key, value in (("sample_id", ""), ("split", "gold"), ("split", "test"), ("messages", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_row({**self.row, key: value})
        self.row["messages"][0]["role"] = "user"
        with self.assertRaises(ValueError):
            validate_row(self.row)

    def test_native_five_fields_and_strategy(self):
        value = json.loads(self.row["messages"][-1]["content"])
        for broken in ({**value, "action_id": "probe"}, {**value, "strategy": "hint"},
                       {**value, "memory": []}, {**value, "reason": ""},
                       {**value, "diagnosis": []}, {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_native_output(json.dumps(broken))
        value["memory"][1]["id"] = "Q1"
        with self.assertRaises(ValueError):
            validate_native_output(json.dumps(value))

    def test_no_silent_truncation(self):
        encoded = encode_row(self.row, self.tokenizer, 4096)
        length = len(encoded["input_ids"])
        self.assertEqual(encode_row(self.row, self.tokenizer, length), encoded)
        with self.assertRaisesRegex(OverlengthError, "no truncation"):
            encode_row(self.row, self.tokenizer, length - 1)

    def test_text_prefix_mismatch(self):
        class BadTemplate(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                return "changed" + text if len(messages) == 3 else text
        with self.assertRaisesRegex(ValueError, "prefix"):
            encode_row(self.row, BadTemplate(), 4096)

    def test_token_prefix_mismatch(self):
        class BadBoundary(FakeTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if text.endswith("<assistant>\n"):
                    result["input_ids"][-1] = 999
                return result
        with self.assertRaisesRegex(ValueError, "prefix"):
            encode_row(self.row, BadBoundary(), 4096)

    def test_audit_counts_without_training_rejected_rows(self):
        long_row = copy.deepcopy(self.row)
        long_row["sample_id"] = "long"
        long_row["messages"][1]["content"] *= 500
        bad = {**self.row, "sample_id": "bad", "dry_run": True}
        features, report = prepare_rows([self.row, long_row, bad], self.tokenizer, 1000, audit=True)
        self.assertEqual(len(features), 1)
        self.assertEqual(report["overlength_rows"], 1)
        self.assertEqual(report["invalid_rows"], 1)
        self.assertGreater(report["supervised_tokens"], 0)
        self.assertEqual(report["coverage"], 1 / 3)
        with self.assertRaises(ValueError):
            prepare_rows([self.row, bad], self.tokenizer, 1000)

    def test_duplicate_and_empty_selected_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_rows([self.row, self.row], self.tokenizer, 4096)
        with self.assertRaisesRegex(ValueError, "no usable"):
            prepare_rows([self.row], self.tokenizer, 4096, split="validation")

    def test_reader_and_default_model(self):
        self.assertEqual(DEFAULT_MODEL, "Qwen/Qwen2.5-7B-Instruct")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            path.write_text("\n" + json.dumps(self.row) + "\n", encoding="utf-8")
            self.assertEqual(read_rows(path), [self.row])
            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                read_rows(path)

    def test_manifest_preflight_checks_membership_split_and_exact_prompt(self):
        seed = {"sample_id": self.row["sample_id"], "split": "train"}
        expected = self.row["messages"][:2]
        with patch.object(pa_opsd_contract, "read_jsonl", return_value=[seed]), \
                patch.object(pa_opsd_contract, "audit_seeds", return_value={"rows": 1}) as audit, \
                patch.object(pa_opsd_contract, "student_messages", return_value=expected):
            self.assertEqual(validate_against_seed_manifest([self.row], "manifest"), {"rows": 1})
            audit.assert_called_with([seed])
            for altered in ({**self.row, "sample_id": "unknown"}, {**self.row, "split": "validation"}):
                with self.assertRaises(ValueError):
                    validate_against_seed_manifest([altered], "manifest")
            contaminated = copy.deepcopy(self.row)
            contaminated["messages"][1]["content"] += " teacher privileged answer"
            with self.assertRaisesRegex(ValueError, "x-only"):
                validate_against_seed_manifest([contaminated], "manifest")
        with patch.object(pa_opsd_contract, "read_jsonl", return_value=[seed]), \
                patch.object(pa_opsd_contract, "audit_seeds", side_effect=ValueError("Gold overlap")), \
                self.assertRaisesRegex(ValueError, "Gold overlap"):
            validate_against_seed_manifest([self.row], "manifest")

    def test_cli_checks_manifest_before_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.jsonl"
            path.write_text(json.dumps(self.row), encoding="utf-8")
            with patch("scripts.train_pa_opsd_sft.validate_against_seed_manifest",
                       side_effect=ValueError("manifest rejected")) as preflight, \
                    patch("scripts.train_pa_opsd_sft.load_tokenizer") as loader:
                with self.assertRaises(SystemExit):
                    main(["train", "--data", str(path), "--seed-manifest", "strict.jsonl"])
                preflight.assert_called_once_with([self.row], "strict.jsonl")
                loader.assert_not_called()

    def test_cli_help_imports_no_ml_dependencies(self):
        script = Path(__file__).resolve().parents[1] / "scripts/train_pa_opsd_sft.py"
        result = subprocess.run([sys.executable, "-S", str(script), "--help"], check=False,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("audit", result.stdout)
        self.assertIn("--max-steps", result.stdout)

    def test_train_refuses_bad_row_before_loading_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejected.jsonl"
            path.write_text(json.dumps({**self.row, "dry_run": True}), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                main(["train", "--data", str(path), "--max-steps", "1"])
            self.assertEqual(caught.exception.code, 2)

    def test_training_validation_split_is_forbidden(self):
        with self.assertRaises(SystemExit) as caught:
            main(["train", "--data", "unused", "--split", "validation"])
        self.assertEqual(caught.exception.code, 2)

    def test_audit_cli_fails_when_every_row_is_overlength(self):
        with patch('scripts.train_pa_opsd_sft.read_rows', return_value=[self.row]), \
                patch('scripts.train_pa_opsd_sft.validate_against_seed_manifest'), \
                patch('scripts.train_pa_opsd_sft.load_tokenizer', return_value=self.tokenizer):
            self.assertEqual(main(['audit', '--data', 'unused', '--max-length', '2']), 1)

    def test_base_model_generation_is_a_supported_control(self):
        with patch('scripts.train_pa_opsd_sft.generate', return_value=0) as generate:
            self.assertEqual(main(['generate', '--data', 'unused', '--model', 'base']), 0)
            self.assertIsNone(generate.call_args.args[0].adapter)

    def test_unlabeled_inference_seeds_validate_membership_and_hide_privileged_fields(self):
        seed = {'sample_id': 's', 'split': 'validation', 'teacher_privileged': 'SECRET'}
        messages = [{'role': 'system', 'content': 'Tutor'}, {'role': 'user', 'content': 'visible'}]
        with patch.object(pa_opsd_contract, 'read_jsonl', return_value=[seed]), \
                patch.object(pa_opsd_contract, 'audit_seeds'), \
                patch.object(pa_opsd_contract, 'student_messages', return_value=messages):
            result = inference_rows([seed], 'manifest')
            self.assertEqual(result, [{'sample_id': 's', 'split': 'validation', 'messages': messages}])
            self.assertNotIn('SECRET', json.dumps(result))
            with self.assertRaises(ValueError):
                inference_rows([seed, seed], 'manifest')
            with self.assertRaises(ValueError):
                inference_rows([{**seed, 'split': 'train'}], 'manifest')

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "torch/transformers unavailable; no local gradient test")
    def test_selected_assistant_logits_match_dense_loss_and_gradients(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from scripts.train_pa_opsd_sft import assistant_token_loss

        torch.manual_seed(42)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32)).eval()
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 0, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
                  "labels": torch.tensor([[-100, -100, -100, -100, 5, 6],
                                          [-100, -100, 7, 8, -100, -100]])}
        dense = model(**inputs).loss
        dense.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        selected, outputs = assistant_token_loss(model, inputs)
        torch.testing.assert_close(selected, dense)
        self.assertEqual(outputs.logits.shape, (2, 4, 64))
        selected.backward()
        for name, p in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad, gradients[name], rtol=1e-4, atol=1e-6)
        model.zero_grad(set_to_none=True)
        offloaded, _ = assistant_token_loss(model, inputs, offload_saved_tensors=True)
        torch.testing.assert_close(offloaded, dense)
        offloaded.backward()
        for name, p in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad, gradients[name], rtol=1e-4, atol=1e-6)
        scaled, _ = assistant_token_loss(model, inputs, num_items_in_batch=8)
        torch.testing.assert_close(scaled, selected / 2)
        # A common-class row and a minority-class row share the same accumulation denominator.
        weights = [0.5, 1.5]
        from scripts.train_pa_opsd_sft import AssistantCollator

        weighted = AssistantCollator(0)([
            {**{key: value[i].tolist() for key, value in inputs.items()}, "sample_weight": weight}
            for i, weight in enumerate(weights)
        ])
        model.zero_grad(set_to_none=True)
        actual, _ = assistant_token_loss(model, weighted, num_items_in_batch=8)
        actual.backward()
        actual_grad = model.lm_head.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        expected = sum(weight * model(**{key: value[i:i + 1] for key, value in inputs.items()}).loss
                       * inputs["labels"][i, 1:].ne(-100).sum() for i, weight in enumerate(weights)) / 8
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, model.lm_head.weight.grad, rtol=1e-4, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "supervised assistant"):
            assistant_token_loss(model, {**inputs, "labels": torch.full_like(inputs["labels"], -100)})

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "torch/transformers unavailable; no local gradient test")
    def test_tiny_local_random_model_gradient(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=1024, n_embd=16,
                                          n_layer=1, n_head=2))
        encoded = encode_row(self.row, self.tokenizer, 1024)
        encoded["input_ids"] = [value % 256 for value in encoded["input_ids"]]
        encoded["labels"] = [value % 256 if value != -100 else -100 for value in encoded["labels"]]
        batch = {key: torch.tensor(value) for key, value in pad_examples([encoded], 0).items()}
        before = model.transformer.wte.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = model(**batch).loss
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        self.assertGreater(model.transformer.wte.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.transformer.wte.weight.detach()))


if __name__ == "__main__":
    unittest.main()
