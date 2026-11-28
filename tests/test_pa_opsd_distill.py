"""Dependency-light contract checks; tensor/gradient checks run only with torch."""

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pa_opsd_collator import (
    PAOPSDDataCollator,
    prepare_pa_opsd_sequences,
    validate_distill_row,
)
from scripts.train_pa_opsd_distill import load_accepted_rows

HAS_TORCH = importlib.util.find_spec("torch") is not None


def example():
    target = {"memory": [{"id": f"Q{i}", "answer": "Unknown"} for i in range(1, 4)],
              "diagnosis": "Conceptual Gap", "reason": "Historical evidence", "strategy": "Correction",
              "content": "Try recalling the definition"}
    return {"sample_id": "test-1", "split": "train", "status": "accepted", "dry_run": False,
            "teacher_privileged": {"standard_answer": "SECRET_REFERENCE"},
            "messages": [{"role": "system", "content": "Tutor"},
                         {"role": "user", "content": "History\n### MEMORY QUERIES\n- Q1: first?\n- Q2: second?\n- Q3: third?"},
                         {"role": "assistant", "content": json.dumps(target)}]}


class Tokenizer:
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, **kwargs):
        self.token_kwargs = kwargs
        return {"input_ids": [ord(char) + 1 for char in text]}


class ContractTests(unittest.TestCase):
    def setUp(self):
        # Keep native five-field/memory validation real; only replace the external
        # official module's optional SDK imports in this dependency-free suite.
        official = types.SimpleNamespace(_validate_output=lambda target, queries: None)
        self.patcher = patch("scripts.pa_opsd_contract.official_module", return_value=official)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_pa_opsd_rollout_restores_checkpointing_options_after_failure(self):
        from contextlib import contextmanager

        tree = ast.parse((ROOT / "opsd_trainer.py").read_text(encoding="utf-8"))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "OPSDTrainer")
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_unwrap_pa_opsd_for_generation")
        options = {"use_reentrant": False}

        class CheckpointedModel:
            def __init__(self, enabled):
                self.is_gradient_checkpointing = enabled
                self.restored = []

            def gradient_checkpointing_disable(self):
                self.is_gradient_checkpointing = False

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self.is_gradient_checkpointing = True
                self.restored.append(gradient_checkpointing_kwargs)

        @contextmanager
        def upstream_unwrap(model, accelerator):
            unwrapped = accelerator.unwrap_model(model)
            self.assertFalse(unwrapped.is_gradient_checkpointing)
            yield unwrapped

        namespace = {"contextmanager": contextmanager, "unwrap_model_for_generation": upstream_unwrap}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "opsd_unwrap", "exec"), namespace)  # noqa: S102 - trusted local trainer method
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                checkpointing = CheckpointedModel(enabled)
                harness = types.SimpleNamespace(
                    args=types.SimpleNamespace(gradient_checkpointing_kwargs=options),
                    accelerator=types.SimpleNamespace(unwrap_model=lambda model: model.module),
                )
                wrapped = types.SimpleNamespace(module=checkpointing)
                with (
                    self.assertRaisesRegex(RuntimeError, "rollout failure"),
                    namespace[method.name](harness, wrapped) as model,
                ):
                    self.assertIs(model, checkpointing)
                    raise RuntimeError("rollout failure")
                self.assertEqual(checkpointing.is_gradient_checkpointing, enabled)
                self.assertEqual(checkpointing.restored, [options] if enabled else [])

    def test_student_has_only_x_teacher_has_x_plus_z(self):
        collator = PAOPSDDataCollator(Tokenizer(), max_length=4096)
        row = example()
        student, teacher = collator.prompt_messages(row)
        self.assertEqual(student, row["messages"][:2])
        self.assertEqual(teacher[:2], student)
        self.assertEqual(teacher[-1]["role"], "user")
        self.assertNotIn("SECRET_REFERENCE", json.dumps(student))
        self.assertIn("SECRET_REFERENCE", teacher[-1]["content"])
        self.assertNotIn(row["messages"][2]["content"], json.dumps(teacher))
        self.assertEqual(len(row["messages"]), 3)

    def test_strict_output_prompt_replaces_invalid_example_for_both_roles(self):
        from scripts.pa_opsd_contract import strict_output_messages

        row = example()
        row["messages"][0]["content"] = (
            'Tutor task and diagnosis-to-strategy mapping\n\n### OUTPUT FORMAT\n'
            'Return strictly a JSON object:\n'
            '{"memory": [{"id": "Q1", "answer": "..."}]\n"diagnosis": "..."}\n'
            'Make outputs concise.\n'
        )
        original = copy.deepcopy(row)
        collator = PAOPSDDataCollator(
            Tokenizer(), max_length=4096, strict_output_prompt=True,
            standard_solutions={"test-1": {"standard_answer": ["SECRET_SOLUTION"]}},
        )
        student, teacher = collator.prompt_messages(row)
        self.assertEqual(student, strict_output_messages(row["messages"][:2]))
        self.assertEqual(row, original)
        self.assertEqual(student[1], original["messages"][1])
        self.assertEqual(student[0], teacher[0])
        self.assertIn("SECRET_SOLUTION", teacher[1]["content"])
        self.assertNotIn("SECRET_SOLUTION", json.dumps(student))
        system = student[0]["content"]
        self.assertTrue(system.startswith("Tutor task and diagnosis-to-strategy mapping\n"))
        start, end = system.index("{"), system.rindex("}")
        schema = json.loads(system[start:end + 1])
        self.assertEqual(set(schema), {"memory", "diagnosis", "reason", "strategy", "content"})
        self.assertEqual([item["id"] for item in schema["memory"]], ["Q1", "Q2", "Q3"])
        self.assertIn("Unknown", system)
        self.assertEqual(
            PAOPSDDataCollator(Tokenizer(), max_length=4096).prompt_messages(row)[0],
            original["messages"][:2],
        )

    def test_standard_solution_teacher_ignores_gold_and_joins_on_current_question(self):
        from pa_opsd_collator import load_standard_solutions

        row = example()
        row["messages"][1]["content"] = row["messages"][1]["content"].replace(
            "History", "History\n[8]\nPrevious problem\n\n### CURRENT QUESTION\n[3]\nCurrent problem"
        )
        row["teacher_privileged"] = {"gold_reference": {"memory": "GOLD_MEMORY", "content": "GOLD_FEEDBACK"}}
        with tempfile.TemporaryDirectory() as folder:
            bank = Path(folder) / "questions.jsonl"
            bank.write_text("\n".join(json.dumps(record) for record in (
                {"id": 8, "info": {"answer": ["PAST_ANSWER"], "analysis": "past"}},
                {"id": 3, "info": {"answer": ["CURRENT_ANSWER"], "analysis": "CURRENT_ANALYSIS",
                                    "diagnosis": "GOLD_DIAGNOSIS"}},
            )), encoding="utf-8")
            solutions = load_standard_solutions([row], bank)
            collator = PAOPSDDataCollator(Tokenizer(), max_length=4096, standard_solutions=solutions)
            student, teacher = collator.prompt_messages(row)
            self.assertEqual(student, row["messages"][:2])
            self.assertEqual([msg["role"] for msg in teacher], ["system", "user"])
            self.assertTrue(teacher[1]["content"].startswith(student[1]["content"] + "\n\n"))
            self.assertNotIn("CURRENT_ANSWER", student[1]["content"])
            self.assertIn("CURRENT_ANSWER", teacher[1]["content"])
            self.assertIn("CURRENT_ANALYSIS", teacher[1]["content"])
            for excluded in ("PAST_ANSWER", "GOLD_MEMORY", "GOLD_FEEDBACK", "GOLD_DIAGNOSIS"):
                self.assertNotIn(excluded, teacher[1]["content"])
            bank.write_text('{"id":3,"info":{"answer":[],"analysis":"worked"}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete standard solution.*3"):
                load_standard_solutions([row], bank)

    def test_precision_does_not_silently_reload_gpu_sft_as_fp32(self):
        from scripts.train_pa_opsd_distill import resolve_precision, weight_precision

        self.assertEqual(resolve_precision('auto', True, True), 'bfloat16')
        self.assertEqual(resolve_precision('auto', True, False), 'float16')
        self.assertEqual(resolve_precision('auto', False, False), 'float32')
        self.assertEqual(resolve_precision('float32', True, True), 'float32')
        with self.assertRaises(ValueError):
            resolve_precision('bfloat16', True, False)
        self.assertEqual(weight_precision('float16', is_adapter=False), 'float32')
        self.assertEqual(weight_precision('float16', is_adapter=True), 'float16')

    def test_quantized_distill_metadata_is_read_by_shared_sft_generation(self):
        from scripts.train_pa_opsd_distill import save_training_metadata
        from scripts.train_pa_opsd_sft import checkpoint_quantized

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data.jsonl'
            data.write_text('{}\n', encoding='utf-8')
            options = types.SimpleNamespace(sft_checkpoint='initial-adapter', load_in_4bit=True,
                max_length=32768, max_completion_length=256, train_jsonl=str(data),
                output_dir=directory, teacher_mode='local-same-model-privileged')
            save_training_metadata(options, 'bfloat16', 1, {'loss': 0.1})
            self.assertTrue(checkpoint_quantized(directory))
            self.assertFalse(checkpoint_quantized(None))
            self.assertTrue(checkpoint_quantized(None, requested=True))

    def test_rejects_unaccepted_and_implicit_or_true_dryrun(self):
        for changes in ({"status": "rejected"}, {"dry_run": True}, {"dry_run": None}, {"dry_run": 0}):
            row = example()
            row.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_distill_row(row)

    def test_rejects_extra_target_fields_and_missing_memory(self):
        for changes in ({"extra": "not-native"}, {"memory": []}):
            row = example()
            target = json.loads(row["messages"][2]["content"])
            target.update(changes)
            row["messages"][2]["content"] = json.dumps(target)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_distill_row(row)

    def test_identical_template_no_truncation_and_budget_failure(self):
        tokenizer = Tokenizer()
        collator = PAOPSDDataCollator(tokenizer, 4096, 100)
        student, teacher = collator.prompt_messages(example())
        collator.encode_prompt(student)
        kwargs = tokenizer.last_kwargs.copy()
        collator.encode_prompt(teacher)
        self.assertEqual(kwargs, tokenizer.last_kwargs)
        self.assertFalse(tokenizer.token_kwargs["truncation"])
        self.assertFalse(tokenizer.token_kwargs["add_special_tokens"])
        self.assertEqual(tokenizer.padding_side, "left")
        with self.assertRaises(ValueError):
            PAOPSDDataCollator(tokenizer, 20, 10).encode_prompt(student)
        with self.assertRaises(ValueError):
            PAOPSDDataCollator(tokenizer, reason_first=True)

    def test_train_split_only_and_sft_contract_gate(self):
        row = example()
        other = copy.deepcopy(row)
        other.update(split="validation", sample_id="test-2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in (row, other)), encoding="utf-8")
            with patch("scripts.train_pa_opsd_sft.validate_row", return_value=row["messages"]) as gate:
                self.assertEqual(load_accepted_rows(path), [row])
                self.assertEqual(gate.call_count, 2)
            with patch("scripts.train_pa_opsd_sft.validate_row", side_effect=ValueError("SFT mismatch")), self.assertRaisesRegex(ValueError, "SFT mismatch"):
                load_accepted_rows(path)

    def test_real_official_prompt_with_native_trailing_instruction(self):
        from scripts.pa_opsd_contract import (
            official_module,
            read_jsonl,
            student_messages,
        )
        from scripts.train_pa_opsd_sft import validate_row

        self.patcher.stop()
        official_module.cache_clear()
        seed = read_jsonl(ROOT / "data/pa_opsd/v0.2_question_disjoint_weak_seed/all.jsonl")[0]
        weak = seed["teacher_privileged"]["longtutor_weak_annotation"]
        target = {"memory": [{"id": f"Q{i}", "answer": "Unknown"} for i in range(1, 4)],
                  "diagnosis": weak["diagnosis"], "strategy": weak["strategy"],
                  "reason": "K/M历史证据", "content": "先回想概念。"}
        row = {"sample_id": seed["sample_id"], "split": seed["split"], "status": "accepted", "dry_run": False,
               "teacher_privileged": {"answer": "TEACHER_ONLY"},
               "messages": student_messages(seed) + [{"role": "assistant", "content": json.dumps(target)}]}
        validate_row(row)
        validate_distill_row(row)

    def test_teacher_forward_is_inside_no_grad_and_alignment_slices_match(self):
        tree = ast.parse((ROOT / "opsd_trainer.py").read_text(encoding="utf-8"))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OPSDTrainer")
        loss = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "compute_loss")
        teacher_blocks = [node for node in ast.walk(loss) if isinstance(node, ast.With)
                          and any("outputs_teacher" in ast.unparse(child) for child in node.body)]
        self.assertTrue(any("torch.no_grad()" in ast.unparse(node.items[0].context_expr) for node in teacher_blocks))
        text = ast.unparse(loss)
        self.assertIn("student_prompt_len - 1:-1", text)
        self.assertIn("teacher_prompt_len - 1:-1", text)
        training = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "training_step")
        branch = next(node for node in ast.walk(training) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "inputs.get('pa_opsd')" and node.orelse)
        self.assertIn("prepare_pa_opsd_sequences", ast.unparse(branch))
        self.assertTrue(branch.orelse, "Legacy mathematical behavior must remain isolated")


@unittest.skipUnless(HAS_TORCH, "torch missing: real tensor/EOS/alignment and gradient verification blocked")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def prepared(self):
        torch = self.torch
        inputs = {"student_prompt_length": 3, "teacher_prompt_length": 5,
                  "student_prompt_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
                  "teacher_prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
                  "teacher_prompts": torch.tensor([[0, 0, 4, 5, 6], [7, 8, 9, 5, 6]])}
        generated = torch.tensor([[0, 2, 3, 8, 0, 0], [1, 2, 3, 8, 9, 0]])
        return prepare_pa_opsd_sequences(inputs, generated, 0, 0)

    def test_fp16_autocast_full_model_master_weights_can_unscale(self):
        from scripts.train_pa_opsd_distill import weight_precision

        torch = self.torch
        model = torch.nn.Linear(2, 1).to(dtype=getattr(torch, weight_precision('float16', False)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Keep this test focused on master-weight unscaling, not dynamic overflow recovery.
        scaler = torch.amp.GradScaler('cpu', init_scale=128)
        with torch.autocast('cpu', dtype=torch.float16):
            loss = model(torch.ones(1, 2)).float().square().sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))

    def test_shared_continuation_eos_and_left_padding(self):
        inputs = self.prepared()
        self.assertEqual(inputs["labels"].tolist(), [[-100, -100, -100, 8, 0, -100], [-100, -100, -100, 8, 9, 0]])
        self.assertEqual(inputs["student_attention_mask"].tolist(), [[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
        self.assertEqual(inputs["teacher_attention_mask"].tolist(), [[0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
        self.assertTrue(self.torch.equal(inputs["student_input_ids"][:, 3:], inputs["teacher_input_ids"][:, 5:]))

    def test_rollout_gate_drops_bad_rows_and_retains_exact_retry_tokens(self):
        from pa_opsd_collator import gate_pa_opsd_rollouts

        torch = self.torch
        payload = json.loads(example()["messages"][2]["content"])
        payload["strategy"] = "Conceptual Explanation"
        payload = json.dumps(payload)
        valid = [ord(char) + 2 for char in payload]
        character = ord("{") + 2
        prompts = torch.tensor([[10, 20], [11, 21], [12, 22]])
        inputs = {"pa_opsd": True, "student_prompts": prompts, "student_prompt_length": 2,
                  "teacher_prompts": torch.tensor([[30], [31], [32]]),
                  "sample_ids": ["initial", "recovered", "skipped"]}
        first = torch.tensor([prompts[0].tolist() + valid + [1],
                              prompts[1].tolist() + [character, 1] + [0] * (len(valid) - 1),
                              prompts[2].tolist() + valid + [0]])
        retry_rows = []

        def retry(subset):
            retry_rows.extend(subset["sample_ids"])
            return torch.tensor([prompts[1].tolist() + valid + [1, 0],
                                 prompts[2].tolist() + [character, 1] + [0] * len(valid)])

        class RolloutTokenizer:
            pad_token_id = 0

            def decode(self, token_ids, skip_special_tokens=False):
                return "".join(chr(token - 2) if token > 1 else ("<eos>" if token else "<pad>")
                               for token in token_ids)

        accepted, generated, attempts = gate_pa_opsd_rollouts(inputs, first, 1, RolloutTokenizer(), retry)
        self.assertEqual(retry_rows, ["recovered", "skipped"])
        self.assertEqual(accepted["sample_ids"], ["initial", "recovered"])
        self.assertEqual(accepted["teacher_prompts"].tolist(), [[30], [31]])
        self.assertEqual(generated.tolist(), [prompts[0].tolist() + valid + [1],
                                              prompts[1].tolist() + valid + [1]])
        self.assertEqual([(a["sample_id"], a["attempt"], a["status"]) for a in attempts],
                         [("initial", 1, "valid"), ("recovered", 1, "invalid_json"),
                          ("skipped", 1, "missing_eos"), ("recovered", 2, "valid"),
                          ("skipped", 2, "invalid_json")])

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "Trainer integration needs Accelerate")
    def test_rollout_gate_skips_empty_optimizer_windows_and_keeps_mixed_gradients(self):
        import random
        from collections import Counter
        from contextlib import nullcontext
        from typing import Any

        from transformers import Trainer, TrainerCallback, TrainingArguments

        torch = self.torch
        tree = ast.parse((ROOT / "opsd_trainer.py").read_text(encoding="utf-8"))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OPSDTrainer")
        gate = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "PAOPSDRolloutCallback")
        step = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "training_step")
        step.decorator_list = []
        wrapper = ast.parse("class TrainingStep(Trainer): pass").body[0]
        wrapper.body = [step]
        namespace = {"Trainer": Trainer, "TrainerCallback": TrainerCallback, "Counter": Counter,
                     "nn": torch.nn, "torch": torch, "Any": Any, "random": random,
                     "gather_object": lambda objects: objects}
        source = ast.fix_missing_locations(ast.Module(body=[gate, wrapper], type_ignores=[]))
        exec(compile(source, "opsd_cpu_smoke", "exec"), namespace)  # noqa: S102 - executes trusted local trainer code
        callback = namespace["PAOPSDRolloutCallback"]()
        valid = json.loads(example()["messages"][2]["content"])
        valid["strategy"] = "Conceptual Explanation"
        optimizer_history = []

        class Snapshot(TrainerCallback):
            def on_step_end(self, args, state, control, optimizer, lr_scheduler, model, **kwargs):
                underlying = getattr(optimizer, "optimizer", optimizer)
                param = next(model.parameters())
                adam_step = underlying.state.get(param, {}).get("step", 0)
                if "step" in lr_scheduler.state_dict():
                    raise AssertionError("Suspending an empty window leaked a method into the scheduler state")
                optimizer_history.append((int(adam_step), lr_scheduler.last_epoch, float(param.detach())))

        class Harness(namespace["TrainingStep"]):
            pa_rollout_gate = callback
            reason_first = False
            use_vllm = False
            generation_config = types.SimpleNamespace(eos_token_id=1)
            _generation_save_frequency = 99

            def _get_train_sampler(self, train_dataset=None):
                return torch.utils.data.SequentialSampler(self.train_dataset)

            def _unwrap_pa_opsd_for_generation(self, model):
                return nullcontext(model)

            def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id):
                sample = inputs["sample_ids"][0]
                self.generation_count[sample] += 1
                good = sample == "7" or (sample == "2" and self.generation_count[sample] == 2)
                completion = [2 if good else 3, 1]
                if sample == "7":
                    completion.insert(1, 0)
                return torch.tensor([[9, *completion]]), None, None

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                self.trained.extend(inputs["sample_ids"])
                self.asserted_labels.append(inputs["labels"].tolist())
                return next(model.parameters()).square().sum()

        def batch(features):
            return {"pa_opsd": True, "sample_ids": [r["sample_id"] for r in features],
                    **{f"{role}_{field}": value for role in ("student", "teacher")
                       for field, value in (("prompts", torch.tensor([[9]])),
                                            ("prompt_attention_mask", torch.ones(1, 1, dtype=torch.long)),
                                            ("prompt_length", 1))}}

        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(output_dir=tmp, max_steps=4, per_device_train_batch_size=1,
                                     gradient_accumulation_steps=2, use_cpu=True, weight_decay=0.1,
                                     save_strategy="no", logging_strategy="no", report_to=[],
                                     disable_tqdm=True, remove_unused_columns=False)
            model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.ones_(model.weight)
            trainer = Harness(model=model, args=args, data_collator=batch, callbacks=[callback, Snapshot()],
                              train_dataset=[{"sample_id": str(i)} for i in range(8)])
            trainer.processing_class = types.SimpleNamespace(
                decode=lambda ids, **kwargs: json.dumps(valid) if ids in ([2], [2, 0]) else "{",
                batch_decode=lambda ids, **kwargs: ["student"] * len(ids), pad_token_id=0)
            trainer._textual_logs = {"prompt": [], "completion": []}
            trainer._generation_outputs_buffer = []
            trainer._on_policy_loss_total = trainer._on_policy_step_equiv = 0.0
            trainer.trained, trainer.asserted_labels = [], []
            trainer.generation_count = Counter()
            trainer.train()
        self.assertEqual(trainer.trained, ["2", "7"])
        self.assertEqual(trainer.asserted_labels, [[[-100, 2, 1]], [[-100, 2, 0, 1]]])
        self.assertEqual([row[:2] for row in optimizer_history], [(0, 0), (1, 1), (1, 1), (2, 2)])
        self.assertEqual(optimizer_history[1][2], optimizer_history[2][2])
        self.assertLess(optimizer_history[3][2], optimizer_history[2][2])
        self.assertEqual(callback.stats["effective_optimizer_steps"], 2)
        self.assertEqual(callback.stats["skipped_optimizer_windows"], 2)

    def test_compute_loss_real_tiny_gradient_and_teacher_stopgrad(self):
        torch = self.torch
        from contextlib import nullcontext
        tree = ast.parse((ROOT / "opsd_trainer.py").read_text(encoding="utf-8"))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OPSDTrainer")
        methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"compute_loss", "generalized_jsd_loss"}]
        namespace = {"torch": torch, "F": torch.nn.functional, "nullcontext": nullcontext,
                     "empty_cache": lambda: None, "is_peft_model": lambda model: False}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "opsd_loss_extracted", "exec"), namespace)  # noqa: S102 - trusted local trainer methods, no external input
        harness = types.SimpleNamespace(use_thinking_machines_loss=False, temperature=1.0,
                                        use_ema_teacher=False, fixed_teacher=False, beta=0.5,
                                        top_k_loss=None, jsd_token_clip=None, frozen_teacher_model=None)
        harness.generalized_jsd_loss = namespace["generalized_jsd_loss"]
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 10)
                self.calls = []
            def forward(self, input_ids, attention_mask, logits_to_keep=None, use_cache=None):
                ids = input_ids if logits_to_keep is None else input_ids[:, logits_to_keep]
                logits = self.embedding(ids)
                self.calls.append((torch.is_grad_enabled(), logits))
                return types.SimpleNamespace(logits=logits)
        model = Tiny()
        inputs = self.prepared()
        full_loss = namespace["compute_loss"](harness, model, inputs).detach()
        model.calls.clear()
        inputs['pa_opsd'] = True
        loss = namespace["compute_loss"](harness, model, inputs)
        torch.testing.assert_close(loss.detach(), full_loss)
        self.assertEqual([logits.shape[1] for _, logits in model.calls], [3, 3])
        loss.backward()
        self.assertEqual([flag for flag, _ in model.calls], [True, False])
        self.assertFalse(model.calls[1][1].requires_grad)
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)

        frozen = Tiny().requires_grad_(False).eval()
        harness.frozen_teacher_model = frozen
        model.calls.clear()
        model.zero_grad(set_to_none=True)
        loss = namespace["compute_loss"](harness, model, inputs)
        loss.backward()
        self.assertEqual([flag for flag, _ in model.calls], [True])
        self.assertEqual([flag for flag, _ in frozen.calls], [False])
        self.assertIsNone(frozen.embedding.weight.grad)
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)

    def test_pa_opsd_rollout_uses_autocast_eval_and_restores_training(self):
        torch = self.torch
        from contextlib import nullcontext, redirect_stdout
        from io import StringIO

        tree = ast.parse((ROOT / 'opsd_trainer.py').read_text(encoding='utf-8'))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OPSDTrainer')
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'generate_on_policy_outputs')
        namespace = {'torch': torch, 'nullcontext': nullcontext}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'opsd_rollout_extracted', 'exec'), namespace)  # noqa: S102 - trusted local trainer method

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(use_cache=False)
                self.dtype = torch.bfloat16
                self.observed = []

            def generate(self, **kwargs):
                self.observed.append((torch.is_autocast_enabled('cpu'), self.training, torch.is_grad_enabled()))
                return types.SimpleNamespace(sequences=kwargs['input_ids'])

        model = Tiny()
        config = types.SimpleNamespace(use_cache=False, max_new_tokens=2)
        harness = types.SimpleNamespace(accelerator=types.SimpleNamespace(
            autocast=lambda: torch.autocast('cpu', dtype=torch.bfloat16)))
        inputs = {'pa_opsd': True, 'student_prompts': torch.tensor([[1, 2]])}
        with redirect_stdout(StringIO()):
            namespace['generate_on_policy_outputs'](harness, model, inputs, config)
        self.assertEqual(model.observed, [(True, False, False)])
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(config.use_cache)


if __name__ == "__main__":
    unittest.main()
