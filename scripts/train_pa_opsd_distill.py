"""Local same-model privileged-context PA-OPSD distillation (not API logits)."""

import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_accepted_rows(path):
    from pa_opsd_collator import validate_distill_row

    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                from scripts.train_pa_opsd_sft import validate_row
                validate_row(row)
                validate_distill_row(row)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid accepted row at line {line_number}: {exc}") from exc
            if row["split"] == "train":
                rows.append(row)
    if not rows:
        raise ValueError("Accepted training JSONL is empty")
    return rows


def resolve_precision(requested, cuda_available, bf16_supported):
    if requested == 'auto':
        return ('bfloat16' if bf16_supported else 'float16') if cuda_available else 'float32'
    if requested == 'bfloat16' and (not cuda_available or not bf16_supported):
        raise ValueError('bfloat16 requires a supported CUDA GPU')
    if requested == 'float16' and not cuda_available:
        raise ValueError('float16 training requires CUDA')
    return requested


def weight_precision(precision, is_adapter):
    return 'float32' if precision == 'float16' and not is_adapter else precision


def save_training_metadata(options, precision, rows, metrics, *, standard_solution_context=None,
                           rollout_validation=None, optimizer_log=None):
    summary = {'stage': 'opsd', 'rows': rows, 'model': str(options.sft_checkpoint),
               'load_in_4bit': options.load_in_4bit, 'precision': precision,
               'max_length': options.max_length, 'max_completion_length': options.max_completion_length,
               'metrics': metrics, 'source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
               'data_sha256': sha256(Path(options.train_jsonl).read_bytes()).hexdigest(),
               'teacher_mode': options.teacher_mode}
    for field in ('learning_rate', 'seed'):
        if hasattr(options, field):
            summary[field] = getattr(options, field)
    if optimizer_log is not None:
        summary['optimizer_log'] = optimizer_log
    if standard_solution_context is not None:
        summary['teacher_context'] = standard_solution_context
    if rollout_validation is not None:
        summary['rollout_validation'] = rollout_validation
    if getattr(options, 'strict_output_prompt', False):
        from scripts.pa_opsd_contract import STRICT_OUTPUT_PROMPT_VERSION

        summary['output_prompt_version'] = STRICT_OUTPUT_PROMPT_VERSION
    (Path(options.output_dir) / 'training_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None, *, validate_manifest=None, gold_run=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--seed-manifest", type=Path,
                        default=ROOT / "data/pa_opsd/v0.2_question_disjoint_weak_seed/all.jsonl")
    parser.add_argument("--sft-checkpoint", required=True, help="Local checkpoint from the contract-checked SFT stage")
    parser.add_argument("--teacher-mode", required=True, choices=["local-same-model-privileged"],
                        help="Explicitly acknowledge the teacher shares student weights and uses extra z; API text is not logits")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument('--precision', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--validate-rollouts', action='store_true',
                        help='Validate on-policy JSON and EOS, retry once, and skip invalid rollouts')
    parser.add_argument('--strict-output-prompt', action='store_true',
                        help='Replace the legacy output example for both student and teacher')
    parser.add_argument('--checkpoint-steps', type=int,
                        help='Save full Trainer checkpoints at this optimizer-step interval; keep the latest two')
    if gold_run:
        parser.add_argument('--standard-solution-bank', type=Path,
                            help='Use only the current problem answer and analysis from the official XES3G5M bank')
        parser.add_argument('--solution-only-addendum', action='store_true',
                            help='Append only the standard solution, without teacher-specific instructions')
    options = parser.parse_args(argv)
    if options.checkpoint_steps is not None and options.checkpoint_steps < 1:
        parser.error('--checkpoint-steps must be positive')
    rows = load_accepted_rows(options.train_jsonl)
    from scripts.train_pa_opsd_sft import validate_against_seed_manifest

    (validate_manifest or validate_against_seed_manifest)(rows, options.seed_manifest)
    bank = options.standard_solution_bank if gold_run else None
    solution_only = gold_run and options.solution_only_addendum
    if solution_only and bank is None:
        parser.error('--solution-only-addendum requires --standard-solution-bank')
    solutions, bank_hash = None, None
    if bank is not None:
        from pa_opsd_collator import load_standard_solutions

        bank_hash = sha256(bank.read_bytes()).hexdigest()
        solutions = load_standard_solutions(rows, bank)
    if options.check_data_only:
        if options.strict_output_prompt:
            from scripts.pa_opsd_contract import (
                STRICT_OUTPUT_PROMPT_VERSION,
                strict_output_messages,
            )

            for row in rows:
                strict_output_messages(row["messages"][:2])
            print(f"Prompt source matched {STRICT_OUTPUT_PROMPT_VERSION} for {len(rows)} rows")
        print(f"SFT/accepted message contract passed for {len(rows)} rows; no model or logits loaded")
        if solutions is not None:
            print(f"Matched {len(solutions)} current-question solutions; question bank SHA-256 {bank_hash}")
        return
    checkpoint = Path(options.sft_checkpoint)
    is_adapter = (checkpoint / "adapter_config.json").is_file()
    if not is_adapter and not (checkpoint / "config.json").is_file():
        raise ValueError("--sft-checkpoint must be an existing local SFT model or PEFT checkpoint")
    solution_context = None
    if bank_hash is not None:
        if not is_adapter:
            raise ValueError("The standard-solution Gold experiment requires a saved Gold SFT adapter")
        anchor_weights = checkpoint / "adapter_model.safetensors"
        solution_context = {
            "mode": "standard_solution", "question_bank_sha256": bank_hash,
            "initial_adapter_sha256": sha256(anchor_weights.read_bytes()).hexdigest(),
        }
        if solution_only:
            solution_context["addendum"] = "solution_only"
    if options.max_steps < 1 or options.batch_size < 1 or options.gradient_accumulation_steps < 1:
        raise ValueError("Training steps and batch/accumulation sizes must be positive")
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl.experimental.gold import GOLDConfig

        from opsd_trainer import OPSDTrainer
        from pa_opsd_collator import PAOPSDDataCollator
    except ImportError as exc:
        raise RuntimeError("Training requires the dependencies pinned in environment.yml; no training was run") from exc

    precision = resolve_precision(options.precision, torch.cuda.is_available(), torch.cuda.is_bf16_supported())
    if options.validate_rollouts and precision == "float16":
        raise ValueError("--validate-rollouts supports BF16 and FP32")
    dtype = getattr(torch, precision)
    # AMP FP16 requires FP32 master weights for full-model trainable parameters.
    # PEFT adapters are promoted separately while their frozen base can stay FP16.
    load_dtype = getattr(torch, weight_precision(precision, is_adapter))
    model_options = {'local_files_only': True, 'torch_dtype': load_dtype}
    if options.load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError('4-bit distillation requires a CUDA GPU')
        from transformers import BitsAndBytesConfig

        model_options.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype), device_map={'': torch.cuda.current_device()})

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), padding_side="left", local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS for unambiguous completion masking")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = PAOPSDDataCollator(tokenizer, options.max_length, options.max_completion_length,
                                 standard_solutions=solutions, strict_output_prompt=options.strict_output_prompt,
                                 standard_solution_guidance=not solution_only)
    # Check every prompt budget before starting any optimizer step.
    for row in rows:
        for messages in collator.prompt_messages(row):
            collator.encode_prompt(messages)
    control_options = {"save_strategy": "no", "logging_steps": 1, "disable_dropout": False} if gold_run else {}
    if options.checkpoint_steps is not None:
        control_options.update(save_strategy="steps", save_steps=options.checkpoint_steps, save_total_limit=2)
    args = GOLDConfig(
        output_dir=options.output_dir,
        max_length=options.max_length,
        max_completion_length=options.max_completion_length,
        per_device_train_batch_size=options.batch_size,
        gradient_accumulation_steps=options.gradient_accumulation_steps,
        max_steps=options.max_steps,
        learning_rate=options.learning_rate,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",
        use_vllm=False,
        bf16=precision == 'bfloat16',
        fp16=precision == 'float16',
        gradient_checkpointing=options.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        lmbda=1.0,
        seed=options.seed,
        data_seed=options.seed,
        **control_options,
    )
    model = str(checkpoint)
    if is_adapter:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM

        adapter_config = PeftConfig.from_pretrained(str(checkpoint), local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path, **model_options,
        )
        if options.load_in_4bit:
            from peft import prepare_model_for_kbit_training

            base = prepare_model_for_kbit_training(base,
                use_gradient_checkpointing=options.gradient_checkpointing,
                gradient_checkpointing_kwargs={'use_reentrant': False})
            base.get_input_embeddings().to(dtype=dtype)
            base.get_output_embeddings().to(dtype=dtype)
        model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True, local_files_only=True)
    else:
        if options.load_in_4bit:
            raise ValueError('4-bit distillation requires a trainable PEFT adapter checkpoint')
        args.model_init_kwargs = model_options
    args.presence_penalty = 0.0
    # TRL may cast a k-bit adapter to BF16 while constructing SFTTrainer.
    # Preserve the SFT checkpoint's FP32 masters for the small matched learning rate.
    masters = ({name: parameter.detach().clone() for name, parameter in model.named_parameters()
                if parameter.requires_grad} if gold_run and options.load_in_4bit else {})
    trainer = OPSDTrainer(
        model=model, args=args, train_dataset=Dataset.from_list(rows),
        data_collator=collator, processing_class=tokenizer,
        reason_first=False, fixed_teacher=False, use_ema_teacher=False,
        student_thinking=False, teacher_thinking=False,
        validate_pa_rollouts=options.validate_rollouts,
    )
    if masters:
        for name, parameter in trainer.model.named_parameters():
            if parameter.requires_grad:
                parameter.data = masters.pop(name).to(device=parameter.device)
        if masters:
            raise ValueError("TRL changed the trainable adapter parameters")
    print("Teacher mode: local-same-model-privileged; no API teacher logits; teacher forward is stop-gradient")
    result = trainer.train()
    trainer._save_generation_outputs(result.global_step)
    effective_steps = result.global_step
    rollout_validation = None
    if (gate := trainer.pa_rollout_gate) is not None:
        effective_steps = gate.stats["effective_optimizer_steps"]
        by_class = defaultdict(lambda: {"initial": 0, "skipped": 0})
        labels = {row["sample_id"]: json.loads(row["messages"][2]["content"])["diagnosis"] for row in rows}
        for sample_id, count in gate.initial_sample_ids.items():
            group = by_class[labels[sample_id]]
            group["initial"] += count
            group["skipped"] += gate.skipped_sample_ids[sample_id]
        rollout_validation = {
            "retry_limit": 1,
            "optimizer_windows": result.global_step,
            "effective_optimizer_steps": effective_steps,
            "skipped_by_gold_diagnosis": by_class,
            "counts": dict(gate.stats),
        }
    if solution_context is not None:
        solution_context["optimizer_steps"] = effective_steps
    if effective_steps == 0 and rollout_validation is not None:
        save_training_metadata(options, precision, len(rows), result.metrics,
                               standard_solution_context=solution_context, rollout_validation=rollout_validation,
                               optimizer_log=trainer.state.log_history)
        raise RuntimeError("No valid PA-OPSD rollouts reached an optimizer update")
    trainer.save_model(options.output_dir)
    tokenizer.save_pretrained(options.output_dir)
    save_training_metadata(options, precision, len(rows), result.metrics,
                           standard_solution_context=solution_context, rollout_validation=rollout_validation,
                           optimizer_log=trainer.state.log_history)


if __name__ == "__main__":
    main()
