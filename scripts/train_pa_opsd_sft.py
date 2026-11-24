#!/usr/bin/env python3
"""Assistant-only PA-OPSD SFT; no changes to the mathematical SFT pipeline."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED_MANIFEST = (Path(__file__).resolve().parents[1] / "data" / "pa_opsd" /
                         "v0.2_question_disjoint_weak_seed" / "all.jsonl")


def validate_against_seed_manifest(rows, seed_path):
    if __package__:
        from .pa_opsd_contract import audit_seeds, read_jsonl, student_messages
    else:
        from pa_opsd_contract import audit_seeds, read_jsonl, student_messages

    seeds = read_jsonl(seed_path)
    audit = audit_seeds(seeds)
    indexed = {}
    for seed in seeds:
        sample_id = seed["sample_id"]
        if sample_id in indexed:
            raise ValueError("duplicate sample_id in official seed manifest")
        indexed[sample_id] = seed
    seen = set()
    for row in rows:
        messages = validate_row(row)
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError("duplicate sample_id in training rows")
        seen.add(sample_id)
        seed = indexed.get(sample_id)
        if seed is None:
            raise ValueError(f"sample {sample_id} is absent from the audited seed manifest")
        if row["split"] != seed["split"]:
            raise ValueError(f"sample {sample_id}: split differs from audited seed manifest")
        if messages[:2] != student_messages(seed):
            raise ValueError(f"sample {sample_id}: student prompt differs from official x-only prompt")
    return audit


class OverlengthError(ValueError):
    pass


def validate_native_output(text):
    if __package__:
        from .pa_opsd_contract import validate_target
    else:
        from pa_opsd_contract import validate_target

    value = json.loads(text)
    try:
        validate_target(value, ["Q1", "Q2", "Q3"])
    except (TypeError, AttributeError) as exc:
        raise ValueError("invalid native target field type") from exc
    return value


def validate_row(row):
    if not isinstance(row, dict):
        raise TypeError("training row must be an object")
    if row.get("status") != "accepted" or row.get("dry_run") is not False:
        raise ValueError("only accepted rows with dry_run=false may enter SFT")
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"].strip():
        raise ValueError("sample_id must be a nonempty string")
    if row.get("split") not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("messages must contain exactly system, user, assistant")
    clean = []
    for message, role in zip(messages, ("system", "user", "assistant")):
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError("messages must be ordered system, user, assistant")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a nonempty string")
        # Pass only these two fields to the tokenizer, never provenance/sidecars.
        clean.append({"role": role, "content": content})
    validate_native_output(clean[-1]["content"])
    return clean


def encode_row(row, tokenizer, max_length):
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    messages = validate_row(row)
    prompt = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not full.startswith(prompt) or not full[len(prompt):].startswith(messages[-1]["content"]):
        raise ValueError("chat template is not an exact assistant continuation prefix")
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("tokenized chat template prefix mismatch or empty assistant")
    if len(full_ids) > max_length:
        raise OverlengthError(
            f"sample {row['sample_id']}: {len(full_ids)} tokens exceed max_length={max_length}; no truncation"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
    }


def pad_examples(examples, pad_token_id):
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    width = max(len(item["input_ids"]) for item in examples)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for item in examples:
        padding = width - len(item["input_ids"])
        result["input_ids"].append(item["input_ids"] + [pad_token_id] * padding)
        result["attention_mask"].append(item["attention_mask"] + [0] * padding)
        result["labels"].append(item["labels"] + [-100] * padding)
    return result


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch

        batch = {key: torch.tensor(value, dtype=torch.long)
                 for key, value in pad_examples(examples, self.pad_token_id).items()}
        if "sample_weight" in examples[0]:
            batch["sample_weight"] = torch.tensor([item["sample_weight"] for item in examples],
                                                  dtype=torch.float32)
        return batch


def read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
    if not rows:
        raise ValueError("training file contains no rows")
    return rows


def prepare_rows(rows, tokenizer, max_length, split="train", audit=False, *, class_weights=None):
    features = []
    report = {"total_rows": len(rows), "selected_rows": 0, "usable_rows": 0,
              "invalid_rows": 0, "overlength_rows": 0, "supervised_tokens": 0,
              "coverage": 0.0, "issues": []}
    seen = set()
    for index, row in enumerate(rows, 1):
        if isinstance(row, dict) and row.get("split") == split:
            report["selected_rows"] += 1
        try:
            validate_row(row)
            if row["sample_id"] in seen:
                raise ValueError("duplicate sample_id")
            seen.add(row["sample_id"])
            if row["split"] != split:
                continue
            encoded = encode_row(row, tokenizer, max_length)
            if class_weights is not None:
                label = json.loads(row["messages"][-1]["content"])["diagnosis"]
                if label not in class_weights:
                    raise ValueError(f"missing SFT class weight for {label!r}")
                encoded["sample_weight"] = class_weights[label]
        except (ValueError, TypeError) as exc:
            category = "overlength_rows" if isinstance(exc, OverlengthError) else "invalid_rows"
            report[category] += 1
            report["issues"].append({"row": index, "kind": category, "error": str(exc)})
            if not audit:
                raise ValueError(f"row {index}: {exc}") from exc
            continue
        features.append(encoded)
        report["supervised_tokens"] += sum(token != -100 for token in encoded["labels"])
    report["usable_rows"] = len(features)
    if report["selected_rows"]:
        report["coverage"] = len(features) / report["selected_rows"]
    if not audit and not features:
        raise ValueError(f"no usable {split} rows")
    return features, report


def load_tokenizer(name, local_files_only=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def assistant_token_loss(model, inputs, num_items_in_batch=None, offload_saved_tensors=False):
    import torch
    from torch.nn import functional as F

    # 第 t 个位置预测 t+1；跳过无监督历史的词表投影，历史本身仍完整参与注意力计算。
    targets = inputs["labels"][:, 1:]
    positions = torch.nonzero(targets.ne(-100).any(dim=0), as_tuple=True)[0]
    if not positions.numel():
        raise ValueError("batch has no supervised assistant tokens")
    targets = targets[:, positions]
    context = (torch.autograd.graph.save_on_cpu(pin_memory=torch.cuda.is_available())
               if offload_saved_tensors else nullcontext())
    with context:
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        logits_to_keep=positions, use_cache=False)
    sample_weight = inputs.get("sample_weight")
    loss = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100,
                           reduction="none" if sample_weight is not None else "sum")
    if sample_weight is not None:
        loss = (loss.reshape_as(targets).sum(dim=1) * sample_weight).sum()
    denominator = targets.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    return loss / denominator, outputs


def train(args, tokenizer, features):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    class AssistantOnlyTrainer(Trainer):
        supervised_tokens_seen = 0

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = assistant_token_loss(model, inputs, num_items_in_batch,
                                                 offload_saved_tensors=args.offload_saved_tensors)
            if model.training:
                self.supervised_tokens_seen += inputs["labels"][:, 1:].ne(-100).sum().item()
            return (loss, outputs) if return_outputs else loss

    torch.manual_seed(args.seed)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model_options = {"torch_dtype": dtype, "local_files_only": args.local_files_only,
                     "attn_implementation": args.attn_implementation}
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit training requires a CUDA GPU")
        from transformers import BitsAndBytesConfig

        model_options.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype), device_map={"": torch.cuda.current_device()})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **model_options,
    )
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        # PEFT 会把冻结的 embedding/head 升为 FP32；恢复计算精度，避免长历史激活也翻倍。
        # 归一化层仍保留 PEFT 设置的 FP32。
        model.get_input_embeddings().to(dtype=dtype)
        model.get_output_embeddings().to(dtype=dtype)
    if args.init_adapter:
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(args.init_adapter, local_files_only=args.local_files_only)
        if Path(config.base_model_name_or_path).resolve() != Path(args.model).resolve():
            raise ValueError('initial adapter and training base model differ')
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True,
                                         local_files_only=args.local_files_only)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_r * 2,
            lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    model.config.use_cache = False
    checkpointing = {"save_strategy": "no"}
    if getattr(args, "checkpoint_steps", None) is not None:
        checkpointing.update(save_strategy="steps", save_steps=args.checkpoint_steps, save_total_limit=2)
    class_weights = getattr(args, "diagnosis_class_weights", None)
    trainer = (AssistantOnlyTrainer if args.assistant_logits_only else Trainer)(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir, max_steps=args.max_steps,
            num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, logging_steps=1,
            report_to=[], seed=args.seed,
            remove_unused_columns=class_weights is None,
            bf16=use_bf16, gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            **checkpointing,
        ),
        train_dataset=features, data_collator=AssistantCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary = {"model": args.model, "rows": len(features), "max_length": args.max_length,
               "init_adapter": args.init_adapter, "seed": args.seed,
               "learning_rate": args.learning_rate, "optimizer_steps": result.global_step,
               "optimizer_log": trainer.state.log_history,
               'optimizer_restarted': bool(args.init_adapter),
               'supervised_tokens_in_dataset': sum(sum(t != -100 for t in f['labels']) for f in features),
               'lora_r': model.peft_config['default'].r,
               "load_in_4bit": args.load_in_4bit, "gradient_checkpointing": args.gradient_checkpointing,
               "assistant_logits_only": args.assistant_logits_only,
               "offload_saved_tensors": args.offload_saved_tensors, "metrics": result.metrics,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    if args.assistant_logits_only:
        summary["supervised_tokens_seen"] = trainer.supervised_tokens_seen
    if class_weights is not None:
        summary["diagnosis_class_weights"] = class_weights
    from hashlib import sha256

    summary['source_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    summary['data_sha256'] = sha256(Path(args.data).read_bytes()).hexdigest()
    (Path(args.output_dir) / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def inference_rows(rows, seed_manifest):
    """Allow B0 and held-out inference without manufacturing accepted teacher labels."""
    if __package__:
        from .pa_opsd_contract import audit_seeds, read_jsonl, student_messages
    else:
        from pa_opsd_contract import audit_seeds, read_jsonl, student_messages

    if all('messages' in row for row in rows):
        validate_against_seed_manifest(rows, seed_manifest)
        return [{'sample_id': r['sample_id'], 'split': r['split'],
                 'messages': validate_row(r)[:2]} for r in rows]
    if any('messages' in row for row in rows):
        raise ValueError('cannot mix seed and accepted inference rows')
    manifest = read_jsonl(seed_manifest)
    audit_seeds(manifest)
    indexed = {r['sample_id']: r for r in manifest}
    seen = set()
    result = []
    for row in rows:
        seed = indexed.get(row['sample_id'])
        if seed is None or row['sample_id'] in seen or row['split'] != seed['split']:
            raise ValueError('unknown, duplicate or wrong-split inference seed')
        if student_messages(row) != student_messages(seed):
            raise ValueError('inference seed differs from official x-only prompt')
        seen.add(row['sample_id'])
        result.append({'sample_id': row['sample_id'], 'split': row['split'],
                       'messages': student_messages(seed)})
    return result


def checkpoint_quantized(adapter, requested=False):
    path = Path(adapter) / 'training_summary.json' if adapter else None
    return requested or bool(path and path.is_file() and json.loads(
        path.read_text(encoding='utf-8')).get('load_in_4bit', False))


def completion_ngram_processors(prompt_length, ngram_size):
    from transformers import LogitsProcessorList, NoRepeatNGramLogitsProcessor

    class CompletionNoRepeatNGram(NoRepeatNGramLogitsProcessor):
        def __call__(self, input_ids, scores):
            return super().__call__(input_ids[:, prompt_length:], scores)

    return LogitsProcessorList([CompletionNoRepeatNGram(ngram_size)])


def generate(args, load_inference=None):
    rows = (load_inference or inference_rows)(read_rows(args.data), args.seed_manifest)
    output_prompt_version = None
    if getattr(args, 'strict_output_prompt', False):
        if __package__:
            from .pa_opsd_contract import (
                STRICT_OUTPUT_PROMPT_VERSION,
                strict_output_messages,
            )
        else:
            from pa_opsd_contract import (
                STRICT_OUTPUT_PROMPT_VERSION,
                strict_output_messages,
            )

        rows = [{**row, 'messages': strict_output_messages(row['messages'])} for row in rows]
        output_prompt_version = STRICT_OUTPUT_PROMPT_VERSION

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    base_model = args.model
    quantized = checkpoint_quantized(args.adapter, args.load_in_4bit)
    if args.adapter:
        config = PeftConfig.from_pretrained(args.adapter, local_files_only=args.local_files_only)
        base_model = config.base_model_name_or_path
    tokenizer = load_tokenizer(args.adapter or base_model, args.local_files_only)
    dtype = torch.float32 if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.bfloat16
    options = {'torch_dtype': dtype, 'local_files_only': args.local_files_only,
               'attn_implementation': args.attn_implementation}
    if quantized:
        if not torch.cuda.is_available():
            raise ValueError('4-bit generation requires a CUDA GPU')
        from transformers import BitsAndBytesConfig

        options.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype), device_map={'': torch.cuda.current_device()})
    model = AutoModelForCausalLM.from_pretrained(
        base_model, **options,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=args.local_files_only)
    if not quantized:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    valid = True
    selected = 0
    results = []
    for row in rows:
        if row["split"] != args.split:
            continue
        if args.limit is not None and selected >= args.limit:
            break
        selected += 1
        started = time.monotonic()
        prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + args.max_new_tokens > args.max_length:
            raise ValueError("prompt plus max_new_tokens exceeds max_length; reduce generation budget")
        generation_options = {}
        if args.completion_no_repeat_ngram_size:
            generation_options = {
                "logits_processor": completion_ngram_processors(
                    prompt_length, args.completion_no_repeat_ngram_size),
                # Checkpoint defaults can otherwise also block n-grams from the prompt.
                "no_repeat_ngram_size": 0,
                "encoder_no_repeat_ngram_size": 0,
            }
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, **generation_options)
        generated = output[0, prompt_length:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        try:
            validate_native_output(text)
            error = None
        except ValueError as exc:
            valid = False
            error = str(exc)
        from hashlib import sha256

        result = {"sample_id": row["sample_id"], "split": row['split'], "output": text,
                  "valid": error is None, "error": error,
                  'prompt_sha256': sha256(prompt.encode()).hexdigest(),
                  'input_tokens': inputs['input_ids'].shape[1], 'output_tokens': len(generated),
                  'seconds': time.monotonic() - started,
                  'stop_reason': 'token_limit' if len(generated) >= args.max_new_tokens and
                  int(generated[-1]) != tokenizer.eos_token_id else 'eos'}
        if output_prompt_version:
            result['output_prompt_version'] = output_prompt_version
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if not selected:
        raise ValueError(f"no usable {args.split} rows for generation")
    if args.generation_summary:
        if __package__:
            from .pa_opsd_contract import contract_metadata
        else:
            from pa_opsd_contract import contract_metadata

        summary = {'model': base_model, 'adapter': args.adapter, 'load_in_4bit': quantized,
                   'dtype': str(dtype), 'data_sha256': sha256(Path(args.data).read_bytes()).hexdigest(),
                   'source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                   'contract': contract_metadata(),
                   'split': args.split, 'selected_rows': selected,
                   'native_valid_rows': sum(r['valid'] for r in results),
                   'max_length': args.max_length, 'max_new_tokens': args.max_new_tokens,
                   'completion_no_repeat_ngram_size': args.completion_no_repeat_ngram_size,
                   'do_sample': False, 'input_tokens': sum(r['input_tokens'] for r in results),
                   'output_tokens': sum(r['output_tokens'] for r in results),
                   'evaluation': 'engineering_format_only_not_official_teaching_score'}
        if output_prompt_version:
            summary['output_prompt_version'] = output_prompt_version
        path = Path(args.generation_summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if valid else 1


def main(argv=None, *, validate_manifest=None, load_inference=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "train", "generate"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--split", choices=["train", "validation"], default=None)
    parser.add_argument("--seed-manifest", default=str(DEFAULT_SEED_MANIFEST))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="outputs/pa_opsd_sft")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--assistant-logits-only", action="store_true",
                        help="Compute vocabulary logits only at supervised positions; requires model logits_to_keep support")
    parser.add_argument("--diagnosis-class-weights",
                        help="JSON object mapping diagnosis labels to positive SFT sample loss weights")
    parser.add_argument("--offload-saved-tensors", action="store_true",
                        help="Store saved forward tensors on CPU; requires --assistant-logits-only")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--adapter")
    parser.add_argument('--init-adapter', help='Initialize trainable LoRA from an existing adapter; fresh optimizer')
    parser.add_argument('--checkpoint-steps', type=int, help='Save training state every N optimizer steps')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--generation-summary')
    parser.add_argument('--strict-output-prompt', action='store_true',
                        help='Use a valid, explicit JSON example during local generation')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--completion-no-repeat-ngram-size", type=int, default=0,
                        help="Block repeated n-grams in generated tokens only (0 disables the filter)")
    args = parser.parse_args(argv)
    weights = None
    if args.diagnosis_class_weights is not None:
        if args.command not in {"train", "audit"} or not args.assistant_logits_only:
            parser.error("--diagnosis-class-weights requires train or audit and --assistant-logits-only")
        try:
            weights = json.loads(args.diagnosis_class_weights)
        except ValueError:
            parser.error("--diagnosis-class-weights must be valid JSON")
        if not isinstance(weights, dict) or not weights or any(
            type(weight) not in (int, float) or not 1.17549435e-38 <= weight <= 3.40282346e38
            for weight in weights.values()
        ):
            parser.error("--diagnosis-class-weights must map labels to positive normal float32 numbers")
        args.diagnosis_class_weights = weights
    if args.checkpoint_steps is not None and (args.command != "train" or args.checkpoint_steps < 1):
        parser.error("--checkpoint-steps requires train and a positive interval")
    if args.strict_output_prompt and args.command != "generate":
        parser.error("--strict-output-prompt applies only to generate")
    args.split = args.split or ("validation" if args.command == "generate" else "train")
    if args.command == "train" and args.split != "train":
        parser.error("training may only consume the train split")
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    if args.completion_no_repeat_ngram_size < 0:
        parser.error("completion-no-repeat-ngram-size must be nonnegative")
    if args.offload_saved_tensors and not args.assistant_logits_only:
        parser.error("offload-saved-tensors requires --assistant-logits-only")
    if args.max_length < 2 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-length >= 2 and max-steps must be -1 or positive")
    try:
        if args.command == "generate":
            return generate(args) if load_inference is None else generate(args, load_inference)
        rows = read_rows(args.data)
        # Fail closed before importing model dependencies or downloading anything.
        if args.command == "train":
            for row in rows:
                validate_row(row)
        (validate_manifest or validate_against_seed_manifest)(rows, args.seed_manifest)
        tokenizer = load_tokenizer(args.model, args.local_files_only)
        features, report = prepare_rows(rows, tokenizer, args.max_length, args.split,
                                        audit=args.command == "audit", class_weights=weights)
        print(json.dumps(report, ensure_ascii=False))
        if args.command == "audit":
            return 1 if report["invalid_rows"] or report['overlength_rows'] or not report["usable_rows"] else 0
        train(args, tokenizer, features)
        return 0
    except ImportError as exc:
        parser.exit(2, f"Missing optional training dependency: {exc}. Use environment.yml on the training host.\n")
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
