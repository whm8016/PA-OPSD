"""Privileged-context collation for accepted PA-OPSD teaching examples."""

import json
import re

from scripts.pa_opsd_contract import student_messages, validate_target


def load_standard_solutions(rows, question_bank):
    """Join each verified training row to the official current exercise, not its history."""
    current_questions = {}
    for row in rows:
        sample_id = row["sample_id"]
        current = row["messages"][1]["content"].rpartition("\n### CURRENT QUESTION\n")[2]
        match = re.match(r"\[(\d+)\]\n", current)
        if match is None:
            raise ValueError(f"Missing native current question ID: {sample_id}")
        current_questions[sample_id] = match[1]

    needed = set(current_questions.values())
    solutions = {}
    with question_bank.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record["id"])
            if qid not in needed:
                continue
            if qid in solutions:
                raise ValueError(f"Duplicate question bank ID: {qid}")
            item = record["info"]
            answers, analysis = item.get("answer"), item.get("analysis")
            if (not answers or not isinstance(answers, list)
                    or not all(isinstance(answer, str) and answer.strip() for answer in answers)
                    or not isinstance(analysis, str) or not analysis.strip()):
                raise ValueError(f"Incomplete standard solution for current question {qid}")
            solutions[qid] = {"standard_answer": answers, "analysis": analysis}
    if missing := needed - solutions.keys():
        raise ValueError(f"Missing standard solutions for question IDs: {', '.join(sorted(missing))}")
    return {sample: solutions[qid] for sample, qid in current_questions.items()}


def standard_solution_messages(student, reference, *, include_guidance=True):
    teacher = [dict(message) for message in student]
    teacher[1]["content"] += (
        "\n\n### TEACHER-ONLY REFERENCE SOLUTION\n"
        + json.dumps(reference, ensure_ascii=False, sort_keys=True)
    )
    if include_guidance:
        teacher[1]["content"] += (
            "\nUse the worked solution to understand the exercise. Base memory answers and diagnosis "
            "on the student's records. Give an appropriate next hint without exposing the full solution. "
            "Return the same requested JSON response."
        )
    return teacher


def validate_distill_row(row):
    messages = row.get("messages", [])
    if len(messages) != 3 or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("PA-OPSD requires messages=[system,user,assistant]")
    if any(not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Message content must be text")
    if row.get("status") != "accepted" or row.get("dry_run") is not False:
        raise ValueError("Only accepted rows with dry_run=false may be distilled")
    if row.get("dryrun"):
        raise ValueError("Dry-run teacher output cannot be used for training")
    if "student_input" in row and messages[:2] != student_messages(row):
        raise ValueError("Student messages differ from the student-visible seed contract")
    target = json.loads(messages[2]["content"])
    # Accepted rows retain the native prompt, not the source seed's weak annotation.
    user_text = messages[1]["content"]
    if "### MEMORY QUERIES\n" not in user_text:
        raise ValueError("Accepted prompt must retain its native memory queries")
    query_section = user_text.split("### MEMORY QUERIES\n", 1)[1].split("\n### ", 1)[0]
    query_section = query_section.split("\n\nAnalyze the data and generate the JSON response.", 1)[0]
    queries = []
    for index, line in enumerate(query_section.strip().splitlines(), 1):
        prefix = f"- Q{index}: "
        if not line.startswith(prefix) or not line[len(prefix):].strip():
            raise ValueError("Invalid native memory query order")
        queries.append(line[len(prefix):].strip())
    if len(queries) != 3:
        raise ValueError("Expected three native memory queries")
    validate_target(target, queries)
    if not isinstance(row.get("teacher_privileged"), dict) or not row["teacher_privileged"]:
        raise ValueError("teacher_privileged must be a nonempty, separate context object")


class PAOPSDDataCollator:
    def __init__(self, tokenizer, max_length=2048, max_completion_length=256, reason_first=False, *,
                 standard_solutions=None, strict_output_prompt=False, standard_solution_guidance=True,
                 teacher_context=None):
        if reason_first:
            raise ValueError("PA-OPSD v1 supports reason_first=False only")
        if max_length <= max_completion_length or max_completion_length < 1:
            raise ValueError("max_length must leave room for prompts and completion")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer requires a pad token")
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.max_prompt_length = max_length - max_completion_length
        self.standard_solutions = standard_solutions
        self.standard_solution_guidance = standard_solution_guidance
        self.strict_output_prompt = strict_output_prompt
        self.teacher_context = teacher_context

    def prompt_messages(self, row):
        validate_distill_row(row)
        student = [dict(message) for message in row["messages"][:2]]
        if self.strict_output_prompt:
            from scripts.pa_opsd_contract import strict_output_messages

            student = strict_output_messages(student)
        if self.standard_solutions is not None:
            reference = self.standard_solutions[row["sample_id"]]
            context = self.teacher_context(student)[0] if self.teacher_context else student
            return student, standard_solution_messages(
                context, reference, include_guidance=self.standard_solution_guidance)
        teacher = student + [{
            "role": "user",
            "content": "Teacher-only privileged context (not student-visible):\n"
            + json.dumps(row["teacher_privileged"], ensure_ascii=False, sort_keys=True),
        }]
        return student, teacher

    def encode_prompt(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids or len(ids) > self.max_prompt_length:
            raise ValueError("Prompt exceeds context budget; PA-OPSD never silently truncates")
        return ids

    def __call__(self, features):
        import torch

        if not features:
            raise ValueError("Cannot collate an empty batch")
        encoded = {"student": [], "teacher": []}
        for row in features:
            student, teacher = self.prompt_messages(row)
            encoded["student"].append(self.encode_prompt(student))
            encoded["teacher"].append(self.encode_prompt(teacher))
        result = {"pa_opsd": True, "sample_ids": [row["sample_id"] for row in features]}
        for role, sequences in encoded.items():
            lengths = [len(ids) for ids in sequences]
            width = max(lengths)
            result[f"{role}_prompts"] = torch.tensor([
                [self.tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in sequences
            ], dtype=torch.long)
            result[f"{role}_prompt_attention_mask"] = torch.tensor([
                [0] * (width - length) + [1] * length for length in lengths
            ], dtype=torch.long)
            result[f"{role}_prompt_length"] = width
            result[f"{role}_prompt_lengths_per_example"] = torch.tensor(lengths, dtype=torch.long)
        return result


def pa_rollout_status(text, terminated):
    """Judge format and completion only, without using Gold answers."""
    if not terminated:
        return "missing_eos"
    from scripts.train_pa_opsd_sft import validate_native_output

    try:
        validate_native_output(text)
    except json.JSONDecodeError:
        return "invalid_json"
    except ValueError:
        return "invalid_native_schema"
    return "valid"


def gate_pa_opsd_rollouts(inputs, generated_ids, eos_token_id, tokenizer, retry):
    """Retry failed student prompts once and retain exact sampled IDs up to EOS."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    if not eos or None in eos:
        raise ValueError("PA-OPSD rollout validation requires explicit EOS token IDs")
    width = inputs["student_prompt_length"]
    sample_ids = inputs["sample_ids"]
    accepted, attempts = {}, []

    def select(indices):
        return {
            key: ([sample_ids[i] for i in indices] if key == "sample_ids"
                  else value[indices] if isinstance(value, torch.Tensor) else value)
            for key, value in inputs.items()
        }

    def check(indices, sampled, attempt):
        failed = []
        for batch_index, (index, tokens) in enumerate(zip(indices, sampled[:, width:].tolist(), strict=True)):
            stop = next((pos for pos, token in enumerate(tokens) if token in eos), None)
            choice = tokens if stop is None else tokens[:stop]
            text = tokenizer.decode(choice, skip_special_tokens=False)
            status = pa_rollout_status(text, stop is not None)
            kept = len(tokens) if stop is None else stop + 1
            attempts.append({"sample_id": sample_ids[index], "row_index": index, "attempt": attempt,
                             "status": status, "completion": text, "completion_tokens": kept})
            if status == "valid":
                accepted[index] = sampled[batch_index, :width + kept]
            else:
                failed.append(index)
        return failed

    failed = check(range(len(sample_ids)), generated_ids, 1)
    if failed:
        check(failed, retry(select(failed)), 2)
    indices = sorted(accepted)
    if not indices:
        return None, None, attempts
    generated = pad_sequence([accepted[i] for i in indices], batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    return select(indices), generated, attempts


def prepare_pa_opsd_sequences(inputs, generated_ids, eos_token_id, pad_token_id):
    """Share the exact sampled continuation, retaining its first terminating EOS."""
    import torch

    width = inputs["student_prompt_length"]
    continuation = generated_ids[:, width:]
    if continuation.shape[1] == 0:
        raise ValueError("PA-OPSD requires at least one generated token")
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    if not eos_ids or any(token is None for token in eos_ids):
        raise ValueError("PA-OPSD requires explicit EOS token IDs for completion masking")
    eos = torch.zeros_like(continuation, dtype=torch.bool)
    for token in eos_ids:
        eos |= continuation.eq(token)
    completion_mask = (eos.long().cumsum(dim=1) - eos.long()).eq(0)
    if pad_token_id is not None and pad_token_id not in eos_ids:
        completion_mask &= continuation.ne(pad_token_id)
    completion_mask = completion_mask.long()
    inputs["student_input_ids"] = generated_ids
    inputs["student_attention_mask"] = torch.cat(
        [inputs["student_prompt_attention_mask"], completion_mask], dim=1
    )
    inputs["teacher_input_ids"] = torch.cat([inputs["teacher_prompts"], continuation], dim=1)
    inputs["teacher_attention_mask"] = torch.cat(
        [inputs["teacher_prompt_attention_mask"], completion_mask], dim=1
    )
    labels = generated_ids.clone()
    labels[:, :width] = -100
    labels[:, width:] = continuation.masked_fill(completion_mask.eq(0), -100)
    inputs["labels"] = labels
    return inputs
