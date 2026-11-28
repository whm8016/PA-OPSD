"""Freeze counterfactual action-score inputs on the already observed JSON prefixes."""

import argparse
from pathlib import Path

from scripts.experiments.action_pilot import current_step
from scripts.experiments.action_pilot import prefix_action_probe as probe


def main():
    args = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "dev", "tokenizer", "out"):
        args.add_argument("--" + name, type=Path, required=True)
    args.add_argument("--rule", choices=("no_examples", "original"), required=True)
    opts = args.parse_args()
    marker = "先确定实际话术，再按话术填最贴切的动作字母。"
    original = current_step.INSTRUCTION
    if original.count(marker) != 1:
        raise ValueError("Unknown intervention format")
    if opts.rule == "no_examples":
        current_step.INSTRUCTION = original.split(marker)[0] + marker
    else:
        # Keep the three teacher views and fixed observed prefix; remove only the added system instruction.
        orig_messages = current_step.messages

        def messages(row, arm):
            system, user = orig_messages(row, arm)
            if not system["content"].endswith("\n" + original):
                raise ValueError("Unexpected system prompt")
            return [
                dict(system, content=system["content"].removesuffix("\n" + original)),
                user,
            ]

        current_step.messages = messages
    probe.prepare(opts)
    p = opts.out / "config.json"
    import json

    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg.update(
        counterfactual_rule=opts.rule,
        rule_builder_sha256=probe.sha(Path(__file__)),
        interpretation="Counterfactual rescoring of the saved public-view structural prefix; no new student trajectory is sampled.",
    )
    probe.save(p, cfg)


if __name__ == "__main__":
    main()
