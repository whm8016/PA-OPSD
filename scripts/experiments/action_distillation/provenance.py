"""Freeze inference provenance before writing predictions and validate it on resumption."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_base(bundle, model):
    bundle = bundle.resolve()
    receipt = bundle / "base_verification.json"
    if not receipt.exists():
        receipt = bundle.parent / "base_verification.json"
    info = json.loads(receipt.read_text())
    if (
        info["base_revision"] != "a09a35458c702b33eeacc393d103063234e8bc28"
        or not info["verified"]
    ):
        raise ValueError("Missing base provenance")
    for name, h in info["verified"].items():
        if digest(model / name) != h:
            raise ValueError("Changed base file " + name)
    if any(
        (model / n).exists()
        for n in ("chat_template.jinja", "chat_template.json", "chat_templates")
    ):
        raise ValueError("Unexpected external chat template")
    return info


def bind(path, evidence, outputs):
    if path.exists():
        if json.loads(path.read_text()) != evidence:
            raise ValueError(
                "Existing output is bound to different model, code, configuration or inputs"
            )
        return
    if any(Path(f).exists() for f in outputs):
        raise ValueError("Unbound predictions already exist")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
