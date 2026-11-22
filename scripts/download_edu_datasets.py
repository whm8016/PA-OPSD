#!/usr/bin/env python3
"""Download education datasets into D:/OPSD/data/. Safe to re-run (skips existing)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    log(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def shallow_clone(repo: str, dest: Path) -> str:
    if (dest / ".git").is_dir():
        return "skipped (already cloned)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if any(dest.iterdir()):
            raise RuntimeError(
                f"{dest} exists but is not a git repo; rename/remove it and re-run"
            )
        dest.rmdir()
    run(["git", "clone", "--depth", "1", repo, str(dest)])
    return "cloned"


def hf_snapshot(repo_id: str, dest: Path, repo_type: str = "dataset") -> str:
    marker = dest / ".download_complete"
    if marker.is_file():
        return "skipped (already complete)"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return f"downloaded to {path}"


def download_url(url: str, dest_file: Path) -> str:
    if dest_file.is_file() and dest_file.stat().st_size > 0:
        return "skipped (file exists)"
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"  GET {url}")
    urllib.request.urlretrieve(url, dest_file)
    return f"saved {dest_file}"


def extract_zip(zip_path: Path, dest_dir: Path) -> str:
    if dest_dir.is_dir() and any(dest_dir.iterdir()):
        return "skipped (extract dir non-empty)"
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return f"extracted to {dest_dir}"


def main() -> None:
    status: dict[str, dict[str, str]] = {}
    DATA.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, callable]] = []

    # LongTutor (large; may take time)
    jobs.append(
        (
            "longtutor/repo",
            lambda: shallow_clone(
                "https://github.com/liano3/LongTutor.git",
                DATA / "longtutor" / "repo",
            ),
        )
    )

    # MathTutorBench (EMNLP 2025) — code + bundled benchmark data
    jobs.append(
        (
            "mathtutorbench/repo",
            lambda: shallow_clone(
                "https://github.com/eth-lre/mathtutorbench.git",
                DATA / "mathtutorbench" / "repo",
            ),
        )
    )

    # TutorBench (arXiv:2510.02663) — full set on Scale HF (not tutorbench/tutorbench 30-sample)
    jobs.append(
        (
            "tutorbench/hf",
            lambda: hf_snapshot("ScaleAI/TutorBench", DATA / "tutorbench" / "hf"),
        )
    )

    # XES3G5M (NeurIPS 2023) — official repo + HF interaction sequences mirror
    jobs.append(
        (
            "xes3g5m/repo",
            lambda: shallow_clone(
                "https://github.com/ai4ed/XES3G5M.git",
                DATA / "xes3g5m" / "repo",
            ),
        )
    )
    jobs.append(
        (
            "xes3g5m/hf_interaction_sequences",
            lambda: hf_snapshot(
                "Atomi/XES3G5M_interaction_sequences",
                DATA / "xes3g5m" / "hf_interaction_sequences",
            ),
        )
    )

    # Big-Math (Albalak et al., 2025) — HF dataset is gated; requires `hf auth login` + accept license
    jobs.append(
        (
            "bigmath/repo",
            lambda: shallow_clone(
                "https://github.com/SynthLabsAI/big-math.git",
                DATA / "bigmath" / "repo",
            ),
        )
    )
    jobs.append(
        (
            "bigmath/hf",
            lambda: hf_snapshot(
                "SynthLabsAI/Big-Math-RL-Verified",
                DATA / "bigmath" / "hf",
            ),
        )
    )

    jobs.append(
        (
            "eedi/dialogues",
            lambda: hf_snapshot(
                "Eedi/Question-Anchored-Tutoring-Dialogues-2k",
                DATA / "eedi" / "dialogues",
            ),
        )
    )

    jobs.append(
        (
            "eedi/misconceptions-graph",
            lambda: hf_snapshot(
                "Eedi/Eedi-Misconceptions-Graph",
                DATA / "eedi" / "misconceptions-graph",
            ),
        )
    )

    jobs.append(
        (
            "eduagentbench",
            lambda: hf_snapshot("eduagentbench/eduagentbench", DATA / "eduagentbench"),
        )
    )

    jobs.append(
        (
            "edubench/repo",
            lambda: shallow_clone(
                "https://github.com/ybai-nlp/EduBench.git",
                DATA / "edubench" / "repo",
            ),
        )
    )

    jobs.append(
        (
            "mmtutorbench/repo",
            lambda: shallow_clone(
                "https://github.com/TangciuYueng/MMTutorBench.git",
                DATA / "mmtutorbench" / "repo",
            ),
        )
    )

    # MMTutorBench HF (jsonl + keyframes; large)
    jobs.append(
        (
            "mmtutorbench/hf",
            lambda: hf_snapshot("Tangchiu/mmtutorbench", DATA / "mmtutorbench" / "hf"),
        )
    )

    jobs.append(
        (
            "ednet/repo",
            lambda: shallow_clone(
                "https://github.com/riiid/ednet.git",
                DATA / "ednet" / "repo",
            ),
        )
    )

    # EdNet contents only (~174KB zip) — full KT1 is ~1.2GB, opt-in via env
    def ednet_contents() -> str:
        z = DATA / "ednet" / "EdNet-Contents.zip"
        r = download_url("http://base.ustc.edu.cn/data/EdNet/EdNet-Contents.zip", z)
        extract_zip(z, DATA / "ednet" / "contents")
        return r

    jobs.append(("ednet/contents", ednet_contents))

    def astra_zenodo() -> str:
        z = DATA / "astra" / "ASTRA_Synthetic_Dataset_v2_N540.zip"
        url = "https://zenodo.org/records/18114909/files/ASTRA_Synthetic_Dataset_v2_N540.zip?download=1"
        r = download_url(url, z)
        extract_zip(z, DATA / "astra" / "extracted")
        return r

    jobs.append(("astra", astra_zenodo))

    log("=== Edu dataset download ===\n")
    for key, fn in jobs:
        log(f"[{key}]")
        try:
            status[key] = {"ok": True, "detail": fn()}
        except Exception as e:
            status[key] = {"ok": False, "detail": str(e)}
            log(f"  FAILED: {e}")
        log("")

    def xes3g5m_official_archive() -> str:
        """Google Drive archive from ai4ed/XES3G5M README (gzip tar, not zip)."""
        archive_path = DATA / "xes3g5m" / "XES3G5M.tar.gz"
        extract_dir = DATA / "xes3g5m" / "official"
        if extract_dir.is_dir() and any(extract_dir.iterdir()):
            return "skipped (extract dir non-empty)"
        if not archive_path.is_file() or archive_path.stat().st_size < 1_000_000:
            try:
                import gdown
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
                import gdown

            file_id = "1eFiIYyh5O2V90RA0brammGH6EpHvPDQe"
            gdown.download(id=file_id, output=str(archive_path), quiet=False)
        if tarfile.is_tarfile(archive_path):
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_dir)
            return f"extracted to {extract_dir}; archive={archive_path}"
        raise RuntimeError(f"{archive_path} is not a gzip tarball after download")

    if __import__("os").environ.get("OPSD_DOWNLOAD_XES3G5M_FULL") == "1":
        log("[xes3g5m/official_zip] OPSD_DOWNLOAD_XES3G5M_FULL=1")
        try:
            status["xes3g5m/official_zip"] = {
                "ok": True,
                "detail": xes3g5m_official_archive(),
            }
        except Exception as e:
            status["xes3g5m/official_zip"] = {"ok": False, "detail": str(e)}
    else:
        status["xes3g5m/official_zip"] = {
            "ok": None,
            "detail": "skipped (set OPSD_DOWNLOAD_XES3G5M_FULL=1 for Google Drive official XES3G5M.tar.gz)",
        }

    if __import__("os").environ.get("OPSD_DOWNLOAD_EDNET_KT1") == "1":
        log("[ednet/KT1] OPSD_DOWNLOAD_EDNET_KT1=1")
        try:
            z = DATA / "ednet" / "EdNet-KT1.zip"
            r = download_url(
                "http://base.ustc.edu.cn/data/EdNet/EdNet-KT1.zip", z
            )
            status["ednet/KT1"] = {"ok": True, "detail": r + " (not extracted; unzip manually)"}
        except Exception as e:
            status["ednet/KT1"] = {"ok": False, "detail": str(e)}
    else:
        status["ednet/KT1"] = {
            "ok": None,
            "detail": "skipped (set OPSD_DOWNLOAD_EDNET_KT1=1 to fetch ~1.2GB KT1)",
        }

    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(DATA),
        "items": status,
    }
    status_path = DATA / "DOWNLOAD_STATUS.json"
    status_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Download status",
        "",
        f"Updated: {summary['updated_at']}",
        "",
        "| Item | OK | Detail |",
        "|------|----|--------|",
    ]
    for k, v in status.items():
        ok = v["ok"]
        ok_s = "yes" if ok is True else ("no" if ok is False else "skip")
        detail = v["detail"].replace("|", "\\|")
        md_lines.append(f"| `{k}` | {ok_s} | {detail} |")
    (DATA / "DOWNLOAD_STATUS.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    log(f"Wrote {status_path} and DOWNLOAD_STATUS.md")
    failed = [k for k, v in status.items() if v["ok"] is False]
    if failed:
        log(f"Some items failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
