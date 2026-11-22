# LongTutor B1 zero-shot baseline on AutoDL

This folder runs the untrained `Qwen2.5-7B-Instruct` model on LongTutor Gold1000 and produces the first B1 baseline table.

## 1. What B1 means

- backbone: `Qwen2.5-7B-Instruct`
- benchmark: LongTutor Gold, 1,000 samples in `human_an_updated.jsonl`
- model input: student history, current question, and three memory queries
- model output: memory answers, diagnosis, strategy, and teaching response
- training: none

Gold diagnosis and Gold teaching references are used only after generation by the metric script.

## 2. One-time setup

```bash
bash autodl/setup_longtutor_b1.sh
```

The script creates and uses a dedicated conda environment named `longtutor-b1`; it does not install packages into AutoDL's base environment.

It performs these steps:

1. find AutoDL's conda executable;
2. create Python 3.10 environment `longtutor-b1` if needed;
3. install LongTutor requirements and vLLM in that environment;
4. obtain the LongTutor official repository;
5. rebuild XES3G5M history features with the correct XES concept segmentation;
6. verify all 1,000 history keys align with Gold;
7. write exact package versions to `autodl/longtutor_b1_freeze.txt`.

### If GitHub clone fails

AutoDL connections to GitHub can terminate with GnuTLS errors. The setup script tries, in order:

1. AutoDL academic acceleration through `/etc/network_turbo`;
2. a direct GitHub clone;
3. an offline seed zip named `LongTutor_official_seed*.zip`.

The seed can be placed beside the project's `autodl/` directory or at `offline/LongTutor_official_seed*.zip`. It contains the official LongTutor scripts and source XES3G5M files, but not the generated 428MB history files; those are rebuilt and verified locally.

For the current failed clone, you can also retry immediately in the same shell:

```bash
source /etc/network_turbo
rm -rf /root/autodl-tmp/longtutor_b1/data/longtutor/repo
bash autodl/setup_longtutor_b1.sh
```

## 3. One-time runtime configuration

After setup succeeds, write all repeated parameters into `autodl/local.env`:

```bash
bash autodl/configure_longtutor_b1.sh
```

The default model path is:

```text
/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master
```

The script asks for the local model directory, judge endpoint/model/key, worker count, context length, and GPU settings. The API key is entered hidden and `local.env` is created with permission `600`.

You do not need to run this configuration again unless the model, judge, or GPU settings change.

## 4. Start Qwen

Terminal 1:

```bash
bash autodl/start_vllm_qwen25_7b.sh
```

The launcher reads `autodl/local.env`, activates `longtutor-b1`, uses the local Qwen snapshot when available, and exposes it as `Qwen2.5-7B-Instruct` at:

```text
http://127.0.0.1:8000/v1
```

For a 24G GPU, configure these values in `autodl/local.env`:

```text
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=4
WORKERS=4
```

## 5. Run B1

First run two samples:

```bash
bash autodl/run_longtutor_b1.sh smoke
```

Check that `summary.json` has:

```text
samples = 2
missing_gold = 0
content.scored = 2
```

Then run all 1,000 samples:

```bash
bash autodl/run_longtutor_b1.sh full
```

If no judge key was configured, the run still saves `preds.jsonl`; rerun the same command after adding the judge configuration to score Teaching.

## 6. Outputs

```text
outputs/longtutor_b1/<run_id>/
├── preds.jsonl
├── metrics.jsonl
├── metrics_run.log
├── summary.json
└── nvidia_smi.txt
```

Use `summary.json` for the B1 report:

- `memory.acc`: Evidence average
- `memory_by_type`: three Evidence subtasks
- `diagnosis.acc`: Diagnosis accuracy
- `diagnosis_macro_f1`: Diagnosis macro-F1
- `content.avg_scores`: Teaching dimensions
- `content.avg_rouge_l`: ROUGE-L

Keep the target prompt, split, judge, temperature, and worker protocol fixed for later B2/PA-OPSD comparisons.
### If PyPI mirror returns 403

AutoDL may be configured with an HTTP Aliyun mirror. If a large wheel such as vLLM is cached but one dependency returns HTTP 403, setup retries the installation with:

1. HTTPS Aliyun mirror;
2. Tsinghua mirror;
3. official PyPI;
4. AutoDL academic acceleration plus official PyPI.

The failed attempt can be resumed directly; conda and already-installed packages are reused.
