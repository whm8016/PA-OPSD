# 本地修复后的恢复运行步骤

当前实例由用户关闭。本文件供再次开启GPU后使用；本轮未自动重启实例。

先解压原始 `pa_opsd_sft_smoke_20261112.zip` 到新的工作目录，再在同目录解压 `pa_opsd_repair_overlay_20261114.zip`。覆盖包携带修复入口、测试、修订数据和57条验证种子，不包含凭据或模型权重。复用已安装的固定版本环境和已下载的 `models/Qwen2.5-7B-Instruct`，或按原SFT说明配置。

验证覆盖包成员哈希：

```bash
.venv/bin/python -c 'import json,hashlib,pathlib; m=json.loads(pathlib.Path("repair_bundle_manifest.json").read_text()); bad=[n for n,h in m.items() if hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print("bundle hashes OK")'
.venv/bin/python -m unittest discover -s tests -v
```

先检查数据：

```bash
.venv/bin/python scripts/train_pa_opsd_sft.py audit \
  --data outputs/pa_opsd/repair_curated_team4_20261114.jsonl \
  --model models/Qwen2.5-7B-Instruct --max-length 32768 --local-files-only
```

B0和后续模型使用同一验证输入、相同NF4精度与解码预算。以下只做4条工程检查，保留不合法输出，不使用Gold调参：

```bash
OMP_NUM_THREADS=8 .venv/bin/python scripts/train_pa_opsd_sft.py generate \
  --data data/pa_opsd/v0.2_question_disjoint_weak_seed/validation.jsonl \
  --split validation --limit 4 --model models/Qwen2.5-7B-Instruct \
  --load-in-4bit --max-length 32768 --max-new-tokens 1536 --local-files-only \
  --generation-summary outputs/b0_validation4_summary.json \
  > outputs/b0_validation4.jsonl 2> outputs/b0_validation4_stderr.log
```

格式检查失败时退出码为1；应检查原始响应，不能把退出码改成0。前置加载/数据错误为2。确认B0确实生成目标分母的全部行后，再进入工程短训。

从基础模型使用修订后的4条数据训练，不继承关闭前含错误计数的中断实验：

```bash
OMP_NUM_THREADS=8 .venv/bin/python scripts/train_pa_opsd_sft.py train \
  --data outputs/pa_opsd/repair_curated_team4_20261114.jsonl \
  --model models/Qwen2.5-7B-Instruct --max-length 32768 --local-files-only \
  --max-steps 48 --batch-size 1 --gradient-accumulation-steps 1 \
  --lora-r 8 --learning-rate 0.0002 --load-in-4bit \
  --gradient-checkpointing --assistant-logits-only --offload-saved-tensors \
  --attn-implementation sdpa --output-dir outputs/repair_curated_sft_step48
```

48步是小样本工程拟合，不是正式SFT实验规模。先在训练侧4条验证格式，再用与B0相同的验证输入检查：

```bash
OMP_NUM_THREADS=8 .venv/bin/python scripts/train_pa_opsd_sft.py generate \
  --data outputs/pa_opsd/repair_curated_team4_20261114.jsonl --split train \
  --adapter outputs/repair_curated_sft_step48 --max-length 32768 \
  --max-new-tokens 1536 --local-files-only \
  --generation-summary outputs/sft_train4_summary.json > outputs/sft_train4.jsonl

OMP_NUM_THREADS=8 .venv/bin/python scripts/train_pa_opsd_sft.py generate \
  --data data/pa_opsd/v0.2_question_disjoint_weak_seed/validation.jsonl \
  --split validation --limit 4 --adapter outputs/repair_curated_sft_step48 \
  --max-length 32768 --max-new-tokens 1536 --local-files-only \
  --generation-summary outputs/sft_validation4_summary.json > outputs/sft_validation4.jsonl
```

单教师生成入口使用 `--teacher-mode single`，团队使用 `--teacher-mode team`。修改源码或配置后必须换新日志名。正式B1/B2比较先取共同合格子集，再锁定数据量和训练/生成预算；不能把两次接受数量不同的试跑直接比较。

OPSD数据契约可以在本地先检查：

```bash
.venv/bin/python scripts/train_pa_opsd_distill.py \
  --train-jsonl outputs/pa_opsd/repair_curated_team4_20261114.jsonl \
  --sft-checkpoint outputs/repair_curated_sft_step48 \
  --teacher-mode local-same-model-privileged \
  --output-dir outputs/repair_opsd --max-length 32768 --check-data-only
```

真实OPSD仍需SFT输出质量过关后单独执行GPU单步，核对散度、掩码、教师无梯度和显存。新增精度/量化选项并不保证双提示长上下文OPSD能装入24GB。
