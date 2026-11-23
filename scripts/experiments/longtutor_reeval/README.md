# LongTutor Gold 140 复评

## 新顺序 SFT 正式复评

使用 `outputs/pa_opsd/longtutor_reeval_gold140/official_base_vs_memory_first_20261123/official_eval_gpu.zip`。包内包含固定开发集 140 道、官方 Gold、从此前 1,000 条官方推理结果提取的同题基座预测，以及训练 105 步的新顺序 SFT 适配器。已核对两组使用的 140 道官方提示词哈希。基座预测复用原始逐题结果。官方脚本固定在 `ba40af3`，两组均使用该版本的原始评分入口。

AutoDL 解压后，从包目录启用已有 vLLM 0.9.2 环境，使用 `base_model.json` 核验权重和 tokenizer。模型服务及 SFT 推理命令如下。先用独立输出文件加 `--limit 2` 试跑，确认后再运行全量。

```bash
vllm serve "$QWEN_MODEL" --served-model-name Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 \
  --max-num-seqs 4 --gpu-memory-utilization 0.91 --enable-lora --max-lora-rank 8 \
  --lora-modules memory_first=adapters/memory_first
```

```bash
PYTHONHASHSEED=0 OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  python -u upstream/eval_ai_tutor.py --input inputs.jsonl --tests gold.jsonl \
  --output sft.jsonl --model memory_first --temperature 0 --max_tokens 3000 --workers 4
```

取回 `sft.jsonl` 后，核对两组均覆盖同一 140 个 ID 并区分模型格式失败和服务故障。在本地配置 DK 的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和模型 ID，分别对 `base.jsonl`、`sft.jsonl` 运行下列原始评分入口。使用不同的逐题评分文件及运行日志，核对实际完成条数和裁判错误后汇总。

```text
python -u upstream/compute_ai_tutor_eval_metrics.py --pred BASE_OR_SFT.jsonl --gold gold.jsonl --out ARM.metrics.jsonl --model DK_MODEL --temperature 0 --workers 4 --limit 140
```

## 历史 Gold SFT、续训和 OPSD 复评

用固定的 140 条验证题，比较原始 Qwen2.5-7B-Instruct、Gold SFT、SFT 续训和 token 级 OPSD。两种续训各更新 35 步，复评直接加载保存的检查点。

## 本地准备

从项目根目录运行。固定哈希种子用于复现基模型推理时的策略选项顺序。

```powershell
$env:PYTHONHASHSEED = "0"
python scripts/experiments/longtutor_reeval/run.py prepare --output outputs/pa_opsd/longtutor_reeval_gold140/longtutor_reeval_gold140.zip
```

准备步骤按样本 ID 核对固定划分、官方数据、基模型结果及提示词哈希。包中包括 140 条官方输入和金标、对应的基模型预测、三个 LoRA 检查点和 32 条已有 Gemini 评分。32 条从验证集有完整 Gemini 评分的样本中按诊断分层，每类 8 条。包内 `manifest.json` 记录文件哈希和运行设置。

## 一台 5090 上复评

解压后进入解压目录。优先复用已有的 vLLM 0.9.2 环境和已校验的 Qwen2.5-7B-Instruct 权重，开两个终端。两个终端使用同一个 Python 环境。

终端一启动模型服务。

```bash
python run.py serve --bundle . --model /path/to/qwen_bf16
```

终端二先验证每个 LoRA 的两题推理，再复评全部 140 条。

```bash
python run.py infer --bundle . --output smoke --smoke
python run.py infer --bundle . --output predictions
```

三组 LoRA 使用同一套 BF16 基模型、官方提示词和官方推理入口。贪心解码，最多生成 3,000 tokens，上下文限制为 32,768。官方脚本记录模型输出格式失败，入口把 API 故障的记录留待重试。原始基模型预测按官方提示词哈希从已完成的 1,000 条中取出。

把 `predictions/` 取回本地运行无 API 评分。

```bash
python run.py score --bundle . --predictions predictions --output diagnosis_rouge.json
```

诊断按固定标签精确计分，同时报告四类 Macro-F1 和仓库的 Macro-F1。仓库在有空预测时会把空标签纳入平均。ROUGE-L 沿用发布评分代码。主要对照是 `sft_cont` 与 `pa_opsd`。

## 裁判校准和评分

配置候选裁判的 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 环境变量后，运行小样本校准。选定裁判后，四组使用相同的裁判模型及提示词。诊断维持本地标签计分，裁判用于证据的语义等价和教学四项。

```bash
python run.py judge --bundle . --arm base --calibration --model JUDGE_MODEL --output judge_calibration
python run.py judge --bundle . --predictions predictions --arm sft_cont --model JUDGE_MODEL --output judge_sft_cont
```

校准输出包含三类证据判定的一致率和双向分歧计数，以及教学四项绝对误差和评分偏差。已有 Gemini 评分记录缺少逐问请求状态，校准沿用保留下来的完整评分行。每次写入已成功的评分，裁判失败会暂停后续批次并保留进度。使用同一输出目录和设置重新运行可续评。裁判命令会请求外部 API，GPU 推理和离线计分可独立运行。

本地可以核验输入和预测对齐、包文件完整性及官方脚本的调用流程。实际 LoRA 加载与显存占用由 5090 上的两题试跑确认。

## Gold SFT 输出格式提示对照

`compare_output_prompts.py` 根据冻结的 Gold 140 开发集原生提示构造原版与 `pa_opsd_strict_json_v1` 两组，共用同一个 Gold SFT 适配器、BF16 推理和 3,000-token 上限。两组都关闭解码器的 JSON 强制约束，才能测出提示对首轮原始输出的影响。最终内容按 LongTutor 的解析、重试和归一化处理。两组均保存每次生成的原文、结束原因、严格校验错误和归一化预测。离线阶段复用 `score_gold140.py` 汇总四类诊断、策略与 ROUGE-L，并记录 Q1–Q3 的精确匹配。语义和反馈裁判可由现有 `judge` 入口使用各组的预测目录另行评分。

`run_output_prompt_comparison.sh` 从新结果目录启动，依次传入上一轮 GPU 实验目录、Gold 140 官方复评包解压目录、含 `.venv` 和 `qwen_bf16` 的推理运行时、冻结的 `development/prompts.jsonl`、已装 ROUGE 的评分解释器。它等待上一轮释放 GPU，先验证两道，再完整跑 140 道和本地计分。`--check` 只校验冻结的输入，`--check --score` 同时检查评分依赖。`--score --output <已完成生成的目录>` 可离线重算。

基模型同提示评测将上一轮目录指定为 Gold SFT 提示对照运行目录，第六个参数传 `base`。它只跑新版提示，逐题核对已完成的 Gold SFT 新版提示哈希与推理设置，验证结果写入 `matched_sft_protocol.json`。
