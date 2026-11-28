# Gold evidence-first 对照

## 当前实验与仓库准备

11 月 23 日的实验使用 420 条训练新顺序 SFT，再在 140 条开发题上比较基座和新顺序 SFT。证据语义及教学反馈已完成官方复评。 [基座与新顺序 SFT 结果](../../../docs/PA-OPSD近期进展与下一轮方案.md#longtutor-官方复评--基座与现有-sft)

代码依赖 [LongTutor](https://github.com/liano3/LongTutor) 官方仓库的脚本和 XES3G5M 数据，放在 `data/longtutor/repo/` 并检出提交 `ba40af3b9e976f5960c42a15a645c0e9c3a7d718`。固定 Gold 划分、派生输入包、模型权重和评测输出不随本仓库提供，运行前需要在本地准备。原始划分默认位于 `outputs/pa_opsd/gold_split_20261120/`，模型校验文件默认为该目录下 `pa_opsd_gold_20261120/base_model.json`。

11 月 23 日重启训练每 5 步保存检查点并保留最近 2 个。以下加权对照复用该保存间隔。

这个入口验证三个问题：在 SFT 目标中先输出历史答案是否改善诊断，标准解能否提供诊断监督，以及续训学习率和 OPSD 是否带来可复现的收益。原 Gold 数据和检查点继续用于历史复核。

| 阶段 | 数据和口径 |
| --- | --- |
| SFT 顺序对照 | 两组用同样的 420 条、同样的输入、相同初始 LoRA 种子及 `2e-5`。旧序 `diagnosis, reason, strategy, content, memory`，新序 `memory, diagnosis, reason, strategy, content`。两组都用同一份显式、合法的 LongTutor JSON 提示。 |
| 教师预检 | 同一 SFT 权重在同一批 140 道开发题上生成两份输出。教师用与 OPSD 训练一致的标准解追加段，学生用普通提示，Gold 参考只进入评分。诊断答对数和四类 Macro-F1 都提升才通过试验前筛选。 |
| 学习率与蒸馏 | 第二阶段仅用另留的 140 条。SFT 续训和 OPSD 从同一 SFT 适配器开始，使用同样的样本和累积配置。可用 `--steps` 固定更短的预算对照 `1e-6`、`5e-6`、`2e-5`。正式比较需要相同学习率与有效更新步数。 |

划分对学生、当前题 ID、归一题干和历史文本指纹分组。独立二阶段保证该阶段的当前题不在 SFT 中担任当前题或监督目标，题目仍可能在其他学生的历史记录中出现。原 140 道开发题保持原样本清单。只读取原 `development/` 下三个文件，不读取封存题。入口校验原始文件哈希、再校验派生包、目标序列、样本交集、教师输入边界及二阶段检查点归属。实际训练前可在 CPU 上检查保存的真实 tokenizer。

在工作区根目录准备输入和 AutoDL 包。

```text
python -m scripts.experiments.gold_evidence_first.run --bundle outputs/pa_opsd/gold_evidence_first/inputs_v1 prepare
python -m scripts.experiments.gold_evidence_first.run --bundle outputs/pa_opsd/gold_evidence_first/inputs_v1 check --tokenizer /path/to/qwen-tokenizer
python -m scripts.experiments.gold_evidence_first.run --bundle outputs/pa_opsd/gold_evidence_first/inputs_v1 pack --output outputs/pa_opsd/gold_evidence_first/gpu_v1.zip
```

归档包包含开发数据和引用，单独上传到授权的训练机。GitHub 只同步代码、公开构造的例子和测试。解压后从包根目录运行以下命令。使用已装好项目依赖的 AutoDL 环境，另外需要上游校验依赖 `openai`、`tqdm`。运行前按包内 `base_model.json` 核对 Qwen2.5-7B-Instruct 权重和配置。

```bash
MODEL=/path/to/Qwen2.5-7B-Instruct
gold() { python -m scripts.experiments.gold_evidence_first.run --bundle inputs "$@"; }

gold check --tokenizer "$MODEL"
gold train memory_first --model "$MODEL" --pin base_model.json --output runs/smoke --smoke
gold train memory_first --model "$MODEL" --pin base_model.json --output runs/memory_first

gold evaluate --model "$MODEL" --pin base_model.json --output runs/base_eval
gold evaluate --model "$MODEL" --pin base_model.json --checkpoint runs/memory_first --output runs/memory_eval
gold compare runs/base_eval runs/memory_eval --output runs/base_vs_memory_first.json

# 以下教师预检和二阶段属于后续研究，按需单独运行。
ANCHOR=runs/memory_first
gold evaluate --model "$MODEL" --pin base_model.json --checkpoint "$ANCHOR" --teacher-pair --output runs/teacher_probe

gold train sft_cont --model "$MODEL" --pin base_model.json --anchor "$ANCHOR" --learning-rate 5e-6 --steps 8 --output runs/cont_screen_5e6
gold train opsd --model "$MODEL" --pin base_model.json --anchor "$ANCHOR" --learning-rate 5e-6 --steps 8 --teacher-report runs/teacher_probe/report.json --output runs/opsd_screen_5e6
```

学习率筛查结束后，用相同入选学习率、不带 `--steps` 训练两组，再分别 `evaluate`、`compare`。每次使用新目录。默认批大小 1、累积 4，SFT 105 步，第二阶段 35 个尝试窗口。OPSD 跳过整窗无有效 JSON 的更新，因此不足 35 实际步时保留诊断结果并阻止按同预算汇总。带 `--smoke` 的适配器只用于工程验证。

训练目录保存真实命令、代码与数据哈希、软件版本、原始日志、学习率和梯度、有效步数、计入损失的 token 及前后权重变化。评测目录保存逐题原文及终止原因、四类诊断及召回、策略匹配、Q1–Q3 精确匹配和反馈字符重合。这里是同一入口的离线对照。正式教学质量仍需复用 LongTutor 裁判协议另评。

## 诊断类别加权对照

在同一个已校验的输入包上从基座训练，使用 `memory_first` 训练数据中的全部 420 道。`--balance-diagnosis-classes` 用各类题数的平方根倒数给每条样本的答案损失加权，并使样本平均权重为 1。类别只从训练目标的 `diagnosis` 读取。一次设置保留 105 步、`2e-5`、批大小 1、累积 4、LoRA 配置、训练输入及输出格式。运行配置和训练摘要记录类别权重，原入口省略该选项即可复现未加权目标。

```bash
gold train memory_first --model "$MODEL" --pin base_model.json --output runs/balanced_smoke --smoke --balance-diagnosis-classes
gold train memory_first --model "$MODEL" --pin base_model.json --output runs/balanced_sft --balance-diagnosis-classes
```

试跑取最长的 4 条，权重同样从全量训练集计算。正式训练结束后，将新适配器接入原始 LongTutor 推理和评分脚本。固定开发集 140 道、BF16 和同一 DK 裁判，复用现有基座与未加权 SFT 的逐题结果。主看诊断 Macro-F1 和各类命中，同时核对证据与教学。 [官方复评入口](../longtutor_reeval/README.md)

## 并行二阶段训练

| 入口 | 教师输入和更新方式 |
| --- | --- |
| `sft_cont` | 同一起点上的 Gold SFT 续训。 |
| `opsd` | 同源教师读全历史和题库标准解。 |
| `opsd_compact` | 同源教师读取精简历史、全历史统计和同一份标准解。按可见的题号和单日日期保留所有匹配记录。记忆题的范围不明确时，教师使用全历史。学生提示保持完整。 |
| `gkd_frozen` | 冻结的同词表 Qwen2.5-14B-Instruct 教师读全历史和标准解。校验本地权重、词表、特殊边界 token、提示分词和 logits 维度。教师前向无梯度。 |

在 `opd_variants/gpu_stage2.zip` 解压出的包根目录执行。各实例使用同一份包并各自运行一个分组，指定新的结果目录。`OPD_PYTHON` 为本地训练环境的解释器路径，`OPD_STUDENT_MODEL` 默认为原 SFT 训练时记录的基座路径。D 组还需设 `OPD_FROZEN_TEACHER`，路径指向 `teacher_14b.json` 固定版本的模型。

```bash
export OPD_PYTHON=/path/to/venv/bin/python
bash scripts/experiments/gold_evidence_first/run_stage2.sh sft_cont runs/sft_cont
bash scripts/experiments/gold_evidence_first/run_stage2.sh opsd runs/opsd
bash scripts/experiments/gold_evidence_first/run_stage2.sh opsd_compact runs/opsd_compact
bash scripts/experiments/gold_evidence_first/run_stage2.sh gkd_frozen runs/gkd_frozen
```

启动脚本默认统一用 `5e-6`。蒸馏组先预检教师，复用已保存的新 SFT 学生输出并重验逐题提示哈希。C 组额外在固定 Gold 24 上保存覆盖和回退情况。完成教师预检后在各组最长的 4 道训练题上试跑一个窗口，随后进行 35 个更新窗口和 140 道开发题评测。独立教师结果许可及有效更新数写进收据。显存、有效步数、离线质量评分待 GPU 运行验证。

## 一条虚构样例

学生可见输入是“上周做对两道通分题。今天把分母改成 12，分子忘了乘。请回答三道历史问题并给出诊断和提示”。教师额外看到“这道题的标准解需要分子、分母同乘 3”，历史和问题保持相同。SFT 目标先回答 `memory`，再写 `diagnosis`、`reason`、`strategy` 和 `content`。SFT 只在助手答案 token 上计算损失。OPSD 先由学生生成回答，教师在保留相同生成前缀的情况下提供分布监督。
