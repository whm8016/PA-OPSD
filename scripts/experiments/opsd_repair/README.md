# 普通 OPSD 复测

Gold 140 验证集上，SFT 续训为 68/140、四类 Macro-F1 为 21.33%，旧 token 级 OPSD 为 66/140 和 20.91%。旧训练每步从学生推理得到轨迹，教师在同一轨迹前缀计算逐 token 分布并停止教师梯度。现有证据指向两个需要独立检验的因素。

- 旧教师从 `teacher_privileged.gold_reference` 读取整份 Gold 教学输出，包括三道记忆题答案、诊断、理由、策略和参考辅导文本。复测新增只读当前题标准答案及解析的教师。当前题从冻结提示词的 `CURRENT QUESTION` 段取 ID，按 ID 连接官方 XES3G5M 题库。学生提示和原训练 JSONL 保持原样。
- 旧日志 140 条 rollout 中有 51 条用了全部 512 tokens。训练轨迹达到预算上限时可能在 JSON 闭合前结束。先以 512 复现教师输入差异。随后从同一 Gold SFT 起点重跑标准解教师，学生轨迹上限改为 1,024 tokens，官方复评保留 3,000 tokens，便于与既有结果比较。

标准解版的 140 条轨迹中有 37 条达到 512 tokens。留存的 124 条里有 34 条封顶且带未结束的无效 JSON。90 条未封顶的轨迹中，61 条符合原生 JSON schema、14 条 JSON 无效、15 条 schema 无效。提高预算可解决部分末尾截断，格式和字段问题还要独立统计。脚本训练完成时把最后一个缓冲区保存到 `generations/`，便于下一轮审计。

```text
python scripts/experiments/opsd_repair/audit_rollouts.py --run outputs/pa_opsd/opsd_repair/autodl_5090_38169/formal --output outputs/pa_opsd/opsd_repair/autodl_5090_38169/rollout_audit --gradient-accumulation-steps 4
```

## 数据与复测入口

从原 Gold 实验目录使用同一个 Gold SFT 适配器、同一个 `continuation.jsonl`、同一 140 条数据顺序、同一学习率与 35 个更新步。保留原实验目录。同步当前仓库的 `opsd_trainer.py`、`pa_opsd_collator.py`、`scripts/pa_opsd_contract.py`、`scripts/train_pa_opsd_distill.py` 与 `scripts/train_pa_opsd_sft.py` 到运行目录对应路径。`continuation.jsonl` 应与旧训练摘要里的 `data_sha256` 相同。保留 `development/` 与 `split_manifest.json` 以便入口核验，并将官方 `XES3G5M/questions.jsonl` 上传到同一主机。

```bash
python -u scripts/run_pa_opsd_gold.py opsd \
  --train-jsonl /path/to/continuation.jsonl \
  --seed-manifest /path/to/split_manifest.json \
  --sft-checkpoint /path/to/gold_sft \
  --output-dir runs/opsd_standard_solution \
  --teacher-mode local-same-model-privileged \
  --standard-solution-bank /path/to/XES3G5M/questions.jsonl \
  --load-in-4bit --max-length 32768 --max-completion-length 512 \
  --batch-size 1 --gradient-accumulation-steps 4 --max-steps 35 \
  --learning-rate 0.000001 --seed 42 --gradient-checkpointing
```

加 `--check-data-only` 可校验冻结数据与题库连接。训练会复查实际分词长度。训练摘要记录题库 SHA-256。原训练 140 条全部匹配到有答案和解析的题库记录，其中 43 条当前题的可见提示含图片占位符。[本地审计](/D:/OPSD/outputs/pa_opsd/opsd_repair/local_audit.json)记录了新旧教师在 768-token 输出预算下的输入长度检查。跑正式 35 步前，先用独立输出目录做一次 GPU 更新，检查现有卡的峰值显存和训练损失。

### 1,024-token 训练复测

在独立工作目录放置相同训练代码、官方 LongTutor 校验依赖、`anchor/`、`continuation.jsonl`、`questions.jsonl`、`split_manifest.json`、`development/` 以及上轮按教师输入分词长度选出的 `smoke_input.jsonl`。从工作目录运行 `bash /path/to/train_1024.sh /path/to/env/bin/python`。入口校验续训文件、题库与初始权重，逐步验证 LoRA 权重确实更新。`sources.sha256` 记录当次试跑样本及核心代码。训练侧较上轮补存最后一批生成轨迹，便于用 `audit_rollouts.py` 检查触顶、EOS、JSON 与字段有效性。

首次 1,024 运行因余额中断，旧目录保留日志。新实例从原 Gold SFT 起点重新开始。正式训练使用 `--checkpoint-steps 3`，保留最近两份 Trainer 检查点，其中包括适配器、优化器、进度和随机数状态。项目尚未实测从这些检查点恢复，生成日志的连续性需要在断点恢复时另行核验。

## 训练前校验试跑

下一轮沿用 Gold SFT 起点和原任务提示，启用 `--validate-rollouts`。`run_gated.py` 复用金标训练入口，按旧训练摘要校验续训文件、题库和初始权重。训练包使用固定 140 道续训题及 Gold 训练清单。官方开发评测另行使用原来的 140 道开发题。

在工作区打包。

```text
python scripts/experiments/opsd_repair/run_gated.py prepare --output outputs/pa_opsd/opsd_repair/rollout_gate_train.zip
```

解压后从包的根目录用训练环境的 Python 分阶段运行。GPU 阶段要求缓存的 Qwen 权重可从 Gold SFT 适配器记录的原路径读取。

```text
python scripts/experiments/opsd_repair/run_gated.py check --output runs/rollout_gate/check
python scripts/experiments/opsd_repair/run_gated.py smoke --output runs/rollout_gate/smoke
python scripts/experiments/opsd_repair/run_gated.py formal --output runs/rollout_gate/formal
```

`check` 在 CPU 上核验打包哈希、训练划分和标准解连接。`smoke` 用一个可见 GPU 跑两个梯度累积窗口。确认训练、显存和适配器变化后，从原 Gold SFT 再启动 `formal`。每轮使用独立输出目录，保存 `config.json`、`status.json`、`runtime.json`、训练日志、GPU 采样、逐条轨迹与 `rollout_audit.json`。审计检查每个窗口的初次采样数量，统计首次通过、重试挽回、最终跳过、各类覆盖和实际计入损失的生成 token。

学生原始输出须生成 EOS 并满足五字段 JSON 契约，通过后使用原始 token 计算损失。只重采失败的题，最多一次。整窗没有有效轨迹时，优化器和学习率均跳过。混合窗口沿用 Trainer 的梯度累积除数，无效小批贡献零梯度。`--max-steps` 控制尝试窗口数，`effective_optimizer_steps` 和 `teacher_context.optimizer_steps` 记录实际更新数。比较两臂时同时核对实际更新数、接受样本数和各类覆盖。逐次记录包含样本 ID、初次或重试、失败原因、结束位置和生成正文。旧版 `audit_rollouts.py` 适用于每题只生成一次的旧日志。新版查看 `rollout_validation` 汇总和逐次记录。

本地已用固定 140 条训练样本核对标准解连接，复查旧的 140 条保存轨迹，并用 CPU 上的 Transformers 4.57.1 验证重试后标签、混合窗口以及全空窗口。正式对照要求 35 个实际更新。若有空窗口导致实际步数不足，训练目录与审计保留为部分结果。真实 Qwen 和 GPU 显存仍需在试跑中验证。

## 输出格式提示对照

原 Gold 提示已包含 LongTutor 的任务说明与诊断策略映射。其输出示例在 `memory` 后缺逗号，格式说明也比训练校验宽。新版在核验冻结输入后，只替换系统提示中的输出格式段。新版示例可直接解析，要求一个 JSON 对象、五个字段和按顺序回答 Q1–Q3。缺证据时填 `Unknown`，诊断及策略使用规定的英文名称。学生和教师使用同一段格式指令，标准解仍仅提供给教师。

- 本地生成对照用 `python scripts/run_pa_opsd_gold.py sft generate`，指定同一验证输入、同一适配器和 `--max-new-tokens 1024`。分别生成旧提示与加 `--strict-output-prompt` 的结果，各用独立日志和 `--generation-summary`。这个入口使用贪心解码，统计首次严格通过率、解析失败、字段失败、触顶及诊断表现。
- 当轨迹过滤后可用数据不足时，可在同设定下单独加 `--strict-output-prompt --validate-rollouts` 做对照。新提示记录为 `pa_opsd_strict_json_v1`。试跑关注随机采样首次通过率、重试挽回数和实际更新步数。
- 官方评测继续采用既有提示词和评测配置。格式对照单独登记，保留旧 SFT 和旧 OPSD 的运行口径。

## 官方口径复评

把新适配器取回本地后，在仓库根目录制作复评包。`--opsd-checkpoint` 将标准解版绑定到包内的 `pa_opsd` 模型名，校验训练文件哈希、Gold SFT 初始权重哈希和 35 个实跑更新步。训练摘要和教师配置随包保存。原复评结果可用于对照。

```powershell
$env:PYTHONHASHSEED = '0'
python scripts/experiments/longtutor_reeval/run.py prepare --opsd-checkpoint /path/to/opsd_standard_solution --output outputs/pa_opsd/opsd_repair/standard_solution_reeval.zip
```

解压复评包后，按 [LongTutor 复评说明](../longtutor_reeval/README.md)启动服务和固定 BF16 基座，在同一运行环境中做两题试跑及 140 条正式推理。以下命令只推理并评分新检查点。

```bash
python run.py infer --bundle . --arms pa_opsd --smoke --output smoke
python run.py infer --bundle . --arms pa_opsd --output predictions
python run.py score --bundle . --arms pa_opsd --predictions predictions --output standard_solution_140.json
```

在同一批 Gold 140 上比较诊断准确率、四类 Macro-F1 和空输出数，并从逐题结果核查回忆失败、迁移不足和图题的变化。标准解是否带来收益以验证结果为准。动作级蒸馏另有训练目标，使用这批对照来定位普通 OPSD 的问题。

把新预测和候选包带回本地，运行配对计分。

```text
python scripts/experiments/opsd_repair/score_gold140.py --candidate-bundle /path/to/candidate_bundle --candidate-predictions /path/to/official_predictions --output /path/to/new_score_directory
```

计分采用相同开发题和官方推理设置，复用历史 SFT 续训与旧标准解 OPSD 1,024 的预测。输出逐题比较、四类诊断、策略、ROUGE-L 和官方重试后的最终格式拒绝数。官方记录只保存归一化预测，因此这组记录可比较最终拒绝数，首次严格 JSON 通过率用单独的原始输出审计。
