# LongTutor 诊断原因排查

用原始 Qwen2.5-7B-Instruct 和 420 道 Gold 普通 SFT 的已存检查点对比，检查候选标签排序和显式历史统计的作用。每组使用固定 stage2 140 道选先验校正强度，固定开发 140 道看结果。烟测从 stage2 取 16 道，每类 4 道。实验指标为四类诊断的离线分析，完整 LongTutor 评测仍使用固定上游的原始流程。

## 本地准备和检验

```text
python -m scripts.experiments.diagnosis_probe.offline prepare --output <新的本地准备目录>
python -m scripts.experiments.diagnosis_probe.offline statistics --bundle <本地准备目录> --output <新的统计基线报告目录>
```

`prepare` 核验已冻结的输入、训练收据和适配器，将原始历史对到逐题输入。统计按当前题与历史记录的共有知识点计算。没有有效分母时正确率记为空，找不到答对记录时时间差也记为空。CPU 逻辑回归需要 scikit-learn。缺失值填充、标准化和逻辑回归只在 stage2 拟合，固定在开发集评分。

准备目录下的 `autodl.zip` 含推理代码、无标签输入和已校验的 LoRA。`references.jsonl` 中的金标留在本地。`local_manifest.json` 记录固定输入、历史来源和训练收据哈希。

## AutoDL 推理

一张 24 GB 或更大显存的卡即可串行完成两组。分机运行时优先克隆同一份带模型缓存的镜像，保证 PyTorch、Transformers 和 PEFT 版本一致。已训练 SFT 的环境为 Transformers 4.57.1、PEFT 0.17.1、PyTorch 2.7.0+cu128。解压同一包，基座用 `base`，现有 SFT 用 `sft`。`MODEL` 指向原来评测用的固定 Qwen2.5-7B-Instruct BF16 权重目录。

```bash
ARM=base
MODEL=/root/autodl-tmp/longtutor_gold_ba40af3/qwen_bf16
python run.py --bundle . --model-dir "$MODEL" --arm "$ARM" --phase smoke --output "runs/$ARM/smoke"
```

先检查两组的 16 道是否能逐题对齐，以及原输入、加统计两种条件是否稳定到达诊断位置并产出四个有限分数。失败的逐题记录也保留。如果大批未到达诊断字段，先看是字段顺序还是生成上限，再决定是否统一修改运行设置。试跑通过后，分别给 `ARM` 赋值为 `base` 和 `sft`，执行以下命令。

```bash
python run.py --bundle . --model-dir "$MODEL" --arm "$ARM" --phase full --output "runs/$ARM/full"
```

全量每个模型做 140 道 stage2 原输入，以及 140 道开发题的原输入、加统计输入，共 420 次前缀生成。原输入是项目固定的 memory 优先提示。同一前缀下累加四个完整标签各自的 token 对数概率，计分到英文标签最后一个词。逐题保存生成原文、诊断前缀、候选分数、原输入诊断位置的最后层向量和运行状态。生成没到诊断位置的题留在分母。断点续跑沿用原目录和运行设置，已写完的题直接跳过。

## 本地对照

将两组的 `settings.json` 和 `predictions.jsonl` 分别拉回本地。检查两组除检查点外的生成配置一致，再做配对汇总。

```text
python -m scripts.experiments.diagnosis_probe.offline compare --bundle <本地准备目录> --base <基座结果目录> --sft <SFT结果目录> --output <新报告目录>
python -m scripts.experiments.diagnosis_probe.offline linear-probe --bundle <本地准备目录> --inference <某一组全量结果目录> --output <新探针报告目录>
```

`compare` 对每个模型用 420 道 SFT 的类别频率作先验，从 stage2 选 `τ`，并输出开发集原始标签排名、相对第一名的分差、四类 Macro-F1、混淆矩阵及每类得失题数。比较原输入与加统计输入时两组都用 `τ=0`。`linear-probe` 直接用已经保存的向量在本地拟合，可以等首轮结果再决定。人工金标证据对照留作后续逐题排查。

## diagnosis 字段加权与金标 memory 对照

`baseline.py prepare` 复用 2026-11-25 普通 SFT 的预测和固定 Gold 划分，生成 `training.zip`、`evaluation.zip`、运行协议与本地审计。可用原模型保存的 tokenizer 预检。两个包解压到同一个新目录。

```text
python -m scripts.experiments.diagnosis_probe.baseline prepare --tokenizer outputs/pa_opsd/gold_evidence_first/opd_variants/anchor --output <新实验目录>
bash scripts/experiments/diagnosis_probe/run_baseline_gpu.sh <解压目录> <已核验基座目录> <训练环境Python> <vLLM环境Python> <新GPU结果目录>
```

- 金标对照冻结旧 420 条 SFT。把旧模型实际生成的 `memory` 和金标 `memory` 渲染成同一种 JSON 前缀，在原公开提示下分别打分四个英文标签。只读取 `memory` 的 `id` 和 `answer`，避免 `reason` 直接提示诊断。主结论来自两组前缀都可用的同题配对。另报告与旧原生前缀的差异，以及 140 题中补齐格式后的结果。若准确 `memory` 没有改善，仍可能是这些历史答案不足以区分错因。
- 加权 SFT 从原 7B 基座训练，用原 420 条、105 步、种子 42 和学习率 `2e-5`。仅诊断标签的交叉熵乘以 `8 × 归一化逆平方根类别权重`。8 在跑开发集前固定，用来提高这一小段标签的损失权重。损失分母仍为所有监督 token 数，其他 token 系数仍为 1。训练会保存短批验证、配置、日志、适配器和有效步数。训练失败时保留原目录，恢复前检查日志和已写入的检查点。
- 加权权重会用与旧 SFT 相同的四选标签前缀打分，并用固定 LongTutor `ba40af3` 原脚本、vLLM 0.9.2 生成 140 条完整教学预测。所有小批和完整输出分别保存。全量正式教学评分使用已核准的同版 DK 裁判，在本地单独执行。GPU 入口只调用本机模型服务。

把 `oracle_full`、`weighted_full`、`train_full` 和正式推理预测与日志回传，再分别对两组运行下列离线计分命令。`--bundle` 使用本地包含 `controls.jsonl` 的原准备目录。

```text
python -m scripts.experiments.diagnosis_probe.baseline score --bundle <本地实验目录> --source <回传的oracle_full或weighted_full> --output <新汇总目录>
```

`score` 写出逐题四标签分数和金标排名、四类 Macro-F1、混淆矩阵和各类得失题数。这些数字只用于诊断排查。封存测试集不参与此次选择。

`baseline_summary.py` 接受两份 `paired_by_id.jsonl` 和新模型的原始 LongTutor 生成结果，按官方输入的当前题号做 10,000 次配对重采样。离线诊断统一保留空预测，报告四类 Macro-F1、准确率与区间，并单独列出正式生成的空诊断数。教学质量仍由单独的官方裁判流程评估。

```text
python -m scripts.experiments.diagnosis_probe.baseline_summary --oracle <金标记忆逐题> --weighted <加权SFT逐题> --generated <official_140.jsonl> --output <新对照目录>
```

11 月 28 日 38789 实例回传验收后，在项目根目录运行 `python -m scripts.experiments.diagnosis_probe.baseline_judge`。该入口核对本轮权重的 140 条原始预测、固定上游提交和存档普通 SFT 分数，随后用原始 LongTutor 评分程序评新模型。裁判沿用旧运行的 `deepseek-flash`、温度 0、4 workers 和供应商缺省思考设置。接入层把上游的生成上限传给 DeepSeek 的 `max_tokens`。结果目录保留逐题评分、配对区间、逐请求用量和 2 元预算的运行账本。
