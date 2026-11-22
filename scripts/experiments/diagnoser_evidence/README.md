# 诊断者历史输入对照

沿用三角色实验的 4 条训练题和 32 条验证题，以及 Gold SFT 检查点、题库标准解和诊断指令。比较三种输入。

| 组 | 原文记录 | 附加统计 |
|---|---|---|
| `full_history` | 199 条历史，复用上一轮诊断者的完整提示 | 无 |
| `full_with_summary` | 同样的 199 条历史 | 从所有历史条目计算的统计 |
| `compact_with_summary` | 最多 12 条，保留原始位置编号 | 同一份全历史统计 |

`evidence.py` 解析当前题和历史中的题号、知识点、时间、正误。统计从完整历史计算，优先展示与当前题重合且在历史中更少见的知识点，最多六个。统计包含作答数、对错数、最近时间和近三次结果。先保留最多八条高匹配记录，再补最多四条最近的相关记录。没有匹配时使用最近四条。按原序号引用，可在历史原文里定位。校验使用原始时间差，防止当前题之后的记录参与统计。

`run.py` 制作带 SHA-256 的样本与特征清单，固定模型和解码参数，保存逐条原始回复、请求时延、输入与输出 token 用量。全历史与全历史加统计的比较用于估计统计的影响。固定统计之后再比较原文压缩的影响。每组也报告诊断准确数、四类 Macro-F1、逐类命中与有效引用。

LongTutor 这批样本的相关历史位置与共享知识点规则一致。重叠次数用来描述检索覆盖与引用分布，证据是否支撑诊断要核对逐条理由和原文。运行使用同一批开发验证题，结果用于选方法。

## 入口

```text
python scripts/experiments/diagnoser_evidence/run.py build --source outputs/pa_opsd/teacher_team_gold32/input --output outputs/pa_opsd/diagnoser_evidence_gold32/input
```

用 [既有 vLLM 启动入口](../longtutor_reeval/README.md) 校验基座和 Gold SFT 适配器，端口为 `8020`。推理机部署此目录、`teacher_team/run.py`、`action_pilot/run.py`、`longtutor_reeval/run.py` 及样本文件。推理读取清单和输入特征，评测用参考文件可以仅留本地。

```text
python scripts/experiments/diagnoser_evidence/run.py infer --bundle /path/to/input/validation --service-bundle /path/to/frozen_service_bundle --output /path/to/validation --port 8020
python scripts/experiments/diagnoser_evidence/run.py score --bundle /path/to/input/validation --output /path/to/validation
```

`decisions.jsonl` 逐条保存三组输出。同目录续跑会检查输入、源码、适配器和服务清单。`score.json` 汇总各组数据与逐题配对，`review.jsonl` 可用于复核遗漏、引用错误和诊断变化。代码和结果的长期入口见 [实验留痕](../../../docs/实验留痕.md)。

## 新 24 道开发题复验

从冻结的 140 道 Gold 开发题按四类诊断各取 6 道，先按样本 ID 排除前一轮的 32 道，再沿用原规则优先取文本可读且有标准解的题。同一题在三组中使用相同的诊断提示、标准解和 Gold SFT。本地保存参考诊断、策略和反馈，推理包只包含历史、当前题、标准解和历史统计。候选中保留图题，清单标明图像占位数量，逐题解读时可按此核对。

```text
python scripts/experiments/diagnoser_evidence/build_followup.py --previous outputs/pa_opsd/diagnoser_evidence_gold32/input/validation --output outputs/pa_opsd/diagnoser_evidence_gold24/input/validation
```

入口需要一个新输出目录。`outputs/pa_opsd/diagnoser_evidence_gold24/` 下保存固定清单、各样本三组提示的本地分词长度和模型推理前的验证。把 `inference_bundle.zip` 解压到推理机，使用已经核验的 Gold 140 服务包启动相同的 Qwen2.5-7B-Instruct 与 `sft` 适配器。三组共 72 个本地 vLLM 请求。

```text
python scripts/experiments/diagnoser_evidence/run.py infer --bundle input/validation --service-bundle /path/to/frozen_gold140_service_bundle --output validation --port 8020
```

模型运行后下载 `validation/` 到一个新的本地实验目录，将其作为评分命令的 `--output`，`--bundle` 使用本地带参考答案的 `outputs/pa_opsd/diagnoser_evidence_gold24/input/validation`。
