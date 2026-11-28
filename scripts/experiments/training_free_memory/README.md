# LongTutor 学生历史 RAG 与教学技能卡

## 从完整回复读懂本次实验

先读 [端到端结果和逐题证据](../../../outputs/pa_opsd/memory_budget_paper/gpu_run1/结果与复现.md)。主方法只有两个步骤。先读 [evidence_executor.py](evidence_executor.py) 的 `execute`，了解公开问题如何选取作答记录、返回答案与来源位置。再读 [answer_with_evidence.py](answer_with_evidence.py) 的 `compose`，看哪些字段会被程序填写，模型的诊断和教学回复如何保留。两个文件均可脱离裁判单独运行。

| 要复现的比较 | 入口 | 归档位置（相对 gpu_run1） |
|---|---|---|
| 140 例对强时序检索 | [adaptive_memory.py](adaptive_memory.py) | `adaptive/`、`system/adaptive/` |
| 全部 560 例对强时序检索 | [transfer_temporal.py](transfer_temporal.py) | `transfer_temporal/system/`、`transfer_temporal/review/` |
| 512 token 下的证据包与错误值干预 | [package_budget.py](package_budget.py)、[packet_followup.py](packet_followup.py) | `results/runs/`、`followup/`、`analysis/` |
| 数值、事件表、数值加表 | [trace_memory.py](trace_memory.py) | `trace/`、`results/trace/` |
| 训练池全史与标注范围差异 | [scope_audit.py](scope_audit.py) | `scope/` |

在项目根目录使用以下命令可从已归档的真实预测和判分重新计算主表，无需 GPU 或裁判调用。

```powershell
python -m scripts.experiments.training_free_memory.adaptive_memory aggregate
python -m scripts.experiments.training_free_memory.transfer_temporal aggregate
python -m scripts.experiments.training_free_memory.trace_memory --stage aggregate
python -m scripts.experiments.training_free_memory.packet_results aggregate
python -m scripts.experiments.training_free_memory.adaptive_memory costs
```

从头复跑时，先在新的运行目录准备输入。相邻运行依赖按 `gpu_run1` 的目录关系放置，冻结输入包内列出了逐文件指纹。GPU 运行包的实际 bash 入口及配置保存在 `gpu_run1/results/`。基座和推理服务参数见对应目录的 `run.tsv`、`models.json`、`version.json`，判分与语义盲评的口径见结果索引。

## 论文主实验从哪里读

先打开 [最终结果与证据](../../../outputs/pa_opsd/memory_budget_paper/实验结果.md)。只想理解方法，先读 [evidence_executor.py](evidence_executor.py) 的 `execute`：输入是题问和原始作答，输出是运算、答案及作答位置。紧接着的 `predict` 比较长短历史并回退到已有模型。再读 [package_budget.py](package_budget.py) 的 `execution_views`，看如何在 512-token 上限内把计算结果放回导师输入。检索强基线在 [budget_curve.py](budget_curve.py)；计分和互斥池复核分别在 [score_executor.py](score_executor.py) 和 [transfer_executor.py](transfer_executor.py)。各脚本从项目根目录运行。

以下为本地复跑示例，结果使用新目录，tokenizer 使用固定版 Qwen2.5-7B。原始输入和旧预测的位置已固化在脚本顶部，可逐个检查。命令依次生成 2,800 份排序选择、280 条程序问答、盲评输入、560 样本复核和 GPU 包。匿名复核的实际判定保存在最终结果目录。

```powershell
$runRoot = 'outputs/pa_opsd/memory_budget_paper/replay'
python -m scripts.experiments.training_free_memory.budget_curve --output "$runRoot/sweep"
python -m scripts.experiments.training_free_memory.evidence_executor --output "$runRoot/predictions.jsonl"
python -m scripts.experiments.training_free_memory.score_executor --predictions "$runRoot/predictions.jsonl" --output "$runRoot/evaluation"
python -m scripts.experiments.training_free_memory.transfer_executor --output "$runRoot/transfer"
python -m scripts.experiments.training_free_memory.package_budget --run "$runRoot/sweep" --packet-tokenizer .codex_tmp/opd_qwen_teacher_tokenizer/tokenizer.json --output "$runRoot/gpu"
```

GPU 包解压后，使用下文服务环境启动 `python service.py serve --model /path/to/Qwen2.5-7B-Instruct`。服务就绪后依次执行 `bash run_inference.sh smoke Qwen2.5-7B-Instruct runs/smoke` 与 `bash run_inference.sh full Qwen2.5-7B-Instruct runs/full`。已有输入与校验和，不含金标答复。此五组尚未实际生成。

在固定 Gold 开发集 140 道上测试免训练增强。每组使用同一模型和解码参数。当前实现历史检索、配对输入构建和官方生成的 AutoDL 入口。教学收益需要真实推理与 DK 评分。

| 组 | 模型收到的历史输入 |
| --- | --- |
| `rag` | 复用知识点与近邻检索，最多 12 条原始作答 |
| `temporal_rag` | 优先保留最近的相关正确、错误记录和证据问题显式点名的记录，再用原检索补齐，并附上从完整历史计算的数值统计 |
| `temporal_rag_card` | 共用 `temporal_rag` 的全部输入，只对有可见支持证据的题附加一条候选教学动作和停用条件 |

被证据题点名的题号可能出现多次，保留全部匹配记录。必须保留的记录超过 12 条时回退完整历史。标准解、Gold 诊断、答案、策略和反馈均留在本地评分阶段。候选卡复用 [教学技能卡](../teaching_skills/README.md) 的回忆与类比卡，原卡中的诊断标签和策略标签留在代码里。没有触发的题两组输入相同。日期范围和隐式计数题的全部原文覆盖仍待验证。

## 本地准备

从项目根目录运行，使用 LongTutor 基座的固定 tokenizer 和 Python 环境中的 `transformers`、`openai`、`tqdm`。

```powershell
$env:PYTHONHASHSEED = '0'
python -m scripts.experiments.training_free_memory.prepare --tokenizer .codex_tmp/opd_qwen_teacher_tokenizer --output outputs/pa_opsd/training_free_memory/gold140_run
```

准备入口默认读取已核验的历史和只含题问的输入，逐条核对公开题问的原文，验证 140 个开发 ID、上游提交、模型 tokenizer、逐题键及上下文预算。结果目录有 `manifest.json`、`routing.jsonl`、`inference.zip` 以及全量和两道试跑的各组输入。两道试跑包含最长的已触发题和另一道最长题。推理包只携带启动代码、原始评测代码、可见数据和运行配置。

## AutoDL 生成

需要 GPU 时，把 `inference.zip` 解压到新目录。在现有 LongTutor BF16 环境下运行，替换本机已有的权重路径。启动入口校验 Qwen2.5-7B-Instruct 权重、tokenizer 和 vLLM 0.9.2，使用 BF16 和 32,768 上下文。

```bash
python service.py serve --model /root/autodl-tmp/longtutor_gold_ba40af3/qwen_bf16
```

模型服务就绪后，在另一个终端启用同一个环境并执行试跑。

```bash
bash run_inference.sh smoke Qwen2.5-7B-Instruct runs/base_smoke
bash run_inference.sh full Qwen2.5-7B-Instruct runs/base_full
```

先检查试跑日志、显存、输出长度和三组输出，再执行全量。比较现有 SFT 时，启动服务加 `--adapter <已保存的 memory_first 适配器目录>`，生成时把模型名换成 `memory_first`，使用独立的结果目录。改用其他端口时同时设置服务的 `--port` 和生成的 `VLLM_PORT`。已有完整历史预测可直接复用。每个新目录保存模型清单、参数、原始逐题输出和日志。

## 本地评分

取回 GPU 生成目录，使用原始 LongTutor `ba40af3` 的官方评分脚本。以 `temporal_rag` 为例，下列命令在推理包解压目录执行。提前在本地配置同版 DK 的环境变量和模型 ID。

```text
python upstream/compute_ai_tutor_eval_metrics.py --pred runs/base_full/temporal_rag.jsonl --gold PATH_TO_FROZEN_BUNDLE/gold.jsonl --out runs/base_full/temporal_rag.metrics.jsonl --model DK_MODEL --temperature 0 --workers 4 --limit 140
```

每组使用相同的完整 Gold 文件、DK 模型和解码参数，保存逐条评分与官方汇总日志。全量对照报告诊断 Macro-F1、四类召回、Q1–Q3 和教学四项。可按本地路由清单复核技能卡触发题的配对变化。普通 RAG 与时序组同时改变了选取策略和统计摘要，估计的是组合效果。时序组加卡对照固定这两项。

## 第二轮时序证据、状态与消融

第二轮复用首轮的固定开发 140 道及两个检索版本，核对公开题问、原始样本键、模型和上游脚本。`rag`、`rag_stats`、`temporal`、`temporal_rag` 比较记录选择和同一个全历史数值摘要。以下两组沿用 `temporal_rag` 的历史和摘要。

| 组 | 加入的可见历史事实 |
| --- | --- |
| `event_ledger` | Q2 点名题号、完整日期或带引号已知知识点匹配的候选事实。按时间保留原历史位置、题号、时间、正误和标签，保留重做记录。候选相并，把交集筛选留给模型。超过 24 行或提示超过 4,096 tokens 时回退 `temporal_rag`。 |
| `learner_state` | 至多六个当前题知识点的总次数、对错次数和最近三次原始事件位置。提示超过 4,096 tokens 时回退 `temporal_rag`。 |

本地用同一固定 tokenizer 准备六组正式输入、最长两道试跑和 16 道同输入重复试验。`routing.jsonl` 保留逐题候选位置、是否加索引或摘要、提示长度。`manifest.json` 保存固定 IDs、调用参数、覆盖与文件校验和。

```powershell
$env:PYTHONHASHSEED = '0'
python -m scripts.experiments.training_free_memory.prepare_followup --tokenizer .codex_tmp/opd_qwen_teacher_tokenizer --output outputs/pa_opsd/training_free_memory/followup_run
```

解压生成的 `inference.zip`，使用本 README 的同一 `service.py` 命令启动基座服务。在 AutoDL 依次运行下列命令，每次使用新结果目录。

```bash
bash run_inference.sh smoke Qwen2.5-7B-Instruct runs/followup_smoke
bash run_inference.sh repeat Qwen2.5-7B-Instruct runs/followup_repeat
bash run_inference.sh full Qwen2.5-7B-Instruct runs/followup_full
```

默认使用首轮的 4 workers。原上游脚本在 1 worker 模式每条等待 40 秒，串行仅适用于另行安排的重复性排查。`repeat` 两臂输入字节相同，检查时按样本键比较输出，考虑官方并发写入顺序。先检查试跑和重复性的差异规模，再运行全量。11 月 26 日已归档[真实模型预测和离线评分](../../../outputs/pa_opsd/training_free_memory/followup_gold140_20261125_verified/autodl_4090_49862_20261126/结果摘要.md)，语义评分待运行。整体评测顺序和论文接入见[免训练记忆实验与论文方案](../../../docs/免训练记忆实验与论文方案.md)。

## 第三轮教学卡先导

使用首轮已经匹配到教学卡、且卡片支撑记录全部保留的 22 道开发题。其中类比 15 道、回忆 7 道。三组共用第二轮纯时序取样的原始历史、当前题和公开证据问题。`no_card` 使用共同输入，`matched_card` 加入路由匹配的卡，`swapped_card` 加入另一张卡。两张卡在同一提示位置使用同一模板，正文有 20 token 长度差。卡正文含适用条件、动作建议和停用条件，Gold 标签、标准解和参考反馈留在本地。

```powershell
$env:PYTHONHASHSEED = '0'
python -m scripts.experiments.training_free_memory.prepare_cards --tokenizer .codex_tmp/opd_qwen_teacher_tokenizer --output outputs/pa_opsd/training_free_memory/card_pilot_run
```

沿用 AutoDL 生成入口，先执行 `smoke` 两道每组的试跑，再执行 `full` 三组各 22 道。使用 `python -m scripts.experiments.training_free_memory.review_cards --prepared <准备目录> --destination <本地备份目录> --sha256 <远程备份校验和>` 核对归档及逐题输出。发给评审的是 `blind_review.jsonl`，它只含题面、共用历史和 A/B/C 匿名的教学回复。对应关系保存于 `internal/blind_key.jsonl`，在评价冻结后再读。该先导以实际首步动作适用性和全文教学风险为主要观察项，使用[动作盲审规程](../../../outputs/pa_opsd/training_free_memory/card_pilot_20261126/动作盲审规程.md)。


## 开题用的两臂裁判和同题统计

基座两臂复用 11 月 26 日的普通检索和纯时序检索预测，限定冻结开发集 140 道。两组使用同一版本的 LongTutor 官方脚本和 deepseek-flash 非思考模式，接口接入保留请求账本。此次授权上限每臂 2.5 元，合计 5 元，对应唯一输出目录 `outputs/pa_opsd/proposal_evidence_20261127/dk`。从项目根目录运行 `python -m scripts.experiments.training_free_memory.judge_retrieval ARM OUTPUT`，ARM 取 rag 或 temporal。已完成样本由上游程序按键复用。未结算请求继续占用其预算预留。日志记录估算金额和保守上界。

同批新顺序 SFT 只更换已核验的模型适配器。GPU 运行包和配置保存在这次的结果目录；与基座共用官方提示、逐题输入、温度 0、四并发和 3,000 token 输出上限。SFT 新生成的两臂使用本地精确匹配。统计命令 `python -m scripts.experiments.training_free_memory.paired_report --sft SFT_FULL_PREDICTION_DIR` 会将四组预测逐题对齐，输出 10,000 次按当前题号整簇重抽样的区间，并在同一模型内区分普通检索和时序检索。Q1 的题号覆盖分析只核对题问里明确标出的方括号数字和输入原文的题号，此项属于解释性分析。
