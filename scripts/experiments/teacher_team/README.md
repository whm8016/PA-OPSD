# 教师团队与单教师对照

## 证据核验与盲复核版

第三版先用程序从当前题前的作答记录构造最多 12 张历史卡。题号和原历史位置仅留作追溯，模型引用 `H01` 等卡片编号。相同题号、时间和正误的重复记录共用一张卡，同题同一时刻的冲突记录暂不引用。卡片上的正误、时间间隔和共有知识点由程序计算。

| 流程 | 推理顺序 | 每题调用 |
| --- | --- | ---: |
| 单教师 | 一次给出首选、备选、引用和反馈 | 1 |
| 教师团队 | 诊断 → 校对引用和正误时间 → 必要时盲复核 → 生成一步反馈 | 2–3 |

两组使用同一个已有 Gold SFT、同样的证据卡、标准解和教学动作表。两组均只能看到当前作答正误。诊断模型要陈述共同方法或题型差异，并解释首选与备选的区别。程序先核对引用存在、正误及时间一致，再筛查四类诊断所需的最低支持。较早成功与较近成功以 4 天划分，作为本轮待测的时间规则。只靠部分标签匹配、同一证据也符合另一类诊断、引用不合法、首选依据不足时启动复核。复核只看输入，不看首轮决定。两次判断仍依赖部分匹配或有类别歧义时，候选需同标签、同关系并共享足够的证据卡。缺图时跳过复核，优先向学生索取题图。

无足够依据时把诊断留作待确认，老师只给一个区分错因的问题或可执行的子目标。程序筛查常见直报答案和声称看到未提供的错误步骤。机器通过仅说明达到预检条件，事实之外的题型语义、数学正确性、提示价值和间接泄题需要人工核对。`review.jsonl` 提供判断、证据、反馈和参考。核对时优先盲看同题的两条反馈，分别标记证据是否支持、提示是否可执行、数学是否正确、是否直接或间接泄题，再揭晓组别。通过预检的输出暂不作为蒸馏数据。

- `regression32` 保留上一版同一批开发题。四道已知失败案例在前，用于检查卡片编号、小时与天的换算、以及重复或相互矛盾的历史引用。
- `holdout24` 使用另外 24 条固定开发样本，学生与回归 32 道不重合，当前题号有部分重合。24 道里 8 道当前题缺图，包括全部 6 道迁移不足。缺图题可用于验证澄清策略，迁移诊断增益需有相应题图或额外可读题。
- 本地预检分开记录原始首选、双方共用的事实与最低证据筛查、各组最终采纳及覆盖率、按四类和题图可见性划分的结果、调用数和 token。团队的额外复核机会影响采纳率，应结合共用筛查与人工配对复核看结果。正式 LongTutor 指标沿用原始评测脚本。

### 本地准备与 AutoDL 试跑

```text
python -m scripts.experiments.teacher_team.grounded build --cohort regression32 --source outputs/pa_opsd/teacher_team_gold32/input/validation --output outputs/pa_opsd/teacher_team_grounded_20261125/input/regression32
python -m scripts.experiments.teacher_team.grounded build --cohort holdout24 --source outputs/pa_opsd/diagnoser_evidence_gold24/input/validation --output outputs/pa_opsd/teacher_team_grounded_20261125/input/holdout24
```

使用 [上一轮 vLLM 服务设置](#运行)，解压新版 `inference_bundle.zip`，在解压目录运行。推理包只带输入与清单，逐题金标留在本地。先跑前四道回归题，合计 12–16 次生成。确认引用与反馈质量后，才去掉 `--batch` 继续同一输出目录。需要核验新开发样本时，切换到 `input/holdout24` 和新的输出目录。

```bash
python -m scripts.experiments.teacher_team.grounded infer --bundle input/regression32 --service-bundle "$GOLD140_SERVICE" --port 8020 --output predictions/regression32 --batch 4
```

取回 `settings.json` 和 `decisions.jsonl` 后，本地评分。完整批次去掉 `--partial`，报告会标出实际分母。AutoDL 4090D 首批四道已完成，两组均未通过证据核验。[逐题复核](../../../outputs/pa_opsd/teacher_team_grounded_20261125/autodl_4090d_22532/逐题复核.md)

```text
python -m scripts.experiments.teacher_team.grounded score --bundle outputs/pa_opsd/teacher_team_grounded_20261125/input/regression32 --output outputs/pa_opsd/teacher_team_grounded_20261125/predictions/regression32 --partial
```

## 共享证据版

这一步检验教师的诊断、证据和教学建议，供后续决定是否用于蒸馏。从固定开发集复用历史试验的同一批 32 题，每类诊断 8 题，其中 1 题有图片占位符。两组用同一个既有 Gold SFT 模型、同一题和同一份历史统计、历史原文、标准解、诊断定义和动作规则。金标诊断、相关历史和参考反馈存于本地 `references.jsonl`。选历史时只用当前题和真实作答记录，引用保留原序号。

| 组别 | 流程 | 每题调用 |
| --- | --- | ---: |
| 单教师基线 | 一次给证据、诊断、动作和提示，并自查 | 1 |
| 教师团队 | Diagnoser 给证据和诊断 → Pedagogue 给动作和提示并可提出异议 → Verifier 查原记录和标准解、核对证据和策略、检查数学及泄题，输出最终决定和修订依据 | 3 |

两组采用相同的温度和单次生成上限。团队多两次调用，汇总记录两组各自累积的输入、输出 token 和调用时间。比较用完整的配对样本作分母，记录诊断命中和四类 Macro-F1、相关历史命中、动作候选命中，以及验证者修订前后的诊断变化。动作候选来自 LongTutor 策略对应的可能动作。反馈是否贴题、数学是否正确、是否过早泄题，可以用 `review.jsonl` 的原始引用、标准解和输出逐题核对。验证者自报的检查状态留在逐题记录，正式 LongTutor 指标仍由原官方脚本出分。

### 运行

本地构造冻结输入，可直接复用已生成的 `outputs/pa_opsd/teacher_team_shared_evidence_20261125/input/`。

```text
python -m scripts.experiments.teacher_team.compare build --source outputs/pa_opsd/teacher_team_gold32/input/validation --output outputs/pa_opsd/teacher_team_shared_evidence_20261125/input
```

AutoDL 解压实验目录下的 `inference_bundle.zip`，复用原 [vLLM 0.9.2 服务包](../longtutor_reeval/README.md#一台-5090-上复评)，指定同一缓存 BF16 基座和 `sft` 适配器。以下命令中的变量分别指已解压服务包的路径与已核验的基座目录。

```bash
python "$GOLD140_SERVICE/run.py" serve --bundle "$GOLD140_SERVICE" --model "$QWEN_MODEL" --port 8020
```

在另一终端，从 `inference_bundle.zip` 解压目录运行。先做 4 题，每类 1 题，共 16 次调用。其余 28 题在核对试跑后续跑，同一输出目录无需重跑前 4 题。

```bash
python -m scripts.experiments.teacher_team.compare infer --bundle input --service-bundle "$GOLD140_SERVICE" --port 8020 --output predictions --batch 4
python -m scripts.experiments.teacher_team.compare infer --bundle input --service-bundle "$GOLD140_SERVICE" --port 8020 --output predictions
```

取回 `predictions/`，本地计分。若只取回了前 4 题，显式加 `--partial` 可以检查小样本；报告记录实际分母。评分产物为 `score.json` 和 `review.jsonl`，原文和停止原因保存在 `decisions.jsonl`。输入、检查点、服务端口及代码校验值保存在 `settings.json`。

```text
python -m scripts.experiments.teacher_team.compare score --bundle outputs/pa_opsd/teacher_team_shared_evidence_20261125/input --output outputs/pa_opsd/teacher_team_shared_evidence_20261125/predictions
```

## 历史全量作答记录版

在同一个 Gold SFT 模型上比较一次独立决策与诊断者、教学策略者、验证者的三次串联决策。两组共用原有历史、当前题及作答结果、题库标准解和简短教学量规。验证者能修订前两步，并给出最终诊断、动作和反馈。环境复用固定 Qwen2.5-7B-Instruct BF16 基座、Gold SFT LoRA 和 vLLM。实验衡量教师决策，作为动作级 OPSD 的开工依据。

## 数据和指标

- 训练集 4 条用于工程试跑，验证集按四类诊断固定取 32 条。按样本 ID 哈希排序，优先选择当前题的文本记录。迁移不足的文本题少于 8 条，所以验证集有 1 条当前题保留图片占位符。
- `inputs.jsonl` 包含原始历史、带作答正误的当前题和题库标准解。`references.jsonl` 保存 Gold 诊断、策略、反馈和 LongTutor 发布数据里匹配到的相关历史序号。推理只读取 `inputs.jsonl` 和校验清单。
- 逐阶段报告准确率所需的正确数、四类 Macro-F1、逐类命中、有效历史序号及与发布相关历史的重叠。历史重叠衡量引用是否碰到发布记录中的相关条目。
- 参考策略对应的候选动作只衡量与宽泛策略的一致性。`review.jsonl` 将每条单导师和团队决定、选中的历史原文、参考反馈并列，以便复核动作与泄题。每个角色的生成预算和生成量均有记录，团队有三次判断机会。
- 32 个样本中有 27 个不同的当前题 ID，题目重复时来自不同学生。比较以学生样本为单位，作为小批可行性结果。

## 运行

先生成固定样本。

```text
python scripts/experiments/teacher_team/run.py build --output outputs/pa_opsd/teacher_team_gold32/input
```

使用 [LongTutor 复评服务](../longtutor_reeval/README.md) 的固定基座和 Gold SFT 适配器，以 `--port 8020` 启动服务。把本目录 `run.py`、`action_pilot/run.py`、`longtutor_reeval/run.py` 和 `input` 下两个分组的 `inputs.jsonl`、`manifest.json` 按原相对路径部署。评分文件可以仅留在本地。

```text
python scripts/experiments/teacher_team/run.py infer --bundle outputs/pa_opsd/teacher_team_gold32/input/train --output outputs/pa_opsd/teacher_team_gold32/train_smoke --port 8020
python scripts/experiments/teacher_team/run.py infer --bundle outputs/pa_opsd/teacher_team_gold32/input/validation --output outputs/pa_opsd/teacher_team_gold32/validation --port 8020
python scripts/experiments/teacher_team/run.py score --bundle outputs/pa_opsd/teacher_team_gold32/input/validation --output outputs/pa_opsd/teacher_team_gold32/validation
```

推理核对 `sft` 适配器文件并逐条保存每阶段响应。使用相同模型服务和配置重跑同一输出目录时，只请求缺失响应。每轮按阶段并发独立题，各阶段完成后才进入下一阶段。服务启动入口会校验基座、tokenizer 和生成配置。评分依赖 `settings.json` 和 `decisions.jsonl`，参照本地固定标签生成 `score.json` 与 `review.jsonl`。

原验证 32 条中，单导师有 1 条达到 256 tokens，三角色各阶段均无触顶。若复测，可沿用原输入目录，在两条 `infer` 命令上加 `--max-new-tokens 1024` 并指定各自的新输出目录，使单导师和团队各阶段采用相同上限。`settings.json` 记录实际预算，评分仍逐阶段计数 `finish_reason=length`。
