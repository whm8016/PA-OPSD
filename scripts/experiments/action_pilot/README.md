# Gold 64 教学动作与教师对照

用同一个 Gold SFT 检查点，在同一批 64 条训练样本上比较学生可见输入和附加教师依据两种条件。记录四类诊断的改变、少数类答对数和输出动作与参考动作的匹配。两组均用相同的诊断和动作定义、温度 0 与输出预算。

## 数据来源

- 四种诊断各 16 条，来自固定 560 条训练集。当前题有 XES3G5M 标准答案与解析，排除当前题显式引用图片的记录。固定哈希排序优先覆盖不同当前题，最终有 64 位学生、62 个当前题 ID。两个重复题 ID 来自迁移不足类，学生各不相同。
- 两个条件共用当前题和完整可见历史。原始记录只有正误时，可见输入保留该信息。每题的教师条件还加入题库标准解、通用教学量规和至多五个原历史中的相关记录位置。Eedi 误区候选及已验证技能本轮均留空。
- 金标诊断、教学策略、参考反馈、动作标注和复核状态保存在单独的评分文件。动作初标按参考反馈的实质教学动作填写。遇到复合动作，优先标记推进学生下一步的首个实质动作，并保留原文和需要复核的地方。
- LongTutor 当前题取自作答错误的场景。64 条按诊断均衡，动作覆盖服从原始反馈。历史沿用公开文本，其中保留原始图片占位符。动作草稿用于查看初步匹配，有歧义或过度提示的行先放在待复核状态，复核后才进入已复核动作统计。
- 这批题曾用于 Gold SFT 初训。学生条件与教师条件的差值用于查整套补充依据有无信号。各字段的独立贡献需要再做对照，泛化结果要在独立验证数据上测。

## Gold 策略与教学动作

LongTutor 的 `strategy` 描述教学意图。八类动作描述这条反馈具体做了什么。先用官方策略查候选，再读参考反馈划出第一个实质教学动作。诊断只用于抽样分层和诊断评分。官方发布脚本把四类诊断一一对应到四种策略，因此每个策略仍需结合反馈才能区分具体动作。

| Gold 策略 | 候选 | 原文如何判断 |
|---|---|---|
| Retrieval Practice（提取练习） | C、D | 请学生回忆是 C。反馈直接交代规则是 D。 |
| Conceptual Explanation（概念解释） | D、F | 直接解释概念是 D。用独立例题或情境教学是 F。 |
| Stepwise Scaffolding（分步支架） | E、G、H | 修正可核实的错误步骤是 E。指定中间目标是 G。给出当前题的实际推导是 H。 |
| Analogical Transfer（类比迁移） | F、C、D、G | 对比另一个实例是 F。让学生找对应关系、说明共同原理、或先做一个局部类比目标分别可能是 C、D、G。 |

候选表用于审阅。一条 Gold 策略允许出现多个行动，也允许复核后保留候选外的实际动作。A、B 依赖学生当前状态和反馈原文，当前的错题抽样没有得到 A、B 初标。

打开生成的 `动作标注初稿.md`，先看表后附的待复核案例。根据当前题、作答记录和原文判断提示是否给出了实际步骤、是否确实能定位学生的错误、是否已经给完答案。确认后，在标注源文件中按案例的样本 ID 更新 `action`、`quote`、`status`。支持保留原动作或改标，确认可用时填 `reviewed`。如果反馈已经泄题，或原始记录无法支撑该动作，保持 `review_required` 并记录原因。当前脚本会将有歧义或初标超出策略候选的行暂时排除，已复核动作单独计分。

## 八类教学动作

| 输出 | 动作 | 如何区分 |
|---|---|---|
| A | 不干预 | 暂时观察或等待，保留学生当前尝试。 |
| B | 确认并推进 | 确认已展示的正确思路，请学生继续。重复寒暄或肯定历史表现的开场语在后续动作中计。 |
| C | 追问 | 让学生自己给出关键判断、回忆规则或解释。 |
| D | 概念提示 | 直接提供可迁移的概念、关系或法则。 |
| E | 程序性修正 | 指出可定位的错误步骤，并给出针对该步的纠正方向。 |
| F | 举例 | 用另一个具体情境或题例说明当前关系。 |
| G | 重新聚焦子目标 | 将学生引向一个关键的中间目标，等待学生处理它。 |
| H | 部分解答 | 给出了当前题实际推导、具体算式或中间结果，并给学生留下关键工作。 |

按原文找证据。问题式表达如果已经给出了关键步骤，按实际支持程度标注。完整给出解题过程和终值的反馈保留原文待复核。将确认过的 `action_annotation.status` 所对应的源文件状态改为 `reviewed` 后重建。标注输入为 `data/pa_opsd/action_pilot_gold64/action_drafts.jsonl`。

## 运行与输出

从仓库根目录执行准备。

```text
python scripts/experiments/action_pilot/run.py build --output outputs/pa_opsd/action_pilot_gold64
```

- `probe_inputs.jsonl` 保存同一题的可见输入和结构化教师依据。
- `review_targets.jsonl` 保存金标诊断、完整参考反馈、动作初标和所引用的历史记录。
- `动作标注初稿.md` 便于按题阅读。`manifest.json` 记录固定的样本 ID、分布和文件校验值。

对照入口与已加载 Gold SFT 的 vLLM 服务在同一主机运行，默认端口 8000。沿用 [LongTutor 复评的模型服务](../longtutor_reeval/README.md#一台-5090-上复评)，使用同一 BF16 基座与 32,768 tokens 的上下文。用该基座的 tokenizer 核验后，最长教师输入为 29,375 tokens，生成预算为 96 tokens。脚本通过模型卡上的适配器目录比对原始 Gold SFT 的文件校验值。可以先用 `--limit 4` 和单独的输出文件做连通性检查。正式的均衡对照覆盖 64 条。

```text
python scripts/experiments/action_pilot/run.py probe --pilot outputs/pa_opsd/action_pilot_gold64 --model sft --output outputs/pa_opsd/action_pilot_gold64/decisions.jsonl
python scripts/experiments/action_pilot/run.py score --pilot outputs/pa_opsd/action_pilot_gold64 --predictions outputs/pa_opsd/action_pilot_gold64/decisions.jsonl --output outputs/pa_opsd/action_pilot_gold64/probe_results.json
```

每道题两侧完成后原子写入结果。同一输出文件由一个进程写入，使用同一配置可续跑完整配对。API 失败留待续跑。诊断和动作分别计分，另报格式校验结果。评分区分动作初标匹配和已复核动作匹配，列出逐类诊断和两侧各自答对的样本数，并记录运行设置和评分目标的 SHA-256。

## 路线 A 的教师首步验收

使用已有的 Gold 64 道训练题。每道的当前题和 199 条记录须与冻结输入逐条一致，标准解重新核对题库。三组固定原 Gold SFT 检查点、纯时序检索出的原始题目与历史，以及 A–H 动作定义。教师另读标准解，或同时读从同一可见原文生成的事实索引。索引最多三条，优先覆盖同题、末级知识点匹配的最近答对和最近答错。它只说记录中的题号、位置、时间、正误和匹配标签，不推断解法或错因。未挑到证据时，两种教师提示相同。该独立动作探针不输入 LongTutor 的三个证据问题。

三组输出动作字母、对学生说的首个教学动作以及引用位置。评分根据实际话术，单列模型报出的标签是否与动作一致。盲评只收到题面、共用历史和匿名话术。可用的首步必须在可见题意下可执行、数学正确，且为学生留下工作。分别标记误引及无依据的学生细节、数学或题意错误、直接报答案。空字符串代表选择不干预，JSON/格式失败单列。旧的 64 道动作初标和参考反馈留在本地，供复核争议时使用。

冻结的工程门槛要求事实索引组至少 45/64 道实际首步可用且无上述三类风险，净胜公开输入至少 7 道。在事实索引有内容的子集上，还须净胜只给标准解至少 4 道。误引和直接报答案均须相对两个对照不增加。教师组有格式失败即无法完整判定风险，留待排查。另固定每类 4 道共 16 道完全同输入重复，至少 12 道两次都安全可用、13 道实际动作一致。门槛用于决定是否投入小批训练，评估总体泛化和正式 LongTutor 效果使用后续独立数据和官方评测。

```powershell
python -m scripts.experiments.action_pilot.evidence_probe prepare --pilot outputs/pa_opsd/action_pilot_gold64 --tokenizer .codex_tmp/opd_qwen_teacher_tokenizer --output outputs/pa_opsd/action_pilot/teacher_evidence_run
```

将 `inference.zip` 解压到 AutoDL 的新目录并校验 `SHA256SUMS`。从解压根目录运行。沿用固定基座模型服务，`memory_first` 是服务接入的历史别名，此处哈希固定为原始 Gold SFT 检查点。先在一个终端启动服务，替换权重入口路径。然后用另一个终端先运行四道试跑，再生成 64 道全量对照和 16 道重复，共 208 条正式输出。

```bash
python service.py serve --port 8808 --model PATH_TO_PINNED_BASE --adapter PATH_TO_ORIGINAL_GOLD_SFT
python -m scripts.experiments.action_pilot.evidence_probe probe --bundle . --phase smoke --output smoke.jsonl &&
python -m scripts.experiments.action_pilot.evidence_probe probe --bundle . --phase full --output full.jsonl
```

每个成功请求先写入 `.progress.jsonl`。遇到传输错误，使用同一配置和输出目录可只补缺失请求。格式失败保留原文，试跑有此类失败则退出。模型服务的基座、适配器父模型、vLLM 版本、实际配置与权重校验由打包的服务入口核对。

```text
python -m scripts.experiments.action_pilot.score_evidence_probe blind --bundle PREPARED --predictions FULL_JSONL --output NEW_REVIEW_DIR
python -m scripts.experiments.action_pilot.score_evidence_probe score --bundle PREPARED --review REVIEW_DIR --ratings RATING_JSONL --output GATE_JSON
```

向评审仅提供 `blind.jsonl`。同题完全相同的话术只评一次，随后将评分回配所有对应生成。评分 JSONL 字段为 `case_option`（拼接盲审包标识、连字符、案例及选项，例如 `<packet>-1B`）、`observed_action`、`usable`、`unsupported_history`、`math_or_task_error`、`direct_answer` 和 `reason`。无可评内容用空动作和四项 false，理由说明缺失。冻结评分后汇总与揭盲。教师达标后才启动动作级蒸馏训练。

## 分开使用历史依据与学生端提示

`grounded_teacher.py` 从 Gold 560 训练题另取 24 道纯文字题，与前轮 Gold 64 的学生和当前题号均无交集。回忆失败、概念缺口、步骤错误各 8 道。迁移不足类经过同样筛选后没有合格的纯文字题。四组分别使用公开输入、加标准解、加标准解与事实索引、同一索引加教师规约。规约组只在内部用历史挑动作，向学生给当前题的具体子目标或关键一步。引用至多一个索引内位置，生成和盲评使用同一校验。原始输出完整保留。对照测量整套教师规约的效果。

固定原 Gold SFT 检查点，先生成 4 道四组试跑以核对接入、配置和写入，再生成 24 道四组及 8 道规约组同提示复测，共 104 条。格式失败保留在分母。先导门槛为规约组 17/24 首步安全可用，比三组分别净胜至少 3、2、2 道，规约组全部可评，三项风险均不增加。8 道复测至少 6 道两次都可用，7 道动作相同。此项采用同题逐条盲评，先导达标后再扩大验收，开发和封存集继续按原方案使用。

```text
python -m scripts.experiments.action_pilot.grounded_teacher prepare --tokenizer TOKENIZER --output NEW_BUNDLE
python -m scripts.experiments.action_pilot.grounded_teacher probe --bundle BUNDLE --phase smoke --output runs/smoke.jsonl
python -m scripts.experiments.action_pilot.grounded_teacher probe --bundle BUNDLE --phase full --output runs/full.jsonl
python -m scripts.experiments.action_pilot.grounded_teacher blind --bundle BUNDLE --predictions runs/full.jsonl --output NEW_REVIEW_DIR
python -m scripts.experiments.action_pilot.grounded_teacher score --bundle BUNDLE --review REVIEW_DIR --ratings RATINGS_JSONL --output GATE_JSON
```

AutoDL 解压校验包内 `SHA256SUMS` 后，用上节的固定服务入口启动。生成保存原文、配置，故障时保存逐臂归档的 `.progress.jsonl`。本地保留匿名包、逐条评分、配对结果与门槛判定。 [冻结清单](../../../outputs/pa_opsd/action_pilot/grounded_teacher_20261126_frozen/manifest.json)

## 时序历史与分类器联合对照

用原 Gold SFT 教师预测 24 道纯文字训练题的第一条提示。按学生、题号、规范化题面避开前两轮，覆盖回忆失败 3 道、概念缺口 2 道和步骤错误 19 道。迁移不足没有满足这一筛选条件的样本。分类器复用原 10 项历史统计与逻辑回归，在原 stage2 中扣除与 24 道有关联的学生、题号和规范化题面后重拟合。拟合 ID、移除 ID、统计和概率留在本地。教师两组共用时序历史、核验过的历史索引和标准解，处理组额外看到分类器预测的单个类别。模型输出动作和提示，来源由本地索引记录。

本地准备、匿名评阅和汇总：

```text
python -m scripts.experiments.action_pilot.classifier_teacher prepare --tokenizer TOKENIZER --output NEW_BUNDLE
python -m scripts.experiments.action_pilot.classifier_teacher blind --bundle BUNDLE --predictions runs/full.jsonl --output NEW_REVIEW_DIR
python -m scripts.experiments.action_pilot.classifier_teacher score --bundle BUNDLE --review REVIEW_DIR --ratings RATINGS_JSONL --output GATE_JSON
```

先安装原诊断探针使用的 scikit-learn，并准备本地已核验的分词器。AutoDL 在新目录解压 `inference.zip`，核验 `SHA256SUMS`，再执行：

```bash
bash scripts/experiments/action_pilot/run_classifier_teacher.sh VLLM_PYTHON BASE_MODEL_DIR GOLD_SFT_ADAPTER_DIR
```

4 道双臂试跑每组至少 3 条格式有效，无截断后跑 24 道双臂和 8 次处理组重复。盲评依据题面和时序原始记录，逐条标具体动作、可用性、无依据历史陈述、数学或任务错误、直接报答案。预先固定处理组至少 17/24 条安全可用且动作标签一致、比对照净增至少 3 条、重复两次均安全至少 6/8、重复动作相同至少 7/8、处理组格式全有效且三类风险不增。另报每类与分类器正确或错误时的配对结果。这里是进入小批动作训练的工程门槛。运行归档 `gpu_results.tar.gz`，本地核验备份后按用户要求关机。


## 当前题一步提示与标准解消融

目标是在同一原 Gold SFT 教师上分开检验两项变化。原规约与新规约共用标准解和可见事实索引，只改变系统提示。第二个比较固定新规约及可由学生可见历史计算的事实索引，移除标准解。三组使用相同的题面、历史和输出字段。主指标是匿名话术是否给出具体、适当、无误引、无数学错误且未泄题的第一步。另报格式、自报动作与实际行为的一致性、同输入重复。评阅依据是当前题和学生可见历史，不向评阅者提供教师的标准解或分组名。

从仓库根目录执行 `python -m scripts.experiments.action_pilot.current_step prepare NEW_BUNDLE`。冻结清单包含上一轮 24 道回归题及训练池中按学生、题号和规范化题面排除三轮先导后剩余的可用纯文字题，两部分分别计分。解压生成的 inference.zip 后，运行 `bash scripts/experiments/action_pilot/run_current_step.sh VLLM_PYTHON PINNED_BASE ORIGINAL_GOLD_SFT`。脚本先做五题三臂试跑，再写全量与重复记录。回传后使用 `current_step blind BUNDLE --predictions FULL_JSONL --out REVIEW`，在 REVIEW 下盲评并锁定 ratings.jsonl，再运行 `current_step score BUNDLE --out REVIEW`。评分列及动作字典沿用第一轮。该先导旨在识别可执行的话术修订；训练决策还需跨类确认和特权信息本身的增益。

开发确认集用 `python -m scripts.experiments.action_pilot.confirm_current_step TRAIN_BUNDLE NEW_DEV_BUNDLE` 构造，只使用开发划分并剔除与全部训练学生、题号和规范化题面重合的行。为复算已归档批次的盲评和汇总，请从该批的 inference.zip 解压目录调用其中对应版本的模块。

### 真实输出前缀上的八动作分布

`prefix_action_probe prepare` 使用训练池新增 10 题及开发确认 23 题的公开组已生成回复，截取首个动作字母之前的 JSON 前缀。准备参数是 `--train`、`--dev`（两批含原始预测归档的本地目录）、`--tokenizer` 与新目录 `--out`。用 `python -m scripts.experiments.action_pilot.prefix_action_probe` 调用。正确标准解、公开输入和同组长度相近的错配标准解共享同一输出前缀。GPU 子命令 `infer --bundle INPUT --model BASE --adapter GOLD_SFT --out OUTPUT` 对 A 到 H 的单 token 候选进行前向打分，保存条件概率、首选动作和八字母的总概率质量。先进行小批试跑。

`python -m scripts.experiments.action_pilot.action_rule_ablation` 接受相同准备参数，另加 `--rule no_examples` 或 `--rule original`，分别去掉提示末尾的字母示例或撤去整段新增规约。沿用相同的 GPU 打分入口。比较前核对每题的身份、错配解提供者和结构前缀。两组反事实只计算概率，不产生新的教学话术。实际运行脚本、分布与样本清单见[动作消融归档](../../../outputs/pa_opsd/proposal_evidence_20261127/action_prefix/)。
