# 教学动作级蒸馏与三轮技能验证

本目录实现开题 V3 的两项研究。学生只看题面和该学生的可见历史。训练时，同源教师可收到标准解、量规、来源明确的 Eedi 误概念候选及三角色意见。研究点二另测通过独立准入的技能能否沉降进参数。正式 LongTutor 推理使用独立的原版官方入口。

## 数据和模型

从仓库根目录运行 `python -m scripts.experiments.action_distillation.prepare NEW_RUN`。固定划分为 Gold 训练 560、开发 140、封存 300。本次只打开训练与开发参考。按现有时序检索选择最近三条入选原文，模型在这些可见位置中选依据，可选 0 表示证据不足。训练参考标签仅用于分层、Gold 对照监督或离线检验。三角色推理的消息构造只读取公开数据与已声明的特权字段。

主对照从有解析的纯文字训练题按固定散列顺序四类轮换，各取 18 道；全量扩展使用符合相同条件的全部 373 道。前者固定 72 个样本、36 次更新，后者 373 个样本、187 次更新。两种规模都从同一原 Gold SFT 适配器开始。所有相同规模的权重组共享训练行顺序与学习率。基座 Qwen2.5-7B-Instruct 固定到 `a09a35458c702b33eeacc393d103063234e8bc28`。输入、逐项哈希、模型版本、shell 脚本与实际使用的源码快照保存在[本轮输出目录](../../../outputs/pa_opsd/v3_full_20261127_run1/)。

## 训练入口

1. `train.py` 提供首轮工程对照，动作和内容均用 JSD，另含 Gold 文本 SFT、固定教师离策略分布匹配、错配及等长无关输入。此轮较早的错配算法允许重用供体，正式 V3 对照另用一次一人的最小代价匹配。
2. `eedi.py NEW_RUN data/eedi/misconceptions-graph/data/misconception-map.csv` 产出知识点到明确 construct 和 misconception ID 的对照。候选描述来自原图谱，不作为学生诊断标签。
3. `team.py --bundle NEW_RUN --model BASE --adapter GOLD --out NEW_RUN/team` 逐阶段批量运行诊断者、策略教师和审核者。另行去掉误概念输入，重新生成受影响题的三个角色。扩展训练可用 `--train-only` 跳过开发和反事实生成。
4. `v3.py --arm pa --bundle NEW_RUN --model BASE --adapter GOLD --count 72 --out NEW_RUN/v3_pa` 执行 PA-OPSD。学生按温度 0.7 生成证据、诊断、动作与话术。教师分支在学生的同一续写前缀上打分并停止梯度。动作损失为八类上的教师到学生 KL，话术损失为对称 JSD，格式锚定只对固定结构字符计算交叉熵。联合系数分别为 0.5、0.5、0.05。动作独立、内容独立、教师文本 SFT、错配、等长无关、仅标准解和去 Eedi 均有独立入口与权重。供体通过匈牙利算法按特权 token 长度匹配，禁止同题或同解。教师文本 SFT 遇审核输出不可解析时，使用冻结教师重新生成结构化目标，并留痕。

### 教师参数与官方 OPSD 的对应

以作者仓库提交 [`ae7d2519e94920c4eb6206c0c26de46d9c50abae`](https://github.com/siyan-zhao/OPSD/tree/ae7d2519e94920c4eb6206c0c26de46d9c50abae) 为参照。官方主实验在教师前向时关闭学生新增的 LoRA，以固定第 0 步策略，并在 `torch.no_grad()` 下给学生轨迹打分。PA-OPSD 从已训练的 Gold SFT LoRA 起跑。为保留第 0 步 Gold SFT 策略，`v3.py` 在公共冻结基座上载入一份冻结的初始适配器。`--teacher-mode` 只接受 `fixed`，也可省略。学生采样与反传均使用学生适配器，输出只保存学生适配器。`receipt.json` 记录教师模式、教师适配器哈希和上游参照提交。

11 月 27 日的 72 与 373 题 PA-OPSD 权重及技能循环采用共享适配器的动态教师，源码与配置保存在[原实验归档](../../../outputs/pa_opsd/v3_full_20261127_run1/weights_and_training.tgz)，仅用于复现对应历史结果。固定教师版本的完整 GPU 训练和 LongTutor 评测待跑。教师文本 SFT 始终使用固定来源。V3 的动作 KL、话术 JSD、格式锚定和受限生成属于本项目的变体。官方原版示例使用 token 级前向 KL（`--beta 0`）和 `--fixed_teacher`。

## 技能快慢循环

`cycles.py freeze --bundle NEW_RUN --prior PRIOR_DEV_STRATA` 在看本轮结果前锁定三轮来源、准入、更新、确认四池。各池之间禁止学生、题号和规范化题干交叉，确认池还排除上次 23 道确认题。准入池取自原始 Gold SFT 曾经见过的训练划分，当前轮次只对抽取来源独立。使用 `cycles.py execute --bundle NEW_RUN --model BASE --out NEW_RUN/cycles`，入口会核验联合模型 `v3_pa` 的固定教师回执和权重哈希。每轮固定本轮开始时的教师适配器，并在该轮的 `result.json` 中记录对应哈希。

提炼从六种教学话术模板中选择，记录触发条件、来源例及反例。准入比较现有技能库与加入候选后的技能库，使用独立 Gold 诊断、动作相容性和明确报答案标记。通过后进入记忆。未通过候选也可生成标记为 shadow 的实验性权重，仅用于量化风险，不得继承。确认时比较同一批学生在关闭检索后的联合命中、诊断 Macro-F1、显式报答案和推理开销；学生聚类重抽样求 95% 区间。额外检索增益不到 5 个百分点时，不计算保持率。明确报答案标记是文本模式，只衡量安全性的一个侧面。三个确认池均缺少迁移不足类，单轮数据不足以外推全部诊断类型。

## 开发与外部评测

- `evaluate.py` 在固定的 140 道上保存各权重的完整结构化输出。诊断正确、动作与 Gold 策略相容、二者同时命中为工程指标。公开端不读取标准解、教师意见或技能卡。结构化诊断使用多次前向，不能当作单次官方 API 调用的成本。
- `official.py` 直接调用指定 LongTutor 版本的 `eval_ai_tutor.py`。每个模型先小批再全量，保留格式失败。活跃服务模型映射、适配器实际权重、输入与预测一并核验。`judge.py` 在本地调用相同上游的 `compute_ai_tutor_eval_metrics.py`。DeepSeek 六个主组共用 5 元总额，各组记录逐请求用量及不可挪用的初始份额。全量扩展仅在主组全部结清后使用余额。
- `external_benchmark.py` 直接使用 MathTutorBench 提交 `6faed173ec2bef55cb899b2a3e0f93982f9cb176` 的题目构造、提示、解析与计分。StepVerify 和 GSM8K 的正式数据传输改为按哈希读取本地副本。四项任务为普通与困难教学提问、错步定位及苏格拉底式提问。GPU 批量生成后将完全匹配的请求与答案送回上游主入口。前两项的官方指标仅检验提问表达，不能独立证明数学内容正确。
- `analyze.py NEW_RUN` 汇总逐题结果、原始计量、技能轮次，并按当前题号整簇重抽样做配对区间。官方汇总及分母保留在 `official_dk/*.judge.log`；未完成的评分单列。

## 历史干预与报告复算

`history_probe.py --bundle RUN --model BASE --sft GOLD --out RUN/history_probe` 在 GPU 读取四组冻结权重的诊断和八动作概率。预测前在输出目录放入本轮的 `design.json`。固定前缀使用原 Gold SFT 在真实输入上生成的证据序号与诊断。自由前缀在真实、干预两种视图分别重新选择。仅翻转展示历史中的正误标记，当前题完整保留；这个干预没有教学金标。自然同题异生对照使用真实历史与已有标签，按当前题等权。复现实测环境、版本、命令和逐项结果见[干预归档](../../../outputs/pa_opsd/v3_full_20261127_run1/history_mechanism.tgz)。

从仓库根目录以 `python -m scripts.experiments.action_distillation.<模块名>` 调用：`analyze RUN` 和 `analyze RUN/scale` 读取对应逐项输出；`history_analysis RUN` 核对前缀并汇总干预；`scale_comparison RUN` 生成同题变化与答辩图；`audit RUN` 复核本轮六个归档、权重与正式预测；`report RUN docs` 在 docs 生成原稿补充版和独立实验报告。合法的官方缺分保留在原始记录中，部分组仅对共同有分的题目描述均值。

本轮裁判使用量按厂商回执计量，已结算请求根据缓存命中量按高峰价取上界，未决请求保留整笔预留。接近总限额后停止新请求。[厂商价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) · [本轮结果与覆盖范围](../../../outputs/pa_opsd/v3_full_20261127_run1/结果索引.md)
373 题教师 SFT 的 140 条评分已全部归档。项目根目录运行 `python -m scripts.experiments.action_distillation.compare_teacher_sft outputs/pa_opsd/v3_full_20261127_run1` 可从已有逐题评分重新生成同规模对照、区间和计费汇总。输出为 `scale/teacher_sft_completion/paired_summary.json` 和 `paired_by_id.jsonl`。实际裁判运行使用 `judge v3_teacher_sft --scale --complete-teacher-sft`，复核结果只需运行本地比较。[完整结论](../../../outputs/pa_opsd/v3_full_20261127_run1/scale/teacher_sft_completion/同规模训练结论.md)

