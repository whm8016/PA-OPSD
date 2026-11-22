# 教师团队 Gold 32 对照

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
