# 教学技能卡与历史证据路由

这是《开题报告_OPSD自进化_v3》研究点二的本地原型。每张卡描述何时适用、推荐的实际教学动作和停用条件。当前实现「间隔后先提示回忆」和「用已做对的题搭桥」两张候选卡。路由只读当前题和该学生的历史，找不到充分线索时留空。

回忆卡使用同题或知识点标签组完全相同的历史。类比卡使用近期已做对的同组异题，交给教师确认题型或条件是否真的变化。四天和历史正确率参考 [LongTutor 附录中的诊断规则](https://aclanthology.org/2026.acl-long.1371.pdf)，严格匹配标签组和保留空选项是这版原型的取舍。同一标签组可能覆盖不同解法，真实间隔和正误仅用于选取可检查的历史。技能卡目前是人工编写的规则，结合已发布策略和已有动作定义。

## 本地复跑

从工作区根目录执行。输入复用已冻结的 64 道训练题与 24 道开发题；脚本校验输入哈希及样本所属划分。Gold 参考仅在离线路由完成后读入用于对照。

```text
python -m scripts.experiments.teaching_skills.run --split outputs/pa_opsd/gold_split_20261120/split_manifest.json --train outputs/pa_opsd/action_pilot_gold64 --dev outputs/pa_opsd/diagnoser_evidence_gold24/input/validation --output outputs/pa_opsd/teaching_skills/local_repeat
```

| 输出 | 用途 |
| --- | --- |
| `cards.json` | 卡片定义、对应的策略与动作 |
| `逐题核对.md` | 两组逐题触发或留空的原因、原历史位置、Gold 对照和开发题原文示例 |
| `*.audit.jsonl` | 可机读的逐条路由与核对结果，仅供本地分析 |
| `dev24.teacher_pairs.jsonl` | 已触发开发题的教师预检输入；两组共用历史片段、原始序号和时间提示，实验组额外收到卡片 |
| `run.json` | 参数、固定输入和代码哈希、触发数、模型调用次数 |

教师配对文件是诊断与教学动作的工程预检。正式 LongTutor 指标继续由锁定的官方推理与评分代码生成。当前只验证了卡片路由和输入构造，教师增益、技能入库和 PA-OPSD 蒸馏需要实测。
