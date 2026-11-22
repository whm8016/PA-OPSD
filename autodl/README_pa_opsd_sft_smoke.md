# PA-OPSD 单卡 SFT 工程验收

本说明对应 2026-11-12 的 4090D 单卡实例。当前工作是完整长历史下的短训、保存和重载验收；单条样本的工程结果不能证明教学质量提升，也不是 OPSD 或自进化实验结果。

## 实际环境和数据

- GPU：RTX 4090D，24,564 MiB。
- 工作目录：`/root/autodl-tmp/pa_opsd_20261112`。
- 独立环境：工作目录下 `.venv`，复用实例已有的 CUDA PyTorch，未替换其基础环境。
- 实例版本：Python 3.12.3、torch 2.8.0+cu128、transformers 4.57.1、peft 0.17.1、bitsandbytes 0.48.2、accelerate 1.11.0。
- 本机全量回归：56 项通过；AutoDL 固定版本的 SFT 专项测试：18 项通过，含损失和梯度等价测试。
- 输入：`outputs/pa_opsd/sft_engineering_one_20261112.jsonl`，取自动通过记录中的 q1066 一条，约 2.5 万 tokens；不使用此前标出的三条质量疑点记录。
- 官方学生消息、Gold 隔离检查和长度校验保持不变；Gold 文件仅用于核对隔离，不进入训练输入。

运行包 `pa_opsd_sft_smoke_20261112.zip` 内包括训练入口、契约、专项测试、种子清单及隔离核对所需的 LongTutor 文件。不包含本地 `.env` 或 API 凭据。上传后校验压缩包 SHA256 和全部 12 个成员文件的 SHA256。

## 本轮代码调整

`scripts/train_pa_opsd_sft.py` 增加四个可选开关，原有调用不必增加参数：

- `--load-in-4bit`：冻结的基础权重采用 NF4 双重量化，只训练 LoRA 参数。
- `--gradient-checkpointing`：反向传播时重新计算部分中间结果，降低激活显存。
- `--assistant-logits-only`：完整历史仍经过所有 Transformer 层，只省去无监督历史位置上的词表投影。第 t 个位置预测 t+1，padding 和 prompt 标签仍为 -100。
- `--offload-saved-tensors`：采用 PyTorch 的 save_on_cpu，把前向保存、反向还要用的张量临时放到 CPU；以额外数据传输换取显存，要求同时开启 assistant logits 选项。

新增的 `assistant_token_loss` 选择所有批次中有监督的因果位置，保留有效 assistant token 的同一交叉熵。跨梯度累积使用 Trainer 传入的监督 token 总数归一化。小型真实 Qwen 模型验证损失和逐参数梯度与完整 logits 计算一致，同时覆盖变长 padding 和空监督拒绝。

`--attn-implementation sdpa` 使用 PyTorch 内置实现；本轮不额外编译 FlashAttention。训练结果保存 `training_summary.json`，包含训练指标、实际配置和 CUDA 分配显存峰值。

真实首轮遇到 OOM，PEFT 会将非量化参数升为 FP32；因此在准备后将冻结 embedding/head 恢复为原先加载的 BF16，norm 仍为 FP32。第二轮前向已通过，但反向 checkpoint 重算时还需 898 MiB，剩余约 657 MiB，仍 OOM。最终配置继续加入 CPU 暂存。失败日志分别保存为 training_pipeline_before_bf16_fix.log 和 training_pipeline_before_cpu_offload.log。

## 运行命令

模型从 Qwen 的公开 ModelScope 仓库获取，下载至 `models/Qwen2.5-7B-Instruct`。采用本地模型路径，训练时使用 `--local-files-only`。实际包版本保存在 `environment_freeze.txt`。

新实例需要先确认已有的 `torch` 可用 CUDA，再创建独立环境。以下命令复用了本实例的 torch 2.8.0+cu128；其他镜像应先核对实际版本。

```bash
/root/miniconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install transformers==4.57.1 peft==0.17.1 \
  accelerate==1.11.0 datasets==3.6.0 bitsandbytes==0.48.2 openai tqdm modelscope
.venv/bin/python -c 'from modelscope.hub.snapshot_download import snapshot_download; snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="models/Qwen2.5-7B-Instruct")'
```

```bash
cd /root/autodl-tmp/pa_opsd_20261112

.venv/bin/python scripts/train_pa_opsd_sft.py audit \
  --data outputs/pa_opsd/sft_engineering_one_20261112.jsonl \
  --model models/Qwen2.5-7B-Instruct \
  --max-length 32768 --local-files-only

OMP_NUM_THREADS=8 .venv/bin/python scripts/train_pa_opsd_sft.py train \
  --data outputs/pa_opsd/sft_engineering_one_20261112.jsonl \
  --model models/Qwen2.5-7B-Instruct \
  --max-length 32768 --local-files-only \
  --max-steps 1 --batch-size 1 --gradient-accumulation-steps 1 \
  --lora-r 8 --learning-rate 0.0002 \
  --load-in-4bit --gradient-checkpointing --assistant-logits-only --offload-saved-tensors \
  --attn-implementation sdpa \
  --output-dir outputs/pa_opsd_sft_smoke_step1
```

首轮只执行 1 步，成功后再用同一条样本从基础模型执行 3 步短训，输出目录改为 `outputs/pa_opsd_sft_smoke_step3`。这不是在一步检查点上继续训练，两个运行分别从同一基础模型初始化。

## 2026-11-12 实测结果

- 真实输入 24,814 tokens，其中 assistant 监督 353 tokens，完整保留历史。
- 单步训练成功：18.1945 秒，loss 1.324035，梯度非零，已保存。
- 3 步短训成功：50.9297 秒，逐步日志 loss 1.3240、1.2689、1.2098；只反映同一训练样本的拟合，不能解释成评测提升。
- 3 步运行的 PyTorch 分配显存峰值为 19,485,648,384 bytes，约 18.15 GiB。不是整台 GPU 的总显存占用统计。
- LoRA 权重均为有限值，lora_B 的绝对值总和约 622.44；LoRA B 从零初始化后产生非零更新。
- 新进程从同一基础模型重新加载 3 步 LoRA 并成功生成回复，但原生 JSON 校验失败：LaTeX 的 `\(` 被直接写入 JSON 字符串，造成非法转义。回复还出现当前题解题方程，需要教学安全复核。
- 训练退出码为 0，重载生成的格式校验退出码为 1。训练链路通过，模型输出质量未通过；未修改校验器放行。
- 本机最终全量测试 56 项通过，lint 通过；服务器固定版本专项测试 18 项通过；独立复审未发现已实现优化中的 P0/P1/P2。

训练已结束，GPU 当前空闲；实例保持开启。权重在服务器工作目录的 `outputs/pa_opsd_sft_smoke_step1` 和 `outputs/pa_opsd_sft_smoke_step3`。日志、两个检查点和结构化验收报告另存 `gpu_acceptance_20261112.zip` 并备份回本机。

重载检查使用以下命令。它在训练样本上做工程检查，不计入正式评测；当前模型输出未通过原生格式校验，因此退出码为 1。

```bash
.venv/bin/python scripts/train_pa_opsd_sft.py generate \
  --data outputs/pa_opsd/sft_engineering_one_20261112.jsonl \
  --adapter outputs/pa_opsd_sft_smoke_step3 --split train \
  --max-length 32768 --max-new-tokens 1024 --local-files-only
```

后续正式训练要先解决候选教师数据的质量疑点并固定实验配置；9 条自动接受不能当作 9 条人工 Gold。当前未执行 OPSD 或技能自进化训练。

## 排查

- 连接成功却 `python: command not found`：非交互 SSH 没有激活 conda；使用 `.venv/bin/python` 的完整路径。
- 提示 tokens exceed max_length：先检查完整聊天模板后的实际长度，禁止静默截断历史或监督输出。
- CUDA out of memory：保存失败日志与当前配置，核对量化、checkpointing、assistant logits 和 CPU 暂存四个开关；不要把长度从 32768 悄悄改为 4096。
- 模型下载未完成：等待所有 safetensors 分片和 tokenizer 文件齐全，检查 `download.log`，不要把 `.incomplete` 文件当成完整模型。
- 一步后生成不符合教学格式：不等于训练链路失效；先检查 loss、LoRA 更新和重载，一步不承担完整格式学习的验收目标。

## 调研依据

- [QLoRA，arXiv:2305.14314](https://arxiv.org/abs/2305.14314) 和 [PEFT 量化指南](https://huggingface.co/docs/peft/developer_guides/quantization)：NF4、双重量化与低秩适配器。
- [Transformers 4.57.1 Qwen2 源码](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py)：支持在指定位置计算 `logits_to_keep`。
- [PyTorch 2.8 save_on_cpu 文档](https://docs.pytorch.org/docs/2.8/autograd.html#torch.autograd.graph.save_on_cpu)：保存张量时放到 CPU，反向需要时恢复到原设备；小模型检查暂存前后的损失及梯度一致。
- [Qwen 官方模型说明](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)：当前 config 默认支持 32768，当前输入无需额外扩展 RoPE。
- 对标 [TRL，约 19.7k stars](https://github.com/huggingface/trl)、[Unsloth，约 79.4k stars](https://github.com/unslothai/unsloth)、[Axolotl，约 12.7k stars](https://github.com/axolotl-ai-cloud/axolotl)、[Liger Kernel，约 6.7k stars](https://github.com/linkedin/Liger-Kernel)，star 为本会话查询时页面展示值。

工程选择：在现有 Trainer 上组合量化、checkpointing 和原生受监督位置投影，减少框架迁移。Liger 的融合损失是可评估的另一方案，但会增加内核依赖；普通 packing 不能让一个完整长历史样本自动变短。当前组合仍须以单卡实测为准。
