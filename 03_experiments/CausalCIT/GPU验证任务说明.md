# GPU 服务器验证任务说明 (v2 — 已改为多seed版本，请使用本版本，忽略之前的旧版本)

> 目的：验证门控注意力 bug 修复后，`Full CausalCIT` 能否在 ETTh1 长序列预测 (pred_len=336)
> 上反超 `w/o Gate`（全连接注意力），修复之前的实验结果是 **-0.66%（输）**。
>
> **重要更新：第一轮GPU结果不可信，已发现并修复问题，请务必重新跑一次（见下方"更新说明"）。**
>
> 本文档写给在服务器上执行任务的 AI/操作者，请**严格按步骤执行**，不要自行修改代码逻辑。

---

## 更新说明（第一轮结果为什么不算，这次改了什么）

第一轮跑出来的 `Full CausalCIT` 在 `pred_len=336` 仍是 `-0.98%`（没有翻正），但同时发现
`w/o Gate` 这个**完全没有改动过代码的变体**，这次也从修复前的 `+5.51%` 变成了 `-0.99%`。
排查后确认：整个代码库之前**没有固定任何随机种子**（模型初始化、DataLoader shuffle 全是随机的），
导致两次实验的差异很可能只是训练随机性噪声，而不是bug修复的真实效果，**第一轮结果不能用来下结论**。

现在已经给 `run_ablation.py` 加了 `--seed` 参数（每个变体训练前都会重置种子，保证公平对比）。
**请重新同步代码后按下面的新步骤运行**，第4步的命令已经改变，请注意。

---

## 0. 背景（可跳过，仅供理解）

`CausalCIT_demo/models/causal_channel.py` 里的通道注意力门控实现有一个数值 bug：

```python
# 旧代码（有bug）：硬mask式惩罚
attn = attn * gate_mask + (1 - gate_mask) * (-1e9)
```

只要 `gate_mask < 0.9999`，惩罚项就会把 attention logit 压到 `-1e8` 量级，
softmax 后权重几乎为 0 —— 等价于把本该是"软"的 sigmoid 门控强行二值化成开/关，
导致门控机制形同虚设甚至有害。这是之前消融实验里 **Full CausalCIT 在 pred_len=336
反而输给 w/o Gate（全连接注意力）** 的根本原因。

已修复为对数域软惩罚：

```python
# 新代码（已修复，本地已验证方向正确）：
attn = attn + torch.log(gate_mask.clamp(min=1e-4))
```

惩罚幅度与门控值平滑对应（`g=0.5` 时惩罚 ≈ `-0.69`，与 attn logits 量级匹配），
不再强制二值化。本地用 CPU + 等效小规模场景验证过，方向已经反转为正贡献。
**现在需要在 GPU 上跑一次完整规模的消融实验，确认在真实论文用的规模/数据集下也成立。**

---

## 1. 第一步：确认代码已经是修复后的版本

打开文件：`03_experiments/CausalCIT/CausalCIT_demo/models/causal_channel.py`，
搜索 `CausalChannelAttention` 类的 `forward` 方法，应该看到类似下面这一行（约在
第 130-140 行之间）：

```python
attn = attn + torch.log(gate_mask.clamp(min=1e-4))
```

**如果看到的是 `attn = attn * gate_mask + (1 - gate_mask) * (-1e9)`，说明代码没同步成功，
请先重新同步代码，不要继续往下跑实验。**

可以用命令快速检查（Linux）：

```bash
cd <项目根目录>/03_experiments/CausalCIT
grep -n "torch.log(gate_mask" CausalCIT_demo/models/causal_channel.py
```

如果这条命令有输出，说明修复已生效，继续下一步；如果没有任何输出，说明代码是旧版本，停下来重新同步。

---

## 2. 第二步：准备环境

如果服务器上还没有配置过环境，在 `03_experiments/CausalCIT/` 目录下执行：

```bash
cd <项目根目录>/03_experiments/CausalCIT
bash setup.sh --env-only --name causalcit --python 3.10
conda activate causalcit
```

如果环境已经配置好（之前跑过实验），直接激活已有环境即可，例如：

```bash
conda activate causalcit
```

确认 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

应该输出 `True` 和显卡名称（如 RTX 4090）。**如果输出 `False`，先停下来排查 CUDA / driver 问题，不要用 CPU 硬跑这个实验（太慢）。**

---

## 3. 第三步：确认数据集存在

需要 `ETTh1.csv`。检查是否已经存在：

```bash
find <项目根目录>/03_experiments/CausalCIT -name "ETTh1.csv"
```

如果没找到，下载：

```bash
cd <项目根目录>/03_experiments/CausalCIT
python download_data.py --dataset ETTh1
```

---

## 4. 第四步：运行完整消融实验（核心步骤，注意：本次需要跑3个不同的seed）

**重要：这次必须跑 3 次，分别用 `--seed 42`、`--seed 123`、`--seed 2024`**，
每次用不同的输出目录，**不要覆盖旧的 `output/` 目录**（那是最早修复前的结果，继续留作参考）：

```bash
cd <项目根目录>/03_experiments/CausalCIT/CausalCIT_ablation

python run_ablation.py --exp all --device cuda --seed 42   --output_dir ./output_seed42
python run_ablation.py --exp all --device cuda --seed 123  --output_dir ./output_seed123
python run_ablation.py --exp all --device cuda --seed 2024 --output_dir ./output_seed2024
```

- 参数说明：`--exp all` 表示合成数据 + ETTh1 都跑；`--device cuda` 强制用 GPU；
  `--seed` 是本次新加的参数，**必须显式传**，否则默认也是42（等于只跑了一组，不够）。
- 为什么要跑3个seed：上一轮跑的时候没固定随机种子，导致同一个"完全没改代码"的变体
  (`w/o Gate`) 前后两次训练结果从 `+5.51%` 变到 `-0.99%`，说明单次训练的随机噪声
  比我们要观察的效果还大，**必须多个seed取平均才能下结论**，单跑一次不算数。
- 3次依次跑，**请顺序执行，不要并行跑（会抢显存）**。跑完一个再跑下一个。
- 预计耗时：每次约 **45 分钟**，3次约 **2.5小时**。可以用 `nohup` / `tmux` 后台跑，
  比如：`nohup python run_ablation.py --exp all --device cuda --seed 42 --output_dir ./output_seed42 > seed42.log 2>&1 &`
- 跑的过程中终端会持续打印每个变体的训练日志和最终 MSE/MAE，属于正常现象，不要中断。
- 如果中途报错（比如显存不足），可以加 `--batch_size 16` 降低批大小后重试（3次都要用同样的batch_size），不要更改模型代码。

跑完之后，产出文件在 `./output_seed42/`、`./output_seed123/`、`./output_seed2024/` 下，
每个目录里重点是这个文件：

```
CausalCIT_ablation/output_seed42/ablation_report.md
CausalCIT_ablation/output_seed123/ablation_report.md
CausalCIT_ablation/output_seed2024/ablation_report.md
```

---

## 5. 第五步：核对结果（不需要你自己下结论，但可以简单看一眼）

三次跑完后，分别打开 `output_seed42/ablation_report.md`、`output_seed123/ablation_report.md`、
`output_seed2024/ablation_report.md`，找到 **`## ETTh1 真实数据消融` → `### pred_len = 336`** 这一段表格。

**不需要你自己判断修复是否成功**——因为现在看的是3个seed的分散程度，需要我把3份数据放在一起算
平均值和方差才能下结论，你只要确认3个文件都完整生成、没有报错即可。

---

## 6. 第六步：把结果反馈回来

请把以下内容**原文粘贴**回复给我（不要总结、不要省略数字、不要只贴一份）：

1. `output_seed42/ablation_report.md`、`output_seed123/ablation_report.md`、
   `output_seed2024/ablation_report.md` 这**3个文件的完整内容**（缺一份都不够做统计）。
2. 3次运行终端里每个变体打印的最终 `MSE` / `MAE` 那几行（如果日志还在，比如用了nohup重定向的.log文件）。
3. 如果中途有任何报错或修改了默认参数（比如改了 batch_size），请一并说明改了什么、为什么改，
   并确认3次跑的参数（除了seed）是否完全一致。

**不需要**做额外分析或写结论，只需要把上面3项原始数据发回来，后续的分析和结论判断由我这边来做。

---

## 常见问题

- **Q: 需要改代码吗？**
  A: 不需要。代码已经在本地修复好（包括新加的 `--seed` 参数），服务器只需要用同步过去的最新版本直接跑实验即可。
- **Q: 可以只跑 ETTh1，不跑合成数据吗？**
  A: 可以，3次命令都用 `--exp real` 替代 `--exp all`，能省一半时间，但请仍然完整贴出3份 `ablation_report.md`。
- **Q: 可以只跑1个seed吗，3个太慢了？**
  A: 不行。上一轮就是因为只跑了1次、没固定种子，结果被随机噪声误导，这次的核心目的就是要看多个seed下结果是否稳定，跑1个seed等于没解决问题。
- **Q: 3个seed的输出目录能不能用默认的 `./output`？**
  A: 不要。请严格按 `./output_seed42`、`./output_seed123`、`./output_seed2024` 命名，`./output` 目录下是最早修复前的旧结果，覆盖了就没法参考了。
- **Q: 3次运行之间除了seed还能改别的参数吗？**
  A: 不能。3次除了 `--seed` 不同，其余参数（batch_size等）必须完全一致，否则无法做公平对比。
