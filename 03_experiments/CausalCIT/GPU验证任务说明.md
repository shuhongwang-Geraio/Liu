# GPU 服务器验证任务说明

> 目的：验证门控注意力 bug 修复后，`Full CausalCIT` 能否在 ETTh1 长序列预测 (pred_len=336)
> 上反超 `w/o Gate`（全连接注意力），修复之前的实验结果是 **-0.66%（输）**。
>
> 本文档写给在服务器上执行任务的 AI/操作者，请**严格按步骤执行**，不要自行修改代码逻辑。

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

## 4. 第四步：运行完整消融实验（核心步骤）

**不要覆盖旧的 `output/` 目录**（里面是修复前的结果，用来做对比基准），
用一个新目录 `output_gatefix_gpu`：

```bash
cd <项目根目录>/03_experiments/CausalCIT/CausalCIT_ablation
python run_ablation.py --exp all --device cuda --output_dir ./output_gatefix_gpu
```

- 参数说明：`--exp all` 表示合成数据 + ETTh1 都跑；`--device cuda` 强制用 GPU。
- 预计耗时：RTX 4090 上约 **45 分钟**（合成数据 ~25min + ETTh1 ~20min）。
- 跑的过程中终端会持续打印每个变体（PatchTST / w/o Gate / w/o EnvSplit / w/o HSIC / Full CausalCIT）的训练日志和最终 MSE/MAE，属于正常现象，不要中断。
- 如果中途报错（比如显存不足），可以加 `--batch_size 16` 降低批大小后重试，不要更改模型代码。

跑完之后，产出文件在 `./output_gatefix_gpu/` 下，重点是这个文件：

```
CausalCIT_ablation/output_gatefix_gpu/ablation_report.md
```

---

## 5. 第五步：核对结果（关键判断标准）

打开 `ablation_report.md`，找到 **`## ETTh1 真实数据消融` → `### pred_len = 336`** 这一段表格。

对比标准（旧结果来自
`03_experiments/CausalCIT/experiments/2026-06-03_initial/ablation/report.md`）：

| 变体 | 修复前 vs PatchTST | 本次需要确认的方向 |
|------|------|------|
| w/o Gate | +5.51% | 应该仍然明显为正（不该变差太多） |
| **Full CausalCIT** | **-0.66%（输给baseline）** | **本次应该变成正数**，即修复生效 |

**判断是否修复成功的核心标准：**
`pred_len=336` 表格里 `Full CausalCIT` 这一行的 `vs PatchTST` 列，
数值应该从原来的 `-0.66%` 变成**正数**（理想情况下应接近甚至超过 `w/o Gate` 的 `+5.51%`）。

同时也看一下 `pred_len=96` 和"合成数据消融"两个表格，正常预期：
- pred_len=96：改动前后差异不大（原本就接近0，是预期内的）。
- 合成数据：Full CausalCIT 相对 PatchTST 应该由负转正（原来是 `-2.41%`）。

---

## 6. 第六步：把结果反馈回来

请把以下内容**原文粘贴**回复给我（不要总结、不要省略数字）：

1. `CausalCIT_ablation/output_gatefix_gpu/ablation_report.md` 的**完整内容**。
2. 运行 `run_ablation.py` 时终端里每个变体打印的最终 `MSE` / `MAE` 那几行（如果日志还在）。
3. 如果中途有任何报错或修改了默认参数（比如改了 batch_size），请一并说明改了什么、为什么改。

**不需要**做额外分析或写结论，只需要把上面 3 项原始数据发回来，后续的分析和结论判断由我这边来做。

---

## 常见问题

- **Q: 需要改代码吗？**
  A: 不需要。代码已经在本地修复好，服务器只需要用同步过去的最新版本直接跑实验即可。
- **Q: 可以只跑 ETTh1，不跑合成数据吗？**
  A: 可以，用 `--exp real` 替代 `--exp all`，能省一半时间，但请仍然完整贴出 `ablation_report.md`。
- **Q: 输出目录能不能用默认的 `./output`？**
  A: 不要。默认 `./output` 目录下是修复前的旧结果，覆盖了就没法对比了。务必用 `--output_dir ./output_gatefix_gpu`。
