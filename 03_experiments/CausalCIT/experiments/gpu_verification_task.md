# GPU 服务器验证任务说明 (v3 — 修复版协议，2026-08-11 更新)

> **重要：本版本已全面重写。** 旧版 (v2) 针对的是 `CausalCIT_demo` 的数值 bug 修复验证
> （`attn = attn + torch.log(gate_mask)`），该验证已于 2026-08-08 完成归档，请勿再执行旧命令。
>
> 当前主线：**门 1 静态诊断确认的根因 1&2 已修复（修 A+B）**，修复版 `full_v2_fixed`
> 是论文正式采用的协议。本文档写给在 GPU 服务器上执行任务的 AI/操作者，请严格按步骤执行。
>
> **核心判据（门 2）**：修复版 8-seed 重跑 weather/electricity/traffic，看负收益是否翻正。
> 负转正 → 抢救 CausalCIT 为主；仍负 → 转向想法 1（跨环境风险厌恶 DRO）。

---

## 0. 背景与关键须知

- 门 1 静态诊断（2026-08-11，0 GPU）确认三个根因：
  - 根因 1：RFF σ 硬编码 → `proj.std` 随 d_model 单调 (≈√d_model)；
  - 根因 2：未归一化 HSIC → 核坏导致依赖强度无区分度（动态范围仅 1.1–1.2×）；
  - 根因 3：非语义环境切分 → `cv≈0.005`，稳定性项无信息。
- **修 A+B 已实现并 CPU 验证**：`causal_channel.py` 新增 `rff_sigma_mode='median'` +
  `cka_normalize=True`（均默认关闭，不破坏旧行为）。验证：`proj.std` 3.5–7.9 → 0.8–1.1；
  HSIC 区分度 1.1× → **5.6–9×**；`cv≈0.005`（根因 3 仍在，本批 GPU 任务不涉及）。
- 修复版是新协议，**与 `output_large_v2` 旧数字不可直接对比**；traffic(d16) 核"最不坏"
  → 修复后提升可能收窄，属预期。
- 论文正式采用 `full_v2_fixed`；⚠️ 所有旧数字（含 P0-1 前的）在 seed 修复 + 修复版重跑前
  **不可最终采信**。

---

## 1. 第一步：环境准备

在项目根目录 `03_experiments/CausalCIT/` 下（首次需建环境）：

```bash
cd <项目根目录>/03_experiments/CausalCIT
bash setup.sh --env-only --name causalcit --python 3.10
conda activate causalcit
```

已有环境则直接激活，并确认 GPU 可用：

```bash
conda activate causalcit
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

应输出 `True` 和显卡名。若 `False` 先排查 CUDA/driver，不要用 CPU 硬跑。

---

## 2. 第二步：确认数据集存在

需要 `weather.csv`、`electricity.csv`、`traffic.csv`、`ETTh1.csv`、`ETTm1.csv`、
`exchange_rate.csv`、`ILI.csv`。检查：

```bash
find <项目根目录>/03_experiments/CausalCIT -name "*.csv" | head -20
```

缺失则下载：

```bash
cd <项目根目录>/03_experiments/CausalCIT
python download_data.py --dataset weather    # 按需替换 electricity/traffic/ETTh1/ETTm1/exchange_rate/ILI
```

---

## 3. 第三步：验证靶场（最高优先，门 2 判据）

**目的**：修复版 `full_v2_fixed` 8-seed 重跑 weather/electricity/traffic，看负收益是否翻正。

```bash
cd <项目根目录>/03_experiments/CausalCIT/CausalCIT_ablation

# 1) 生成 job（3 shard）
python run_large.py gen --datasets weather electricity traffic \
    --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 \
    --num_shards 3 --output_dir ./output_large_v3

# 2) 每张卡跑一个 shard（3 张卡并行；2 张卡则先跑 shard0/1，跑完再补 shard2）
CUDA_VISIBLE_DEVICES=0 CIT_THREADS=8 python -u run_large.py run --device cuda:0 \
    --job_file ./output_large_v3/jobs_shard0.json --result_csv ./output_large_v3/results_shard0.csv &
CUDA_VISIBLE_DEVICES=1 CIT_THREADS=8 python -u run_large.py run --device cuda:1 \
    --job_file ./output_large_v3/jobs_shard1.json --result_csv ./output_large_v3/results_shard1.csv &
CUDA_VISIBLE_DEVICES=2 CIT_THREADS=8 python -u run_large.py run --device cuda:2 \
    --job_file ./output_large_v3/jobs_shard2.json --result_csv ./output_large_v3/results_shard2.csv &

# 3) 全部结束后汇总
python run_large.py summarize --output_dir ./output_large_v3
```

- 建议加 `--amp` 提速（HSIC/门控仍走 fp32 保精度）。
- 产出：`output_large_v3/improvement_vs_patchtst.md` 等汇总表。
- 判据：weather/electricity 负收益翻正 → 通过；仍负 → 最多再做一轮修 C 即止损。

---

## 4. 第四步：P0-1 重跑主表（seed 修复 + 修复版，科学严谨性硬伤）

**背景**：spawn seed bug 已修复（`_train_one`/`_train_syn_ood` 内补 `set_seed(job['seed'])`），
但主表还是旧协议结果，必须重跑才可采信。

```bash
cd <项目根目录>/03_experiments/CausalCIT/CausalCIT_ablation

# 1) 生成 job（6 数据集 × 6 变体 × 8 seed；--dump_gates 给 traffic/electricity 高维热图供数）
python run_large.py gen --datasets traffic electricity etth1 ettm1 weather exchange ili \
    --variants patchtst full_v2_fixed capacity_match gate_prior_only no_env \
    --seeds 42 123 2024 7 13 99 2023 31 \
    --num_shards 3 --output_dir ./output_large_v3 \
    --dump_gates

# 2) 每张卡一个 shard（同第三步写法）
# 3) 汇总
python run_large.py summarize --output_dir ./output_large_v3
```

**验收**：traffic/electricity 的显著提升在 seed 真正生效下依然成立；之后用
`plot_bootstrap_ci.py --results_dir ./output_large_v3` 重新出图（新图替代旧图投稿）。

> 注意：主表与验证靶场共用 `output_large_v3` 目录。建议先只跑验证靶场并确认判据，
> 再补全主表（gen 会合并新 job，run 有断点续跑，结果 csv 按 key 去重，重复 job 自动跳过）。

---

## 5. 第五步：P1 系列（GPU 空闲后按序）

### P1-1 敏感性分析网格（证明结论不依赖超参脆点）
traffic 上 8-seed，逐组跑（每组 gen → run → summarize）：

```bash
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --n_envs 2 --num_shards 3 --output_dir ./output_sens_nenvs2
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --rff_dim 64 --num_shards 3 --output_dir ./output_sens_rff64
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --prior_weight 0.1 --num_shards 3 --output_dir ./output_sens_prior01
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --temperature 1.0 --num_shards 3 --output_dir ./output_sens_temp10
```

验收：提升率方向/幅度不因超参显著翻转。

### P1-3 熵正则正式实验（traffic）
`gate_prior_only` 无门控熵接口，熵正则不会作用于它（已知限制，不必跑该变体）：

```bash
python run_minimal_falsifiable.py --dataset traffic --seeds 42 123 2024 7 13 99 2023 31 \
    --device cuda:0 --entropy_weight 0.01 --output_dir ./output_falsifiable_entropy
```

（对照 entropy_weight ∈ {0, 0.1}，判断单调趋势）

### P1-2 baseline 评测（iTransformer / DLinear）
代码已接入 `create_ablation_model`（`--variants dlinear itransformer`），CPU 前向已验。
GPU 上与 P0-1 同协议跑 6 数据集 × 8 seed：

```bash
python run_large.py gen --datasets traffic electricity etth1 ettm1 weather exchange ili \
    --variants dlinear itransformer \
    --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 --output_dir ./output_large_v3_baseline
```

---

## 6. 第六步：结果反馈

请把以下内容原文发回（不要总结、不要省略数字）：

1. 验证靶场 `output_large_v3/` 下的 `improvement_*.md` 汇总表全文
   （weather/electricity 的 full_v2_fixed 相对 patchtst 提升率与 Wilcoxon 显著性）；
2. 各 shard 的 `results_shard*.csv`（或至少其中的 MSE 列）；
3. 中途任何报错 / 参数修改（如改 batch_size），说明改了什么、为什么；
   同一组实验内除指定差异外参数必须完全一致。

**不需要**做额外分析或下结论，原始数据发回即可，分析由主控侧完成。

---

## 常见问题

- **Q: 可以并行跑多个 shard 吗？**
  A: 可以，`run_large.py` 的 shard 设计就是为多卡并行（每卡一个 shard）。但同一 shard 内
  单 job 在独立 spawn 子进程执行，shard 之间抢显存自行调度；显存不足时加 `--batch_size` 降低。
- **Q: 跑一半断了怎么办？**
  A: 重新执行 run 命令即可，结果 csv 按 (dataset, pred_len, variant, seed) 去重续跑，不会重复计算。
- **Q: 可以跳过验证靶场直接跑主表吗？**
  A: 可以合并，但不建议。门 2 判据是决策点：验证靶场结果决定后续资源投向。
- **Q: 旧文档 (v2) 的命令还能用吗？**
  A: 不能。CausalCIT_demo 的数值 bug 验证已完成，旧协议已归档可从 git 恢复。
- **Q: 根因 3（语义环境切分）什么时候修？**
  A: 不在本批 GPU 任务内。修 C 需要时间戳/真实数据评估 weather/electricity 可行性，
  属 0 GPU 工作，由主控侧另行安排。

---

## 历史记录

- **v2 (2026-08-08)**：CausalCIT_demo 门控数值 bug（`-1e9` 硬 mask）修复验证，3-seed × ETTh1；
  已执行完毕，结论归档（修复方向正确，后续被更严重的 spawn seed bug 掩盖）。
- **v3 (2026-08-11)**：全面重写。反映门 1 诊断 + 修 A+B + spawn seed 修复后的最新协议，
  主线为验证靶场（门 2 判据）+ P0-1 主表重跑 + P1 系列。
