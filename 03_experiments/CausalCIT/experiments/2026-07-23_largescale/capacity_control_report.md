# 控制容量对照实验报告 (Capacity-Control Ablation)

> 2026-07-23 · 直接回应评委质疑 1.3 ("参数膨胀 25 倍: 可能只是大模型胜利").
> 数据: output_capacity/ (learned_gate) + output_traffic/ + output_large/ (其余变体). 0 错误, 3 seeds.

## 0. 动机
reviewer_critique.md §1.3: full_v2 在 traffic 上 776k vs PatchTST 31k (~25x), 而 no_gate (32k) 只 +4.1%, 故"增益可能来自容量而非因果逻辑".
目的: 隔离 (a) 参数容量 (b) 因果稳定性逻辑. 设计 learned_gate —— 与 full_v2 参数严格匹配, 但门控由纯学习 N×N 矩阵决定、不走 HSIC/因果逻辑.

## 1. 实验设计
| 变体 | 门控来源 | 因果 | 参数量级 (traffic/electricity) |
|---|---|---|---|
| patchtst | 无(CI) | - | ~31k/~17k (小) |
| no_gate | 标准注意力无门控 | 否 | ~32k/~32k (小) |
| learned_gate | 纯学习 N×N (sigmoid(0.05·prior)) | 否 | 776k/186k (大, ∝N²) |
| full_v2 | 因果稳定性门控 (gate_mlp(HSIC)+同规模prior) | 是 | 776k/186k (大, ∝N²) |

learned_gate 与 full_v2 骨架完全相同, 参数差仅 51 (traffic). 唯一区别: gate 由纯学习矩阵 vs HSIC 驱动的 gate_mlp.

## 2. 结果 (MSE, 3 seeds mean)
### Traffic (862 通道)
| horizon | patchtst | no_gate(32k) | learned_gate(776k) | full_v2(776k) |
|---|---|---|---|---|
| pl96 | 0.5559 | 0.5328(+4.14%) | 0.5024(+9.62%) | 0.4952(+10.92%) |
| pl192 | 0.5432 | 0.5270(+2.98%) | 0.5222(+3.87%) | 0.5104(+6.05%) |
### Electricity (321 通道)
| horizon | patchtst | no_gate(32k) | learned_gate(186k) | full_v2(186k) |
|---|---|---|---|---|
| pl96 | 0.1718 | 0.1654(+3.72%) | 0.1614(+6.02%) | 0.1634(+4.87%) |
| pl192 | 0.1767 | 0.1734(+1.88%) | 0.1732(+2.02%) | 0.1735(+1.83%) |

## 3. 关键发现
### 3.1 参数容量是增益主驱动 (证实评委 1.3)
小参数 no_gate(~32k) 增益远小于大参数变体(186k-776k): traffic pl96 +4.14%→+9.6%/+10.9%; elec pl96 +3.72%→+6.0%/+4.9%. "可学习 N×N 大容量门控"本身就是同分布增益主因, 随 N² 单调上升.

### 3.2 因果逻辑 vs 纯学习(同容量): 同分布无稳定优势
learned_gate vs full_v2 (参数匹配):
- traffic pl96: full_v2 +10.9% > learned +9.6% (因果优 1.3%)
- traffic pl192: full_v2 +6.05% > learned +3.87% (因果优 2.2%)
- elec pl96: learned +6.02% > full_v2 +4.87% (纯学习优 1.2%)
- elec pl192: learned +2.02% ≈ full_v2 +1.83%
净结论: 同分布下因果稳定性门控相对同等容量纯学习门控无一致显著额外收益 (~1-2%, 方向不一致).

## 4. 对论文初稿的冲击 (必须修改)
- §5.4 核心论证被证伪: "full_v2 相对 no_gate 额外增益证明 v2 改进是结构性而非容量" —— 实际可由参数量(776k vs 32k)解释; learned_gate(同776k,无因果)同样>no_gate 且≈full_v2.
- §5.1/§5.2 公平性声明需重写: 不能只对比 PatchTST 声称因果有效; 同分布下任何同等容量通道门控都能达类似增益.
- "CausalCIT" 同分布优势叙事需降级.

## 5. 结论与生死线: OOD
full_v2 相对 learned_gate 的唯一可能差异化价值在 OOD 鲁棒性. learned_gate 无约束会过拟合训练分布→OOD 退化; full_v2 因果约束应抑制过拟合→OOD 保持. 这正是 IRM/StableNet 经典模式.
- 若 OOD 上 full_v2 >> learned_gate → 论文成立 (因果约束=分布鲁棒性, 相对 Adapformer/普通注意力的真正差异化).
- 若 OOD 上不优 → 核心论点崩塌.

## 6. 下一步 (P0)
1. 立即补 OOD 实验 (ILI-COVID/Exchange/合成漂移), 四变体全跑. 决定论文生死.
2. 把 learned_gate 作为"容量匹配过拟合对照": 若 OOD 下 learned_gate 崩溃而 full_v2 不崩, 即证因果约束价值.
3. 修正 §5.4: 删"v2 改进是结构性而非容量", 改"容量必要, 但因果约束是 OOD 鲁棒性来源".
4. method 部分诚实说明参数量随 N² 膨胀, 讨论其为因果门控代价但 OOD 收益补偿.

## 复现
- 变体: models_ablation.py → PureLearnedGate / LearnedGate_ChannelInteraction / 工厂 'learned_gate'
- 生成: python run_large.py gen --datasets traffic electricity --variants learned_gate --num_shards 3 --output_dir ./output_capacity
- 合并报告: output_capacity_full/ → large_scale_report.md + improvement_heatmap.png
