# 01_proposal — 跨环境风险厌恶的时序通道建模 (DRO 式)

> 编号: idea 04 (`dro_risk_aversion`)
> 状态: 提案 (待 review)
> 日期: 2026-08-12
> 上游: `00_spark.md`; 脑暴源 `00_inbox/2026-08-11_new_directions.md` 想法 1;
>       语义环境可行性 `CausalCIT_ablation/docs/diagnostics/2026-08-12_env_split_feasibility.md`

---

## 1. 动机

CausalCIT 的完整证据链 (2026-07~08):
1. **识别对、没利用上**: 门控能区分因果/虚假边 (syn_ood 上因果边门控权重高 20–100×),
   但 MSE 反而 -1.21%; PCD 对比 `full_v2 ≈ pcd_gate` (差异<0.001)。
2. **收益来自容量而非门控**: `capacity_match ≈ full_v2` → 通道混合本身带来提升,
   稳定性门控没有额外贡献。
3. **ERM 不奖励稳定性**: 训练目标只有逐样本 MSE, 跨环境稳定通道对的"稳定"特征
   对 loss 无梯度激励 → 门控学到的稳定性信号无法转化为收益。

推论: **架构层的不变性约束会被 ERM 优化掉**。IRM / DRO / 风险最小化谱系的教训是
不变性必须显式出现在目标函数中。因此把"跨环境稳定性"写进训练目标,
而不是继续在架构上加门控。

## 2. 方法

### 2.1 目标函数

```
L(θ) = E_e[ ℓ_e(θ) ] + λ · Var_e[ ℓ_e(θ) ]
```

其中 e 遍历**语义环境** (季节 / 昼夜 / 一天时段, 按数据频率选择), ℓ_e 为该环境下
batch 的 MSE。λ 控制风险厌恶强度 (λ=0 退化为 ERM)。

可选变体 (与标准 DRO 谱系对应):
- `L = max_e ℓ_e` (min-max, 最保守);
- `L = E_e ℓ_e + λ·std_e ℓ_e` (mean-variance, 等价上述 Var 形式);
- `L = CVaR_α(ℓ_e)` (条件风险值, 尾部环境加权)。

### 2.2 架构

采用 **capacity_match** 的完整通道注意力 (无稳定性门控):
- 无坍缩问题、无 batch 依赖、无门控超参 (temperature/prior_weight/stability_bias 全部消失);
- 通道混合的"准入判据"由损失函数的环境风险惩罚隐式提供。

### 2.3 环境定义 (直接复用修 C 的语义切分)

- 数据层: Dataset 保留 date → 解析语义标签 (season / daynight / tod), 返回 `(x, y, env_label)`;
- 训练: 每个 batch 按标签分组, 分组计算 ℓ_e;
- 语义方案经 `assess_env_split.py` 验证有信息 (ETTh1 昼夜 13.7×, weather 季节 4.2× vs 随机)。

### 2.4 与 CausalCIT 的继承关系

| 组件 | 来源 | 复用方式 |
|------|------|---------|
| 语义环境切分 | 修 C (本提案核心前置) | `env_mode='semantic'` 数据管线 |
| 通道注意力 | `capacity_match` 变体 | 原样 |
| HSIC / RFF | CausalCIT | 暂不用于本提案主目标 (可在诊断中对比) |
| run_large 基础设施 | CausalCIT_ablation | 原样 (新增 `--risk_lambda`) |

## 3. 创新点与差异化

1. **不变性约束从架构层移到目标层** (MTSF 领域少见的 framing): 现有通道交互方法
   全部在架构上做文章 (iTransformer 相关性注意力 / Crossformer Router / CausalCIT 门控),
   本方法首次把"跨环境稳定"显式编码进 loss, 与 IRM 思想对齐。
2. **语义环境的实证支撑**: 用数据验证了语义环境切分的信息量 (本次 0 GPU 产出),
   而非拍脑袋选环境。
3. **简单性即卖点**: 无门控超参、无坍缩、无 batch 依赖 —— 直接回应 CausalCIT
   的批评 (可复现性 / 超参脆点)。

## 4. 关键实验 (决策门 门3)

| # | 实验 | 判据 |
|---|------|------|
| 1 | weather/electricity 上 λ ∈ {0, 0.1, 1} 消融 | λ 增大 → 增益单调或不减 (不要求单调, 要求存在 λ*>0 显著优于 λ=0) |
| 2 | traffic 验证 (高维漂移场景) | full_v2_fixed 已翻正, 若 DRO 在 traffic 也 ≥ full_v2_fixed → 新主线成立 |
| 3 | 环境消融: season vs daynight vs tod | 选择信息量最大的方案 (结合 assess_env_split 结果) |
| 4 | 对照: min-max vs mean-variance vs CVaR | 选最稳形式, 防过拟合单一 DRO 变体 |

**止损规则**: 若所有 λ>0 在 weather+electricity 上都 ≤ λ=0 (ERM), 直接放弃,
不投入主表规模; 转而考虑想法 2 (可逆解耦, 调研支持最强)。

## 5. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 环境内 batch 样本不足 → ℓ_e 高方差 | 环境按语义标签跨 batch 聚合 (EMA) 而非单 batch; 或用全部训练数据的固定环境划分 |
| λ 选择脆 | 3 档消融 + 最小描述长度式早停; 明确写止损线 |
| 语义环境外推弱 (weather 仅 4×) | 若 ℓ_e 在语义环境下无区分, 提前放弃并记录 (避免无限加方案) |
| 与 CausalCIT 门控重复造轮子 | 明确本提案不叠加门控; 对比基线含 full_v2_fixed |

## 6. 0 GPU 前置已完成 / 待办

- [x] 语义环境切分可行性评估 (`2026-08-12_env_split_feasibility.md`): 语义有信息 (4–14×);
- [x] 立项 (本文档);
- [ ] review: 与想法 2 (可逆解耦) 的成本/收益对比, 决定并行线优先级;
- [ ] 实现: `env_mode='semantic'` 数据管线 (修 C 的实现同时服务本提案);
- [ ] GPU: λ 消融 (决策门 门3)。

## 7. 一句话

**把"跨环境稳定"从架构愿望变成损失函数要求 —— 环境有信息 (已实证), 目标有约束 (DRO), 收益应可见。**
