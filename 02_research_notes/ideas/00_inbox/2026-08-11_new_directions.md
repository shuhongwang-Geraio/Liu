# 新方向脑暴归档 (2026-08-11)

> 来源: 2026-08-11 对 `01_external/` + `02_research_notes/` 的系统重读,
> 以及 `07_scope_and_publication_risk_analysis.md`、PCD 初步发现之后的讨论。
> 性质: 未拆分脑暴合集, 供后续按决策门拆分为独立 idea 文件夹。
> 关联: `07_scope_and_publication_risk_analysis.md`(范围/发表风险)、
>       `01_adaptive_channel/05_major_improvement.md`(三根因诊断)、
>       `ideas/03_invertible_decouple/04_report_final.md`(可逆解耦调研)。

---

## 0. 最重要的前置发现（先于一切新想法）

**CausalCIT 可能从未按设计运行过 —— 三个根因至今未修**（2026-08-11 代码核实，与
`05_major_improvement.md`(2026-08-06) 诊断一致）：

| 根因 | 代码位置 | 未修状态 |
|------|---------|---------|
| 1. RFF 核带宽硬编码 σ=1, 从不适配 d_model | `causal_channel.py:26` 默认 sigma=1.0; `:73` 不传 | ❌ 全库无 median heuristic |
| 2. 未归一化 HSIC 淹没稳定性信号 | `:181` `stability = hsic_mean/(1+cv+bias)` | ❌ 无 CKA 归一化 |
| 3. "环境"= 窗口内均分, 非真实机制变化 | `:136` `env_size = patch_num//n_envs` | ❌ 每环境仅 ~3 patch |

**推论（关键）**:
- 效果对 **d_model 完全单调**（traffic d16 +7.9% > electricity d32 +3.3% > weather d64 −0.6%），
  对通道数不单调 → "高维有效"很可能是"d_model 小→核未失效"的**混淆变量**；
- 根因 2 使门控退化 ≈ 相关性强度门控 → **`full_v2 ≈ pcd_gate` (差异<0.001) 不是机制无效的
  证据, 而是根因 2 的预言**;
- 结论修订: 机制**从未被公平检验**。"识别成功利用失败" 更可能源自 3 根因, 而非机制本身不成立。

---

## 1. 五个新想法（按推荐度）

### 想法 1 ★★★★★: 换目标函数, 不换架构 —— 跨环境风险厌恶 (DRO 式)
- **一句话**: 放弃门控式架构, 用完整通道注意力(capacity_match 已验证), 把因果性写进目标:
  `L = E_e[l_e] + λ·Var_e[l_e]` (或环境 CVaR / min-max); 环境用**语义定义**(季节/星期/工作日)。
- **依据**: capacity_match≈full_v2 → 收益来自混合容量而非门控选择; ERM 不奖励稳定性 →
  识别对了却没收益。IRM/DRO 的教训: 不变性必须在目标层。
- **优势**: 架构更简单(无坍缩/batch依赖/门控超参); 有 DRO 泛化理论; **把负面消融变成动机**;
  环境切分+HSIC 代码全可复用。
- **关键实验**: weather/electricity 上 λ∈{0,0.1,1} 对比; traffic 上验证。
- **风险**: 低。是 CausalCIT 失败的最自然继承者。

### 想法 2 ★★★★☆: 复活可逆正交解耦 (Idea 03) —— 免疫 CausalCIT 全部致命伤
- **一句话**: 可学习正交 $W$: $Z=XW$ → RFF-HSIC 约束 $Z$ 各维独立 → CI 独立预测 → $W^{-1}$ 还原。
- **依据**: `ideas/03_invertible_decouple/04_report_final.md` 判定 5 子问题中 3 个"未覆盖";
  通道维正交解耦是空白 (OLinear 在时间维); 无人比较"线性去相关 vs 统计独立"。
- **优势**: 无门控(无坍缩/batch依赖); 可逆变换在前向路径(必须使用); 机制可直接测(HSIC 热图);
  RFF-HSIC 代码可复用。
- **关键实验**: 消融 白化 vs 完整 RFF-HSIC (量化非线性依赖价值) — 天然空白对照。
- **风险**: 862 通道正交矩阵的开销/数值稳定; 需先在 weather(21)/electricity(321) 验证可行性。

### 想法 3 ★★★★☆: 通道交互失效模式审计 —— 把诊断工具箱变成论文
- **一句话**: 用 batch_dep_score / 坍缩判据 / capacity_match 容量对照 / PCD 静态对照 /
  RFF 带宽混淆, 审计 5–8 个开源通道交互方法, 检验"报告的是机制收益还是容量收益/假象/泄漏"。
- **依据**: 自己全部工具 + `01_external/` 现成代码 (iTransformer/SOFTS/Crossformer/CCM/TimeXer/
  ModernTCN/CMamba)。
- **优势**: 负面结果从负债变论据; d_model 混淆核带宽是可推广的普适警告。
- **风险**: 中。需确保审计协议严谨(否则自己会被反审)。

### 想法 4 ★★★☆☆: 滞后感知的跨环境稳定性
- **一句话**: 现有 HSIC 是同期(patch 内), 真实因果有传导延迟(traffic 行程时间差) →
  同期相关性系统性错过因果结构。补 滞后 维度 × 跨环境稳定性。
- **依据**: LIFT (arXiv 2401.17548) 做前导指标, 但"滞后×跨环境稳定性"组合无人做。
- **地位**: CausalCIT 的修复方向, 也可独立成题。

### 想法 5 ★★★☆☆: 多速率预测 (Idea 02, 换赛道备选)
- `03_final_no_rg.md` 已打磨成熟 (Interpolation-Free Multi-Rate via Continuous-Time
  Cross-Attention, 风电场景, 含 MVE 止损线)。核心风险: 打不赢 Concat baseline。
- 与当前主线距离远, 除非彻底换方向, 否则暂缓。

---

## 2. 方向把控：决策门（写入 do.md）

```
[门1] 第0步静态诊断 (0 GPU, 立即, ~30min)
      └─ 打印 proj.std / hsic_mean 动态范围 / log(hsic_mean) vs log(1/(1+cv)) 方差占比
         weather(d64)/electricity(d32)/traffic(d16) 各一次
         ↓
[门2] 根因1&2 定性成立?
      ├─ 是 → 修 A(median heuristic) + B(CKA) → 现有 8-seed 协议重跑 weather/electricity
      │        ├─ 负转正 → CausalCIT 抢救成功 → 主攻 CausalCIT 完善
      │        └─ 仍负   → 走 [门3] 想法1
      ├─ 否 → 机制真不成立 → 走 [门3] 想法1 (换目标函数)
      └─ 同时: 若想开新线 → 想法2 (可逆解耦, 调研支持最强) 或想法3 (审计)
[门3] 想法1 (DRO式) 快速验证: weather/electricity 上 λ 消融
      ├─ 有效 → 作为新主线 (继承 CausalCIT 基础设施)
      └─ 无效 → 止损, 转想法2/3 择一投入
```

**止损规则**（防"再补一个实验"无限循环）:
- 每个门都有明确判定标准与时限;
- 门2 的"仍负"分支, 最多再做一轮修复(C)即止损, 不再无限加变体;
- 想法1 若 λ 消融无单调趋势 → 直接放弃, 不投入主表规模实验。
