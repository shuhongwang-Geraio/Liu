# 与 Baseline / 竞品的差异论证（论文 Related Work 与实验差异章节草稿）

> 依据: `04_baseline_literature/report_final.md`（2026-08-10 调研）+ `method_assessment.md`（2026-08-08 实验）。
> 状态: 草稿。**投稿前需用 P0-1 重跑后的数字替换所有性能引用。**
> 写作原则: 诚实界定边界（场景依赖的有效改进），不宣称"全面超越"。

---

## 1. 一句话定位

CausalCIT 与所有主流通道交互方法（iTransformer、Crossformer、SOFTS、CSformer、Adapformer）
的根本区别在于**通道交互的准入判据**：

> 现有方法以**相关性强度**（训练分布下的统计相关）驱动通道混合 —— "只要相关就交互"；
> CausalCIT 以**跨环境稳定性**（HSIC 度量的相关性在不同环境切分下是否保持）作为准入门槛 ——
> "只有跨环境稳定（暗示因果）的通道对才允许交互"。

这一哲学差异对应三个层面的机制区别（下表）：

| 层面 | 现有方法 (iTransformer 等) | CausalCIT |
|------|---------------------------|-----------|
| 信号来源 | 训练分布内的统计相关性 | 跨环境不变性度量 (HSIC) |
| 交互方式 | 全量 / 启发式混合 | 基于因果判据的选择性交互 |
| 泛化保障 | 经验风险最小化 (ERM) | 跨环境不变性约束 |

---

## 2. 与各 baseline 的具体差异及实验预期

### 2.1 PatchTST (backbone) —— CI 基线
- **差异**: PatchTST 完全不做通道交互 (CI)；CausalCIT 在 CI 之上选择性恢复因果通道交互。
- **实验上如何体现**: 高维多通道 + 短中 horizon (traffic pl96/192、electricity pl96) 上，
  CausalCIT 显著优于 PatchTST (+7.9% / +3.9%, Holm p<0.05)；低维弱依赖 (ETTh1, ILI)
  上两者接近或 CausalCIT 更差 —— 门控退化为噪声，与假设一致。
- **为什么**: 稳定性门控抑制了随漂移消失的虚假相关通道对，只保留稳定因果通道对的信息。

### 2.2 iTransformer —— 相关性驱动的全量交互（直接对立面）
- **差异**: iTransformer 用倒置 self-attention 让所有通道对按相关性强度全量混合；
  CausalCIT 用 HSIC 稳定性门控做选择性混合。
- **实验上如何体现**: 在**分布漂移/环境波动剧烈**的场景 (traffic 空间相关性随天气/节假日漂移、
  electricity 用电模式季节漂移)，iTransformer 的通道注意力权重由训练统计相关决定，
  测试分布变化时可能把错误的高相关通道对放大；CausalCIT 的门控仅保留跨环境稳定的通道对。
  预期: traffic/electricity 上 CausalCIT > iTransformer；低维稳定场景 iTransformer 的
  全局建模能力可能更占优（已作为待跑 baseline，结果待 P1-2 补）。
- **可证伪点**: 若在 traffic 上 iTransformer 与 CausalCIT 持平或更好，则"稳定性门控
  优于相关性门控"在真实数据上不成立 —— 我们在 P1-2 直接对标检验。

### 2.3 DLinear —— 极简 CI（放弃交互换取稳定）
- **差异**: DLinear 完全不交互 (CI) + 分解，以线性表达换取稳定性；
  CausalCIT 尝试"有选择的交互"。
- **实验上如何体现**: 强非线性通道因果依赖场景下 CausalCIT 应优于 DLinear
  (DLinear 无法建模跨通道非线性)；低维/近似独立时两者接近。已接入 baseline（P1-2 待跑）。

### 2.4 Crossformer —— 如何交互（效率） vs 是否交互（准入）
- **差异**: Crossformer 用 Router/TSA 解决"如何高效交互"；
  CausalCIT 解决"是否应该交互"（准入判据）。
- **实验上如何体现**: 依赖结构随环境漂移时，Crossformer 的 Router 仍会路由那些
  高相关但随环境漂移的虚假连接；CausalCIT 的稳定性门控将其过滤。
  预期在 traffic 类高维漂移场景 CausalCIT 更鲁棒。

### 2.5 Adapformer —— 直接竞品（"自适应" vs "因果稳定"）
- **差异**: Adapformer 的通道管理 (ACE/ACF) 是 **ERM 框架下的相关性自适应**；
  CausalCIT 是 **HSIC 跨环境稳定性准入**。"任务相关 ≠ 因果稳定"。
- **实验上如何体现**: 在存在环境干扰（传感器故障、政策突变、季节性政策变动）的数据集上，
  Adapformer 的适配器可能过拟合到下游环境特定特征，CausalCIT 门控则保持不变性。
  预期在漂移剧烈场景 CausalCIT 鲁棒性更强。代码未发布，暂无法直接对标
  （可用其公开数字作定性对照）。
- **诚实边界**: 目前无 Adapformer 直接对照实验，该差异论证是机制层面 + 定性预期，
  不作为主结论依据。

### 2.6 CSformer —— 先 CI 后 CD（启发式混合 vs 理论准入）
- **差异**: CSformer 在架构上承认 CI 价值（第一阶段 CI），但第二阶段 CD 是全量启发式混合；
  CausalCIT 为 CD 阶段的交互范围提供了 HSIC 理论准则。
- **实验上如何体现**: 漂移场景下 CSformer 的 CD 混合仍会引入训练集特有虚假依赖；
  CausalCIT 用稳定性判据限定混合范围。代码未发布，暂定性对照。

### 2.7 TimeXer / SOFTS / ModernTCN（简要）
- TimeXer: 依赖**人工指定**内生/外生变量角色；CausalCIT 是**数据驱动**的因果发现。
  当外生变量因果链断裂（政策突变）时，CausalCIT 门控能自动切断失效交互，TimeXer 缺乏
  自适应切断机制。
- SOFTS: 全局统计聚合 (STAR)，"交互的效率"；CausalCIT 是"交互的质量与稳定性"。
  异常通道"毒化"全局表示时 CausalCIT 门控可隔离。
- ModernTCN: 固定卷积核的静态通道混合；CausalCIT 稳定性门控可作为插件提升 OOD 泛化。
  （这几项属 Related Work 定性对比，不作为主实验对标。）

---

## 3. 与 Related Work 的关系（稳定学习/不变学习谱系）

| 工作 | 机制 | 与 CausalCIT 的关系 |
|------|------|---------------------|
| FOIL (ICML'24) | 时序环境推断 + 不变风险最小化 | 特征层不变性 vs 通道交互关系层稳定性；其环境推断可为我们的环境切分提供参考 |
| StableNet (CVPR'21) | RFF 去相关 + 样本重加权 | 跨环境稳定学习的先驱，我们把稳定性思想迁移到通道交互选择 |
| Koopa (NeurIPS'23) | Koopman 算子的时不变/时变分解 | 概念呼应（稳定 vs 虚假），但关注动力学分解而非通道选择 |
| COGS (AAAI'26) | 因果表示学习 + 显式因果图 | 全局因果图搜索（重）vs 轻量 HSIC 门控 |

关键区分句（论文可用）:
"Unlike FOIL/COGS which enforce invariance over the learned feature space,
CausalCIT enforces invariance over the **channel-interaction graph** —
the set of inter-channel dependencies that survive across environments.
This is the first time, to our knowledge, that cross-environment stability
is used as the admission criterion for channel mixing in MTSF."

### 3.1 PCD (ICASSP'26) — 从"威胁"转"资产"（方案 4，2026-08-12 落实）

PCD（Pattern-Cognizant Decoupling，ICASSP'26）独立报告**与我们完全相同的维度规律**：
高维数据集（PEMS，170–883 通道）+12.7%~40.2%，低维数据集（ETTh1/2）仅 0.3%~2.8%。
而我们的主表（旧协议）同样显示 traffic/electricity（高维）显著为正、ETT/ILI（低维）不占优。

**论文措辞（不回避、主动引用）**:
> "两项独立工作在完全不同的实现（PCD: 静态相关掩码; CausalCIT: RFF-HSIC 跨环境稳定性门控）
> 与不同数据集族上观测到同一维度规律 —— 这提示 **'通道交互的价值取决于数据集依赖结构'
> 是一个真实现象**，而非某一方法的实现细节。我们的贡献是给出**解释该现象**（跨环境稳定性门控）
> 并能**事先判定**它何时有效的诊断框架（训练前适用性判据，见论文 §X）。"

这样 PCD 从"我们的机制与简单掩码打平"的威胁，变成"现象真实存在"的独立佐证。

---



## 4. 论文中如何呈现（推荐结构）

1. **Related Work 段**: 用上面"信号来源/交互方式/泛化保障"三列对比表 + 一段文字。
2. **实验段**: 主表补 iTransformer/DLinear 两列（P1-2 待跑）→ 在 traffic/electricity
   上与 CausalCIT 直接对标，证明"稳定性门控优于相关性门控/全量交互"。
3. **Limitation 段**: 明确低维/长 horizon 不占优（方法假设的预期行为）；
   Adapformer/CSformer 无代码，仅定性对照。
4. **诚实红线**: 任何"全面超越"表述都要删除；只报告场景依赖的有效改进
   + 机制消融（no_env 显著变差 + gate_prior_only 坍缩）证据链。

---

## 5. 待办（写进 do.md）

- [ ] P1-2: traffic/electricity/ETTh1 上跑 iTransformer、DLinear 与 CausalCIT 直接对标
      （6 数据集 × 8 seed，与 P0-1 同协议）。
- [ ] 读 Adapformer / CSformer 论文原文，确认 benchmark 数字与机制细节（调研 agent 已给摘要，
      投稿前需读原文确认引用准确）。
- [ ] P0-1 重跑后，用新数字重写本文件的性能引用。
