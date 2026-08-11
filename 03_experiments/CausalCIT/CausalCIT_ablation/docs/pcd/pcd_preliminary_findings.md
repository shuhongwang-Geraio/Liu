# PCD 对比实验 — 初步发现 (2026-08-11)

> 相关: `docs/pcd/pcd_compare_argument.md` (实验设计), `method_assessment.md` (整体评估),
> `output_pcd_full/` (GPU 机器 5-seed 正式版, 进行中), `output_pcd_smoke/` (3-epoch smoke)。

## 1. 数据来源

| 来源 | 变体 × seed | epoch | 说明 |
|------|-------------|-------|------|
| `output_pcd/ckpt/` (50-epoch, 已有) | patchtst ×4 (42/123/2024/5), full_v2 ×2 (42/123), pcd_gate ×2 (42/123) | 50 | 本机此前训练, 无 results.csv 汇总 |
| `output_pcd_smoke/` (3-epoch) | 3 变体 × 2 seed | 3 | 协议 smoke, 2026-08-10 |

本文档基于**已有 50-epoch checkpoint 的 test 集评估** (syn_ood pl96, OOD test: 虚假相关反转)。

## 2. 结果 (MSE, syn_ood test)

| 变体 | 50-epoch ckpt (seeds) | 3-epoch smoke (s42/123) |
|------|------------------------|--------------------------|
| patchtst | 0.3189 / 0.3189 / 0.3194 / 0.3180 | 0.320446 |
| full_v2 | 0.32084 / 0.32067 | 0.320754 |
| pcd_gate | 0.32054 / 0.32034 | 0.320441 |

## 3. 初步结论

1. **full_v2 ≈ pcd_gate** (差异 < 0.001, 50-epoch 与 3-epoch 两个协议一致):
   在 syn_ood 上, PCD 注入的虚假相关结构 (情形 A/B/C 设计) **没有让稳定性门控产生可测收益**。
2. **门控类方法 (full_v2 / pcd_gate) 在 syn_ood 上均略差于纯 PatchTST** (~0.5–0.8%):
   与 `method_assessment.md` 记录的 full_v2 在 syn_ood 上 −1.21% 显著变差一致。
3. 即: **稳定性门控在真实高维数据 (traffic/electricity) 有效, 但在 syn_ood 这个
   "机制测试" 场景里没有兑现"切断虚假边带来 OOD 收益"的承诺** —— 这正是 P2 需要排查的
   方向: 是 syn_ood 的构造 (spurious_strengths 配置) 未触发门控的区分能力,
   还是门控的容量/训练问题, 而非简单的"门控有效"。

## 4. 对论文叙事的影响

- 这一发现**限制**了论文可以宣称的范围: 不能宣称"因果门控带来 OOD 鲁棒性" (syn_ood 机制
  测试未通过); 只能报告"在高维真实数据上的场景依赖改进 + 门控行为诊断证据链"。
- 与 do.md P2 "syn_ood 负结果排查" 合并处理。

## 5. 待办

- [ ] GPU 机器上完成 `output_pcd_full/` 的 5-seed 完整版 (patchtst/full_v2/pcd_gate × 5 seeds),
      并补全至 5 seeds 做配对 Wilcoxon。
- [ ] 逐情形 (A/B/C) 检验: 目前混在一起; 需单独看"虚假边跨环境反转"下门控是否切断。
- [ ] 排查 syn_ood 构造: 若门控理论上应区分而实际无差异, 检查 spurious_strengths 是否
      足够强/测试反转是否触发 (与 do.md P2 相同条目合并)。
