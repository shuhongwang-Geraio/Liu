# 熵正则 smoke test 记录 (2026-08-10, CPU, syn_ood)

> 目的 (用户指示"稍微验证一下"): 确认 `--entropy_weight>0` 代码接线是否真正生效、
> 对门控行为是否有可观测影响, 为 GPU 空闲后的正式实验 (P1) 提供依据。

## 配置

- 脚本: `run_minimal_falsifiable.py` (与 run_large 同一代码路径)
- 数据: `syn_ood` (内置合成数据, 不需 csv), pred_len=96, seed=42, epochs=2 (未收敛)
- 对比: `entropy_weight=0.0` vs `0.01`
  - baseline: `../output_entropy_smoke_0/minimal_falsifiable_report.md`
  - entropy: `./minimal_falsifiable_report.md`

## 结果: 门控行为 (off_mean / off_std, 越大越分化)

| 变体 | entropy=0 off_mean | entropy=0 off_std | entropy=0.01 off_mean | entropy=0.01 off_std | 熵正则是否生效 |
|------|--------------------|-------------------|-----------------------|----------------------|----------------|
| full_v2 | 0.1868 | 0.1619 | **0.9538** | 0.1673 | ✅ 明显生效 (门控被推向 0/1 极端) |
| full_v2_fixed | 0.1359 | 0.0208 | — | — | (未重跑, 与 full_v2 同门控路径) |
| gate_prior_only | 0.1756 | 0.0033 | 0.1756 | 0.0033 | ❌ 完全无变化 |

## 结论

1. **熵正则代码接线正确且对 full_v2 有效**:
   `--entropy_weight=0.01` 使 full_v2 门控均值从 0.19 升到 0.95 (sigmoid 输出被推向果断的 0/1 极端),
   方向符合"鼓励果断选择"的设计意图。正式实验 (traffic 8 seed) 值得跑。
2. **gate_prior_only 不受熵正则影响 (关键发现)**:
   `CausalCIT.get_gate_entropy()` 只从 `CausalStabilityGate.last_entropy` 取熵
   (`CausalCIT_demo/models/causal_channel.py:423-425`), 而 `gate_prior_only` 走
   `PriorOnly_ChannelInteraction`, 无该接口 → trainer 的 `gate_entropy is None` 分支跳过。
   ⇒ **"熵正则对症 gate_prior_only 坍缩"的假设不成立**, 该变体坍缩只能靠结构性修改解决
   (或接受其作为对照本就不该被救)。
3. **P0-2 判据不一致再次复现**: gate_prior_only 在 syn_ood 上 off_std=0.0033,
   按 `run_minimal_falsifiable.py` 的 `<1e-4` 判为"未坍缩", 按 `analyze_gates.py` 的 `<0.01`
   判为"坍缩"。统一判据仍是必须项。

## 备注

- smoke 仅 2 epoch/1 seed, 数字不用于正式结论; 正式实验需 traffic + 30 epoch + 8 seed。
- 若想让熵正则也覆盖 gate_prior_only 之类变体, 需给 `PriorOnly_ChannelInteraction`
  补 `last_entropy`/`get_gate_entropy` 接口 (属于代码修改, 待与 P0-2 一并处理)。
