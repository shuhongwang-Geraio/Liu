# syn_ood 识别-利用脱节排查方案 (2026-08-12)

> 对应: `07_scope_and_publication_risk_analysis.md` 方案 3b; do.md P2 "syn_ood 负结果排查"。
> 前置代码已就绪: `run_large.py` 新增 `--alpha_init` / `--fusion_alpha` 透传 (2026-08-12, CPU 验证通过)。
> 状态: 方案已定, **GPU 待跑** (P0-1 主表完成后排)。

## 1. 背景与线索

syn_ood 机制测试未通过: `full_v2` MSE **−1.21%** 显著变差; PCD 对照打平
(`full_v2 ≈ pcd_gate`, 差异<0.001)。但同一批实验**门控结构识别成功**:
因果边相对虚假/独立边高 0.10–0.35 (对照组 20–100 倍), 且未坍缩。

→ **"识别对了, 却没转化成预测收益"** 的识别-利用脱节。

工程上两个最可能的利用侧原因:
1. `alpha_init=-2.0` → `sigmoid(-2)≈0.12`, 通道混合分支初始几乎关闭;
2. 残差融合 `fusion_alpha=0.3` 的稀释效应 (混合分量只占 30%)。

## 2. 排查实验设计

在 syn_ood (pl96) 上, 固定其余参数与 full_v2_fixed 协议一致, 单因子扫描:

| 轮次 | 变量 | 取值 | 目的 |
|------|------|------|------|
| 1a | `--alpha_init` | {-2.0, -0.5, 0.0, 1.0} | 混合分支初始权重是否过低 |
| 1b | `--fusion_alpha` | {0.1, 0.3, 0.5, 1.0} | 残差融合是否稀释机制收益 |
| 2 | 组合 (若 1a/1b 有信号) | alpha_init∈{0,1} × fusion_alpha∈{0.5,1.0} | 确认叠加效应 |

判据:
- **任一配置 syn_ood 提升转正 (或显著优于 patchtst)** → 机制成立但利用不足 → 可修,
  问题从"机制不成立"变为"下游利用不足";
- 全负 → 支持"机制本身在 syn_ood 构造下无收益", 配合构造侧审查 (方案 3a)。

## 3. GPU 命令 (P0-1 完成后执行)

```bash
cd <项目根目录>/03_experiments/CausalCIT/CausalCIT_ablation
for ai in -2.0 -0.5 0.0 1.0; do
  python run_large.py gen --datasets syn_ood --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 \
    --alpha_init $ai --output_dir ./output_synood_alpha
done
# 每卡一 shard: python run_large.py run --device cuda:0 --job_file ./output_synood_alpha/jobs_shard0.json --result_csv ./output_synood_alpha/results_shard0.csv
# (fusion_alpha 同理 --output_dir ./output_synood_fusion)
# 汇总: python run_large.py summarize --output_dir ./output_synood_alpha
```

注意事项:
- 只动 syn_ood + full_v2_fixed, 不污染主表输出目录;
- 每轮次 8 seed, 与主表同协议 (seed 配对对比);
- `--alpha_init` / `--fusion_alpha` 已透传并经 CPU gen 验证 (job 内 `model_kwargs` 正确写入)。

## 4. 若翻正 → 后续动作

- 把最优 (alpha_init, fusion_alpha) 纳入论文正式协议;
- syn_ood 从"机制测试失败"变为"机制在正确利用下成立" → 论文 OOD 叙事升级;
- 该最优配置需在 traffic/electricity 上复查不引入退化 (防止过拟合 syn_ood)。

## 5. 若仍全负 → 止损转向

- 结合方案 3a (构造侧审查: 虚假通道在测试期是否主动有害);
- 若构造也无问题 → 机制测试结论定为"不成立", 触发 claim 降级决策 (方案 5, do.md P2)。
