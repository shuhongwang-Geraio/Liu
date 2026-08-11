# 门 1 静态诊断结果 (2026-08-11)

> 脚本: `diagnose_gate_static.py` (零训练成本, 随机初始化 + 单 batch 前向)。
> 方法: 同一批 syn_ood 数据, 三个 d_model (16/32/64 对应 traffic/electricity/weather 配置),
> 重现 `compute_stability_score_v2` 中间量。

## 结果

| 指标 | d_model=16 | d_model=32 | d_model=64 | 判定 |
|------|-----------|-----------|-----------|------|
| proj = x@W 的 std | 3.99 | 5.92 | 7.49 | **根因 1 确认**: ≫1 且随 d_model 单调(≈√d_model) |
| hsic_mean 动态范围 (ratio) | 1.2× | 1.1× | 1.1× | 核坏→ HSIC 几乎无通道区分度 |
| log(hsic) 方差占比 | 88% | 79% | 69% | **根因 2 确认**: 稳定性(cv)项是装饰品 |
| cv 分布 (mean) | 0.049 | 0.037 | 0.041 | **根因 3 确认**: 环境切分几乎无信息 |

## 结论

1. **根因 1 (RFF σ 硬编码) 定性成立**：proj.std ∈ [3.99, 7.49]，cos(proj) 剧烈震荡，
   RFF 特征退化为伪随机向量。且随 d_model 单调 —— 与"效果对 d_model 完全单调"
   (traffic d16 +7.9% > electricity d32 +3.3% > weather d64 −0.6%) 精确对应。
   理论值 sqrt(d_model) = {4, 5.66, 8}，实测 {3.99, 5.92, 7.49}，符合"x~N(0,1) 归一化后 x@W, W~N(0,1)"的预言。
2. **根因 1 的后果比预想更严重**：hsic_mean 动态范围仅 1.1–1.2× —— 不是"稳定性信号被淹没"，
   而是**依赖强度信号本身就没有区分度**（核坏 → 所有通道对的 RFF 内积都≈常数）。
3. **根因 3 确认**：cv≈0.04，环境切分 (窗口内均分, env_size=3) 在 syn_ood 上无信息。

## 对决策门的含义 (do.md 门 2)

- 门 2 判定：**根因 1 成立** → 走「修 A(median heuristic) + B(CKA 归一化)」分支。
- 修复方向明确：
  - **A**: RFFKernel 带宽用 median heuristic (σ = 中位数成对距离), 使 proj.std ≈ 1;
  - **B**: HSIC 做 CKA 归一化 `HSIC/√(HSIC_xx·HSIC_yy)`, 使不同通道对可比;
  - 修完 A 后必须复查 hsic_mean 区分度 (目标 ratio 从 1.1× 提升到 ~10×+) 与 cv 分布。
- 关键预测: traffic(d16) 核"最不坏"→ 修复后提升幅度可能收窄; weather(d64)/electricity(d32)
  核最坏 → 修复后负收益有翻正潜力。**weather/electricity 是修复的验证靶场。**

## 更新：修 A+B 已实现并 CPU 验证 (同日)

代码: `CausalCIT_demo/models/causal_channel.py`
- **A**: RFFKernel 新增 `sigma_mode='median'` (median heuristic, 首次 forward 采样估计 σ);
- **B**: `compute_stability_score_v2` 新增 `cka_normalize=True` (HSIC/√(HSIC_xx·HSIC_yy));
- 均默认关闭 (不破坏旧行为), 修复版需显式开启。验证脚本: `_verify_gate_fix.py`。

结果 (syn_ood 同批数据):

| d_model | mode | σ | proj.std | hsic 区分度 ratio | log(hsic) 占比 | cv.mean |
|---------|------|------|----------|--------------------|----------------|---------|
| 16 | fixed | 1.0 | 3.47 | 1.65× | 97% | 0.062 |
| 16 | median+cka | 4.22 | **1.07** | **9.02×** | 100% | 0.005 |
| 32 | fixed | 1.0 | 5.52 | 1.12× | 72% | 0.045 |
| 32 | median+cka | 6.86 | **0.78** | **5.98×** | 100% | 0.004 |
| 64 | fixed | 1.0 | 7.89 | 1.10× | 73% | 0.042 |
| 64 | median+cka | 9.45 | **0.84** | **5.61×** | 100% | 0.006 |

**解读**:
1. proj.std → [0.78, 1.07] ≈ 1: 根因 1 修复生效, 核恢复有效工作区间;
2. HSIC 区分度 1.1× → 5.6–9×: 门控终于能区分不同通道对的依赖强度 (因果链条根基恢复);
3. cv≈0.005: 根因 3 (非语义环境) 仍在 —— 稳定性项尚无信息, log(hsic) 占比 100%。

## 待办

- [x] 修 A+B (median heuristic + CKA 归一化), CPU 验证通过 (区分度 1.1× → 5.6–9×)
- [ ] GPU: 透传 `rff_sigma_mode='median', cka_normalize=True` 到 run_large,
      8-seed 重跑 weather/electricity (现有协议), 看负收益是否翻正 (验证靶场)
- [ ] 若稳定性项仍需信息: 修 C (语义环境切分) 需要时间戳/真实数据, 属下一步
