# CausalCIT 待办清单 (2026-08-11 更新)

> 本文档是**详细待办执行清单**（决策门、GPU 命令、止损规则）；
> 进度概览与"单一事实来源"见 `PROGRESS.md`（按 research-org 规范，会话结束必更新）。

> 上版 (2026-08-08) 之后的状态变化:
> - `output_large_v2/` (6 数据集×6 变体×8 seed = 720 结果) 与 `output_falsifiable_full/`
>   (traffic 门控诊断 80 条) 均已跑完, 汇总进 `method_assessment.md`。
> - 2026-08-08 修复了 `run_large.py` 的 spawn seed bug (见 P0-1)。
> - 2026-08-10 完成了多项 P1 准备工作 (见文末"本轮已完成")。
> - 2026-08-11 归档战略分析 (07_scope_and_publication_risk_analysis.md) 与
>   新方向脑暴 (00_inbox/2026-08-11_new_directions.md); 新增本节「方向决策门」。

## 0. 当前定位 (一句话, 2026-08-11 修订)

**CausalCIT 的机制可能从未被公平检验** —— `05_major_improvement.md` 诊断的三个根因
(RFF σ 硬编码 / 未归一化 HSIC / 非语义环境) 至今未修, 导致:
- "高维有效"很可能是 **d_model 混淆变量** (traffic d16 +7.9% > electricity d32 +3.3% >
  weather d64 −0.6%, 对 d_model 完全单调);
- 门控退化 ≈ 相关性强度 → `full_v2 ≈ pcd_gate` 打平是**根因的预言**而非"机制无效"。
因此当前所有"负收益/窄范围"结论**在跑完第 0 步静态诊断前都不构成最终判断**。
论文正式采用 `full_v2_fixed`; ⚠️ 数字仍基于 seed bug 修复前协议, P0-1 重跑前不可最终采信。

## 0.5 方向决策门 (最高优先, 2026-08-11)

**门 1 — 第 0 步静态诊断 (0 GPU, 立即, ~30min)**
- 内容: 随机初始化 + 单 batch 前向, 打印 `proj.std`、`hsic_mean` 动态范围、
  `log(hsic_mean)` vs `log(1/(1+cv))` 的方差贡献占比; weather(d64)/electricity(d32)/traffic(d16) 各一次。
- 脚本: 按 `05_major_improvement.md` 第 0 步写 (或复用 `CausalCIT_demo` 单 batch 前向)。

**门 2 — 根因 1&2 定性成立?** → ✅ **2026-08-11 门 1 诊断已确认** (见
`CausalCIT_ablation/docs/diagnostics/2026-08-11_gate_static_diagnosis.md`)
- proj.std = {3.99, 5.92, 7.49} 随 d_model 单调 (≈√d_model) → **根因 1 成立**;
- hsic_mean 动态范围仅 1.1–1.2× → 核坏导致依赖强度无区分度 (比预想更严重);
- log(hsic) 方差占比 69–88% → **根因 2 成立** (cv 项是装饰品);
- cv≈0.04 → **根因 3 成立** (环境切分无信息, 至少 syn_ood 上)。
→ ✅ **修 A+B 已实现并 CPU 验证** (2026-08-11): `causal_channel.py` 新增
  `rff_sigma_mode='median'` + `cka_normalize=True` (均默认关闭, 不破坏旧行为)。
  验证: proj.std 3.5–7.9 → 0.8–1.1; HSIC 区分度 1.1× → **5.6–9×**;
  cv≈0.005 (根因 3 仍在)。
→ ✅ **GPU 待办已就绪** (2026-08-11): 参数已透传全部链路
  (`run_large.FULL_V2_KWARGS` → `create_ablation_model` → `CausalCIT` →
  `CausalCIT_backbone` → `CausalChannelInteraction` → `CausalStabilityGate`),
  修复版 1-epoch 训练 smoke 通过 (median 初始化 + 反向传播 OK)。
→ **GPU 唯一待办**: 8-seed 重跑 weather/electricity (验证靶场), 看负收益是否翻正。
  注意: 修复版是新协议, 与 `output_large_v2` 旧数字**不可直接对比**;
  traffic(d16) 核"最不坏"→ 修复后提升可能收窄, 属预期。

**门 3 — 想法 1 (跨环境风险厌恶 DRO 式, ★★★★★)** — 换目标函数不换架构
- 快速验证: weather/electricity 上 λ∈{0,0.1,1} 消融; 无单调趋势即止损。
- 或并行开新线: 想法 2 (可逆解耦, 调研支持最强) / 想法 3 (失效模式审计)。

**止损规则**: 门 2 "仍负"最多再做一轮修复(C)即止损; 想法 1 无单调趋势直接放弃;
不无限追加变体。

---

## 一、GPU 空闲时优先跑 (按顺序)

### P0-1 ★重跑主表 (科学严谨性硬伤, 最高优先)
- 背景: spawn seed bug 已修复 (`_train_one`/`_train_syn_ood` 内补 `set_seed(job['seed'])`),
  但主表还是旧协议 (seed 从未真正控制随机初始化) 的结果。
- 动作: 用修复后的代码 + `full_v2_fixed` 重跑 6 数据集 × 8 seed:
```bash
python run_large.py gen --datasets traffic electricity etth1 ettm1 weather exchange ili \
    --variants patchtst full_v2_fixed capacity_match gate_prior_only no_env \
    --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 --output_dir ./output_large_v3
# 每张卡一个 shard
CUDA_VISIBLE_DEVICES=0 python run_large.py run --device cuda:0 --job_file ./output_large_v3/jobs_shard0.json --result_csv ./output_large_v3/results_shard0.csv
...
python run_large.py summarize --output_dir ./output_large_v3
```
- 验收: traffic/electricity 的显著提升在 seed 真正生效下依然成立; 把
  `plot_bootstrap_ci.py --results_dir ./output_large_v3` 重新出图 (新图替代旧图投稿)。

### P1-3 熵正则正式实验 (traffic)
- smoke test 已确认 `--entropy_weight>0` 对 full_v2 门控生效 (off_mean 0.19→0.95)。
- 动作: traffic 30 epoch × 8 seed, 对比 entropy_weight ∈ {0, 0.01, 0.1}:
```bash
python run_minimal_falsifiable.py --dataset traffic --seeds 42 123 2024 7 13 99 2023 31 \
    --device cuda:0 --entropy_weight 0.01 --output_dir ./output_falsifiable_entropy
```
- 注意: `gate_prior_only` 无门控熵接口, 熵正则**不会**作用于它 (见 smoke 记录)——
  若想覆盖, 需先给 `PriorOnly_ChannelInteraction` 补 `last_entropy` 接口 (非 GPU 工作)。

### P1-1 敏感性分析网格 (run_large.py 已支持透传参数)
- 背景: `n_envs`/`rff_dim`/`prior_weight`/`temperature` 原为硬编码, 现已加
  `--n_envs --rff_dim --prior_weight --temperature` (默认 None=不变, 不影响旧结果复现)。
- 动作 (建议 traffic, 每组 8 seed, full_v2_fixed):
```bash
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --n_envs 2 --num_shards 3 --output_dir ./output_sens_nenvs2
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds ... --rff_dim 64 --num_shards 3 --output_dir ./output_sens_rff64
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds ... --prior_weight 0.1 --num_shards 3 --output_dir ./output_sens_prior01
python run_large.py gen --datasets traffic --variants full_v2_fixed \
    --seeds ... --temperature 1.0 --num_shards 3 --output_dir ./output_sens_temp10
```
- 验收: 提升率方向/幅度不因超参显著翻转 → "结论不依赖超参脆点"。

### P1-2 baseline 评测 (iTransformer / DLinear)
- 官方代码已 clone: `01_external/iTransformer/code/`, `01_external/DLinear/code/`
  (iTransformer = thuml/iTransformer; DLinear = cure-lab/LTSF-Linear)。
- 前置 (非 GPU): 把它们适配进 `models_ablation.py` 的 `create_ablation_model`
  变体注册表 (DLinear 极简; iTransformer 需按我们 data/trainer 接口包一层),
  用 `--quick` 在 CPU 上 smoke 通。
- GPU 动作: 与 P0-1 同协议跑 6 数据集 × 8 seed, 结果进主表与消融对照。

---

## 二、不需要 GPU 的剩余工作

### 0 GPU 已完成 (2026-08-12, 本轮)

- [x] **修 C 可行性评估** (数据驱动, 结论: 可行): `assess_env_split.py` +
      `docs/diagnostics/2026-08-12_env_split_feasibility.md`。
      语义环境切分信息量 = 随机均分的 **4–14×** (ETTh1 昼夜 13.7×, weather 季节 4.2×);
      工作日/周末单独无信息 (1.2–1.9×)。→ 修 C (语义环境切分) 有数据支撑。
- [x] **想法 1 立项** (DRO 式风险厌恶): `02_research_notes/ideas/04_dro_risk_aversion/`
      (00_spark + 01_proposal)。语义环境评估直接支撑其"环境用语义定义"假设。
- [x] **方案 1 训练前适用性判据**: `compute_pre_train_stats.py` (0 GPU, 任意数据路径),
      已算 ETTh1/ETTm1/weather/exchange 统计量 (依赖密度/语义环境信息量/稳定通道对占比),
      json 落盘。**待 P0-1 主表后与增益做严格对应** (7 数据点, 启发性证据)。
- [x] **`run_large.py` 新增 `--alpha_init` / `--fusion_alpha` 透传** (CPU gen 验证通过):
      3b syn_ood 排查的前置, GPU 可跑。
- [x] **方案 3b 排查方案文档**: `docs/diagnostics/2026-08-12_synood_utilization_plan.md`
      (alpha_init/fusion_alpha 单因子+组合扫描, 判据与 GPU 命令齐备)。
- [x] **方案 4 PCD 转资产**: `vs_difference_argument.md` §3.1 (主动引用 PCD 独立复现维度规律)。
- [x] **论文草稿 §2.8 训练前适用性判据**: `06_paper_chapter_draft.md`。
- [x] **清理**: 删除根目录 `research-org.zip`; 修正冗余数据副本 (评估统一用 01_external 已有数据)。
- [x] **修 C 实施（代码就绪, CPU smoke 3/3 通过）**: 语义环境切分管线全链路 —
      `data.py` (Dataset 返回 (x,y,env_label), `env_scheme`=season/daynight/tod/wd),
      `causal_channel.py` (`CausalStabilityGate env_mode='semantic'` 按语义标签分组估 HSIC),
      `causalcit.py`+`models_ablation.py` (forward 透传 env_labels),
      `trainer.py` (三元组 batch), `run_large.py` (`--env_mode`/`--env_scheme`)。
      默认 `uniform`/`None` 保持旧行为, 不破坏 P0-1 复现。
- [x] **想法 1 DRO 实现（代码就绪, CPU smoke 3/3 通过）**: `trainer.py` `risk_lambda` +
      `L = mean_e(ℓ_e) + λ·var_e(ℓ_e)` 按环境分组损失 (环境数<2 退化为 ERM);
      `run_large.py` (`--risk_lambda`)。syn_ood 无时间戳, DPO 自动退化为 ERM (无害)。

### GPU 待跑 (P0-1 主表完成后; 前置代码均已就绪并 CPU 验证)

> P0-1 状态 (2026-08-12 16:31 b1403aa): 已回传 219/816 (27%) 快照, 高维 full_v2_fixed
> 8-seed 全面翻正 (traffic pl192 +12.1%)。剩余 ~597 job 待跑, ⚠️ 服务器是否仍在跑需核实
> (详见 PROGRESS.md「P0-1 部分结果快照」)。以下任务在 P0-1 收尾后执行。

1. **修 C 验证** (weather/electricity, uniform vs semantic):
   ```sh
   python run_large.py gen --datasets weather electricity --variants full_v2_fixed \
       --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 \
       --env_mode semantic --env_scheme season --output_dir ./output_fixC_semantic
   # run + summarize; 与 output_large_v3 (uniform) 对比, 判据: semantic 的 cv 提升 / 收益改善
   ```
2. **3b syn_ood 网格** (`--alpha_init`/`--fusion_alpha`): 见 `docs/diagnostics/2026-08-12_synood_utilization_plan.md`
3. **DRO λ 消融** (weather/electricity, capacity_match):
   ```sh
   for lb in 0 0.1 1; do python run_large.py gen --datasets weather electricity \
       --variants capacity_match --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 \
       --env_scheme season --risk_lambda $lb --output_dir ./output_dro_lambda_$lb; done
   ```
4. **方案 1 补测** (近 0 GPU): 服务器上对 traffic/electricity/ILI 跑
   `compute_pre_train_stats.py` (脚本任意路径可用)

### P0-2 统一 collapsed 判据 (审稿硬伤) — ✅ 2026-08-10 完成
- 两处已统一为常量 `COLLAPSED_STD_THRESHOLD = 0.01`
  (`run_minimal_falsifiable.py` 顶部 + `analyze_gates.py` 顶部, 互有注释指认)。
- smoke 验证: gate_prior_only syn_ood off_std=0.0033 → 按 0.01 判"坍缩" (两脚本口径一致)。
- 待办: 旧的涉 collapsed 报告 (如 `output_falsifiable_full/`) 若投稿引用, 需用新判据重生成。

### baseline 论文阅读 + 差异论证 (审稿人 re2 第 3 条点名) — ✅ 调研已完成
- 调研 agent 产出已归档: `02_research_notes/surveys/04_baseline_literature/`
  (`plan.md` / `stage1_research_preparation.md` / `report_final.md` /
  `appendix_baseline_survey_condensed.md`)。
- 论文/代码已下载至 `01_external/` (新增 Crossformer、TimeXer、FOIL、Koopa 的
  `paper/`+`code/`, Adapformer、CSformer 的 `paper/`)。
- 下一步: 基于 report_final 写"vs CausalCIT 差异论证"正式段落 (写作任务)。

### baseline 代码接入 — ✅ 2026-08-10 完成 (待 GPU 评测)
- `models_ablation.py` 新增 `DLinear` / `iTransformerModel` 自包含实现,
  `create_ablation_model` 支持 `--variants dlinear itransformer`;
  统一接口 [bs, seq_len, nvars] → [bs, pred_len, nvars]。CPU 前向验证全部通过。
- 待办: GPU 上按 P0-1 同协议跑 6 数据集 × 8 seed (P1-2)。

### 熵正则接口扩展 — ✅ 2026-08-10 完成
- `AblationBackbone`/`AblationModel` 补 `get_gate_entropy()`, 转发到通道交互模块的
  `get_last_entropy()` (无接口的变体安全返回 None)。
- 验证: gate_prior_only 在 entropy=0.01 下门控 off_mean 0.18→0.99 (被熵正则推向果断),
  修复前完全不受影响。已更新 smoke 记录 `output_entropy_smoke_2/`。

### 高维门控矩阵 dump + 聚类热图
- 脚本已写好: `plot_gate_heatmaps.py` (含高维子采样) + 说明 `plot_visualization_README.md`。
- 缺数据: traffic/electricity 的 `full_v2_fixed` 门控矩阵未 dump
  (run_large 只在 n_vars≤21 保存)。需写 eval dump 脚本 (加载 checkpoint 取 `get_gate_matrix()`)
  或放宽 dump 条件重跑。低维样例图已用旧数据验证可用。

### 战略补救：训练前适用性判据 (方案 1, 0 GPU) — 见 `02_research_notes/ideas/01_adaptive_channel/07_scope_and_publication_risk_analysis.md`
- 只从原始数据计算统计量: 通道依赖密度 / 依赖强度跨环境离散度 / 稳定通道对占比,
  与 7 数据集×horizon 实测增益做对应 → 能否训练前预测增益正负号。
- 目的: 把"范围窄"从 B 类(无解释)变成 A 类(有原则、可预测); 是审稿人"方法太窄"的正面回答。

### 战略补救：syn_ood 识别-利用脱节排查 (方案 3b, 可并入 P1-1) — 同上文档 §3
- 线索: syn_ood 上门控**结构识别成功**(因果边高 20-100 倍) 但 **MSE -1.21% 更差** → "识别对、没利用上"。
- 动作: 扫 `alpha_init`(-2.0 初始几乎关闭混合分支) / `fusion_alpha`(0.3 稀释),
  看 syn_ood 负收益能否翻正。若成立, 问题从"机制不成立"变"利用不足", 可修。

---

## 三、P2 剩余事项 (故事定位与诚实边界)

- [ ] **OOD 结论谨慎处理**: `syn_ood` 上 full_v2 为 -1.21% 显著变差, 尚不能宣称
  "因果门控带来 OOD 鲁棒性"。先排查 syn_ood 机制测试为何失败 (spurious_strengths 配置 or 容量),
  否则此章会被审稿人反杀。
  - **进展 (2026-08-11)**: PCD 对比实验初步发现 — 已有 50-epoch ckpt 评估显示
    `full_v2 ≈ pcd_gate` (差异<0.001) 且均略差于 patchtst (~0.5-0.8%),
    与 `output_pcd_smoke` (3-epoch) 一致 → PCD 注入的虚假结构未让门控产生收益,
    机制测试确认未通过。见 `docs/pcd/pcd_preliminary_findings.md`。
    待 GPU 机器完成 `output_pcd_full/` 5-seed 正式版 + 逐情形(A/B/C)检验后再定稿。
- [ ] **最小可证伪测试收尾 / claim 降级决策**: 若 P0-1 + P1-1 后因果增益仍只在
  traffic/pl192 等窄条件下成立 → 正式把定位从 "causal channel interaction" 降级为
  "stability-regularized channel attention", 去掉 "causal" 字眼 (自评 §6 P0 一直没执行)。
- [ ] **写作**: 按 "场景依赖有效改进 + 修复版批不变性 + 门控结构诊断" 三条主线,
  把 `method_assessment.md` 扩展为论文核心章节; limitation 明确低维/长 horizon 边界。
- [ ] **可视化补强**: traffic/electricity 高维门控矩阵聚类热图 (等数据 dump);
  full_v2_fixed 因果/虚假/独立边箱线图 (需合成数据真值边对齐)。

---

## 四、本轮已完成 (2026-08-10, 无 GPU 的 P1 准备)

- [x] `run_large.py` 敏感性分析参数透传 (`--n_envs --rff_dim --prior_weight --temperature`)。
- [x] clone iTransformer / DLinear 官方代码 → `01_external/{iTransformer,DLinear}/code/`。
- [x] 可视化脚本: `plot_bootstrap_ci.py` (bootstrap CI 误差棒图 + 数值表, 已跑通,
      产出 `output_large_v2/improvement_bootstrap_ci.{png,md}`);
      `plot_gate_heatmaps.py` (门控热图 + 诊断图, 已跑通, 产出 `vis_output/` 10 张图);
      说明文档 `plot_visualization_README.md`。
- [x] 熵正则 CPU smoke test: `output_entropy_smoke_{0,1}/`, 结论见
      `output_entropy_smoke_1/README_smoke_entropy.md` (熵正则对 full_v2 生效,
      对 gate_prior_only 无效 → 需接口扩展; P0-2 判据不一致复现)。
- [x] 文献调研 prompt: `02_research_notes/literature_review_prompt.md` (待转发)。
- [x] **P0-2** 统一 collapsed 判据 (0.01 常量, 两脚本同步)。
- [x] **baseline 接入**: `models_ablation.py` 新增 DLinear/iTransformerModel,
      `create_ablation_model` 支持 `--variants dlinear itransformer` (CPU 前向验证通过)。
- [x] **熵正则接口扩展**: AblationBackbone/AblationModel 补 `get_gate_entropy()`,
      gate_prior_only 熵正则生效验证通过。
- [x] **调研归档**: → `02_research_notes/surveys/04_baseline_literature/`。
- [x] **外部材料下载**: 01_external 新增 Crossformer/TimeXer/FOIL/Koopa (paper+code)
      及 Adapformer/CSformer (paper)。

## 四b、第二轮已完成 (2026-08-10 晚, 0 GPU 收尾)

- [x] **run_large 全流程 smoke** (syn_ood, `output_pipeline_smoke2/`):
      gen → run(cpu) → summarize 全链路跑通; 敏感性参数 (n_envs=2) / 熵正则
      (entropy_weight=0.01) / baseline (dlinear/itransformer) 全部在真实训练循环中验证。
- [x] **修复 --epochs 未生效 bug**: build_kwargs 里 job['epochs'] 之前写死
      `cfg['epochs']`, 覆盖参数未落地 (第一次 smoke 20 个 job 全按 50 epoch 跑,
      patchtst 用了 1082s)。已改为 `epochs if epochs is not None else cfg['epochs']`,
      二次验证 "Epoch 1/2" 生效, 单 job 降至 9–76s。
- [x] **run_large.py 新增 `--epochs` 覆盖** (本地 smoke/GPU 试跑调短用)。
- [x] **run_large.py 新增 `--dump_gates`** (强制 n_vars>21 也保存门控矩阵,
      配合 P0-1 给 traffic/electricity 高维热图供数)。
- [x] **`plot_gate_edge_boxplot.py`**: 因果/虚假/独立边门控权重箱线图 + 汇总表
      (syn_ood 示例已出图, `vis_output/gate_edge_boxplot.{png,md}`)。
- [x] **`dump_gates_eval.py`**: 从已有 checkpoint eval 提取门控矩阵 (不重训),
      syn_ood 验证通过 (8 个矩阵导出, 无门控变体正确跳过)。
- [x] **写作**: `02_research_notes/surveys/04_baseline_literature/vs_difference_argument.md`
      (与各 baseline 的差异论证, 审稿 re2 §3 交付物);
      `02_research_notes/paper_method_chapter_draft.md` (方法+实验章节草稿,
      数字标 P0-1 重跑后替换)。

## 五、历史遗留 (已修复/已完成, 供追溯)

- [x] spawn seed bug (2026-08-08): 见 P0-1 说明。
- [x] `output_large_v2` 720 结果 + `output_falsifiable_full` 80 条诊断 (2026-08-06/08)。
- [x] Wilcoxon + Holm 显著性取代 t-test; full_v2_fixed running_stats 修复。

## 六、项目整理记录 (2026-08-11, 按 research-org 规范)

- [x] `02_research_notes/` 根目录散文件全部归档:
      reviewer 评审链 (re1/re2/ood_diagnostic) → `ideas/01_adaptive_channel/02_review_*`;
      论文草稿 → `ideas/01_adaptive_channel/06_paper_chapter_draft.md`;
      调研 prompt → `surveys/04_baseline_literature/appendix_review_prompt.md`。
- [x] `01_external/iTransformer/iTransformer_paper.pdf` → `iTransformer/paper/`
      (与 code/ 并列, 符合 paper+code 同目录规范)。
- [x] `03_experiments/CausalCIT/GPU验证任务说明.md` → 重命名并归档至
      `experiments/gpu_verification_task.md` (英文命名)。
- [ ] 遗留待清理 (用户决定): 根目录 `research-org.zip`、空中文文件夹
      `转发文献调研任务至调研agent，支撑论文差异论证/` (内容已归档, 可删空壳)。
- [ ] 可选: 清理全项目 `__pycache__/` 与 `*.pyc` (无害缓存, 规范允许)。
