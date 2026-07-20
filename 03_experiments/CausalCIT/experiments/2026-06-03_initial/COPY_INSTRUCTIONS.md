# 复制二进制文件指引

以下文件需要从旧位置手动复制到新位置（文本文件已自动迁移）。

## 从 results_feedback/ -> experiments/2026-06-03_initial/

### v2 可视化图片
copy "03_experiments\results_feedback\result_v2\enhanced_synthetic_results.png" → "03_experiments\CausalCIT\experiments\2026-06-03_initial\v2\enhanced_synthetic_results.png"
copy "03_experiments\results_feedback\result_v2\real_data_results.png" → "03_experiments\CausalCIT\experiments\2026-06-03_initial\v2\real_data_results.png"

### ablation 可视化图片
copy "03_experiments\results_feedback\result_ablation\ablation_synthetic.png" → "03_experiments\CausalCIT\experiments\2026-06-03_initial\ablation\ablation_synthetic.png"
copy "03_experiments\results_feedback\result_ablation\ablation_etth1.png" → "03_experiments\CausalCIT\experiments\2026-06-03_initial\ablation\ablation_etth1.png"

### ablation 门控矩阵 (小文件, 分析需要)
copy "03_experiments\results_feedback\result_ablation\gate_matrices\" → "03_experiments\CausalCIT\experiments\2026-06-03_initial\ablation\gate_matrices\"

## 从 CausalCIT_demo/moveResult/ -> experiments/2026-06-03_initial/demo/
copy "03_experiments\CausalCIT\CausalCIT_demo\moveResult\synthetic_results.png" → "03_experiments\CausalCIT\experiments\2026-06-03_initial\demo\synthetic_results.png"
copy "03_experiments\CausalCIT\CausalCIT_demo\moveResult\ood_results.png" → "03_experiments\CausalCIT\experiments\2026-06-03_initial\demo\ood_results.png"

---

执行完成后，确认文件结构：
experiments/2026-06-03_initial/
├── README.md
├── environment.txt
├── cmd_v2.txt
├── cmd_ablation.txt
├── COPY_INSTRUCTIONS.md  (本文件，确认后可删除)
├── demo/
│   ├── report.md
│   ├── synthetic_results.png   ← 需手动复制
│   └── ood_results.png         ← 需手动复制
├── v2/
│   ├── report.md
│   ├── enhanced_synthetic_results.png  ← 需手动复制
│   └── real_data_results.png          ← 需手动复制
└── ablation/
    ├── report.md
    ├── ablation_synthetic.png  ← 需手动复制
    ├── ablation_etth1.png      ← 需手动复制
    └── gate_matrices/          ← 需手动复制整个目录
