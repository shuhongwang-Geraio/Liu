# CausalCIT 实验诊断摘要

生成时间: 2026-06-04 21:00:04

## 1. 文件清单

### result_v2/
- 总计 916
- drwxrwxr-x 2 wangsh wangsh   4096  6月  4 21:00 .
- drwxrwxr-x 5 wangsh wangsh   4096  6月  4 21:00 ..
- -rw-rw-r-- 1 wangsh wangsh    216  6月  4 21:00 运行参数.txt
- -rw-rw-r-- 1 wangsh wangsh  31897  6月  4 20:53 cmd.txt
- -rw-rw-r-- 1 wangsh wangsh 583727  6月  4 21:00 enhanced_synthetic_results.png
- -rw-rw-r-- 1 wangsh wangsh    930  6月  4 20:53 experiment_report_real.md
- -rw-rw-r-- 1 wangsh wangsh    735  6月  4 20:25 experiment_report_synthetic.md
- -rw-rw-r-- 1 wangsh wangsh   1386  6月  4 20:53 experiment_report_v2.md
- -rw-rw-r-- 1 wangsh wangsh 293425  6月  4 21:00 real_data_results.png

### result_ablation/
- 总计 480
- drwxrwxr-x 3 wangsh wangsh   4096  6月  4 21:00 .
- drwxrwxr-x 5 wangsh wangsh   4096  6月  4 21:00 ..
- -rw-rw-r-- 1 wangsh wangsh    218  6月  4 21:00 运行参数.txt
- -rw-rw-r-- 1 wangsh wangsh 148477  6月  4 21:00 ablation_etth1.png
- -rw-rw-r-- 1 wangsh wangsh   1767  6月  4 21:00 ablation_report.md
- -rw-rw-r-- 1 wangsh wangsh    904  6月  4 21:00 ablation_report_real.md
- -rw-rw-r-- 1 wangsh wangsh    841  6月  4 20:56 ablation_report_synthetic.md
- -rw-rw-r-- 1 wangsh wangsh 285898  6月  4 21:00 ablation_synthetic.png
- -rw-rw-r-- 1 wangsh wangsh  21048  6月  4 21:00 cmd.txt
- drwxrwxr-x 2 wangsh wangsh   4096  6月  4 21:00 gate_matrices

## 2. 实验结果摘要

### 合成数据 (v2)
**MSE改进: -2.11%**

参数量开销: +11.2%

## 结论


### 合成数据 (ablation)

| 变体 | MSE | MAE | Params | Time(s) |
|------|-----|-----|--------|---------|
| PatchTST (no interaction) | 0.482741 | 0.526161 | 225,646 | 25 |
| w/o Gate (full attention) | 0.495676 | 0.532914 | 250,735 | 26 |
| w/o EnvSplit (global HSIC) | 0.488840 | 0.529456 | 250,833 | 31 |
| w/o HSIC (Pearson corr) | 0.488861 | 0.531066 | 250,834 | 35 |
--

### 各组件边际贡献 (MSE降低)

| 组件 | MSE降低 | 贡献占比 |
|------|--------|---------|
| 通道注意力 | -0.012935 | 0.0% |
| 门控选择 | 0.006835 | 0.0% |
| 环境划分 | -0.000020 | 0.0% |
| HSIC检验 | -0.005507 | 0.0% |

## 3. 错误记录
无错误记录

