# PatchTST 数据集目录

本目录存放时间序列预测的基础数据集（csv）。**这些 csv 不进 git**（体积大，可重新下载）。

## 包含的文件中
- `electricity.csv` (92M) — 电力负荷
- `traffic.csv` (131M) — 加州交通
- `ETTh1.csv` / `ETTh2.csv` — 电力变压器（小时级）
- `ETTm1.csv` / `ETTm2.csv` — 电力变压器（15 分钟级）
- `exchange_rate.csv` — 汇率
- `ILI.csv` — 流感样疾病（含 `ILI.csv.bak` 备份，已 gitignore）
- `weather.csv` / `weather_ood` 切分由上游代码生成（见 CausalCIT 的 OOD 协议）

## 重新获取
从 PatchTST 官方仓库下载 `dataset/` 整体放入本目录：
https://github.com/yuqinie1998/PatchTST

> 注意：本目录的 csv 已被根 `.gitignore` 的 `01_external/PatchTST/code/dataset/*.csv` 规则排除。
