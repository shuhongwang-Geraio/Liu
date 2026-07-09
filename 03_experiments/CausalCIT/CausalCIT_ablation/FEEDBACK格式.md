# CausalCIT_ablation 结果回传格式

跑完实验后，请将以下文件打包到一个文件夹（如 `result_ablation/`）中回传：

## 必须包含的文件

```
result_ablation/
├── cmd.txt                    # 终端完整输出（从运行命令到结束）
├── ablation_report.md         # 自动生成的消融报告
├── ablation_synthetic.png     # 合成数据消融可视化（如果跑了 --exp synthetic 或 all）
├── ablation_etth1.png         # ETTh1消融可视化（如果跑了 --exp real 或 all）
└── 运行参数.txt               # 你实际使用的运行命令（见下方模板）
```

## cmd.txt 获取方式

把终端里从运行命令开始到"消融实验完成！"结束的 **全部输出** 复制粘贴保存即可。
核心信息：5个变体（PatchTST / w/o Gate / w/o EnvSplit / w/o HSIC / Full CausalCIT）的 MSE、MAE、参数量。

## 运行参数.txt 模板

```
运行时间: xxxx-xx-xx
设备: cpu / cuda
运行命令: python run_ablation.py --exp all --device cpu
实际总耗时: 约 xx 分钟
是否遇到报错: 无 / 有（描述）
```

## 建议运行顺序

消融实验比较耗时（5个变体），建议：

1. **先跑合成数据**（核心消融，结果最关键）：
   ```bash
   python run_ablation.py --exp synthetic
   ```
2. **再跑真实数据**（补充验证）：
   ```bash
   python run_ablation.py --exp real
   ```

这样即使第二步中断，第一步的结果也不会丢。

## 注意事项

- 合成数据消融（5个变体 × 50 epochs）CPU约2.5小时，GPU约25分钟
- 如果中途报错中断了，cmd.txt 中已有的输出也请传回来
- **ablation_synthetic.png 是最重要的**，它包含门控矩阵对比和组件贡献分析
