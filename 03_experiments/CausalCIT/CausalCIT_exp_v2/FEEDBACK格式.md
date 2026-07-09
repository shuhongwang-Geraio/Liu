# CausalCIT_exp_v2 结果回传格式

跑完实验后，请将以下文件打包到一个文件夹（如 `result_v2/`）中回传：

## 必须包含的文件

```
result_v2/
├── cmd.txt                          # 终端完整输出（从运行命令到结束）
├── experiment_report_v2.md          # 自动生成的实验报告
├── enhanced_synthetic_results.png   # 合成数据可视化（如果跑了 --exp synthetic 或 all）
├── real_data_results.png            # 真实数据可视化（如果跑了 --exp real 或 all）
└── 运行参数.txt                     # 你实际使用的运行命令（见下方模板）
```

## cmd.txt 获取方式

把终端里从运行命令开始到"全部实验完成！"结束的 **全部输出** 复制粘贴保存即可。
关键信息包括：每个模型的参数量、每个epoch的loss、最终的MSE/MAE、MSE改进率。

## 运行参数.txt 模板

```
运行时间: xxxx-xx-xx
设备: cpu / cuda
运行命令: python run_enhanced.py --exp all --device cpu
实际总耗时: 约 xx 分钟
是否遇到报错: 无 / 有（描述）
```

## 注意事项

- 如果 Weather 数据集太大跑不动或太慢，可以只跑 `--exp synthetic`，真实数据部分后面再单独跑
- 如果中途报错中断了，把 cmd.txt 中已有的输出也传回来，我可以帮你排查
