# Surveys Directory Convention

每个调研项目是一个独立的编号文件夹 `NN_topic/`。

## File Structure

```
plan.md             执行计划（分几个阶段、每个阶段做什么）
stage1_*.md         第一阶段产出
stage2_*.md         第二阶段产出
stage3_*.md         第三阶段产出
report_final.md     最终综合报告
appendix_*.md       附加报告/技术方案/补充调研
assets/             图片、图表等附件
```

## Rules

- **新调研**：`mkdir NN_topic/`，从 `plan.md` 开始
- **编号递增**：下一个用 `03_`
- **主线清晰**：`plan → stage → final` 不可少，其余归 `appendix_`
- **assets 统一管理**：所有非文本附件放 `assets/`
- **跨项目通用**：不绑定具体领域

## Current Surveys

| 编号 | 名称 | 内容 |
|------|------|------|
| 01 | adaptive_cicd | 自适应 CI/CD 通道交互机制调研
| 02 | literature_overview | 时序预测文献综合分析
