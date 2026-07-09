# Ideas Directory Convention

每个 idea 是一个独立的编号文件夹 `NN_shortname/`，按以下生命周期组织：

## Lifecycle Stages

```
00_spark.md         原始火花：想到什么写什么，不需要格式
01_proposal.md      正式提案：动机/方法/创新点/预期优势
02_review/          批评与迭代（可多轮）
    rN_critique.md  第 N 轮自我批评
    rN_response.md  第 N 轮回应与修订
03_final.md         当前整合版本（可以有多个变体）
04_competitive.md   竞品调研：是否有类似工作、是否已被抢先
```

## Rules

- **新 idea**：`mkdir NN_shortname/`，从 `00_spark.md` 开始
- **编号递增**：下一个用 `03_`，不会与已有冲突
- **00_inbox/**：未分类的脑暴合集，等酝酿成熟后再拆分为独立 idea 文件夹
- **文件名全英文**：便于跨平台和版本管理
- **跨项目通用**：不绑定具体领域，换研究方向照用

## Current Ideas

| 编号 | 名称 | 状态 |
|------|------|------|
| 01 | adaptive_channel | 已展开（proposal + competitive）
| 02 | multiscale_rg | 完整生命周期（spark → review → final）
