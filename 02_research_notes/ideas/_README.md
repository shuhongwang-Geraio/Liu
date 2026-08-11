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
| 01 | adaptive_channel | 已展开（proposal + competitive + major improvement + review 链 + 战略分析 07）
| 02 | multiscale_rg | 完整生命周期（spark → review → final）
| 03 | invertible_decouple | 完整调研（report_final, 5 子问题中 3 个未覆盖）
| 00_inbox | 脑暴合集 | `brainstorm_6ideas.md`（6 想法, 含检索增强★★★★★/自适应Patch）+ `2026-08-11_new_directions.md`（5 新方向 + 决策门 + 三根因发现）

> 新方向决策门（2026-08-11）: 见 `00_inbox/2026-08-11_new_directions.md` §2 与 `do.md`「方向决策门」——
> 先做第 0 步静态诊断（验证 RFF σ 与 HSIC 归一化两根因），据结果分支到 抢救 CausalCIT / 想法1(DRO) / 想法2(可逆解耦)。
