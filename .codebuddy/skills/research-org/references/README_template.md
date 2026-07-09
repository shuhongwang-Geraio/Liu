# Research Project Organization Paradigm

适用于以「读论文 → 做调研 → 写代码」为核心循环的科研项目。

## Three-Layer Structure

```
project/
├── 01_external/          # Layer 1：别人的东西
├── 02_research_notes/    # Layer 2：我们的想法与调研
└── 03_experiments/       # Layer 3：我们自己的代码
```

---

## Layer 1: `01_external/`

**定义**：外部资料——论文、别人的项目源代码、参考数据。

```
01_external/
├── PatchTST/             # 有论文 + 有代码
│   ├── paper/
│   └── code/
├── DLinear/              # 只有论文
│   └── paper/
├── SOFTS/                # 只有论文（后续可加 code/）
│   └── paper/
└── CCM/                  # 单篇论文也归档
    └── CCM_paper.pdf
```

**规则**：
- 按项目名平铺，论文和代码放同一个文件夹下
- 有代码就加 `code/` 子目录，没有就不加
- 不改动原始内容

---

## Layer 2: `02_research_notes/`

**定义**：自己的阅读笔记、调研报告、研究构想。

```
02_research_notes/
├── paper_reading/        # 论文逐篇解读
│   ├── review_1_*.md
│   ├── review_2_*.md
│   └── meta_analysis_*.md     # 跨论文横向对比
│
├── ideas/                # 研究构想
│   ├── _README.md        # ← 规范说明
│   ├── 00_inbox/         # 未拆分的脑暴合集
│   ├── NN_shortname/
│   │   ├── 00_spark.md          # 原始火花
│   │   ├── 01_proposal.md       # 正式提案
│   │   ├── 02_review_rN_*.md    # 批评与迭代
│   │   ├── 03_final.md          # 整合版本
│   │   └── 04_competitive.md    # 竞品调研
│   └── ...
│
└── surveys/              # 系统性调研
    ├── _README.md        # ← 规范说明
    └── NN_topic/
        ├── plan.md               # 执行计划
        ├── stage1_*.md           # 阶段产出
        ├── report_final.md       # 最终报告
        ├── appendix_*.md         # 附加材料
        └── assets/               # 图片等附件
```

### Idea Lifecycle

```
00_spark ──→ 01_proposal ──→ 02_review ──→ 03_final
                                      ↑          │
                                      └─ 可多轮 ─┘
                            04_competitive（可穿插）
```

### Survey Lifecycle

```
plan ──→ stage1 ──→ stage2 ──→ stage3 ──→ report_final
                                            │
                                    appendix（附加产出）
```

**规则**：
- 每个 idea/survey 是独立的编号文件夹
- 编号递增（01、02、03…），新增直接顺延
- 文件前缀数字表示阶段顺序
- 每个层级有 `_README.md` 自述规范

---

## Layer 3: `03_experiments/`

**定义**：自己写的实验代码、模型实现、训练脚本。

```
03_experiments/
└── ModelName/
    ├── ModelName_demo/       # 最小可行版本
    ├── ModelName_ablation/   # 消融实验
    ├── ModelName_exp_v2/     # 增强实验
    └── output/               # checkpoint、日志、可视化
```

**规则**：
- 一个模型一个顶层文件夹
- demo → ablation → scale-up，渐进式
- 实验产物（checkpoint/日志）统一放 `output/`

---

## Naming Conventions

| 规则 | 示例 |
|------|------|
| 目录全英文 | `01_external/`, `02_research_notes/` |
| 编号前缀 | `01_`, `02_`, `03_` 表示顺序 |
| 文件前缀数字 | `00_spark.md`, `01_proposal.md` 表示阶段 |
| 附录加前缀 | `appendix_*.md` |
| `_README.md` | 每层的自述规范（下划线避免排最前） |

---

## Adding New Content

| 场景 | 操作 |
|------|------|
| 加新论文/代码 | `mkdir 01_external/ProjectName/` → `paper/` + `code/`（按需） |
| 读后写解读 | 放到 `02_research_notes/paper_reading/` |
| 有新 idea | `mkdir ideas/03_name/` → 写 `00_spark.md` |
| 新调研 | `mkdir surveys/03_topic/` → 写 `plan.md` |
| 新实验 | `mkdir 03_experiments/ModelName/` |

---

## Why This Works

1. **三层隔离**：外部/思考/代码互不污染
2. **编号递增**：天然有序，新增不破坏旧结构
3. **生命周期显式化**：idea 从火花到成型的全过程可追溯
4. **自文档化**：每层的 `_README.md` 让新人或未来的自己秒懂
5. **跨项目通用**：换研究方向、换领域，结构不变
