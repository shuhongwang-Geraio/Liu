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

## Root Documents（根目录文档）

```
project/
├── 01_external/          # Layer 1
├── 02_research_notes/    # Layer 2
├── 03_experiments/       # Layer 3
├── README.md             # 本模板：三层结构规范 + 项目快照
├── PROGRESS.md           # 进度单一事实来源（必建）
└── 功能性文档（按需，非状态类）
    ├── DATA.md           # 数据地图：数据/产物在哪、怎么获取、哪些进 git
    └── do.md             # 详细执行 runbook：具体命令、决策门、止损规则
```

**PROGRESS.md**：项目根目录必建，作为「我们在哪 / 下一步做什么」的**单一事实来源**。
内容契约：`项目目标 / 当前状态（已完成·未完成·有问题）/ 下一步（按优先级，含具体命令）/ 备注`。
保持简洁——不复制长篇分析，用链接或一句话摘要指向 `03_experiments/` 下的报告。

**功能性文档与 PROGRESS 的分工**：

| 文档回答的问题 | 归谁 |
|----------------|------|
| 我们在哪、下一步做什么 | `PROGRESS.md`（唯一） |
| 数据/产物在哪、怎么复现 | `DATA.md`（保留） |
| 具体命令、参数、止损规则 | `do.md` / runbook（保留） |

**去留判定**：文档若回答"我们在哪/下一步"→ 必须并入 `PROGRESS.md`；若回答
"数据在哪/怎么执行"→ 可保留，但 `PROGRESS.md` 必须链接引用它、不复制其内容。
删除过时文档时，在 `PROGRESS.md` 的 `备注` 记录删除（内容可从 git 恢复）。

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
6. **根文档分层**：README 管规范、PROGRESS 管进度（单一事实来源）、DATA/do 管执行细节——
   各司其职，状态文档不散落、功能文档不丢
