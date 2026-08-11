---
name: research-org
description: Organizes research project directories into a standardized three-layer structure. Layer 1 for papers and external code, Layer 2 for reading notes, ideas, and surveys following a lifecycle convention, Layer 3 for own experiment code. Use when the user asks to organize, restructure, or clean up a research project, or when setting up a new project from scratch. Triggers include requests about organizing project folders, restructuring research files, or setting up a research repository.
---

# Research Project Organizer

Organize research project directories into a clean, extensible three-layer structure.
All directory and file names use English only. Use `git mv` when available to preserve version history.

## The Three-Layer Paradigm

```
project/
|-- 01_external/          # External: papers and others' code
|-- 02_research_notes/    # Internal: our notes, ideas, surveys
|-- 03_experiments/       # Internal: our experiment code
```

A full README template is available at `references/README_template.md` -- copy it to the project root after organizing.

Note: papers and code for the same project stay together under one folder (e.g., `PatchTST/paper/` and `PatchTST/code/`), NOT separated into global `papers/` and `code/` dirs.

## Project Progress Tracking: `PROGRESS.md`

Every research project must keep a `PROGRESS.md` at the project root (alongside the README).
It is the single source of truth for "where we are" and "what to do next".

**Content contract** (keep concise):
- `## 项目目标` -- one-paragraph goal
- `## 当前状态` -- two sub-lists:
  - `### 已完成` -- finished work (checkmarks, with commit hashes / key artifacts when useful)
  - `### 未完成 / 有问题` -- known gaps, broken pieces, stalled items (with details)
- `## 下一步 (按优先级)` -- ordered actionable next steps, each with the concrete entry command/path
- `## 备注` -- project-specific gotchas, conventions, archived/deleted-file notes

**Rules**:
- Update `PROGRESS.md` at the **end of every work session** (or after any significant change):
  mark what was completed, what regressed, and refresh the "下一步" list accordingly.
- Keep it factual and minimal -- do not copy full analysis here; link or summarize instead.
- When deleting old/obsolete documents (e.g. outdated review notes), record the deletion in
  `## 备注` with the note that content can be recovered from git history.
- Do NOT delete `PROGRESS.md`; it replaces ad-hoc **status** documents.

**Functional docs may coexist with `PROGRESS.md`**:
PROGRESS.md replaces *status* documents ("where we are / what's next"), NOT functional
documents that answer different questions:
- `DATA.md` -- data map (where data/artifacts live, how to reproduce): keep
- runbook / `do.md` -- detailed executable commands, decision gates, stop-loss rules: keep
Rule of thumb: if a doc answers "where are we / what to do next" it must fold into
`PROGRESS.md`; if it answers "where is the data / how exactly to run X" it may stay,
but `PROGRESS.md` should reference (link) it instead of duplicating its content.
Root-level doc layout: `README.md` (conventions + snapshot) + `PROGRESS.md` (status,
single source of truth) + optional functional docs (`DATA.md`, `do.md`).

## Layer 1: `01_external/` -- Papers & External Code

```
01_external/
|-- PatchTST/               # has paper + code
|   |-- paper/
|   |-- code/
|-- DLinear/                # paper only
|   |-- paper/
|-- SOFTS/                  # paper only (can add code/ later)
|   |-- paper/
|-- CCM/                    # standalone PDF also gets a folder
    |-- CCM_paper.pdf
```

**Rules**:
- One folder per project/paper -- flat, no separate papers/ and code/ buckets
- If a paper has corresponding code, add a `code/` subdirectory
- If only a paper, just `paper/` or the PDF itself
- Never modify original content

## Layer 2: `02_research_notes/` -- Reading Notes, Ideas, Surveys

```
02_research_notes/
|-- paper_reading/           # per-paper reading notes
|   |-- review_1_*.md
|   |-- review_2_*.md
|   |-- meta_analysis_*.md   # cross-paper comparison
|
|-- ideas/                   # research ideas with lifecycle
|   |-- _README.md           # convention doc for this level
|   |-- 00_inbox/            # undifferentiated brain dumps
|   |-- NN_shortname/
|       |-- 00_spark.md           # raw initial thought
|       |-- 01_proposal.md        # formal proposal
|       |-- 02_review_rN_*.md     # critique & iteration (multi-round)
|       |-- 03_final.md           # integrated version(s)
|       |-- 04_competitive.md     # competitive landscape check
|
|-- surveys/                 # systematic literature surveys
    |-- _README.md           # convention doc for this level
    |-- NN_topic/
        |-- plan.md               # execution plan
        |-- stage1_*.md           # stage outputs
        |-- stage2_*.md
        |-- stage3_*.md
        |-- report_final.md       # consolidated report
        |-- appendix_*.md         # supplementary materials
        |-- assets/               # figures, data
```

### Idea Lifecycle Convention

Each idea folder follows a numbered stage progression:

| Stage | File | Purpose |
|-------|------|---------|
| 0 | `00_spark.md` | Raw brainstorm, no format required |
| 1 | `01_proposal.md` | Motivation, method, novelty, advantages |
| 2 | `02_review_r1_critique.md` | Self-critique round 1 |
| 2 | `02_review_r1_response.md` | Response & revision round 1 |
| 2 | `02_review_r2_*.md` | Round 2 (repeat as needed) |
| 3 | `03_final*.md` | Current integrated version (can have variants) |
| 4 | `04_competitive.md` | Check if similar work already exists |

### Survey Convention

Each survey follows a plan -> stages -> final pattern:

| File | Purpose |
|------|---------|
| `plan.md` | Execution plan with progress tracking |
| `stage1_*.md` | Phase 1 output |
| `stage2_*.md` | Phase 2 output |
| `stage3_*.md` | Phase 3 output |
| `report_final.md` | Final consolidated report |
| `appendix_*.md` | Supplementary reports/design docs |
| `assets/` | Figures, data, attachments |

## Layer 3: `03_experiments/` -- Experiment Code

```
03_experiments/
|-- ModelName/
    |-- ModelName_demo/       # minimal viable version
    |-- ModelName_ablation/   # ablation studies
    |-- ModelName_exp_v2/     # scaled-up experiments
    |-- output/               # checkpoints, logs, visualizations
```

**Rules**:
- One model per top-level folder
- Progressive: demo -> ablation -> scale-up
- Keep outputs in dedicated `output/` dir

## Naming Rules

1. **Directories**: English only, numbered prefix (`01_`, `02_`, `03_`)
2. **Files**: numbered prefix for stage ordering (`00_spark.md`, `01_proposal.md`)
3. **Appendix files**: prefix with `appendix_`
4. **Convention docs**: `_README.md` at each layer (underscore keeps it near top)

## Workflow

When asked to organize a project:

1. Survey existing directories -- identify which files belong to which layer
2. Identify irrelevant files to purge (cache snapshots, `.tar.gz`, `.pyc` if applicable)
3. Create the three-layer directories
4. Move files with `git mv` (if git repo) or regular move
5. Write `_README.md` convention docs for `ideas/` and `surveys/`
6. Write root `README.md` using the template in `references/README_template.md`
7. Write root `PROGRESS.md` (see "Project Progress Tracking" section) capturing current
   status and next steps
8. Keep any necessary functional docs (data map `DATA.md`, runbook `do.md`) that
   `PROGRESS.md` references; fold pure status docs into `PROGRESS.md`
9. Clean up old empty directories

In addition, whenever working in any research project (not just organizing), update the
project's `PROGRESS.md` at the end of the session per the rules in the
"Project Progress Tracking" section.

## Cleanup Rules

- Platform cache files (`.tar.gz`, `.gz` snapshots from AI platforms) are NOT research content -- delete them
- `__pycache__/` and `.pyc` files can be cleaned but are harmless
- After moving all files out of old directories, remove the empty shells
