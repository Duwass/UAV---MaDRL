# LaTeX Report Preview

This directory contains a preview LaTeX draft for the final UAV-MADRL research report. It currently includes:

- Chapter I: Introduction
- Chapter II: Theoretical Background and Related Work
- Chapter III: System Model and Problem Formulation
- Chapter IV: Proposed Method
- Chapter V: Experimental Setup, Results, and Discussion

The draft follows the current implemented/evaluated Scenario 4 setting:

- 2 UAVs
- 15 IoT devices
- 1 mobile jammer
- frame length: 10
- episode length: 200
- flat action dimension: 864
- hierarchical high-level action dimension: 10
- QMIX local observation dimension: 97
- QMIX global state dimension: 70

Older proposed parameter tables are planning references only and should not be used as final experimental settings.

## Files

- `main.tex`: report skeleton with title placeholders, table of contents, list of figures, list of tables, active Chapters I--V, and bibliography.
- `chapters/chapter1_introduction.tex`: Vietnamese academic draft of Chapter I.
- `chapters/chapter2_background_related_work.tex`: Vietnamese academic draft of Chapter II.
- `chapters/chapter3_system_model_problem.tex`: system model and problem formulation draft.
- `chapters/chapter4_proposed_method.tex`: proposed method draft.
- `chapters/chapter5_experiments_results_discussion.tex`: experimental setup, results, discussion, and temporary conclusion placeholder.
- `references.bib`: initial bibliography with verified core RL/QMIX entries and TODO placeholders for domain references that still need metadata verification.
- `build_notes.md`: compilation notes and LaTeX environment status.
- `preview_report_notes.md`: preview status, included artifacts, and remaining TODOs.
- `build/report_preview_current.pdf`: compiled current preview PDF when available.

## How To Compile

Recommended engine:

```powershell
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Run these commands from `docs/report_latex/`.

If XeLaTeX is unavailable, pdfLaTeX may work only if the local TeX installation includes Vietnamese font support for T5 encoding.

## Status

This is not the final report. It is a full preview to check structure, tone, academic flow, tables, figures, and chapter integration before Phase 4.5 / Stage 5 final writing.
