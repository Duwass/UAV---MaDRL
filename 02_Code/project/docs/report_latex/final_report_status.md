# Final Report Status

## What Is Complete

- Full LaTeX report draft with Vietnamese abstract and Chapters I--V.
- Chapter I introduction polished with gap, objectives, scope, and contributions.
- Chapter II background polished with UAV-IoT, jamming, RF-powered/backscatter, RL, DDQN, hierarchical RL, MaDRL, and QMIX.
- Chapter III system model and problem formulation with compact notation, equations, and system architecture figure.
- Chapter IV proposed method with hierarchical action diagram, QMIX diagram, equations, and pseudocode.
- Chapter V experimental setup, results, discussion, conclusion, and future work.
- All Phase 3 publication tables and figures included and referenced.
- Figure/table audit completed.
- Finalization audit completed.
- Final PDF compiled successfully.

## What Remains TODO

- Replace title page placeholders with official university/student/supervisor information.
- Verify and complete bibliography metadata for TODO references.
- Confirm final citation style and university formatting requirements.
- Human proofread Vietnamese terminology and chapter flow.
- Verify two conceptual/report-level equations marked by `% TODO: verify exact implementation form`.

## Known Limitations of the Report

- Some bibliography entries remain placeholders because exact metadata was not verified locally.
- The title page is not yet official.
- The report uses a technical report layout, not a university-specific final template.
- The Vietnamese hyphenation warning remains in the local TeX build environment.

## Known Limitations of the Experiment

- Final evaluated setting is 2 UAVs, 15 IoT devices, and 1 mobile jammer.
- Results are simulation-based.
- Backscatter is modeled as a system-level abstraction, not a fully validated physical-layer model.
- Executor is rule-based and uses environment-level information.
- Some baselines are single-run while QMIX has three-seed validation.
- Fairness remains moderate and is not fully solved.
- No hardware or hardware-in-the-loop validation is included.
- No retraining was performed in Phase 4.5.

## Compilation Status

- PDF compiled successfully with XeLaTeX and BibTeX.
- Final draft PDF path:
  - `docs/report_latex/build/uav_madrl_final_draft.pdf`

## Build Warnings

- No undefined citations or undefined references after the final build.
- No overfull/underfull hbox warnings after the final build.
- Non-blocking warning: Vietnamese hyphenation patterns are not preloaded in MiKTeX.
