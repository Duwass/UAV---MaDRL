# Phase CTDE Next Steps

## Immediate Next Step

Checkpoint and commit the CTDE foundation on `phase-ctde-foundation` after review. The commit should include CTDE source, config, smoke script, tests, and audit docs, but should not include results, checkpoints, `__pycache__`, `.pyc`, or the pre-existing deleted PDF unless explicitly intended.

## Next Technical Phase

Design a controlled CTDE short-run experiment protocol before making any performance claims.

The protocol should define:

- fixed seed or seed list;
- short train budget;
- evaluation budget;
- exact metrics to collect;
- output directory policy;
- checkpoint policy;
- pass/fail smoke criteria;
- statement that single-run results are not paper claims.

## Later Phases

- Run short controlled CTDE smoke experiments.
- Add multi-seed experiments only after the smoke protocol is stable.
- Revisit fairness/drop reward behavior in a separate phase.
- Add CTDE/decentralized execution leakage audit to the final report.
- Update final LaTeX only after the method and claims are settled.
