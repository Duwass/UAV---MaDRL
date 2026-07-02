# Phase 3D Next Steps

## Next Phase

The next phase should be CTDE on the 3D environment foundation.

This phase should not be started until the 3D environment audit is reviewed and accepted.

## Why CTDE Comes After the 3D Foundation

CTDE needs stable environment contracts:
- Entity positions must be consistently 3D.
- Channel, coverage, and jammer distance modes must be correct.
- Local observation and global state dimensions must be fixed.
- Action dimension must be fixed.
- Metrics must confirm that altitude and vertical motion are visible during rollouts.

Those contracts are now documented in `docs/phase_3d_environment_audit.md`.

## Dimensions for CTDE Review

Flat primitive-action 3D environment:
- `obs_dim = 114`
- `state_dim = 89`
- `action_dim = 1056`

Hierarchical wrapper:
- High-level `action_dim = 10`
- The base environment still has primitive `action_dim = 1056`
- Any CTDE design using the hierarchical wrapper must audit decentralization and information leakage from the executor.

Legacy Scenario 4:
- `obs_dim = 97`
- `state_dim = 71`
- `action_dim = 864`

## CTDE Work Not Included Here

Do not implement CTDE in Phase 3D-6.

Future CTDE work should decide:
- Whether agents train on flat primitive actions or hierarchical actions.
- Which global state fields the centralized critic or mixer can see.
- How to avoid leaking centralized-only information into decentralized execution.
- How old checkpoints should be handled when dimensions differ.
- Whether additional served-target 3D distance metrics are needed.
