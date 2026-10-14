# E3 N1 generated-resource source union on current main

Status: **source/tests/docs candidate; all protected and E3 §9 gates HOLD**.

This isolated union starts from main `ba4954d26892d2dbc6331f441777c0fb74b510b3`
(tree `9e7af8efb4817ecd192881d6f991f7ce553cf73c`) and carries the 56
N1-specific paths from the independently accepted descendant
`5dc1e475b9533b9dddf329a4529f80dc7806bdcb` (tree
`35f79d9642c9aa5f1f4856e76b332faeccad9ccf`): 28 `docs/e3` records,
three broker tests, and 25 private Python source/test files. The ancestor
`62f1e2a5` parent-dirfd readback repair and `794fe465` actor census/schema
correction are included. No existing main file is replaced or deleted.
The older [b983 union](E3_N1_B983_UNION.md) records its own historical base;
this successor preserves the newer main source, CI, and E3 §9 files.

One source adjustment makes the synthetic destination restriction independent
of worktree location: `observe_synthetic` rejects a destination under its own
repository even when an isolated checkout itself lies under `/tmp`. The
existing repository-target negative test caught this case. The generated
attempt contract also clarifies that denial of four listed spawn syscalls in
fixed roles is narrower than whole-graph spawn and migration control.

Focused host checks on this union pass 112 N1 private tests, 24 broker tests,
107 CI source tests, CI topology, Python AST parsing for 28 N1 files, and
local-link resolution for all 24 N1 Markdown files. The credentialed observer
cases require Linux Unix credentialed sockets; this sandbox denies those
sockets with `EPERM`, so the passing N1 host run used the normal host facility.
These checks are generated-only source evidence. They do not read a real OCI
source or establish protected pre-read custody, trusted whole-attempt actor
closure, the 30-second/268,435,456-byte bound, installed behavior, or any
E3 §9 acceptance row. The distinct-UID/private-scope prerequisite and fresh
supported proof remain open.
