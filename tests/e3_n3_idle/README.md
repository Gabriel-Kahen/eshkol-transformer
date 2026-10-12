# E3 N3 idle real-model source candidate

This noninstalled candidate uses an authentic fresh seed-1729 M3 model and the
existing I2 decoder and whole-batch copy plan. It checks the 15 logical paths,
14 distinct live f32/cpu owners, the one head/token-embedding tie, exact shapes,
4,736 parameter bytes, unchanged gradient words/metadata and initializer
successor, and zero parameter words after commit. The optional third scenario
field seeds one real nonzero gradient before the load. The old two-field
`e3-test-run` entry is untouched. Each successful case gets a fresh model,
retained in a bounded eight-slot process root because M3T has no reviewed
postconstruction release.

**HOLD: this source is not AOT reachable yet.** `scripts/e3-private-package-policy.sh`
admits only its current exact tuples; `scripts/build-e1b-consumer.sh` exposes
test-only parameter/gradient observers only for the mode/graph tuple. The next
dependency is an independently reviewed exact N3 tuple with source and native
closure, objects, imports/exports/strings, private rename, test-only native
observer flags, and near-match rejection. The candidate bridge and root are
deliberately not added to canonical package manifests or installed exports.

Source guards: `python3 -B -m unittest tests.e3_n3_idle.test_source_contract`.
They establish source order and tuple isolation only. The generated AOT path,
preprepare negative/fault cases, full 17-mode and typed RNG observations,
role/logit transcript comparison, and §9 rollback/retention gates remain open.
The absent `tests/e3_private/retention_driver.esk` must not be treated as an
available observer; this candidate uses the current test-only observers in
`src/eshkol_transformer/e3_frame.c`.
