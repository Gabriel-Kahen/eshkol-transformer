# CPU-v1 current-main source reconciliation — 2026-11-25

Status: **source candidate / HOLD for CPU-v1 admission**.

This branch composes draft #214 (`c71d39a`) and its #216 admission-review and
mutant-gate successor (`70220bbe`) onto main `8d185344`. The only textual
source-merge conflict was the CLI3 installed-entry comparison test. Current
main's exact D1-route reconstruction was retained; its eight tests pass.
The #216 E3 frame digest repair matches the composed `e3_frame.c` bytes and
restores the first-evaluation mutant test. No public API, dtype, shape, or
checkpoint contract was changed in this reconciliation.

The dependency graph remains explicit:

- #218 then #226 provide private P2/G2 pending-carrier prerequisites for
  same-call terminal output. They establish neither public terminal generation
  nor trained save/reload continuation.
- #221 is the G3/C2 one-use heavy-gate controller on #216. Its generated
  Eshkol replay and authentic SAVE/reader results are still unrun.
- #223 is a separate E3 stage-3 partial-pin launcher on #216. It does not
  discharge the complete section-9 numerical, mode, gradient, failure, and
  resource matrix.

Source checks on this composed tree: 62 focused Python tests pass, including
CLI3, E3 fault/retention/mutant and G3 SAVE/reader tests; CI topology passes
23 shared suites and 38-command coverage; changed shell files pass `bash -n`;
the G3 replay host gate passes its source/mutation and small C native-state
checks. The host gate uses stubbed pin/cache dependencies and does not compile
the generated Eshkol replay. Existing #214 hosted CLI3 and E3 results are
bounded to their original exact tree; they are not current-tree CI evidence.
In particular, #214's fixed-profile resumed/uninterrupted checkpoint parity
does not prove full trained continuation. Main's installed binary establishes
one-update, one-byte generation, not terminal multi-token or EOS output.

The next G3/C2 generated qualification requires a fresh exact-source #221
preparation and independent recipe/seal review. Its controller command is
`python3 -B -m tests.g3r.one_use_gate /absolute/fresh/package EXPECTED_SEAL_SHA256`.
The controller resides on the separate #221 branch and must be reconciled to
the frozen source tree before use.
Under both heavy-build locks, it requires **16 GiB available RAM, 8 GiB free
swap, and 32 GiB free on actual scratch/output storage**. The measured
standard hosted runner fails the RAM and swap floors; no new full matrix was
launched. E3 #223 separately requires at least 32 GiB free on its actual
scratch filesystem and remains subject to the preserved stage-8 host floor
and complete section-9 evidence requirements. Resource checks must be
repeated under the locks on the actual execution host; historic readings are
not a qualification.
