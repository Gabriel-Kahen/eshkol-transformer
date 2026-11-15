# Wave 3 source integration — 2026-11-15

**Source integration reviewed; generated runtime gates deferred; full Wave 3 HOLD.**
Draft [PR #221](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/221)
stacks on PR #216. Existing PR #214/#216/#218 branches are unchanged.

## Composition and scoped fix

The isolated `codex/et-wave3-integration-20261115` branch starts at PR #216
`70220bbe0d27a9a8110deeedaf873055f11ca9a6` and cherry-picks reviewed PR #218
`23c544b66406b1e8a5397c3d0d566c56c178e3e3` as `5ca258e1`.
All seven non-roadmap #218 files are byte-identical; #216 source/test contracts
are retained. No production API, tensor shape, dtype or checkpoint format changes.
The inherited private pending-allocation anchor retains its bounded abort-only
scope; this integration adds no public G2 or continuation result.

Combined checks exposed a Q0 omission. Commit
`ff1ef2c175f2e4135ae703ee557778da35a47156`, tree
`b7d604f3eb036fa59f4816cf9e4315df5fbdd44d`, admits only
`scripts/check-g3c4-p2g2-terminal-pending-anchor.py` to the development-script
allowlist. The existing generic near-miss test checks suffix, nested, source
and native variants for this new entry. Independent review accepts the union
and the one-line Q0 fix.

## Evidence and limits

- 70 focused tests pass after the fix; the original Q0 failure is retained.
- Pending-anchor checker: seven mutation controls pass. Replay host/source
  checks, shell syntax, `git diff --check`, and 21-suite/36-command CI topology pass.
- Lightweight container preflight passes on Ubuntu 22.04 / Clang-LLVM 21.1.8,
  image `sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99`.
  It verifies exact source, pinned fe9 runner/archive/provenance hashes, Q0
  and synthetic oracle controls. Its 72,900 KiB peak measures preflight only.
- Generated Eshkol replay and authentic C2 SAVE/reader **did not run**.
  Both heavy requests returned deferred status 75 before compilation.

Raw commands, failures, receipts and checksums are retained in
`build/evidence/wave3-integration-20261115/` in the integration worktree.
The final preparation receipt pins the published documentation tree after
the source-tested `ff1ef2c1` tree; that final commit changes documentation only.
Historical immutable one-use packages remain untouched and supply recipes
and read-only dependencies, not runtime evidence for this source.

## Resource decision and next gate

Disk now satisfies the 32 GiB floor, but the observed available RAM/free swap
were 7,139,496/423,072 KiB against required 16,777,216/8,388,608 KiB.
No measured gate-specific alternative justifies lowering those floors.
Scratch and `TMPDIR` are under `/home`; `/tmp` holds only the shared lock.

The fresh preparation is **PREPARATION_NOT_RELEASED**, and heavy launch is
disabled even if resources recover. Before execution it still needs reviewed
one-use output/seal enforcement, a resource recheck while holding the shared
locks, and container custody through terminal results. Preserve the recorded
wrapper-only preflight failures when preparing that successor.

Then run genuine generated Eshkol replay and
`scripts/test-g3r-c2-authentic-save.sh FRESH_OUTPUT_DIRECTORY`, each with
normal/repeat/native-provider ASan/UBSan/LSan evidence. Final sanitizer linkage
does not instrument generated Eshkol IR. Independently review the retained
results; E3 §9 numerical/mode/gradient/failure/resource admission remains a
separate required gate. No fresh hosted-CI result is claimed for this branch.
