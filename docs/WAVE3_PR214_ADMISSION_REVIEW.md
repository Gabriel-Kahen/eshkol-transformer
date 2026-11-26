# PR #214 bounded admission review — 2026-11-14

**Full Wave 3 admission remains HOLD.** Independent source review found no
production defect in the CLI3/G3/C2/E3 union. Focused tests found one stale
test checksum, corrected below. Existing supported CI proves more runtime
behavior than the previous roadmap recorded, but leaves the generated
Eshkol G3-R/C2 SAVE and E3 §9 gates open.

## Exact source and supported evidence

Reviewed [PR #214](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/214)
head `c71d39a98411d2285fa099f63d1c56170069da3c` against fetched main
`ee24ef461ef7e313c36c6c6f97ba4f3d8155bac7`.
Supported [run 38661209451](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/38661209451)
has 24 successful checks. The logs identify checkout
`5e3b77cde6043ebfeb1d0455d7bd4ebb885efe79`; fetching `refs/pull/214/merge`
and comparing it to the head proves both have tree
`34ae8017fcb037548974073926bb26ef5fee7fdc` (`git diff --exit-code` passes).
This is identical-tree evidence, not an inference from check names.

| Gate | Inspected supported result | Bound |
| --- | --- | --- |
| [CLI3 commands](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/38661209451/job/116183228215) | `CLI3-PRODUCTION-GENERATE-PASS` and `CLI3-CROSS-COMMAND-CHAIN-PASS` | Compiled production executable with D1 dispatch, pretrain/evaluate/generate, complete resumed/uninterrupted C2 byte equality, matching evaluation/generation JSON, identity/corruption negatives and unchanged input images; fixed CPU f32/context-two/byte profile only. |
| [E3 native gates](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/38661209451/job/116183228243) | Cleanup mutant rejected; frame normal and leak-enabled ASan/UBSan each pass 22,223 checks; five-case parity/repeat/sanitizer gate passes. | Native frame/provider evidence; not generated Eshkol fault/retention-tuple or §9 admission. |

Downloaded job logs and run/check metadata are retained in the assigned
worktree at `build/evidence/pr214-admission-20261114/`, with `SHA256SUMS`
(SHA-256 `dc70f09fac2384e06817bf674bffb4233b9f22773950ebbea3069329d7aa3a7d`).
GitHub returned no retained artifacts for this run. Per-command redirected
JSON/checkpoint files are therefore not independently available from that
hosted run; the verdict is bounded to the inspected scripts, job logs and
successful job exits. Historical preparations and failed evidence are untouched.

## Scoped repair and local checks

`c0d1edfe` moved a guarded partial-pin forward declaration into `e3_frame.c`.
The first-evaluation mutant test still pinned its preceding bytes. The only
test change updates that expected SHA-256 from `248e9149…` to `848317ac…`;
the full-file equality assertion and exact manifest/driver checks remain.
No production file, API, tensor shape, dtype or checkpoint format changes.
The new test/docs tree itself has no hosted run claimed here.

- Before repair: 65 focused tests, one failure at the stale checksum; raw failure retained.
- After repair: the same 65 tests pass, including CLI3 source contract,
  G3 authentic SAVE/file-reader source tests, E3 fault/teardown/overflow/
  first-evaluation/sanitizer/unresolved-policy tests, M3CG contract and Q0 isolation.
- `python3 -m unittest tests.e3_private.test_retention_first_eval_mutant`: 7 pass.
- `bash scripts/test-g3r-replayed-p1-g1-host.sh`: source/mutations and native state pass;
  its stubbed pin/cache dependencies do not prove generated Eshkol replay.
- Changed shell files: `bash -n` passes. `bash scripts/check-ci-topology.sh`:
  21 suites / 36 commands pass. `git diff --check` passes.

## Remaining gates and resource decision

1. Same-registry generated Eshkol replayed P1/G1 fixture, normal/repeat/
   sanitizer, with genuine pin/cache/sampler dependencies.
2. `scripts/test-g3r-c2-authentic-save.sh FRESH_OUTPUT_DIRECTORY` on the
   exact candidate: compiled no-SAVE control, fresh authentic public SAVE
   bytes, derived companion, same-image reader and normal/repeat/sanitizer
   results. This is the next bounded heavy gate, not proof of restored
   model/public continuation.
3. E3 §9 generated Eshkol numerical/mode/gradient/failure/resource gates,
   including normal, poison and leak-enabled sanitizer profiles; remaining
   stage cuts and resource horizons still need their own evidence.
4. Independent review of retained runtime results before broader G3/TR3,
   CLI3 or Wave 3 admission.

Existing authentic-SAVE and stage-8 one-use preparations have empty outputs
and no runtime result; their older source pins cannot be relabeled as this
union. They require a fresh exact-source preparation. Both prescribe a
32 GiB free-disk floor. The workspace had only 8.6 GiB free at this review,
so no multi-GB compile, claim or release was started. No existing run was
repeated. Any later heavy attempt must first satisfy its resource checks and
hold `flock /tmp/eshkol-transformer-heavy-build.lock` for the whole build.
