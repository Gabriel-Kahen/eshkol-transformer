# Bounded E3 stage-3 admission successor

Status: source candidate; compiled result pending. E3 §9 remains HOLD.

The baseline is reviewed PR216 `70220bbe0d27a9a8110deeedaf873055f11ca9a6`.
The selected delta is the stage-3 package/launcher from `188169be`, one-shot
report isolation from `761cb395`, and fault-tuple sanitizer routing from
`48bc75a6`. Existing production source, numerical predicates and PR216's
causal second-normalization verifier remain unchanged. The noninstalled fault
report bypasses the separate 1,024-call arena reconciliation only for site 4,
ordinal 7, occurrence 1; all other scenarios retain their existing report.

This gate covers **one partial-pin denial after seven pins are acquired**,
absent and seeded-present gradients, with same-authority retry against a twin.
Normal, repeat, poison, and ASan/UBSan with leak detection must all complete.
It does not cover site-3 external-borrow denial, all fourteen pin prefixes,
other stage cuts, complete-call retention, or the full R1/R2 matrix.

## Candidate selection

| Candidate | Decision |
|---|---|
| `b8defea9` section-9 source union | Defer the broad mutant/N5 union; distinct later leaf evidence exists. |
| `ea3b38aa` evidence matrix | Consult its accepted-result references; do not replace the current matrix wholesale. |
| `dc565e1d` measured mode imports | Preserve its accepted bounded predecessor result; no unchanged rerun. |
| `48bc75a6` stage-3 fault sanitizer | Select with its two prerequisite source deltas above. |
| `fc049b9f` stage-7 sanitizer | Defer this separate stage-7 gate. |
| `9bdf7572` boundary sanitizer imports | Diagnostic import admission does not repair the retained arena allocation. |
| `6a2e3376` N3 mixed correctness | Defer the distinct authentic-correctness experiment. |

The independently accepted T2 numerical F source is `44c0418d`, tree
`70706c0a25cf2055815de363eec6539823a8fca8`: five layouts, normal A/B,
poison, sanitizer, three mutant kills, exit 0 in 3,750.864 seconds. Its review
is `/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/e3/numerical-a-review/REVIEW.md`,
raw seal `f6b04ae125ae58710893ef5c32d3420de7063fabd2f844a140046e1c01d6e2c1`.
This remains exact-source predecessor evidence, not a current-source carry
claim. Its earlier rejected attempt remains nonaccepting. The bounded
`dc565e1d` mode/gradient-image result likewise does not establish full R2 or
an E3 backward operation; E3 is forward-only. Complete-call retention remains
nonaccepting at `failure-arena-bytes`; this stage-3 gate cannot discharge it.

## Execution contract

After source review, freeze a clean commit/tree and invoke:

```sh
python3 scripts/launch-e3-r1-stage3-compiled.py COMMIT TREE /home/gabe/.cache/et-e3-admission-20261115-evidence/stage3
```

The launcher serializes on `/tmp/eshkol-transformer-heavy-build.lock`.
Only the lock resides under `/tmp`; `TMPDIR=/out/tmp` and every generated
artifact reside in the `/home` evidence directory. A lock-time check requires
at least 32 GiB free on that actual scratch filesystem. The source/common Git,
fe9 dependencies and pinned Q0 environment mount read-only. The network-disabled
read-only f31 image is pinned by full digest. Three CPUs, 6 GiB memory and
7 GiB memory-plus-swap are explicit tighter experimental caps, not minimum
sufficiency claims. Record actual Docker HostConfig, guest cgroup v2 limits,
peak memory/events, command GNU-time RSS, host memory/swap, scratch space,
compiler/runtime stderr and terminal custody. An OOM or timeout is a preserved
resource failure, never a numerical pass or reason to loosen assertions.
No new RAM/swap minimum has been invented from the historical 12 GiB cap.

The existing compiler timeout remains 900 seconds per package compiler call,
5,400-second package/AOT command bounds, and 14,400-second global bound.
Failure output is retained. A successful guest marker still requires host
custody eligibility and independent result review before admission.
