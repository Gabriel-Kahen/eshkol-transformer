# P1 post-index retention profile

On exact `07dbc6e`/tree `c53e5600`, the private TR3-C retention fixture ran
genuine canonical overwrite SAVE, C2 LOAD, and joint restore in one process
after identity prewarm. The pinned recovered f31 runner and LLVM 21 container
compiled test native objects and generated AOT with ASan/UBSan and `-pg`;
LSan ran at exit. The heap limit was explicitly 4 GiB. Both runs preserved
the complete source/receiver image and flat native and Scheme live-authority
counts. The authoritative append-only P1 registry was never reset.

| Cycles | Checks | SAVE | LOAD/restore | Wall | Peak RSS | Arena increase |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 283 | 33.018 s | 25.485 s | 60.03 s | 716,668 KiB | 156,440,264 B |
| 256 | 539 | 78.255 s | 67.630 s | 147.59 s | 927,296 KiB | 312,914,320 B |

The 256 test-only fixture adds that horizon to the existing accepted-horizon
predicate and changes no production code or measured operation. The original
fixture's attempted 256 invocation rejected at that predicate before entering
the cycle loop. Wall time grows 2.46 times for twice the cycles, while arena
growth is almost exactly twice. The `-pg` timing includes instrumentation and
should not be compared directly with the separate non-profiled 1,024-cycle
proof. No sanitizer error was reported.

The earlier dominant `named_let_loop_363` lookup scan is absent after P1's
advisory shell index. In the 128/256 flat profiles, generated P1
`scan-states_392` grows from 2.08 to 7.66 self seconds and `scan-modules_396`
from 1.97 to 6.62; together they grow 3.53 times. The raw-shell creation
`find_364` grows 1.82 to 6.02 seconds, state-entry `find_365` grows 1.05 to
3.65, and native `find_record` grows 0.90 to 3.79. These source attributions
are inferred from uniquely named loops and the source, because gprof has no
Scheme line map. Scheme C1 SHA arithmetic also remains a large fixed cost.
The full profiles and input/build provenance are sealed at
`/home/gabe/.codex/evidence/eshkol-transformer/p1-post-index-profile-20260925/SHA256SUMS`
(`cbda5773...`) and
`/home/gabe/.codex/evidence/eshkol-transformer/p1-post-index-profile-256-20260925/SHA256SUMS`
(`8991e7f2...`). The separate, non-profiled 1,024-cycle 4 GiB sanitizer proof
is sealed at `p1-index-retention-1024-4g-20260925/SHA256SUMS`.

## Candidate active-record contract

The next possible optimization is an active-only state/module record index,
but it needs an accepted contract before code changes. The append-only shell
registry remains authoritative for shell identity, stale rejection, and
rollback. A derived index may contain only exact registry records for live
states or admitted modules. Publication must prepare capacity and promoted
records before exposing the new shell, and either finish both publications in
a nonraising tail or invalidate the index so reads use the registry scan.
State release must remove the active state before invoking provider cleanup;
aborted construction must remove its module records when their registry raw
fields become dead markers. Neither transition may remove terminal registry
records or restore authority to a stale shell. A missing, invalid, or
incomplete index must fall back to the full authoritative scan.

A separate implementation proof should cover capacity/allocation failure at
publication, mid-construction rollback, pre/post native state-release
failpoints, stale and forged shells, module abort/seal, and exact old-or-new
owner state under callback failure. It must compare 128/256 genuine-cycle
time and counts before and after, then run the still-open 8,192 live-authority
horizon under a measured resource budget. The existing 1,024 result proves
that horizon only on the current indexed lookup source. No active-record
implementation or performance improvement is claimed here.
