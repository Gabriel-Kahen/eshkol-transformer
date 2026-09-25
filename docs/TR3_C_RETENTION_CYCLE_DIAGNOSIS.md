# TR3-C private cycle retention diagnosis

The test-only `tr3_c_retention_cycle_root.esk` source-composes the accepted
TR3 snapshot/SAVE, C2 LOAD, and joint restore entries. It interposes the
accepted last recoverable O2 check before loading the joint restore source;
the real check runs on every successful cycle. The fixture creates a genuine
`A=2`, `K=1` source and an independent receiver, then performs a genuine
SAVE/LOAD/restore to prewarm lazy P1/C2 tensor-shell identities. It injects
one pre-seal failure, verifies both complete 42-tensor/control images remain
unchanged, and retries the same checkpoint and receiver successfully.

After that prewarm, each measured cycle overwrites one canonical C2 file from
the source and LOADs and jointly restores it into the receiver. Native F32,
O2, and P1 live authority counts, I2 owned-clone count, C2 live image count,
Scheme C2/O2 owner and restore-plan active counts, and enrolled trainer count
remain equal to their post-prewarm baselines. The source and receiver images
remain exact after every measured horizon. The Clang 21 build instruments
the test-native objects and generated AOT code with ASan/UBSan; LSan is
enabled at process exit. The pinned Eshkol runtime archive is uninstrumented.

| Cycles | Checks | F32 retired-control bytes | P1 tombstones | O2 dead states | C2/O2 Scheme records | I2 plan records | Arena increase | Runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 91 | 96,800 → 658,464 | 271 → 2,223 | 9 → 73 | 9 → 75 | 3 → 35 | 38,848,184 bytes | 12.66 s |
| 128 | 283 | 96,800 → 2,343,456 | 271 → 8,079 | 9 → 265 | 9 → 267 | 3 → 131 | 155,392,696 bytes | 73.82 s |

The F32 retained-control slope is exactly 17,552 bytes per cycle, P1 adds 61
tombstones per cycle, native O2 adds two dead states per cycle, and the I2
Scheme plan registry adds one terminal record per cycle. The C2/O2 Scheme
counts include two final observation snapshots; the loop itself adds two
terminal records per cycle. These are reachable retained records, so a clean
LSan result cannot establish flat total retention. The accepted
`f32_tensor.c` retired-control registry, `p1_identity.c` tombstone registry,
`o2_optimizer.c` terminal state registry, and Scheme `c2-training-state-states`,
`o2-states`, and TR3 restore-ledger lists explain the measured growth.

The pinned f31/LLVM21 network-disabled gate passed on clean `cd2899c` (tree
`81a9980`), with no ASan, UBSan, or LSan report. Compilation took 1:44.75,
peaked at 5,401,604 KiB RSS, and used no swap. The full evidence is
`tr3-retention-cycle-cd2899c-20260925/SHA256SUMS`, SHA-256
`b8280e9066e8ba988a132338d5181491a07d1802b7c7b1f770816d30f000a1d7`.

The 1,024 and 8,192 horizons remain unrun. The 128-cycle point reaches
642,088 KiB RSS and 73.82 seconds. Section 13 requires no growth in *live*
authorities at those horizons; the accepted joint implementation plan already
permits linear terminal records and explicitly forbids a flat-total-retention
claim. This test establishes the shorter live-authority result and measures the
terminal cost. It does not establish either required long horizon or a bound on
total process memory.

## Integrated cost investigation

Test-only successor `b9824e9` adds monotonic timing around the genuine
overwrite SAVE and C2 LOAD/joint restore calls. It changes no accepted runtime
source and preserves the live-authority, exact-image, and terminal-count checks.
The pinned ASan/UBSan/LSan gate passes again:

| Cycles | Wall time | SAVE | LOAD/restore | Peak RSS | Arena increase |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 13.53 s | 7.284 s | 5.133 s | 277,292 KiB | 38,848,192 bytes |
| 128 | 71.63 s | 37.721 s | 32.418 s | 638,752 KiB | 155,392,704 bytes |

The source/test evidence is sealed at
`tr3-retention-phase-b9824e9-20260925/SHA256SUMS` (`392787c0...`).
Both real phases get slower as terminal identities accumulate. The prior
1,024-cycle timeout is CPU-bound (1,183.89 user seconds in 1,205.12 wall
seconds), with no sanitizer completion or final authority census.

A separate `-pg` diagnostic build of the same test source passed 128 cycles.
Its largest generated function took 15.58 of 65.95 sampled CPU seconds
(23.62%). A breakpoint at its 20,001st invocation, inside a measured joint
restore, shows the path through TR3 lease authentication and the accepted P1
fixed-set recheck. Source inspection attributes the loop to traversal of P1's
append-only shell registry; this attribution uses the call stack and source,
not a compiler line mapping. The P1 `scan-states` and `scan-modules` loops
account for another 5.20% self samples. Scheme C1 SHA arithmetic is also a
substantial fixed cost. Native P1 `find_record` is 1.33% self samples. The
profile and stack are sealed at
`tr3-retention-gprof-b9824e9-20260925/SHA256SUMS` (`4639f62b...`).

The horizon cannot be made equivalent by clearing terminal registries,
restarting processes, skipping real SAVE/LOAD/restore, or bypassing P1
authentication: each removes the growing retained state under test. The
current 1,200-second gate does not establish 1,024, and 8,192 remains unrun.
An accepted P1 identity-indexing contract and implementation, or a separately
measured larger runtime budget, is the next dependency. Neither long-horizon
live-authority result nor flat total retention is claimed.
