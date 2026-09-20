# M3 retention gate: measured failure and accepted prerequisite fix

Status: after the accepted I2 prerequisite fix, the full canonical and clean
instrumented local gates **pass all five modes at both 1024/8192 horizons**, with
unchanged production flags and 600-second caps plus five-second kill escalation.
Exact-head supported Ubuntu22/LLVM21 CI and independent M3-R remain required.

The initial canonical forward/8192 run failed its unchanged cap, exit 137, without
a completion marker or exact arena counter; its remaining modes did not run.
That failure remains historical evidence below. It is not relabeled a pass.

## Historical evidence before the failed gate

Local evidence uses the pinned Eshkol source with explicit CachyOS / LLVM-Clang
22.1.6 compatibility permission. It is not supported Ubuntu22/LLVM21 evidence.
Canonical and two fresh packages agree byte-for-byte, with 93 globals, 87 exports,
seven facades and all ten new private seams localized. Public logits, lifetime,
privacy and arity checks passed. The separately instrumented exact public
wrappers passed all 14 unique gradients in all three cases, exact initializer
bits and count/weight metadata. Its numerical stdout also agrees byte-for-byte
with the diagnostic native-O2 variant described below.

The native gate passed 76,484 checks and 1,000 lifecycle cycles in 101.420s.
ASan/UBSan/LSan passed 20,679 checks and 100 cycles in 1.928s. Both modes retain
the complete semantic/failure cases; the sanitizer lifecycle trajectory is
deliberately shorter. The separate existing-I2 1,000-cycle probe passes both.
Native-test fixtures inject readiness/slot bytes and do not execute a C model
schedule. Whole-model numerical evidence comes from the public Eshkol path.

## Historical bounded timing and stack observations

The diagnostic variant rebuilds only the seven private native translation units
at `-O2`. It reuses the same trusted Eshkol object, boxed bridges and provider
objects, retaining finite/fenv flags. Its 97 test globals are unchanged; the
compiler substitutes `bcmp` for `memcmp` in its undefined inventory. This is a
test-only diagnostic artifact, not an accepted production build or replacement
for the failed canonical gate.

| Public operation | Iterations | Original native flags, wall seconds | Diagnostic native O2, wall seconds |
|---|---:|---:|---:|
| Forward/release | 10 | 0.14 | 0.04 |
| Forward/release | 100 | 1.92 | 0.38 |
| Forward/release | 300 | 12.05 | 3.20 |
| VJP | 10 | 0.17 | 0.04 |
| VJP | 100 | 1.40 | 0.16 |
| VJP/zero-grad | 10 | 0.19 | 0.05 |
| VJP/zero-grad | 100 | 1.70 | 0.30 |
| Rejected nonfinite-seed VJP | 10 | 0.12 | 0.04 |
| Rejected nonfinite-seed VJP | 100 | 0.83 | 0.12 |

Every bounded probe exits successfully and has identical original/O2 native
counts. Forward100→300 triples iterations but increases time 6.28× original and
8.42× O2. Optimization reduces constants without eliminating superlinear cost.

Twenty timed GDB interruptions over approximately ten seconds of a separate
original instrumented forward process found 19 stacks inside
`storage_aliases_live` / `ranges_overlap` / `pointer_span_fits`, and one inside
native tensor cloning. No sample stopped directly in Eshkol. This locates the
dominant sampled work; it is not an exhaustive profiler or a proof that Eshkol
lookup contributes no cost. That diagnostic process was deliberately terminated,
without disturbing the canonical horizon run.

## Historical retention evidence and limitations

Canonical forward/1024 completes with exact Eshkol arena allocation
**5,177,344 bytes** and peak RSS **95,476 KiB**. Approximately 132.38 seconds is
estimated from output-file creation to final write, not a stopwatch measurement.
The timed-out canonical8192 process reached **97,580 KiB** peak RSS; no completed
arena trajectory can be inferred from that RSS observation.

The separate instrumented forward/1024 matches all 36 source-derived native
fields: zero live graphs/logits/frames/plans/borrows, 108 live I2 tensors and 14
parameters, 32,868 f32 payload bytes plus 3,952 metadata bytes, and two I1 tensors
with 32 payload bytes. These remaining tensors belong to the deliberately live
model and reusable workspace. Fixed provider admission also retires eight
temporary I2 tensors, one 40-byte copy plan and one 40-byte copy builder; the
checker derives and includes that baseline from P1 provider admission.

The source-derived per-iteration retained native control costs are forward1768,
logits1888, VJP48, VJP/zero-grad128 and rejected-seed VJP0 bytes. The public reset
cost includes a 48-byte gradient plan, 40-byte reset plan and **40-byte reset
builder**. These are not completed 1024/8192 measured slopes: the full native
and Eshkol trajectories remain unmet. Fixed K1/P1 allocations, Eshkol identities
and errors are separate accounting. No flat-memory or performance claim follows.

## Accepted prerequisite fix and local validation

Integration [accepted the narrow I2 allocation envelope for implementation](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5744597535)
in a separate prerequisite-fix commit after runtime checkpoint
`4a317d4d12d050944a2c20ed9c283438629a9297`. This acceptance does not approve
the runtime or waive any retention, numerical, ownership or supported-CI gate.

The implementation adds a conservative allocation envelope to the existing I2
f32 implementation.
A read-only audit found that every protected span visited by
`storage_aliases_live` originates in its one `f32_calloc` helper: tensor shells,
shapes, strides and data; parameter and borrow shells; copy-plan assignments;
gradient-plan entries and prepared values; reset-plan tables; and all retained
shells. External identities, provider statics and M3T stack guards are not scan
members today, and this fix does not change their coverage.

Maintain monotonically expanding address bounds after every successful nonzero
allocation. At the alias scan entry, return false early only for a nonempty,
non-null, representable candidate span wholly outside those bounds. For every
other case, run the existing complete scan unchanged. If successful-allocation
size/address arithmetic cannot be represented, permanently disable the shortcut
without changing the allocator result. Never shrink/reset the envelope, including
after failed construction, free, retirement or test allocator reset.

This adds 24 bytes of local static integer state on the supported x86-64 ABI,
accounted separately from per-iteration retired controls. It adds no heap
allocation, reclamation or public API, and preserves ownership, authentication
and numerical behavior. It
is constant-time only outside the envelope; address spread and inside-envelope
queries retain the full cumulative cost. It is not a promised solution to 8192
until measured on the canonical build. The existing 24-entry N3K predecessor
source inventory contains no entry for `native/f32_tensor.c`; all 24 pins pass
unchanged. Integration confirmed that this zero-entry result requires no new pin.
The exact source hashes are:

- Before: `c95fb958445bceb565516511690074a1e6139fae07faa63553670549488bd9ce`.
- After: `cf66922cf4c814396c4eb9e67a843babba3f45b8c3cedb35cb265ff66d7aff64`.

The original scan body is unchanged after its private rename. Independent local
object inspection finds identical defined/undefined global inventories and one
24-byte local BSS envelope. The independent differential suite passes four fresh
processes in each of normal and ASan/UBSan/LSan builds: coverage 837 checks,
product-overflow 849, endpoint-overflow 849 and uninitialized disablement 22
(2,557 per build). It explicitly constructs 13 live storage classes and all six
retired shell classes through existing APIs, checks scalar/empty storage and
boundary cases, preserves empty-registry invalid-span behavior, and tests
partial failures, freed holes, allocator reset and zero-byte allocation counters.
A separate reviewer found no blocker in the test inventory or oracle. These are
local compatibility results. The full existing `I2_ASAN_DETECT_LEAKS=1
scripts/test-i2.sh` gate also passes, including C/C++ ABI, exact-f32 storage,
parameters, gradients and transactions, K1 views, borrow/stale/alias/failpoint
cases, public Eshkol integration, private/duplicate-authority rejection, hostile
fresh-build determinism and ASan/UBSan/LSan. Its log is
`build/i2-alias-full-gate.log`. Independent source, ABI and sanitizer review found
no blocker in the prerequisite delta. The full canonical M3 result is recorded below.

The unchanged M3 native gate also passes after the fix: normal 1,000 cycles /
76,484 checks in 1.502 seconds; ASan/UBSan/LSan 100 cycles / 20,679 checks in
0.075 seconds. The separate I2 plan probe passes 1,000 cycles in each variant.
Retired-control counts are unchanged, including 1,496,000 I2 bytes for the native
1,000-cycle fixture. Log `build/m3-native-envelope.log` has SHA256
`768ddc666b44f7679e84f5db41115e021a9f9631bfca7e612e06966fbd1afa7d`.

A composed bounded probe recompiles only the changed I2 integration translation
unit with unchanged production native flags, reusing the original instrumented
Eshkol object, bridges and providers. Its exact 97 globals, undefined inventory
and source closures pass. All three cases' 14 gradients, initializer bytes and
metadata, and all 36 native fields at each bounded horizon are byte-identical
to the original. Forward/release takes 0.03 / 0.10 / 0.42 seconds at 10 / 100 /
300 iterations, versus the original 0.14 / 1.92 / 12.05 seconds. This supports
attempting the unchanged canonical gate; it is composed compatibility evidence,
not fresh-package determinism, a full retention trajectory or supported CI.
Artifacts are in `build/m3-alias-envelope-probe`.

The completed local proof includes differential comparison against the original scan across all
live/retired storage classes, interior/straddling/endpoints, envelope holes,
outside spans, zero/null and overflowing spans; allocation failure, freed-history
and permanent-fallback cases; existing I2 alias/stale/ABA/atomicity gates; then
unchanged M3 numerical/ownership and canonical retention gates. No timeout,
horizon, optimization flag or public-contract change substitutes for that proof.

The original failed checkpoint evidence is preserved with a hash manifest under
`build/m3-pre-envelope`; it must not be replaced by subsequent successful evidence.

Detailed local logs remain in `build/m3-public-evidence`, `build/m3-numerical`,
`build/m3-native-o2-probe`, `build/m3-profile/stacks.log`, and
`build/m3-native-final.log`. The native gate log SHA256 is
`19586d55541b1030c44c8452f085549a154611bcd5737c7bd96ac10549a777a5`.

## Clean instrumented result after the prerequisite fix

A single clean `scripts/test-m3-numerical.sh` invocation passes all three public
numerical cases, all 14 unique gradients, exact initializer bytes and metadata,
plus all five modes at both 1,024 and 8,192 iterations. Each case completes within
the unchanged 600-second timeout and matches all 36 source-derived native fields;
stderr is empty. Exact 97-global, undefined-symbol and source/native-closure
checks pass. Numerical stdout is byte-identical to the original. These are fresh
final-source results, separate from the earlier composed probe.

The complete measured native retained-control slopes are 1,768 bytes/iteration
for forward, 1,888 for logits, 48 for VJP, 128 for VJP/reset, and zero for rejected
nonfinite-seed VJP. The reset slope includes the 40-byte public reset builder.
All modes end with zero live graphs, logits copies, frames, plans and borrows;
the deliberately live model/workspace retain the same 32,868 f32 payload bytes.
The I2 envelope's 24 static bytes, fixed K1/P1 storage and Eshkol arena are separate.

Stable evidence is in `build/m3-numerical`, including exact raw outputs, toolchain
provenance and a clean completion status. The full invocation log is
`build/m3-envelope-clean-numerical.log`. This script uses a timeout, not GNU time;
no elapsed-time measurement is claimed for its individual instrumented cases.
CachyOS / LLVM22 compatibility does not establish supported Ubuntu22 / LLVM21 CI.

## Complete canonical result after the prerequisite fix

One refreshed canonical build and one full `scripts/test-m3-package.sh` invocation
pass on runtime commit `c41d0d240184a0426068347f3b4cdcc6fac3e33e`. The gate
checks two fresh packages against the canonical artifact, exact one-member
93-global/87-export closure and ten localized seams, source/native dependency
closures, caller and stdout determinism, all three public numerical cases,
reverse imports, privacy, 15 arity negatives, registry collisions and lifetime.
All five modes complete at both contracted horizons under the original caps.

| Mode | Arena bytes: 1024 / 8192 | Peak RSS KiB: 1024 / 8192 | GNU elapsed seconds: 1024 / 8192 |
|---|---:|---:|---:|
| forward | 5,177,344 / 7,274,496 | 97,316 / 113,540 | 3.83 / 325.62 |
| logits | 5,505,024 / 9,633,792 | 97,528 / 115,948 | 4.81 / 315.27 |
| vjp | 4,915,200 / 4,915,200 | 95,832 / 96,036 | 0.37 / 3.03 |
| reset | 13,369,344 / 72,548,352 | 105,044 / 167,872 | 2.23 / 87.73 |
| failure | 9,830,400 / 44,236,800 | 99,468 / 134,612 | 0.51 / 3.74 |

These are actual GNU-time elapsed/RSS fields and exact reported Eshkol arena
allocation totals. Arena differences reflect measured allocation trajectories,
including allocation granularity; they are separate from the source-derived and
independently measured native control slopes above. Reset and failure have
substantial cumulative Eshkol allocation despite releasing temporary payloads.
The envelope does not establish flat memory, general performance or training
retention guarantees.

Raw evidence is retained in `build/m3-public-evidence`; the parsed table and
per-file checksums are in `build/m3-envelope-public-result.json`. Full log
`build/m3-envelope-public-gate.log` has SHA256
`daf65c75d05840f96af562026580022c8ae42fb2f23d3cae896baba98d052820`.
The canonical object SHA256 is
`45324d8bc855f1116736ad9741252a75cc70f88dc3f01102bc8c35e52ed8f950`;
archive SHA256 is
`b5cb08b9fcaa4d17b22b08fb0652b194973e84619349c41da6374e248de4eda0`.
Local timing uses GNU time 1.10 with the explicit configured executable path.
All measurements use the verified pinned Eshkol source under CachyOS / LLVM22
compatibility settings; supported CI and independent M3-R remain pending.
