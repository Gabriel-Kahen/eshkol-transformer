# M3 retention gate: measured failure and accepted prerequisite fix

Status: canonical forward/8192 **failed its unchanged 600-second cap**, followed
by the five-second kill escalation, exit 137. No completion marker or exact
8192 arena counter was produced. Remaining modes did not run. Supported CI is
withheld until a credible canonical local gate passes. The accepted 1024/8192
horizons and production compilation semantics remain unchanged.

## Evidence before the failed gate

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

## Bounded timing and stack observations

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

## Retention evidence and limitations

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

## Accepted prerequisite fix, pending validation

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
no blocker in the prerequisite delta. The full canonical M3 gate remains pending.

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

Required proof is differential comparison against the original scan across all
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
