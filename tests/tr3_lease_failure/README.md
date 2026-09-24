# TR3 lease failure witness

This test-only witness links a pinned production runtime archive with GNU linker
wrappers. It does not rebuild or enable runtime failpoints. Each object failure
temporarily exhausts the real allocation arena and calls the real constructor.
Promotion bookkeeping failures throw from the real operator-new call boundary,
promotion target failures exhaust the real quiet arena allocator, and the
handler case denies the real malloc needed after the runtime handler pool has
been exhausted with live guards.

For each supported prefix, the witness constructs an authentic D2/M3T/P1/O2
tuple before arming, then verifies the prior trainer head, staging slot,
acquisition guard, component values and exact bits, gradients, moments, counters,
model mode/frame, and dataset cursor. Ordinary public mutations and a successful
retry use that same tuple. A separate short-region success counts every wrapped
allocator between canonical staging and enrollment and requires zero calls.

The object, promotion, and target-copy matrices are exhaustive for their
measured call streams: the shell gate starts a fresh process for each ordinal
and increases it by one until the first unhit ordinal succeeds. Every hit case
still verifies the failed state and retries the same authentic tuple before its
process exits. Process isolation prevents the successful retries retained by the
lease registry from exhausting the ordinary arena during enumeration. The gate
preserves per-case stdout, stderr, exit status, and input hashes, reports the
observed prefix counts, and fails if an ordinal is silently skipped. The handler
failure and post-staging census each run in their own process as well.

With `--allocation-class object`, the final81298 gate injects every vector and
cons prefix and records the other direct arena calls without injecting them.
This is partial evidence: final81298 generated O0 code dereferences null results
from `arena_allocate`, `arena_allocate_closure_with_header`, and
`arena_allocate_string_with_header` before the installed language guard can
observe or catch a condition. A separate preserved bounded-null counterexample
records exit 139 for all three families.

With `--allocation-class all`, the gate requires the reviewed repaired-runtime
commit and the exact source through `63756bd`; it injects every prefix
across all eight direct arena allocator symbols in one measured stream. The
caller selects O0 or O2 with `--optimize`; each level gets separate evidence.
On the frozen `f602a66` lease source with production-profile `de0b249`, the
first 950 full-matrix hit cases passed, but ordinal 950 translated a genuine
bounded substring allocation failure to X1 `unsupported`. The cold
post-staging census also found 31 allocations; the first real operator-new
failure there raised an uncaught `std::bad_alloc`. Those original artifacts
remain preserved. The focused O0 repair probe confirms ordinal 950 now has
typed E1 category `internal` and operation `config-fingerprint`, while cold
and repeated post-staging acquisitions have zero wrapped allocations. Full
O0 subsequently reached ordinal 1053, where a real bounded allocation failure
in T1 fingerprint copying was reclassified as D2 `invalid-argument`. The shared
D2 identity boundary now rejects foreign and missing tokenizers before calling
T2, while preserving T2's typed `internal/tokenizer-fingerprint` failure for an
admitted tokenizer. A focused O0 ordinal-1053 retry probe passes; the public D2
boundary still reports operation `token-dataset-open`. The supported public
D2 aggregate/executable passed 24 compiled rejection and recovery checks with
empty compile and run diagnostics. Independent review approved the repaired
source and runner. Canonical O0 passed all 1,145 cases/54,921 checks and its
manifest verifies. The first O2 run passed the same runtime matrix, but the
runner rejected four exact LLVM loop-vectorization warning pairs in compile
stderr; its failed-gate artifact is sealed separately. A pinned warning
allowance and passing canonical O2 rerun remain required for acceptance.

The test measures exact component bytes and counters around each failure. It
does not claim flat process memory, sanitizer coverage, package localization,
hostile-link resistance, thread safety, or complete TR3 behavior.
