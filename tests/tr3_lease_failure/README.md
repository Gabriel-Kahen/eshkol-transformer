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
commit and injects every prefix across all eight direct arena allocator symbols
in one measured stream. The caller selects O0 or O2 with `--optimize`; each
optimization level gets a separate artifact and evidence directory. This mode
must pass before any claim that the upstream repair closes the three crashing
families for TR3 lease acquisition. On the production-profile `de0b249` runtime,
the first 950 full-matrix hit cases pass, but ordinal 950 turns a real bounded
string allocation failure in `eshkol_utf8_substring` into an X1 `unsupported`
error. The condition check correctly rejects that error. A separate cold
post-staging census records 31 wrapped allocations, starting with symbol
interning in P1 final recheck; denying the first real operator-new call there
raises an uncaught `std::bad_alloc`. The full gate remains blocked. The
counterexample artifacts and runtime build provenance are recorded in the TR3
roadmap entry.

The test measures exact component bytes and counters around each failure. It
does not claim flat process memory, sanitizer coverage, package localization,
hostile-link resistance, thread safety, or complete TR3 behavior.
