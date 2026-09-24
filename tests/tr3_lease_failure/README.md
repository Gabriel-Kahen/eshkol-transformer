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

The successor gate accepts only `--allocation-class all`. It binds the approved
22-slot lease source and failure fixture to a separate production-OFF
`de0b249` runner and archive. Their paths, hashes, clean source commit/tree,
root/build provenance, CMake profile, artifact manifest, container digest, and
LLVM version are recorded in `failure_runtime_candidate.tsv` and checked before
compilation. This runner supplies the exhaustive allocation-prefix evidence; it
is distinct from the recovered final81298 functional runner.

The functional gate uses the accepted recovered final81298 runner with SHA-256
`d5c23f1a59bf8fa96fd54fe4f1e47ce9ae903db2dd0f93408f41345ab34014fa`.
Its manifest separately records the unavailable historical runner with SHA-256
`4a0e6303f7b85ed06fb753b52b62155235a3a77bca6c32aeb17241a28ed80be1`;
the recovered runner is not represented as equivalent to that lost artifact.
It proves the 1024/8192 functional horizons,
while this witness proves allocation-prefix rollback and successful retry.

The successor fixture additionally verifies that each successful retry owns a
22-slot idle record whose epoch-start cursor equals the pre-enrollment cursor,
whose RNG is `(philox4x32-10, 1, 1729, 0, 0)`, and whose T/U/E counters are zero.
The O2 diagnostic-only mode exists solely to capture the new compiler warning
bytes for review. A full O2 matrix refuses to start until the reviewed byte
count and SHA-256 replace the `pending` manifest fields. Prefix counts are
discovered by the run and are not inherited from older evidence.

The test measures exact component bytes and counters around each failure. It
does not claim flat process memory, sanitizer coverage, package localization,
hostile-link resistance, thread safety, or complete TR3 behavior.
