# A2 / PA4 comparison v1

This development-tool extension compares explicitly linked providers
`et_a2_kernel_provider_v1` and `et_pa4_kernel_provider_v1`. It introduces separate
`eshkol-attention-comparison-definition` and `eshkol-attention-comparison-report`
version-1 canonical SHA-256 envelopes; existing B0 smoke and A2 attention formats
are unchanged. It does not change production providers, the A2 shape contract, or
the PA4 admission rule.

## Frozen comparison contract

`benchmarks/attention_compare_v1.json` fixes all ten A2 shapes, f32 CPU inputs,
forward and backward operations, and the exact dyadic/positions/selective-mask
recipe in [BENCHMARK_FORMAT.md](BENCHMARK_FORMAT.md). Both providers receive the
same input and output tensor views, buffers, counts, masks and positions. The
baseline source is extracted from the historical commit and checked against the
SHA-256 frozen in [PA4_ATTENTION.md](PA4_ATTENTION.md). The candidate is built from
the composed current source with `ET_PA4_ATTENTION_CANDIDATE`; default-provider
source is not rewritten. Both objects link into the same executable, with
`-O2 -ffp-contract=off`, strict warnings and no LTO. Preparation/build is excluded.

The checked-in definition predeclares 100 warmup calls before each timed batch,
10,000 calls per batch, 12 same-process timing pairs and four separate-process
RSS pairs per exact shape/operation/scope. Both pair counts must be even. Counts
may change only through a newly checksummed definition within validator bounds.
An AB pair measures A2 then PA4; the next BA pair reverses that order. This repeats
without randomization or result-dependent sample removal. No core Eshkol build,
Python numerical operation, CPU fallback, or GPU execution participates.

For each shape/operation, two separate fresh-process cases measure:

- `dispatch`: synchronous K1 dispatch, including K1 validation, provider preflight
  and invocation, plus the batch loop and clock overhead.
- `invoke`: provider `invoke_call` only, after successful discovery, capability
  resolution, K1 dispatch and provider validation outside timing. All call
  metadata and input data stay unchanged; only disjoint outputs are overwritten.

The primary paired timing process initializes both providers and uses the same
call/buffer addresses for both arms. Each batch ends with consumption of every
output word outside timing. The native protocol emits all final output bits;
the runner independently rejects malformed, nonfinite or all-zero witnesses and
requires exact bit parity across every pair, both providers, both scopes and the
separate RSS processes. The report retains one full witness per case and its
SHA-256 plus the digest on each raw sample. This is observed bit parity, while
A2 numerical gradients and PA4 reference/negative tests remain separate gates.

## RSS and process timing

Primary paired processes also report total process time and peak RSS, shared by
both providers. A separate RSS arm launches the same binary with only one provider
initialized and used, with identical buffers, preflight, warmups and a timed
batch. Fresh selected-provider processes follow four AB/BA pairs per case. Every
raw RSS sample stores selected-provider batch time, whole process elapsed time,
and Linux `wait4.ru_maxrss` in KiB. The existing pidfd helper enforces timeout and
completion; signal, nonzero exit, stderr and malformed output reject the run.

These RSS samples include code, runtime, stack, setup, warmup and serialization.
Both implementations are linked into the binary; only the selected provider is
initialized/invoked in that RSS process. RSS does **not** isolate attention
scratch, allocations or incremental working set. Equal RSS cannot disprove a
scratch difference, and lower process RSS does not prove a scratch reduction.
No memory subtraction or per-dispatch memory estimate is made.

## Predeclared analysis and limits

Every raw pair remains in measurement order. Per exact case the report recomputes
baseline/candidate batch-time minimum, lower median and maximum; the
candidate/baseline ratio minimum, lower median and maximum; separate AB and BA
ratio ranges; and the ratio of total candidate time to total baseline time.
Ratios below one describe faster observed candidate batches. Separate provider
RSS minima/lower medians/maxima describe fresh-process variability.

The four operation/scope aggregates divide summed candidate times by summed
baseline times across all ten shapes, each receiving equal dispatch counts.
This is this bounded shape mix, not a representative training distribution.
Ratios use decimal half-even rounding to six places. Raw timing remains integral
nanoseconds and raw RSS integral KiB. Ratio ranges are observed ranges, **not**
confidence intervals. No timing/RSS sample is trimmed and no acceptance threshold
is introduced. CPU affinity, CPU model, kernel, OS, logical CPU count and compiler
identity are retained; load, scheduler/frequency/cache effects and code layout
can affect observations even with balanced order and warmup.

Backward uses the unchanged analytic A2 implementation and is an explicit
control. A performance conclusion must examine every forward shape/scope,
regressions, AB/BA variation and backward-control variation. Invoke-only benefit
is narrower than K1 dispatch benefit. A tiny aggregate ratio below one by itself
does not establish a reliable end-to-end gain. PA4's existing rule remains exact
parity plus measured speed **or** memory improvement on the same workload; this
tool reports evidence and does not automatically admit PA4.

The report seals definition identity, project commit/dirty state, source closure,
frozen baseline commit/hash, compiler path/hash/version/flags, baseline/candidate
object and target/helper hashes. Sources, compiler and binaries are checked again
after measurement. The selected container-image digest is caller-supplied
provenance, not in-process proof of container identity. A
`supported-native-component` label requires that declared known immutable image,
observed Ubuntu 22.04/x86_64 and Clang 21.1.8. Every other environment is
`compatibility-only`. Neither label establishes exact-head full Eshkol integration,
hosted CI or release admission. Failed runs preserve an existing report; reports
replace destinations atomically after validation.

## Commands

Use `/home` scratch because `/tmp` may be full. The wrapper sets `TMPDIR` under
`build/benchmarks` and holds `/tmp/eshkol-transformer-heavy-build.lock`.

```bash
CC=clang-21 bash scripts/benchmark-attention-compare.sh
python3 -B tests/b0/attention_compare.py verify \
  benchmarks/attention_compare_v1.json \
  build/benchmarks/attention-compare-v1-report.json
TMPDIR="$PWD/build/benchmarks/compare-tmp" \
  flock /tmp/eshkol-transformer-heavy-build.lock \
  python3 -B -m unittest tests.b0.test_attention_compare -v
```

A supported native container invocation supplies
`BENCHMARK_CONTAINER_IMAGE=sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99`
after the orchestrator verifies its actual container image. Optional CPU pinning
must be applied before invoking the wrapper and remains visible in report
`cpu_affinity`. Generated reports/binaries stay outside Git. The [supported native results](ATTENTION_COMPARISON_RESULTS.md) record exact
measurement identity, raw-report checksum, forward gains, control regressions and
uncertainty. Independent source/test review approved the bounded harness; the
full B0 discovery gate passes 59 tests. Full PA4 admission remains separate.
