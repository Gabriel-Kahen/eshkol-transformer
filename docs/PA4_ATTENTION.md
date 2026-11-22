# PA4 bounded attention candidate

Status: **opt-in candidate with measured joint forward and backward improvement
on the fixed ten-shape attention contract**.

## Contract and frozen baseline

The baseline is `native/a2_attention_provider.c` at commit
`ee24ef461ef7e313c36c6c6f97ba4f3d8155bac7`, SHA-256
`fdb4ca172ca7aa282e787ff42b2998592c8b77a30f67d35680f7e38b2f550b02`.
The candidate is an explicitly linked, separate `et_pa4_kernel_provider_v1`
compiled from the current source with `ET_PA4_ATTENTION_CANDIDATE`.
The default A2 build retains `et_a2_kernel_provider_v1`. PA4 uses row scratch
for forward and backward attention invocation and preflight. RoPE and the ten
advertised attention and six RoPE capability rows follow the A2 contract.
[A2_ATTENTION.md](A2_ATTENTION.md) describes that contract. Provider and attention implementation
identity identify this as `pa4.cpu-f32.row-scratch.candidate` and
`pa4.cpu-f32.row-scratch`; its evidence string explicitly says it is not admitted.

Only exact `cpu` f32 dense zero-offset tensors and the A2 shapes, masks, positions,
arithmetic order, allocation-free dispatch, and failure atomicity are claimed.
Three-element stack arrays cache scores and softmax weights. Forward preflight
and backward also cache admission flags and probabilities. Backward caches the
upstream/value dot products and their probability-weighted mean. Forward and dQ
sum in key order. Each dK and dV element sums in query-head-major, then query
position order. Backward preflight uses four head dimensions of gradient scratch
per KV head and checks each accumulation. All advertised shapes fit one tile.
Wider direct calls repeat the row calculation per tile to bound stack use.
The candidate validator rejects direct forward and backward calls with `Tk > 3`
and rejects nonfinite preflight intermediates before writing outputs.

`scripts/test-pa4.sh` builds the frozen baseline and candidate separately, runs
the A2 reference and finite-difference provider suite against the candidate,
then checks exact output bits against the baseline for forward and backward on
all ten advertised shapes. It also covers fully masked rows, intermediate and
gradient overflow with untouched outputs, a multi-tile direct call, and direct
`Tk > 3` rejection in both directions. The script compares same-basename `-O2`
default and frozen A2 objects byte for byte, and checks the candidate's export and
undefined allocator symbols. The test links both providers explicitly.

## Bounded validation, 2026-11-22

The supported Ubuntu 22.04.5 and Clang 21.1.8 run tested clean commit
`2913b1b3d6e5ab6f9e3537c3538282ad40fe2e87`, tree
`f01cb44c83cba89600942b809d7084c2ba827d33`.

- `CC=clang-21 bash scripts/test-pa4.sh` passed 1,045 provider checks, bitwise
  parity on all ten advertised shapes, direct-call edge cases and all five
  predecessor-pin tests. The provider and parity executables also passed with
  ASan, UBSan and LSan enabled.
- `cmp` passed for same-basename frozen and default `-O2 -ffp-contract=off`
  objects, both SHA-256 `85b917fab18c716f66143afcb954ee55b5f0c628343ba34652c2893ab2d03cc5`.
  `nm` found no allocator references in the candidate and only the expected
  public entry point. Clang `-fstack-usage` reported static frames of 312 bytes
  for backward preflight, 120 bytes for row preparation and 568 bytes for
  the provider invoke function after inlining.
- `bash scripts/test-b0.sh` passed all 66 tests. The CI topology check and
  all five Python isolation tests also passed.

## Initial candidate validation, 2026-11-14

An independent source/tests/script review approved the candidate after correcting
RoPE capability attribution; four metadata assertions cover that correction.
Native execution in the cached `eshkol-r0-debug:20260928` container used Ubuntu
22.04.5 and Clang 21.1.8 with `-ffp-contract=off`:

- `CC=clang-21 bash scripts/test-pa4.sh`: 1,045 provider reference, numerical
  gradient, capability and negative checks; exact forward/backward parity against
  the frozen baseline on all ten admitted attention shapes.
- Both candidate test executables also passed with ASan/UBSan and leak detection.
- The default A2 provider passed its 1,041 checks. Its `-O2` object was byte-identical
  to the frozen baseline when compiled with equal source basenames.
- The candidate object exported only `et_pa4_kernel_provider_v1` and had no
  allocator dependencies. The B0 format/runner suite passed all 41 tests.

These are native component results. No canonical Eshkol AOT, full repository,
hosted CI, training, performance, or memory-reduction result is established.
No attention benchmark was run: the B0 tensor contract below remains a dependency.

## Current-source predecessor inventory follow-up

The four live N3K, G3N, G3S, and G3-C4 predecessor inventories now pin the
current A2 source, and G3N pins the resulting N3K inventory digest. The CI
topology test `tests/ci/test_attention_predecessor_pins.py` checks exact tracked
inventory membership and rejects stale, missing, extra, or duplicate direct and
nested pins. All
tracked predecessor inventories pass `sha256sum -c` locally. The frozen PA4
baseline above remains pinned to its historical source. This inventory repair
does not add performance or full integration evidence.

## Measurement gate

PA4 acceptance requires parity **and measured speed or memory improvement** on
the same workload. B0 v1 in [BENCHMARK_FORMAT.md](BENCHMARK_FORMAT.md) admits only
the F0 smoke process, dtype `not-applicable`, and empty shapes. It must reject
an attention workload. A reviewed B0 tensor-workload extension or new version is
therefore a prerequisite to a PA4 acceptance report; B0 smoke timing cannot be
relabelled as attention evidence.

The comparison extension measures K1 dispatch and prevalidated invocation in
same-process AB/BA pairs, with RSS measured in separate fresh processes. Its
opt-in joint definition times forward followed by backward in each iteration
and verifies both outputs against the individual cases. [B0 comparison contract](ATTENTION_COMPARISON.md)

Across the ten shapes, joint dispatch time was 51.934% lower and joint invoke
time was 42.132% lower. Backward dispatch and invoke time were 62.151% and
38.775% lower, respectively. Two joint dispatch shape ranges crossed parity.
Process RSS ranges overlapped in 59 of 60 cases and do not isolate scratch.
Training throughput and wider contexts remain unmeasured. [Supported comparison results](ATTENTION_COMPARISON_RESULTS.md)

[PR #224](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/224) carries
the explicit candidate, parity gates and comparison together. Its integration
keeps default A2 selection and the bounded candidate evidence identity. The
previous [supported CI](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/43760593834)
passed for the original comparison branch. The backward and preflight update
awaits independent review, exact-tree repository and hosted CI, and an
integration decision. Default provider selection remains A2.
