# PA4 bounded attention candidate

Status: **source reviewed; bounded native parity verified**, with no performance
or memory admission.

## Contract and frozen baseline

The baseline is `native/a2_attention_provider.c` at commit
`ee24ef461ef7e313c36c6c6f97ba4f3d8155bac7`, SHA-256
`fdb4ca172ca7aa282e787ff42b2998592c8b77a30f67d35680f7e38b2f550b02`.
The candidate is an explicitly linked, separate `et_pa4_kernel_provider_v1`
compiled from the current source with `ET_PA4_ATTENTION_CANDIDATE`.
The default A2 build retains `et_a2_kernel_provider_v1`. The candidate changes
only forward attention invocation; backward attention, RoPE, validation preflight,
and the ten advertised attention and six RoPE capability rows remain the A2
contract in [A2_ATTENTION.md](A2_ATTENTION.md). Provider and attention implementation
identity identify this as `pa4.cpu-f32.row-scratch.candidate` and
`pa4.cpu-f32.row-scratch`; its evidence string explicitly says it is not admitted.

Only exact `cpu` f32 dense zero-offset tensors and the A2 shapes, masks, positions,
arithmetic order, allocation-free dispatch, and failure atomicity are claimed.
For each forward row, a fixed three-element stack scratch stores scores and
unnormalized weights. Score accumulation and scaling, maximum-subtracted softmax
sum order, probability division, and output accumulation order are unchanged.
The candidate validator rejects direct calls with `Tk > 3` as unsupported before
any output write. This excludes an unverified A2 direct-provider case; it does
not enlarge the K1-advertised shape set or introduce a fallback. Preflight still
rejects nonfinite intermediate values before invocation.

`scripts/test-pa4.sh` builds the frozen baseline and candidate separately, runs
the A2 reference and finite-difference provider suite against the candidate,
then checks exact output bits against the baseline for forward and backward on
all ten advertised shapes. The paired test also exercises direct `Tk > 3`
rejection. A2 baseline and candidate are selected explicitly at link time;
neither is dynamically discovered. These native checks alone do not establish
full A2 integration or PA4 acceptance.

## Bounded validation, 2026-11-14

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

## Measurement gate

PA4 acceptance requires parity **and measured speed or memory improvement** on
the same workload. B0 v1 in [BENCHMARK_FORMAT.md](BENCHMARK_FORMAT.md) admits only
the F0 smoke process, dtype `not-applicable`, and empty shapes. It must reject
an attention workload. A reviewed B0 tensor-workload extension or new version is
therefore a prerequisite to a PA4 acceptance report; B0 smoke timing cannot be
relabelled as attention evidence.

The future definition must pin exact semantic shape, masks, positions, f32 input
bit patterns, forward or backward operation, backend/device, work-unit count,
warmups, repetitions, compiler flags and source/binary hashes for both providers.
Measure whole K1 dispatch (including validation and preflight) separately from
provider `invoke_call` on a prevalidated immutable call. Use an identical
process and repeated inputs with consumed output, report raw samples and the
measurement source, and report host RSS separately from scratch size. Any
invoke-only result describes a narrower region than end-to-end dispatch. No
acceleration, device-memory, deterministic timing, or threshold claim follows
from the current native parity tests. Supported Ubuntu 22.04 / Clang 21.1.8
and exact-head repository gates are still required for final admission.
