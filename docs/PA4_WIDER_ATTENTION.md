# PA4-Wider CPU-f32 causal attention

PA4-Wider is an opt-in pair of K1 ABI 1.0 providers. Link
`build/pa4_wider/libeshkol_transformer_pa4_wider.a` and obtain the descriptor
with `et_pa4_wider_ref_provider_v1()` or `et_pa4_wider_online_provider_v1()`.
A K1 resolver can return either descriptor when asked for the K1 provider
symbol. Each descriptor publishes the verified, deterministic
`kernel.causal-attention` capability with `causal-attention.forward` and
`causal-attention.backward`.

## Tensor and validation contract

The request shape is `[N,Hq,Hkv,Tq,Tk,Dh]`. Supported extents are
`N=1..2`, `Hq=1..4`, `Hkv=1..Hq` with `Hq % Hkv == 0`,
`Tq=1..64`, `Tk=1..64` and `Dh=1..16`. K1 advertises eight ranges, one
for each permitted head pair. MHA uses `Hq == Hkv`, GQA shares a KV head
within each group of `Hq/Hkv` adjacent query heads, and MQA uses one KV
head. Batches have common allocated extents. Positions and the dense mask
supply per-batch and per-query admission for rectangular, padded or
selectively masked work.

| Role | Order | Shape | Dtype |
|---|---:|---|---|
| Q | Input 0 | `[N,Hq,Tq,Dh]` | f32 |
| K | Input 1 | `[N,Hkv,Tk,Dh]` | f32 |
| V | Input 2 | `[N,Hkv,Tk,Dh]` | f32 |
| Query positions | Input 3 | `[N,Tq]` | i64 |
| Key positions | Input 4 | `[N,Tk]` | i64 |
| Keep mask | Input 5 | `[N,Tq,Tk]` | bool (one byte) |
| dO | Backward input 6 | `[N,Hq,Tq,Dh]` | f32 |
| O | Forward output 0 | `[N,Hq,Tq,Dh]` | f32 |
| dQ | Backward output 0 | `[N,Hq,Tq,Dh]` | f32 |
| dK | Backward output 1 | `[N,Hkv,Tk,Dh]` | f32 |
| dV | Backward output 2 | `[N,Hkv,Tk,Dh]` | f32 |

All views are CPU, dense row-major, naturally aligned, zero-offset and
exactly sized. Query and key positions each increase strictly within a
batch and lie in `[0,16777215]`. A key is admitted if its mask byte is 1
and its position is at most the query's position. Mask bytes are 0 or 1.
An empty row produces positive zero and contributes no gradients. K and V
must be disjoint. Outputs must be disjoint from each other, all inputs, and
live K1 metadata, including descriptor extensions, table padding, shape
arrays, and strings through their terminator.

Callers provide addressable descriptors, buffers, and NUL-terminated
strings. The K1 dispatcher's diagnostic argument occupies separate
storage because K1 can write it before the provider callback. Direct
provider validation also checks the diagnostic against borrowed spans it
can identify and avoids writing through a detected alias. On failure
output buffers are untouched. Successful validation leaves output buffers
untouched until invocation. Direct invocation requires a successful
validation of the same provider and operation. Keep inputs, descriptors
and floating-point controls fixed between validation and invocation.
Each output is overwritten on invocation.

Both providers require x86-64 IEEE binary32, round-to-nearest, masked FP
exceptions, and gradual underflow. Validation checks x87 and MXCSR
controls and scans every floating input for finiteness, including masked
elements. For input magnitudes up to `2^20`, bounded dimensions prove
finite score, accumulator and gradient arithmetic. This fast preflight
uses integer scans and leaves the floating-point environment untouched.
For larger finite inputs, validation repeats the selected arithmetic on
stack scratch, rejects nonfinite scores or accumulations, and restores
the floating-point environment and `errno` before returning. A negative
infinite shift from subtracting two finite extremes is permitted as a
zero exponential weight. Invocation can raise normal masked floating-point
status flags or set `errno` via libm.

## Forward and backward arithmetic

Dot products accumulate in increasing head-dimension order in binary32.
The score is the dot product times `1 / sqrtf(Dh)`. The reference
forward computes the row maximum, then the denominator
`sum(expf(score - maximum))`, then the sum of normalized weights times
V. It recomputes scores across the three passes.

Online forward reads each admitted key once. It maintains a running
maximum `m`, denominator `l`, and unnormalized vector numerator `acc`.
For a new maximum `s`, it applies `a = expf(m - s)`, updates
`l = a*l + 1` and `acc = a*acc + V`, then sets `m = s`. For other
scores it adds `w = expf(s - m)` to `l` and `w*V` to `acc`.
The first key starts a fresh row. The result is `acc/l`. Forward uses
one 16-float accumulator and constant scalar state with respect to `Tk`.

Backward evaluates `dp = dot(dO,V)`, `mean = sum(p*dp)` and
`ds = p*(dp-mean)`. It adds `ds*scale*K` to dQ,
`ds*scale*Q` to dK and `p*dO` to dV. Reference backward recomputes
row probabilities and upstream/value dot products. Online backward
reuses one 64-float array for scores, weights and probabilities and one
64-float array for `dp`. It computes each admitted Q/K dot, exponential
and dO/V dot once per row, then visits the cached terms once to write all
three gradients. It uses the same max-shifted softmax and addition
order as reference backward. Each dQ element sums in key order. Each
dK and dV element sums in query-head, then query-position order.
The exact backward preflight for large inputs uses a single KV head
tile with two arrays of `64*16` floats and one dQ row. Invoke writes
dK and dV directly. There is no heap allocation or probability matrix
in provider invocation or validation.

Online's unnormalized numerator can overflow for extreme finite V even
when normalized reference arithmetic can return a finite output. In that
case online validation rejects the call. There is no numerical fallback.
The CPU kernels are serial and dense and take
`O(N*Hq*Tq*Tk*Dh)` time. This evidence covers the listed CPU-f32 component
only. GPU, mixed precision, arbitrary head widths, sequences longer than
64, and training or deployment performance are outside its scope.

## Numerical verification

`scripts/test-pa4-wider.sh` runs K1 direct and dispatched conformance,
forward/backward comparison against an independent f64 oracle, central
finite differences, C++17 ABI/linkage, and a conformance rerun under
ASan, UBSan and LSan. The gradient objective is the dot product of the
f32 output with dO, accumulated in f64. It uses symmetric perturbations
of `2^-4` and `2^-5` with Richardson extrapolation to reduce the
truncation term while keeping f32 rounding noise manageable. The test
samples Q, K and V on the six benchmark shapes and checks every
coordinate on an additional odd-width rectangular MHA shape.

### Supported numerical results

Ubuntu 22.04.5 LTS and Clang/Clang++ 21.1.8 passed 36,016 paired output
comparisons and 1,404 central gradient checks in both regular and
sanitized builds. Maximum absolute central-difference discrepancies
were `1.7204181252360173e-05` for reference and
`1.0862701325702745e-05` for online. Maximum absolute error against
the f64 oracle was `1.0550322866453854e-06` for either provider,
including gradients. C++17 linkage, the archive export and dependency
audit, all five Python isolation tests, all five predecessor pin tests,
and CI topology passed.

| Shape `[N,Hq,Hkv,Tq,Tk,Dh]` | O max abs | O max relative | O max ULP distance |
|---|---:|---:|---:|
| `[1,4,2,8,8,8]` | 5.9604644775390625e-08 | 2.3134430617173222e-05 | 208 |
| `[1,4,2,16,16,16]` | 1.1920928955078125e-07 | 0.00012960235640648013 | 1452 |
| `[1,4,2,32,32,16]` | 1.1920928955078125e-07 | 0.00015401738362243075 | 2173 |
| `[1,4,2,64,64,16]` | 1.1920928955078125e-07 | 0.00020985213026546016 | 1891 |
| `[2,4,2,16,32,16]` | 8.9406967163085938e-08 | 0.00047295070051983629 | 5775 |
| `[1,4,1,32,32,16]` | 1.1920928955078125e-07 | 0.00012960235640648013 | 1452 |
| `[2,3,3,5,7,3]` | 5.9604644775390625e-08 | 2.875356735925704e-06 | 30 |

The table compares online and reference. Relative error divides the
absolute difference by the larger magnitude and is zero for two zeros.
ULP distance counts representable binary32 steps, treating both zeros
as equal. Rescaling and normalizing at different points affect f32
rounding. Near cancellation, even a small absolute difference spans
many ULPs. The observed maxima apply to these fixtures. Conformance
and benchmark parity use `3e-7 + 3e-6 * max(abs(ref),abs(online))`
as an absolute error threshold. Every tested dQ, dK and dV matched
bit for bit.

## Supported scaling measurement

The benchmark compares reference (A) and online (B) in the same
process, using identical tensor addresses, deterministic dyadic inputs,
strictly increasing positions, a selective mask, and a fully masked
query where the shape permits. Each case runs 12 pairs alternating AB
and BA. Each arm performs 15 warmups, zeros outputs, then times 100
iterations with `CLOCK_MONOTONIC`. A joint iteration executes forward
then backward with a fixed upstream gradient and disjoint outputs.

Dispatch includes K1 and provider validation plus execution. Invoke
uses the same immutable inputs after untimed dispatch and prevalidation.
Every timed arm is checked against its own untimed output bits and
against the reference tolerance. The benchmark prints all 432 pairs
and AB/BA ratio ranges. The table reports the lower median for each arm
in nanoseconds per iteration, as **reference / online**. It includes
loop and timer overhead. No samples were discarded.

The run used the `eshkol-r0-debug:20260928` image, digest
`sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99`,
on an AMD Ryzen 7 3700X, Linux `7.2.7-1-cachyos`, with CPU affinity 6.
Clang compiled the benchmark and provider with `-O2 -ffp-contract=off`
and no LTO. CPU frequency and activity on the sibling thread were not
controlled.

| Shape | Forward dispatch | Forward invoke | Backward dispatch | Backward invoke | Joint dispatch | Joint invoke |
|---|---:|---:|---:|---:|---:|---:|
| `[1,4,2,8,8,8]` | 4405.78 / 2870.76 | 3316.51 / 1826.57 | 7542.95 / 4358.89 | 5920.96 / 2814.25 | 12142.30 / 7365.31 | 9275.35 / 4622.39 |
| `[1,4,2,16,16,16]` | 24755.39 / 14562.90 | 23398.91 / 13220.05 | 40800.91 / 20611.21 | 38611.85 / 18564.01 | 65687.96 / 35427.19 | 61840.14 / 31747.09 |
| `[1,4,2,32,32,16]` | 95331.94 / 54281.26 | 93209.11 / 52074.35 | 155524.84 / 77490.60 | 152536.96 / 73805.09 | 253672.24 / 132464.27 | 251847.46 / 127651.22 |
| `[1,4,2,64,64,16]` | 387360.14 / 218736.78 | 376681.63 / 207394.00 | 618024.11 / 301025.21 | 624034.34 / 297291.21 | 1010746.85 / 534023.32 | 991440.42 / 507921.07 |
| `[2,4,2,16,32,16]` | 133284.13 / 76187.33 | 128467.27 / 73187.21 | 218439.11 / 104172.56 | 218545.72 / 103003.83 | 352363.27 / 180321.71 | 360819.85 / 175654.03 |
| `[1,4,1,32,32,16]` | 98548.96 / 55412.70 | 96071.25 / 53482.93 | 156907.67 / 77814.82 | 152815.28 / 74588.59 | 252766.31 / 131583.90 | 247593.38 / 126980.74 |

At T=64, median online time was 47.17% lower for joint dispatch and
48.77% lower for joint invoke. The smallest online/reference median
ratio among the joint dispatch cases was 0.511749 and the largest
was 0.606583. All timed pairs in this run favored online. Across all
cases the observed pair ratios ranged from 0.373843 to 0.906043.
These are sample ranges from one machine and one input recipe.

Clang's static frame report showed 536 bytes for forward computation,
984 bytes for the backward head helper, 200 bytes for numeric validation,
and 8,376 bytes for the separately invoked large-input backward preflight.
These are function frames. They exclude called frames and are not
measurements of process memory. Two archive builds at the measured baseline (`5cb41d4a`) produced the same
SHA-256 `27a99413b0e47a08949b40f32e25c7ac4c8e496be47a4acf2a4ba9b0af616783`,
with provider source SHA-256
`b11a709b7c31d5b32e6988c599adad9d21034dcb60c4ba7d0d965b9a681f341e` and full run
log SHA-256
`cdb442ea7a39cbe15d07b7f7f55a81be3ace2a462f4a6642a1c446e01db15d76`. A subsequent
direct-validation alias-hardening commit updated `aliases_table`/`aliases_call`
only (provider source SHA-256
`291afecd59a7ea439f378f3ade7e0213ac3e85d51796f149ac51ca640d581079`, reproducible
archive SHA-256
`d0d564fd90f656d691556415e3e87611015a097882e12aca1d55ab78e9bb68fe`) and
re-verified the full conformance, sanitizer, and 864-sample benchmark suite in
`eshkol-r0-debug:20260928`.

### Reproduce

From the worktree root, bind-mount the worktree and shared Git directory.
The example uses the default user mapping of the measured rootless Docker
installation and pins CPU 6.

```bash
docker run --rm --network none --cpuset-cpus 6 \
  -v "$PWD:$PWD" \
  -v /home/gabe/eshkol-transformer/.git:/home/gabe/eshkol-transformer/.git:ro \
  -w "$PWD" -e PYTHONDONTWRITEBYTECODE=1 eshkol-r0-debug:20260928 \
  bash -ec 'CC=clang-21 CXX=clang++-21 bash scripts/test-pa4-wider.sh
    python3 -m unittest -v tests.q0.test_python_isolation
    python3 -m unittest -v tests.ci.test_attention_predecessor_pins
    bash scripts/check-ci-topology.sh'
```
