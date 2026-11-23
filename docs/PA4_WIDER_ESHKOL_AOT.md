# PA4-Wider Eshkol AOT bridge

The private C11/C++17 bridge in `native/pa4_wider_eshkol_bridge.h`
links [PA4-Wider CPU-f32 causal attention](PA4_WIDER_ATTENTION.md) to
Eshkol AOT programs. `et_pa4_wider_aot_open(variant, &runtime)` discovers one
K1 runtime using variant `0` for reference or `1` for online. It publishes the
runtime only on success. Pass it to `et_pa4_wider_aot_forward` and
`et_pa4_wider_aot_backward`, then call `et_pa4_wider_aot_close`.
Discovery owns K1 allocations. Dispatch builds its descriptors on the stack,
borrows caller buffers, and invokes the selected provider through
`et_kernel_runtime_dispatch`. Each runtime stays on its selected provider,
including when online preflight rejects arithmetic that reference can accept.

## Buffer and status contract

The six signed dimensions are passed as `N, Hq, Hkv, Tq, Tk, Dh`.
Supported extents are `N=1..2, Hq=1..4, Hkv divides Hq, Tq/Tk=1..64,
Dh=1..16`. MHA, GQA, MQA, and rectangular query/key spans share this ABI.

| Buffers | Dense shape | Element type |
|---|---|---|
| Q, O, dO, dQ | `[N,Hq,Tq,Dh]` | f32 |
| K, V, dK, dV | `[N,Hkv,Tk,Dh]` | f32 |
| Query positions | `[N,Tq]` | i64 |
| Key positions | `[N,Tk]` | i64 |
| Keep mask | `[N,Tq,Tk]` | u8, 0 or 1 |

The caller owns aligned, addressable buffers of the specified sizes. The
bridge derives the byte spans from the dimensions. Buffer capacity and
marshaling indices are caller preconditions. K and V are disjoint, and each
output is disjoint from all inputs and other outputs. Positions increase
strictly within each batch and lie in `[0,16777215]`. A key participates if
it is kept and its position is at most the query position. Fully masked
rows produce positive zero and contribute no gradients.

Calls return zero on success or a K1 error category and preserve output
bytes on failure. Representative categories are `UNSUPPORTED` (6) for
out-of-range extents or incompatible head pairs, `SHAPE_MISMATCH` (2) for a
null tensor buffer with nonzero extent, and `INVALID_ARGUMENT` (1) for
bad positions, mask values, or overlapping outputs. The bridge supplies
private storage for K1's structured diagnostics.

The `et_pa4_wider_aot_store_f32` helper rounds Eshkol f64 values to
binary32 under the supported floating-point environment. Magnitudes above
`FLT_MAX` become signed infinity, which dispatch rejects as input.
`et_pa4_wider_aot_load_f32` widens finite binary32 exactly.
`et_pa4_wider_aot_f32_bits` reads the raw bits into an i64 and preserves
signed zero for bitwise comparisons. `et_pa4_wider_aot_store_i64` and
`et_pa4_wider_aot_store_u8` write positions and mask bytes.

## Compiled verification

In `eshkol-r0-debug:20260928` (Ubuntu 22.04, Clang and Clang++ 21.1.8),
the compiler pinned to `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9` built the
Eshkol driver twice with `eshkol-run --strict-types --no-stdlib`. Each run
used a fresh cache directory and linked the PA4-Wider and K1 archives with
the bridge object. The resulting ELF files and stdout matched byte for
byte. The ELF SHA-256 was
`f4fd4a9911db9309b50a1cfc85e61b29bfff7e2e45234ddfe92d8778f1e5e138`.

The compiled driver compares every forward output and every backward
gradient from both providers. All 13,312 gradient comparisons passed
bitwise, including signed zero. Each forward maximum was below `2e-7`.

| Shape `[N,Hq,Hkv,Tq,Tk,Dh]` | Forward max abs diff | dQ / dK / dV bitwise comparisons |
|---|---:|---:|
| `[1,4,2,16,16,16]` | `1.1175870895385742e-08` | 1,024 / 512 / 512 |
| `[1,4,1,32,32,16]` | `1.4901161193847656e-08` | 2,048 / 512 / 512 |
| `[1,4,2,64,64,16]` | `1.1175870895385742e-08` | 4,096 / 2,048 / 2,048 |

The MQA smoke uses the online provider with `T=32`, `Hq=4`, `Hkv=1`,
and `Dh=16`. Eshkol computes Q/K/V projections, half mean squared error
against a deterministic target, projection gradients, and SGD. It checks
two central differences per projection before training and requires
strictly decreasing loss at each of 12 updates. Eshkol also checks error
statuses for both providers and verifies that the malformed calls leave
output sentinels intact.

| SGD updates | Loss |
|---:|---:|
| 0 | 0.01611747045119014 |
| 1 | 0.01587028988052243 |
| 2 | 0.015657682773558102 |
| 3 | 0.015474639646046324 |
| 4 | 0.015316882831055218 |
| 5 | 0.015180758588865185 |
| 6 | 0.015063145232393159 |
| 7 | 0.014961375192963955 |
| 8 | 0.014873168650764483 |
| 9 | 0.01479657672114825 |
| 10 | 0.014729933241584028 |
| 11 | 0.014671813551595345 |
| 12 | 0.01462099949608835 |

Run the verification inside the container with the pinned toolchain
mounted read-only at `/tc`.

```bash
ESHKOL_SOURCE_DIR=/tc/src ESHKOL_BUILD_DIR=/tc/build CC=clang-21 CXX=clang++-21 \
  bash scripts/test-pa4-wider-aot.sh
python3 -m unittest tests/q0/test_python_isolation.py tests/ci/test_attention_predecessor_pins.py
bash scripts/check-ci-topology.sh
```

The script checks bridge exports and imports, C++17 linkage, and C11
conformance in normal and ASan/UBSan/LSan builds. The native test covers
all admitted head pairs, rectangular and maximum extents, malformed
inputs, and zero heap calls from the linked bridge, K1, and provider
during dispatch.

## Limits

Execution requires the providers' x86-64 IEEE binary32 environment with
round-to-nearest, masked exceptions, and gradual underflow. The compiled
evidence covers three parity shapes and one deterministic CPU-f32
single-layer training fixture in the pinned container. The runtime's
limits remain those of PA4-Wider, including its online overflow
rejection. AOT binary identity was measured across clean compilations
within this environment. Cross-toolchain identity, training throughput,
GPU execution, mixed precision, and model-scale training remain
unverified.
