# Mamba / S6 Selective State-Space Model (Causal Conv1D + Selective Scan + O(1) Recurrent Cache)

This standalone module provides an Eshkol-native **Mamba / S6 Selective State-Space Model** block in C11 (`include/eshkol_transformer/mamba_ssm_abi.h`, `native/mamba_ssm.c`) together with an Eshkol AOT transport bridge (`native/mamba_ssm_eshkol_bridge.{h,c}`) and compiled Eshkol AOT driver (`tests/mamba_ssm/test_mamba_ssm_aot.esk`).

## Artifacts and public contracts

- Header: `include/eshkol_transformer/mamba_ssm_abi.h` (`extern "C"`, C11 and C++17 clean)
- Core static archive: `build/mamba_ssm/libeshkol_transformer_mamba_ssm.a`
  - Deterministic archive SHA-256: `7c86f7ab9b0e34332920720d2390b7148c6075726f251426e3dc5776a1205c4b`
  - Exported symbols (11):
    - `et_mamba_validate_config_v1`
    - `et_mamba_causal_conv1d_silu_forward_v1`, `et_mamba_causal_conv1d_silu_backward_v1`
    - `et_mamba_selective_scan_forward_v1`, `et_mamba_selective_scan_backward_v1`
    - `et_mamba_cache_conv_state_bytes_v1`, `et_mamba_cache_ssm_state_bytes_v1`, `et_mamba_cache_total_state_bytes_v1`
    - `et_mamba_cache_init_v1`, `et_mamba_cache_reset_v1`, `et_mamba_cache_step_v1`
  - Zero heap allocations: `malloc`, `calloc`, `realloc`, and `free` are absent from `nm -u` and actively trapped via `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free` in native and sanitizer tests.
- Eshkol AOT bridge archive: `build/mamba_ssm/libeshkol_transformer_mamba_ssm_aot.a`
  - Deterministic archive SHA-256: `b34e96d9cb0bb57dd300a28859cd90dba3d04394225b9000466bc0c9f3e69fb7`
  - Exports only `et_mamba_aot_*` descriptor, stage dispatch, O(1) cache lifecycle, and scalar marshaling symbols with zero allocator imports.
  - Accepts untyped `void *` pointers so raw Eshkol FFI addresses (including intentionally misaligned `ptr-add` addresses) are validated cleanly without UB.
- Compiled Eshkol AOT driver: `tests/mamba_ssm/test_mamba_ssm_aot.esk`
  - Compiled twice from independent fresh directories via `eshkol-run --strict-types --optimize 0 --no-stdlib`.
  - Byte-identical ELF SHA-256: `59e9f8fea1e0494f6f51f11edd4b6dec5eb4837cfc463f539a5b981c9dd758d6`

## Admitted domain and tensor shape contract

| Dimension | Symbol | Admitted Range |
|---|---:|---:|
| Batch size | `N` | `1 .. 4` |
| Sequence length | `T` | `1 .. 64` |
| Inner channel width | `D` (`D_inner`) | `2 .. 32` |
| SSM state width | `S` (`D_state`) | `2 .. 16` |
| Causal conv kernel width | `K` (`K_conv`) | `2 .. 4` |

All tensor operands use dense row-major IEEE-754 binary32 (`float`) storage except the recurrent cache's `ssm_state`, which uses binary64 (`double`) for drift-free step-by-step equivalence with the binary64 internal accumulation of the full-sequence selective scan.

- Stage A (`et_mamba_causal_conv1d_silu_{forward,backward}_v1`):
  - `u`, `x_out`, `dx`, `du_out`: `float[N, T, D]`
  - `conv_w`, `dconv_w_out`: `float[D, K]`
  - `conv_b`, `dconv_b_out`: `float[D]`
- Stage B (`et_mamba_selective_scan_{forward,backward}_v1`):
  - `x`, `dt_raw`, `z_opt` (optional), `out`, `dout`, `dx_out`, `ddt_raw_out`, `dz_out_opt` (optional): `float[N, T, D]`
  - `a_log`, `da_log_out`: `float[D, S]`
  - `b_in`, `c_in`, `db_out`, `dc_out`: `float[N, T, S]`
  - `d_skip`, `dd_skip_out`: `float[D]`
- Stage C (`et_mamba_cache_{init,reset,step}_v1`):
  - `conv_state`: `float[N, K - 1, D]`
  - `ssm_state`: `double[N, D, S]`
  - `u_step`, `dt_raw_step`, `z_step_opt` (optional), `out_step`: `float[N, D]`
  - `b_step`, `c_step`: `float[N, S]`

## Mathematical specification

All internal arithmetic executes in IEEE-754 binary64 (`double`) with `#pragma STDC FP_CONTRACT OFF`, `#pragma STDC FENV_ACCESS ON`, `-ffp-contract=off -fexcess-precision=standard -fno-fast-math`, verified default x87/MXCSR environment (`FE_TONEAREST`, all exceptions masked, FTZ/DAZ disabled), and saved/restored floating-point environment.

### Stage A: Depthwise causal Conv1D + SiLU

For each `n in 0..N-1`, `t in 0..T-1`, `d in 0..D-1`:

```text
c[n, t, d] = conv_b[d] + sum_{k=0}^{K-1} conv_w[d, k] * u_pad[n, t - (K - 1 - k), d]
u_pad[n, tau, d] = u[n, tau, d] if tau >= 0 else 0
x[n, t, d] = SiLU(c[n, t, d]) = c * sigmoid(c)
```

Backward given `dx[N, T, D]`:

```text
SiLU'(c)       = sigmoid(c) * (1 + c * (1 - sigmoid(c)))
dc[n, t, d]    = dx[n, t, d] * SiLU'(c[n, t, d])
dconv_b[d]     = sum_{n, t} dc[n, t, d]
dconv_w[d, k]  = sum_{n, t : tau = t-(K-1-k) >= 0} dc[n, t, d] * u[n, tau, d]
du[n, tau, d]  = sum_{k : t = tau+(K-1-k) < T} dc[n, t, d] * conv_w[d, k]
```

### Stage B: Selective Scan (S6) discretization and forward recurrence

The continuous-time diagonal state matrix is parameterized strictly negative for guaranteed stability:

```text
A[d, s] = -exp(A_log[d, s]) < 0        (with A_log[d, s] <= ET_MAMBA_MAX_A_LOG = 80.0)
```

Overflow-safe softplus discretization and zero-order hold (`h_{-1}[d, s] = 0`):

```text
dt[n, t, d] = softplus(dt_raw[n, t, d])
            = dt_raw + log1p(exp(-dt_raw))   if dt_raw > 0
            = log1p(exp(dt_raw))             if dt_raw <= 0

dA[n, t, d, s] = exp(dt[n, t, d] * A[d, s])
h_t[d, s]      = dA[n, t, d, s] * h_{t-1}[d, s] + dt[n, t, d] * B[n, t, s] * x[n, t, d]
y_t[d]         = sum_{s=0}^{S-1} (h_t[d, s] * C[n, t, s]) + D_skip[d] * x[n, t, d]
out_t[d]       = y_t[d] * SiLU(z[n, t, d])   (if z_opt != NULL, else y_t[d])
```

### Stage B: Exact analytic reverse-time adjoint backward

For each independent `(n, d)` trajectory, replay the forward trajectory `h_t, dt_t, dA_t, y_t` in binary64, then sweep backward `t = T - 1` down to `0` with `dh_next[s] = 0`:

```text
If gated (z_opt != NULL):
  dy         = dout[n, t, d] * SiLU(z[n, t, d])
  dZ[n,t,d]  = dout[n, t, d] * y_t[d] * SiLU'(z[n, t, d])
Else:
  dy         = dout[n, t, d]

dD_skip[d]  += dy * x[n, t, d]
dx_acc       = dy * D_skip[d]
ddt_acc      = 0

For s = 0 .. S - 1:
  h_prev          = (t > 0) ? h_{t-1}[d, s] : 0
  dh              = dh_next[s] + dy * C[n, t, s]
  dC[n, t, s]    += dy * h_t[d, s]
  d_dt_A          = dh * h_prev * dA_t[d, s]
  ddt_acc        += d_dt_A * A[d, s] + dh * B[n, t, s] * x[n, t, d]
  dA_log[d, s]   += d_dt_A * dt_t * A[d, s]        (since dA/dA_log = -exp(A_log) = A)
  dB[n, t, s]    += dh * dt_t * x[n, t, d]
  dx_acc         += dh * dt_t * B[n, t, s]
  dh_next[s]      = dh * dA_t[d, s]

dX[n, t, d]       = dx_acc
dDt_raw[n, t, d]  = ddt_acc * sigmoid(dt_raw[n, t, d])
```

## Stage C: O(1) constant-memory recurrent cache formula

Unlike attention KV-caches whose memory grows linearly as `O(N * T * D)`, `et_mamba_cache_v1` stores only a fixed-size sliding convolution window and recurrent SSM state:

```text
conv_state_bytes  = N * (K_conv - 1) * D_inner * sizeof(float)
ssm_state_bytes   = N * D_inner * D_state * sizeof(double)
total_state_bytes = conv_state_bytes + ssm_state_bytes   (strictly O(1) w.r.t. T)
```

| Configuration `(N, D, S, K)` | `conv_state_bytes` | `ssm_state_bytes` | `total_state_bytes` (T = 8, 32, 64, ...) | Equivalent f32 KV-cache (`2*N*T*D*4`) at T = 64 |
|---|---:|---:|---:|---:|
| `(1, 8, 4, 4)` (Eshkol AOT) | 96 B | 256 B | **352 B** (constant) | 4,096 B (11.6x larger) |
| `(2, 16, 8, 4)` (Native C11) | 384 B | 2,048 B | **2,432 B** (constant) | 16,384 B (6.7x larger) |
| `(4, 32, 16, 4)` (Max admitted) | 1,536 B | 16,384 B | **17,920 B** (constant) | 65,536 B (3.7x larger) |

## Verification results

Verified inside `eshkol-r0-debug:20260928` (`CC=clang-21`, `CXX=clang++-21`, `--network none`) via `bash scripts/test-mamba-ssm.sh`, `python3 -m unittest -v tests/q0/test_python_isolation.py`, and `bash scripts/check-ci-topology.sh`:

| Check | Result | Measured Accuracy / Evidence |
|---|---|---|
| Deterministic archives (`build-mamba-ssm.sh`) | PASS | Byte-identical across fresh dirs; zero allocator imports in `nm -u` |
| C++17 header & linkage (`header_mamba_cpp.cpp`) | PASS | Clean under `-std=c++17 -Wall -Wextra -Werror -Wpedantic` |
| Stage A Conv1D+SiLU FD (`test_mamba_native.c`) | PASS | `checked=156`, `max_abs_err=1.033e-05` (`<= 1e-4`), zero heap allocs |
| Stage B SelectiveScan S6 FD (`test_mamba_native.c`) | PASS | `checked=346` (all 7 tensors, gated & ungated), `max_abs_err=1.292e-05` (`<= 1e-4`) |
| Stage C O(1) cache vs full-seq parity | PASS | `max_abs_err=0.000e+00` (`<= 1e-6`), including max shape `4x64x32x16x4` and reset |
| Independent Python binary64 oracle (`test_mamba_oracle.py`) | PASS | `cases=8`, `values=11520`, oracle binary64 FD `4.484e-13`, C vs oracle `max_abs_diff=2.270e-07` (`<= 1e-6`), cache `max_cache_diff=0.000e+00` |
| Negative fail-closed contracts & sentinels | PASS | Bad dims/K, `A_log` overflow (`85.0`, `800.0`), `NaN`/`±Inf`, misalignment, aliasing, mismatched `z`/`dz`, wrong cache sizes, corrupted cache state, non-default FP env (`FE_UPWARD`, MXCSR FTZ); all outputs and cache state preserved byte-for-byte |
| ASan / UBSan / LSan (`detect_leaks=1:halt_on_error=1`) | PASS | C11 (`--wrap=malloc/calloc/realloc/free`) + C++17 clean |
| Compiled Eshkol AOT driver (`test_mamba_ssm_aot.esk`) | PASS | Dual fresh compilations byte-identical ELF (`59e9f8fe...`); 27 FD checks across all 9 tensors (`max_abs_err=1.546e-06`); 12-step SGD strictly decreasing MSE (`0.480462 -> 0.230504`); O(1) cache `state_bytes=352` at `T=8,32,64` with `max_abs_diff=0` |
| Repo gates (`test_python_isolation.py`, `check-ci-topology.sh`) | PASS | 5/5 Python isolation tests OK; 23 shared CI suites topology PASS |

## Measured limitations and unsupported cases

- **CPU-only serial execution**: Executes on x86-64 CPU in IEEE-754 binary64 internal arithmetic; no GPU/HIP/CUDA offload or parallel prefix-scan kernel is claimed.
- **Bounded dimensions**: `N in 1..4`, `T in 1..64`, `D_inner in 2..32`, `D_state in 2..16`, `K_conv in 2..4` (enforced to guarantee bounded stack staging with zero heap allocation). Out-of-range shapes fail closed with `ET_MAMBA_ERR_UNSUPPORTED`.
- **`A_log` upper bound**: `A_log > 80.0` is rejected with `ET_MAMBA_ERR_OVERFLOW` before `exp(A_log)` can overflow binary32/binary64.
- **Dtype and FP environment**: Only dense row-major binary32 tensor I/O (`float`, with `double` for `ssm_state`) under x86-64 default `FE_TONEAREST` without FTZ/DAZ is supported; non-default FP environments fail closed with `ET_MAMBA_ERR_FP_ENV`.
- **Linear projection layers external**: Input/output linear projections (`in_proj`, `x_proj`, `dt_proj`, `out_proj`) are composed by callers outside the core S6 kernel (`Stage A` + `Stage B` + `Stage C`).
