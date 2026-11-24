# RoPE Long-Context Extension Engine (Standard, Linear PI, NTK-Aware, YaRN)

Standalone, Eshkol-native Rotary Position Embedding (RoPE) long-context extension
engine providing four exact frequency-scaling regimes (`STANDARD`, `LINEAR_PI`,
`NTK_AWARE`, and `YARN` with attention temperature `mscale`), two channel-pair
layouts (`NEOX_SPLIT_HALF` and `INTERLEAVED`), partial rotary dimensions
(`d_rot <= head_dim`), explicit integer position IDs, exact analytic backward
(VJP), relative-position QK dot-product invariance verification, and a compiled
Eshkol AOT driver.

## Artifacts and public contracts

- Public C11/C++17 ABI header: `include/eshkol_transformer/rope_yarn_abi.h`
- Core C11 implementation: `native/rope_yarn.c` ->
  `build/rope_yarn/libeshkol_transformer_rope_yarn.a`
  - Deterministic archive SHA-256:
    `b41cba33971ad392f9a1db4eff6ca12b0ad4b8d15e07298f7d4bc4b31d7ae2b7`
  - Exports only:
    - `et_rope_yarn_validate_config_v1`
    - `et_rope_yarn_compute_freqs_v1`, `et_rope_yarn_compute_freqs_raw_v1`
    - `et_rope_yarn_forward_v1`, `et_rope_yarn_forward_raw_v1`
    - `et_rope_yarn_backward_v1`, `et_rope_yarn_backward_raw_v1`
    - `et_rope_yarn_qk_dot_invariance_check_v1`,
      `et_rope_yarn_qk_dot_invariance_check_raw_v1`
  - Strictly zero heap allocations (`malloc`/`calloc`/`realloc`/`free` absent
    from undefined imports and trapped via `--wrap` during native and sanitizer
    tests).
- Eshkol AOT bridge: `native/rope_yarn_eshkol_bridge.h`,
  `native/rope_yarn_eshkol_bridge.c` ->
  `build/rope_yarn/libeshkol_transformer_rope_yarn_aot.a`
  - Deterministic archive SHA-256:
    `4d7f3b65d3e32b747be2cd813f89ddc77f9d66cc136f95db2985f1772a3011e8`
  - Zero allocator imports; accepts untyped `void *` pointers so intentionally
    misaligned raw pointers from Eshkol FFI are validated cleanly without UB.
- Build and test scripts: `scripts/build-rope-yarn.sh`,
  `scripts/test-rope-yarn.sh`
- Tests:
  - `tests/rope_yarn/header_rope_yarn_cpp.cpp`
  - `tests/rope_yarn/test_rope_yarn_native.c`
  - `tests/rope_yarn/test_rope_yarn_oracle.py`
  - `tests/rope_yarn/test_rope_yarn_aot.esk`

## Tensor-shape and layout contract

- Activation / gradient tensors: dense row-major binary32 (`float`) of shape
  `[N, T, H, Dh]` with:
  - `N in [1, 64]` (batch)
  - `T in [1, 16384]` (sequence tokens)
  - `H in [1, 128]` (attention heads)
  - `Dh in [2, 256]` (head dimension)
  - `d_rot` even in `[2, Dh]` (`d_rot >= 4` required for `NTK_AWARE`)
- Position IDs: dense row-major `int64_t` of shape `[N, T]`, supporting
  arbitrary non-contiguous, shifted, or extended integer positions
  `pos in [0, min(16777216, floor(orig_max_pos * s))]`.
- Channel-pair layouts over the leading rotary prefix `[0 .. d_rot - 1]`
  (`half = d_rot / 2`, `k = 0 .. half - 1`):
  - `ET_ROPE_LAYOUT_NEOX_SPLIT_HALF` (`0`): pair `(x[k], x[half + k])`
  - `ET_ROPE_LAYOUT_INTERLEAVED` (`1`): pair `(x[2*k], x[2*k + 1])`
- Partial rotary (`d_rot < Dh`): trailing channels `d in [d_rot .. Dh - 1]` are
  validated finite and passed through bit-for-bit unchanged in both forward and
  backward.

## Mathematical specification

All inverse frequencies, angles, trigonometric functions, and `mscale` are
evaluated in IEEE-754 binary64 (`double`) with `-ffp-contract=off
-fexcess-precision=standard -fno-fast-math`.

For `i = 0, 1, ..., d_rot / 2 - 1` and base frequency `base > 1.0` (default
`10000.0`), let `inv_freq_orig[i] = base^(-2*i / d_rot)`:

1. **`ET_ROPE_SCALE_STANDARD` (`0`)** (requires `s == 1.0`):
   ```text
   inv_freq[i] = base^(-2*i / d_rot)
   mscale      = 1.0
   ```

2. **`ET_ROPE_SCALE_LINEAR_PI` (`1`)** (Position Interpolation, `s >= 1.0`):
   ```text
   inv_freq[i] = base^(-2*i / d_rot) / s
   mscale      = 1.0
   ```

3. **`ET_ROPE_SCALE_NTK_AWARE` (`2`)** (NTK-Aware base scaling, `s >= 1.0`,
   even `d_rot >= 4`):
   ```text
   base'       = base * s^(d_rot / (d_rot - 2))
   inv_freq[i] = (base')^(-2*i / d_rot)
   mscale      = 1.0
   ```

4. **`ET_ROPE_SCALE_YARN` (`3`)** (Peng et al., 2023; `orig_max_pos >= 1`,
   `s >= 1.0`, `beta_fast > beta_slow > 0`, defaults `beta_fast = 32.0`,
   `beta_slow = 1.0`):
   ```text
   lambda[i]   = 2 * pi / inv_freq_orig[i]
   r[i]        = orig_max_pos / lambda[i]
   gamma[i]    = clamp((r[i] - beta_slow) / (beta_fast - beta_slow), 0, 1)
   inv_freq[i] = (1 - gamma[i]) * (inv_freq_orig[i] / s) + gamma[i] * inv_freq_orig[i]
   mscale      = (s <= 1.0) ? 1.0 : (0.1 * ln(s) + 1.0)
   ```
   High-frequency bands (`r[i] >= beta_fast`, `gamma[i] = 1`) remain unscaled,
   low-frequency bands (`r[i] <= beta_slow`, `gamma[i] = 0`) interpolate by
   `1 / s`, and intermediate bands blend smoothly.

### Forward, backward, and relative-position invariance

For position `pos` and pair `k`:
```text
phi = (double)pos * inv_freq[k]
c_m = mscale * cos(phi)
s_m = mscale * sin(phi)
```

- **Forward** (`et_rope_yarn_forward_v1`):
  ```text
  y0 = (float)((double)x0 * c_m - (double)x1 * s_m)
  y1 = (float)((double)x0 * s_m + (double)x1 * c_m)
  ```
- **Exact analytic backward / VJP** (`et_rope_yarn_backward_v1`):
  Since `y_rot = mscale * R(phi) * x_rot`, the exact transpose is
  `dx_rot = mscale * R(-phi) * dy_rot`:
  ```text
  dx0 = (float)( (double)dy0 * c_m + (double)dy1 * s_m)
  dx1 = (float)(-(double)dy0 * s_m + (double)dy1 * c_m)
  ```
- **Relative-position QK dot-product invariance**
  (`et_rope_yarn_qk_dot_invariance_check_v1`):
  For any valid `pos_m`, `pos_n`, and `shift >= 0` with `pos_m + shift` and
  `pos_n + shift` in range:
  ```text
  <RoPE(q, m), RoPE(k, n)>
    = <RoPE(q, m + shift), RoPE(k, n + shift)>
    = mscale^2 * <R(m - n) q_rot, k_rot> + <q_tail, k_tail>
  ```
  Scaling both `Q` and `K` by `mscale` scales the rotary attention logit
  `Q K^T` by exactly `mscale^2`.

## Safety, validation, and failure atomicity

- Rejects `NULL` pointers, invalid `mode`/`layout`, odd `d_rot`, `d_rot > Dh`,
  `NTK_AWARE` with `d_rot == 2`, `s < 1.0`, `STANDARD` with `s != 1.0`,
  `base <= 1.0`, `beta_fast <= beta_slow`, `beta_slow <= 0.0`,
  `orig_max_pos == 0`, out-of-range positions (`pos < 0` or
  `pos > floor(orig_max_pos * s)`), misaligned pointers, pairwise overlapping
  spans, non-finite inputs (including unrotated tail elements), binary32 output
  overflows (e.g. `FLT_MAX * mscale`), and non-standard x87/MXCSR FP
  environments (`FE_UPWARD`, unmasked exceptions, FTZ, DAZ).
- Executes a dry-run preflight pass before writing any output byte and saves
  and restores `fenv_t` and `errno`, guaranteeing caller output buffers remain
  byte-for-byte untouched on any error.

## Measured verification results

Verified inside `eshkol-r0-debug:20260928` (`clang-21` / `clang++-21` 21.1.8,
pinned Eshkol toolchain `/tc`):

- **Deterministic archives & symbol audit** (`scripts/build-rope-yarn.sh`):
  - Two fresh builds produce byte-identical `.a` archives.
  - Zero allocator imports in both `rope_yarn.o` and
    `rope_yarn_eshkol_bridge.o`: PASS.
- **C++17 header & linkage** (`tests/rope_yarn/header_rope_yarn_cpp.cpp`):
  - Normal `-O2` and ASan/UBSan/LSan: PASS.
- **Native C11 suite** (`tests/rope_yarn/test_rope_yarn_native.c`, linked with
  `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`, normal `-O2` and
  ASan/UBSan/LSan `detect_leaks=1:halt_on_error=1`):
  - 11 cases spanning all 4 modes, both layouts, full and partial rotary up to
    `Dh = 64` and `pos = 16384`, plus `s = 1.0` exact reduction and all 3 YaRN
    frequency bands:
    - 462 central finite-difference checks: `max_fd_err = 1.335e-05`
    - Norm scaling (`||y_rot||^2 == mscale^2 ||x_rot||^2`):
      `max_norm_err = 7.847e-07`
    - Roundtrip (`backward(forward(x)) == mscale^2 x_rot`):
      `max_inv_err = 1.269e-07`
    - QK dot-product shift invariance: `max_shift_err = 8.385e-08`
    - QK dot-product relative identity: `max_rel_err = 1.903e-07`
  - Negative, FP-environment, and failure-atomicity checks: PASS.
- **Independent Python binary64 oracle**
  (`tests/rope_yarn/test_rope_yarn_oracle.py`):
  - 11 cases, 6,600 forward & backward values, 50 independent central FD checks,
    7 negative status witnesses:
    - `max_freq_err = 0.000e+00`
    - `max_mscale_err = 0.000e+00`
    - `max_val_err = 5.948e-08`
    - `max_fd_err = 5.895e-08`
    - `max_inv_err = 2.287e-07`
- **Compiled Eshkol AOT driver** (`tests/rope_yarn/test_rope_yarn_aot.esk`,
  `eshkol-run --strict-types --optimize 0 --no-stdlib`):
  - Negative rejections with untouched sentinels (`checks=8`): PASS.
  - All 4 scaling modes + YaRN `s = 4.0` exact `mscale = 1.138629436111989` and
    all 3 YaRN bands (`f0 = 1`, `f3 = 0.015026497703256057`,
    `f7 = 7.905694150420948e-05`): PASS.
  - Relative-position QK invariance across `pos = 0..8192` (`checked=8`,
    both layouts + partial rotary): `max_abs_err = 2.609512130147956e-07`, PASS.
  - Central FD gradient checks across all 4 modes, both layouts, and partial
    rotary (`checked=48`): `max_abs_err = 7.659196853637695e-06`, PASS.
  - Two fresh compilations from clean directories produced byte-identical ELFs
    and byte-identical stdout:
    - ELF SHA-256:
      `380d364ace0d98e5e7034c35ff1dd296c90e4b9350a20673f4fb7bda5c486440`
- **Repository gates**:
  - `python3 -m unittest -v tests/q0/test_python_isolation.py`: 5/5 OK.
  - `bash scripts/check-ci-topology.sh`: PASS (23 shared suites, 38 commands).

## Limitations and unsupported cases

- Serial CPU execution only on the reviewed x86-64 IEEE-754 floating-point lane
  (no GPU backend, no SIMD claims, no mixed-precision f16/bf16 kernels).
- Bounded tensor extents (`N <= 64`, `T <= 16384`, `H <= 128`, `Dh <= 256`,
  `pos <= 16777216`, `scale_factor <= 1048576.0`) to guarantee zero heap
  allocations via fixed stack tables.
- Dynamic per-step NTK (`s` updated dynamically from runtime `seq_len`) is
  expressed by passing the caller-chosen `scale_factor` in `et_rope_yarn_config_t`;
  automatic sequence-length-driven `scale_factor` mutation is not performed
  implicitly so all frequency scaling remains explicit and deterministic.
