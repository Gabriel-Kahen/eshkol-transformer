# DeepSeek-V3 Sequential Multi-Token Prediction (MTP)

This standalone module implements an Eshkol-native **DeepSeek-V3 Sequential Multi-Token Prediction (`D in 1..4`)** engine in C11 (`include/eshkol_transformer/mtp_head_abi.h`, `native/mtp_head.c`), an Eshkol AOT bridge (`native/mtp_head_eshkol_bridge.{h,c}`), a compiled Eshkol AOT driver (`tests/mtp_head/test_mtp_head_aot.esk`), native C11/C++17 tests, and an independent Python IEEE-754 binary64 reference oracle.

## Artifacts and public contract

- Public ABI header: `include/eshkol_transformer/mtp_head_abi.h` (`extern "C"`, C11 and C++17 standard-layout clean)
- Core static archive: `build/mtp_head/libeshkol_transformer_mtp_head.a`
  - Deterministic archive SHA-256 (`eshkol-r0-debug:20260928`, `clang-21`): `7e92612093f563191cfd11d48696d64bf592316b3586699e2972d7784475c7a9`
  - Exported symbols (20):
    - Configuration & layout: `et_mtp_validate_config_v1`, `et_mtp_compute_layout_v1`
    - Model lifecycle & state: `et_mtp_model_create_v1`, `et_mtp_model_destroy_v1`, `et_mtp_model_init_weights_v1`, `et_mtp_model_get_config_v1`, `et_mtp_model_get_layout_v1`, `et_mtp_model_get_step_v1`, `et_mtp_model_get_params_v1`, `et_mtp_model_set_params_v1`, `et_mtp_model_get_moments_v1`, `et_mtp_model_state_sha256_v1`
    - Inference, forward/backward, optimization & speculative decode: `et_mtp_main_logits_v1`, `et_mtp_forward_v1`, `et_mtp_forward_backward_v1`, `et_mtp_adamw_step_v1`, `et_mtp_draft_verify_v1`
    - Checkpointing & integrity: `et_mtp_checkpoint_save_v1`, `et_mtp_checkpoint_load_v1`, `et_mtp_sha256_bytes_v1`
- Eshkol AOT bridge archive: `build/mtp_head/libeshkol_transformer_mtp_head_aot.a`
  - Deterministic archive SHA-256 (`eshkol-r0-debug:20260928`, `clang-21`): `34dbb86eee22f88a4b81573f6d5d55a57781c1f58c08ef8e2fa990176d4c1717`
  - Exports 35 `et_mtp_aot_*` session, training, FD, self-speculative verification, checkpointing, negative probe, and telemetry symbols.
- Compiled Eshkol AOT driver: `tests/mtp_head/test_mtp_head_aot.esk`
  - Compiled twice from independent fresh directories via `eshkol-run --strict-types --optimize 0 --no-stdlib`.
  - Byte-identical ELF SHA-256: `bf8566b11f3d54d0324bfa77037ef41c3060088510f33b8953ed5db7a06184ae`

## Admitted domain and tensor shape contract

| Dimension | Symbol | Admitted Range |
|---|---:|---:|
| Vocabulary size | `V` | `2 .. 1024` |
| Hidden dimension | `H` | `2 .. 128` |
| SwiGLU FFN width | `F` | `2 .. 256` |
| MTP lookahead depth | `D` | `1 .. 4` |
| Weight tying flag | `tie_weights` | `0` (untied) or `1` (tied `W_lm == E`) |
| Batch size | `B` | `1 .. 16` |
| Sequence length | `T` | `D + 2 .. 256` (`T <= D + 1` rejected with `ET_MTP_ERR_SHAPE_MISMATCH`) |

Tensor layouts (dense row-major IEEE-754 binary32 `float` / `uint32_t`):

- `h0`, `d_h0_out`: `float[B, T, H]`
- `tokens`: `uint32_t[B, T]` with every entry `< V`
- Shared embedding `E`: `float[V, H]`
- Shared output head `W_lm`: `float[V, H]` (when `tie_weights == 1`, `W_lm` aliases `E` and `lm_head_len == 0` in the flat parameter vector)
- Main final norm `gamma_main`: `float[H]`
- Per-depth MTP block `k in 1..D`:
  - `gamma_h_k`, `gamma_e_k`, `gamma_f_k`, `gamma_o_k`: `float[H]`
  - `W_proj_k`: `float[H, 2H]`
  - `W_gate_k`, `W_up_k`: `float[F, H]`
  - `W_down_k`: `float[H, F]`
- Speculative verification inputs (`et_mtp_draft_verify_v1`):
  - `h0_prefix`: `float[H]`
  - `target_logits`: `float[D + 1, V]`
  - `draft_tokens_out`: `uint32_t[D]`
  - `out_tokens`: `uint32_t[out_capacity]` (`out_capacity >= D + 1`)

## DeepSeek-V3 sequential MTP mathematical specification

All internal math executes in IEEE-754 binary64 (`double`) with `#pragma STDC FENV_ACCESS ON`, `#pragma STDC FP_CONTRACT OFF`, `-ffp-contract=off -fexcess-precision=standard -fno-fast-math`, verified x87/MXCSR rounding environment (`FE_TONEAREST`, all exceptions masked, FTZ/DAZ disabled), and saved/restored `fenv_t`.

### Forward pass

Given trunk hidden states `h^0[b, t]` and token IDs `tokens[b, t]` (`t in 0 .. T - 1`):

1. **Main next-token head (`k = 0`)** over `b in 0..B-1`, `t in 0..T-2` (`N_0 = B * (T - 1)`):
   ```text
   o^0[b, t]      = RMSNorm(h^0[b, t], gamma_main)
   logits_0[b, t] = W_lm @ o^0[b, t]
   L_main         = (1 / N_0) * sum_{b, t} CrossEntropy(logits_0[b, t], tokens[b, t + 1])
   ```
2. **Sequential MTP heads (`k = 1 .. D`)** over `b in 0..B-1`, `t in 0..T-2-k` (`N_k = B * (T - 1 - k) >= B`):
   ```text
   u_h            = RMSNorm(h^{k-1}[b, t], gamma_h_k)
   u_e            = RMSNorm(E[tokens[b, t + k]], gamma_e_k)
   z_k            = W_proj_k @ concat(u_h, u_e)
   h^k[b, t]      = z_k + W_down_k @ (SiLU(W_gate_k @ RMSNorm(z_k, gamma_f_k)) * (W_up_k @ RMSNorm(z_k, gamma_f_k)))
   logits_k[b, t] = W_lm @ RMSNorm(h^k[b, t], gamma_o_k)
   L_k            = (1 / N_k) * sum_{b, t} CrossEntropy(logits_k[b, t], tokens[b, t + 1 + k])
   ```
3. **Combined objective**:
   ```text
   L_mtp_avg = (1 / D) * sum_{k=1}^{D} L_k
   L_total   = L_main + lambda_mtp * L_mtp_avg
   ```

### Exact reverse-depth analytic backward (`et_mtp_forward_backward_v1`)

For each independent `(b, t)` vertical chain (`k_limit = min(D, T - 2 - t)`):

1. Forward-replay `k = 1 .. k_limit` into bounded per-position binary64 scratch.
2. Sweep backward `k = k_limit` down to `1` carrying `dh_carry`, summing the depth-`k` output-head gradient (`scale_k = (lambda_mtp / D) / N_k`) with `dh_carry`, backpropagating through the residual SwiGLU FFN, `W_proj_k`, `RMSNorm(E[tokens[b, t + k]], gamma_e_k)` into shared `d_E`, and `RMSNorm(h^{k-1}, gamma_h_k)` into `dh_carry`.
3. Backpropagate main head `k = 0` (`scale_0 = 1 / N_0`) into `d_W_lm` (or `d_E` when tied), `d_gamma_main`, and `d_h0[b, t] = dh_carry + dh_main`. At `t = T - 1`, `d_h0[b, T - 1] = 0`.
4. All parameter gradients accumulate in binary64 before staged finite conversion to binary32.

### Self-speculative MTP drafting and exact verification (`et_mtp_draft_verify_v1`)

Starting from `h^0_t` and `anchor_token` (either explicit `< V` or computed from the main head via `ET_MTP_ANCHOR_FROM_MAIN`):

- Sequential MTP heads `k = 1 .. D` propose `D` draft tokens (`temperature == 0.0` greedy with lowest-index tie-break, 0 RNG draws; `temperature > 0.0` seeded SplitMix64 counter RNG in binary64).
- Exact verification compares `draft_tokens[0..D-1]` against `target_logits[0..D, V]`, emitting `0 .. D` accepted drafts plus 1 correction or bonus token (with immediate EOS early stop when `eos_token_id != ET_MTP_NO_EOS`).
- Greedy verification is provably bitwise identical to step-by-step greedy target decoding.

### Versioned SHA-256 checkpoint format (`ETMT1000` v1.0)

- Layout: 128-byte little-endian header (`"ETMT1000"`, version `1.0`, `V`, `H`, `F`, `D`, `tie_weights`, `total_params`, `step`, all config & AdamW `f64` bit patterns, `payload_bytes`, reserved `0`) + canonical binary32 `params`, `m`, `v` + 32-byte FIPS 180-4 SHA-256 trailer.
- Save: `O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC` temp file with `fchmod(0600)`, `write_all`, `fsync(tmp_fd)`, destination non-symlink check, `renameat`, `fsync(dir_fd)`.
- Load: `O_RDONLY | O_NOFOLLOW | O_CLOEXEC`, regular-file & exact-size checks, magic/version/SHA-256/config verification, staged finite + `v >= 0` validation before atomic commit.

## Measured verification evidence (`eshkol-r0-debug:20260928`)

All checks executed inside `--rm --network none` with `CC=clang-21 CXX=clang++-21`:

- **Deterministic archives**: `cmp`-verified byte-identical across independent fresh builds:
  - `libeshkol_transformer_mtp_head.a`: `7e92612093f563191cfd11d48696d64bf592316b3586699e2972d7784475c7a9`
  - `libeshkol_transformer_mtp_head_aot.a`: `34dbb86eee22f88a4b81573f6d5d55a57781c1f58c08ef8e2fa990176d4c1717`
- **C++17 header & archive linkage**: PASS (`header_mtp_cpp.cpp`)
- **Native C11 exhaustive central FD (`D=1..4`, untied & tied, `fd_eps = 1/2048`)**:
  - Worst parameter max error across all 8 configs: `2.584e-06` (`< 1e-5`)
  - Worst `d_h0` max error across all 8 configs: `1.886e-08` (`< 1e-5`)
  - Shared `W_lm` / `E` gradient linearity across main + `D=3`: `d_W_lm_mtp_l1=3.2450`, `lin_err=1.490e-08`, `e_err=0.000e+00`
- **Independent Python binary64 oracle (`D=1..4` untied + `D=3` tied)**:
  - Forward losses, `d_h0`, all parameter gradients, and 12-step AdamW trajectory: `0.000e+00` max error across all 5 configurations, with exact self-speculative draft/accept parity.
- **Sanitizers**: ASan + UBSan + LSan (`detect_leaks=1:halt_on_error=1`) PASS on native C11 and C++17 binaries.
- **Compiled Eshkol AOT driver (`test_mtp_head_aot.esk`, `V=16, H=8, F=16, D=3, B=1, T=16`)**:
  - Dual fresh compilations byte-identical ELF SHA-256: `bf8566b11f3d54d0324bfa77037ef41c3060088510f33b8953ed5db7a06184ae`
  - Exhaustive central FD (`checked=2024`, `params=1896`): `max_param_err=9.847e-08`, `max_h0_err=3.593e-09`
  - 12-step AdamW strictly decreasing losses:
    - `L_total`: `4.058114 -> 0.091109`
    - `L_main`: `2.618571 -> 0.057087`
    - `L_mtp_avg`: `2.879086 -> 0.068044` (`L_1=0.048361`, `L_2=0.111977`, `L_3=0.043795`)
  - Self-speculative `D=3` greedy draft-verify: `exact=1`, `emitted=14`, `steps=4`, `tokens_per_step=3.500` (`> 1.0`)
  - Fresh-process 6+6 `ETMT1000` checkpoint resume bitwise parity SHA-256: `b0842de3ccf2cb33c497dac01518743bcba5b35536e32ffb085a78d4534d05eb`
  - Negative boundaries & checkpoint tamper/truncation/symlink/config-mismatch rejections: PASS
- **Repository gates**: `tests/q0/test_python_isolation.py` (5/5 OK) and `scripts/check-ci-topology.sh` (PASS).

## Measured limitations and unsupported cases

- **CPU x86-64 IEEE-754 lane only**: requires `FE_TONEAREST` with masked exceptions and FTZ/DAZ disabled; non-default FP environments fail closed with `ET_MTP_ERR_FP_ENV`. No GPU kernel or mixed-precision execution is claimed.
- **Bounded dimensions**: `V <= 1024`, `H <= 128`, `F <= 256`, `D in 1..4`, `B <= 16`, `T in [D + 2, 256]`; out-of-range inputs fail closed with explicit error codes without mutating caller outputs or model state.
- **Single-token SwiGLU MTP block**: each MTP module uses the residual SwiGLU feed-forward block (`z_k + SwiGLU_k(RMSNorm(z_k))`) without an internal cross-position self-attention sublayer, enabling exact `O(D * (H + F))` vertical-chain reverse-depth backpropagation.
- **Standalone delivery**: no existing repository files (`Makefile`, `docs/ROADMAP.md`) were modified; build and verify via `bash scripts/build-mtp-head.sh` and `bash scripts/test-mtp-head.sh`.
