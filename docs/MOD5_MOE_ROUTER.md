# MOD5 Sparse Top-K Mixture-of-Experts (MoE) Router and SwiGLU Dispatch

MOD5 MoE provides a deterministic Sparse Top-K Mixture-of-Experts layer
(`kernel.moe`, operations `moe.forward` and `moe.backward`) through K1 ABI 1.0
and direct C11 helpers, plus an Eshkol AOT bridge and compiled training driver.

## Artifacts and public contracts

- Header: `include/eshkol_transformer/mod5_moe_abi.h` (C11 and C++17 compatible)
- Provider archive: `build/mod5_moe/libeshkol_transformer_mod5_moe.a`
  - Exports only `et_mod5_moe_kernel_provider_v1`, `et_mod5_moe_forward_f32`,
    and `et_mod5_moe_backward_f32`.
  - Zero allocator imports (`malloc`, `calloc`, `realloc`, `free` absent from
    undefined symbols and trapped during dispatch tests).
- Eshkol AOT bridge: `native/mod5_moe_eshkol_bridge.h`,
  `native/mod5_moe_eshkol_bridge.c`,
  `build/mod5_moe/libeshkol_transformer_mod5_moe_bridge.a`
  - Exports only
    `et_mod5_moe_aot_{open,close,forward,backward,store_f32,load_f32,store_i64,load_i64}`.
  - Accepts untyped `void *` tensor pointers so raw Eshkol FFI pointers
    (including intentionally misaligned `ptr-add` addresses) are validated
    cleanly without misaligned-pointer UB.

## Mathematical specification

For batch `N in 1..2`, tokens `T in 1..32` (`S = N * T`), model width
`D in 2..32`, expert hidden width `F in 2..64`, expert count `E in 2..8`,
active experts per token `K = top_k in 1..4` with `K <= E`, and Switch
load-balancing weight `alpha_aux >= 0.0`:

### 1. Router logits, binary64 softmax, and deterministic Top-K selection

For each token `s in 0 .. S - 1` (`x_s in R^D`) and router weights
`W_gate in R^{E x D}`:

```text
R[s, e]  = sum_{d=0..D-1} x[s, d] * W_gate[e, d]          (binary64)
m_s      = max_{e=0..E-1} R[s, e]
P[s, e]  = exp(R[s, e] - m_s) / sum_{m=0..E-1} exp(R[s, m] - m_s)
```

Top-K selection picks `K` distinct expert indices `e_{s,0}, ..., e_{s,K-1}` in
descending order of `P[s, e]`, breaking exact ties deterministically by lowest
expert index `e`. Renormalized routing weights:

```text
S_p[s]   = sum_{j=0..K-1} P[s, e_{s,j}]
w[s, j]  = P[s, e_{s,j}] / S_p[s]
```

### 2. Sparse SwiGLU expert dispatch

Only the `K` selected experts per token execute SwiGLU (`active_expert_evals ==
S * K < S * E` whenever `K < E`); unselected experts are skipped completely:

```text
g[f]     = sum_{d=0..D-1} W_gate_exp[e, f, d] * x[s, d]
u[f]     = sum_{d=0..D-1} W_up_exp[e, f, d]   * x[s, d]
sig[f]   = sigmoid(g[f])                                   (stable binary64)
silu[f]  = g[f] * sig[f]
h[f]     = silu[f] * u[f]
o[d]     = sum_{f=0..F-1} W_down_exp[e, d, f] * h[f]
Y[s, d]  = sum_{j=0..K-1} w[s, j] * o_{s, j}[d]
```

### 3. Switch load-balancing auxiliary loss

With per-expert routed token counts `expert_counts[e]` summing to `S * K`:

```text
f_e       = expert_counts[e] / (S * K)
P_mean_e  = (1 / S) * sum_{s=0..S-1} P[s, e]
L_aux     = alpha_aux * E * sum_{e=0..E-1} f_e * P_mean_e
```

### 4. Exact analytic backward

Given upstream task gradient `dY = dL_task / dY` (`f32[N, T, D]`), `moe.backward`
computes exact analytic gradients of `L_total = L_task(Y) + L_aux` (away from
discrete top-K tie boundaries):

- Selected expert `e = e_{s, j}` on token `s` (unselected experts contribute
  zero and are skipped):
  ```text
  dw[s, j]         = sum_d dY[s, d] * o_{s, j}[d]
  do[d]            = w[s, j] * dY[s, d]
  dW_down_exp[e]  += do * h^T
  dh[f]            = sum_d W_down_exp[e, d, f] * do[d]
  du[f]            = dh[f] * silu[f]
  dg[f]            = (dh[f] * u[f]) * sig[f] * (1 + g[f] * (1 - sig[f]))
  dW_gate_exp[e]  += dg * x_s^T
  dW_up_exp[e]    += du * x_s^T
  dX[s, d]        += sum_f (W_gate_exp[e, f, d] * dg[f] + W_up_exp[e, f, d] * du[f])
  ```
- Router probability gradient combining top-K renormalization (`w -> P`) and
  Switch auxiliary loss (`L_aux -> P_mean -> P`):
  ```text
  dP_aux[s, e]     = (alpha_aux * E * f_e) / S
  dP_route[s, e_j] = (dw[s, j] - sum_{m<K} w[s, m] * dw[s, m]) / S_p[s]
  dP[s, e]         = dP_aux[s, e] + (dP_route[s, e] if e in top_k(s) else 0)
  dR[s, e]         = P[s, e] * (dP[s, e] - sum_{m<E} P[s, m] * dP[s, m])
  dW_gate[e, d]   += dR[s, e] * x[s, d]
  dX[s, d]        += dR[s, e] * W_gate[e, d]
  ```
  Note that when `K == 1`, `w[s, 0] == 1` identically so `dP_route == 0` and
  router gradients come exclusively from `L_aux`.

## K1 ABI 1.0 contract and validation

- Provider / implementation: `mod5.moe.cpu-f32.serial`, version `1.0`,
  evidence `MOD5:cpu-f32-moe-v1`, `VERIFIED`, `deterministic = 1`, compute
  dtype `f32`, device `cpu`.
- Request shape `[N, T, D, F, E, K]` across 4 verified K1 shape-range rows
  encoding `K in 1..4` and `K <= E`.

| Operation | Inputs in order | Outputs in order |
|---|---|---|
| `moe.forward` | `x f32[N,T,D]`, `w_gate f32[E,D]`, `w_gate_exp f32[E,F,D]`, `w_up_exp f32[E,F,D]`, `w_down_exp f32[E,D,F]`, `alpha_aux f32[]` | `y f32[N,T,D]`, `aux_loss f32[]`, `expert_counts i64[E]`, `active_expert_evals i64[]` |
| `moe.backward` | Same 6 inputs + `dy f32[N,T,D]` | `dx f32[N,T,D]`, `dw_gate f32[E,D]`, `dw_gate_exp f32[E,F,D]`, `dw_up_exp f32[E,F,D]`, `dw_down_exp f32[E,D,F]` |

Validation enforces:
- Out-of-range dimensions and `top_k > E` reject with `UNSUPPORTED`.
- NULL or misaligned pointers (`_Alignof(float)` / `_Alignof(int64_t)`),
  input-input overlap, output-input/output-output/metadata/`error` aliasing,
  negative or non-finite `alpha_aux`, non-finite elements in any input/weight
  (including unselected experts), and binary32/binary64 overflow reject with
  `INVALID_ARGUMENT`.
- Non-standard x87/MXCSR FP environments (non-RN rounding, unmasked exceptions,
  FTZ, DAZ) reject with `UNSUPPORTED`.
- Dry-run preflight restores `fenv_t` and `errno` and guarantees all output
  buffers remain byte-for-byte untouched on any failure.

## Validation results

Run `bash scripts/test-mod5-moe.sh` in the pinned Ubuntu 22.04 / Clang 21.1.8
/ Eshkol toolchain environment (`eshkol-r0-debug:20260928`):

- Deterministic archive & symbol allowlist audit (`mod5_moe_provider.o`,
  `mod5_moe_eshkol_bridge.o`, zero allocator imports): PASS.
- C++17 header and linkage check (`clang++-21 -std=c++17 -Wall -Wextra -Werror -Wpedantic`): PASS.
- Native C11 suite (normal `-O2` and ASan/UBSan/LSan with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`
  and `--wrap=malloc/calloc/realloc/free` trapping allocations during dispatch):
  - 7 shape cases from `[1, 1, 2, 2, 2, 1]` up to max boundary
    `[2, 32, 32, 64, 8, 4]`.
  - 446 central finite-difference checks across `dX`, `dW_gate` (with
    `alpha_aux > 0`), `dW_gate_exp`, `dW_up_exp`, and `dW_down_exp`:
    `max_fd_err = 2.715e-04`.
  - Sparse skip proof (mutating finite unselected expert weights produces
    bit-identical outputs and gradients; NaN in unselected expert still rejects),
    full and partial lowest-expert-index tie-breaking verification, linear
    `dW_gate` decomposition (`route + aux`), and full negative suite with
    untouched output sentinels: PASS.
- Independent binary64 Python oracle (`tests/mod5_moe/test_mod5_moe_oracle.py`):
  - 6 cases, 10,062 forward, telemetry, `aux_loss`, and 5-tensor backward values:
    exact `expert_counts` and `active_expert_evals` match,
    `max_abs_diff = 1.529e-06` (`<= 1e-5`).

## Compiled Eshkol AOT 4-expert Top-2 MoE driver

`tests/mod5_moe/test_mod5_moe_aot.esk` compiles with
`eshkol-run --strict-types --optimize 0 --no-stdlib` linking
`libeshkol_transformer_mod5_moe.a`, `libeshkol_transformer_k1.a`, and
`mod5_moe_eshkol_bridge.o`.

Configuration: `N=1, T=8, D=8, F=16, E=4, top_k=2, alpha_aux=0.0625, lr=0.25`.
Active sparse expert evaluations per forward: `16` (`< 32` dense).

- Malformed FFI checks (`top_k > E`, `E=9`, misaligned raw pointers via
  `ptr-add`, negative `alpha_aux`, output aliasing input, non-finite input):
  PASS with all 4 output sentinels untouched.
- Central finite-difference verification across all 4 parameter tensors and
  input `X` (20 checks): `max_abs_err = 1.0102487237872992e-06`, PASS.
- Two fresh compilations produce byte-identical ELFs and byte-identical stdout:
  - ELF SHA-256:
    `130342d0b3d6b029000f646a9cf2fca1462c7b43cba2432dab58b88efb4d5074`

| SGD Step | Total Loss (`task + aux`) | Task Loss (MSE) | Switch Aux Loss | Active Evals | Expert Counts (`E0,E1,E2,E3`) |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.3186614908876929 | 0.2558897384110007 | 0.0627717524766922 | 16 | `3,4,4,5` |
| 1 | 0.31640163526995113 | 0.2536333473132365 | 0.06276828795671463 | 16 | `3,4,4,5` |
| 2 | 0.31420629896203073 | 0.2514421386269668 | 0.06276416033506393 | 16 | `3,4,4,5` |
| 3 | 0.31206487014756296 | 0.24930546328291986 | 0.0627594068646431 | 16 | `3,4,4,5` |
| 4 | 0.30996806878024824 | 0.24721401888305433 | 0.06275404989719391 | 16 | `3,4,4,5` |
| 5 | 0.3079077354221167 | 0.24515962363765853 | 0.06274811178445816 | 16 | `3,4,4,5` |
| 6 | 0.30587663151458044 | 0.24313503153756397 | 0.06274159997701645 | 16 | `3,4,4,5` |
| 7 | 0.3038683301992823 | 0.24113377847151052 | 0.06273455172777176 | 16 | `3,4,4,5` |
| 8 | 0.3018770367366369 | 0.23915007715049338 | 0.0627269595861435 | 16 | `3,4,4,5` |
| 9 | 0.2998975556712668 | 0.23717870976739336 | 0.06271884590387344 | 16 | `3,4,4,5` |
| 10 | 0.29792516003602637 | 0.23521494935506476 | 0.06271021068096161 | 16 | `3,4,4,5` |
| 11 | 0.2959555810595672 | 0.23325450479041743 | 0.06270107626914978 | 16 | `3,4,4,5` |
| 12 | 0.29398489866557254 | 0.23129346344771515 | 0.06269143521785736 | 16 | `3,4,4,5` |

Final weighted parameter checksum: `-4.946719116534162`.

## Limitations and scope

- Bounded to serial CPU binary32 storage with binary64 internal reductions on
  the reviewed x86-64 FP lane for `N in 1..2`, `T in 1..32`, `D in 2..32`,
  `F in 2..64`, `E in 2..8`, and `top_k in 1..4` (`top_k <= E`).
- Analytic backward assumes perturbations do not cross discrete top-K tie
  boundaries (where `f_e` and expert selection have zero subgradient / jump
  discontinuities).
- GPU execution, expert capacity-factor token dropping, mixed precision,
  arbitrary shapes, and throughput claims are outside this scope.
