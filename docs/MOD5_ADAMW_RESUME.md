# MOD5 AdamW Optimizer, LR Scheduling, Gradient Clipping, and Fresh-Process Resume

This document specifies the MOD5 CPU-f32 AdamW optimizer, global L2 gradient-norm
clipping, deterministic learning-rate schedules, and versioned SHA-256-checksummed
binary training-state snapshot (`ETM5TRN1` v1.0) verified in compiled Eshkol AOT.

## 1. Delivered Surface

- Public C11/C++17 header: [include/eshkol_transformer/mod5_adamw_abi.h](../include/eshkol_transformer/mod5_adamw_abi.h)
- Native implementation: [native/mod5_adamw_state.c](../native/mod5_adamw_state.c)
- Build and test gates: [scripts/build-mod5-adamw.sh](../scripts/build-mod5-adamw.sh), [scripts/test-mod5-adamw.sh](../scripts/test-mod5-adamw.sh)
- Native, C++17, Python oracle, and compiled Eshkol AOT tests:
  - [tests/mod5/test_mod5_adamw_native.c](../tests/mod5/test_mod5_adamw_native.c)
  - [tests/mod5/header_adamw_cpp.cpp](../tests/mod5/header_adamw_cpp.cpp)
  - [tests/mod5/test_mod5_adamw_oracle.py](../tests/mod5/test_mod5_adamw_oracle.py)
  - [tests/mod5/test_mod5_adamw_resume_aot.esk](../tests/mod5/test_mod5_adamw_resume_aot.esk)

The archive `build/mod5_adamw/libeshkol_transformer_mod5_adamw.a` is compiled with
`-std=c11 -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math -fvisibility=hidden`
and audited against an exact exported-symbol list (`et_mod5_adamw_*`) and an explicit
undefined-symbol allowlist (no threading, foreign runtimes, `dlopen`, or `exec`).

## 2. Numerical Contract

All numerical operations require default IEEE-754 round-to-nearest-even (`FE_TONEAREST`)
and on x86-64 require MXCSR round-to-nearest-even with FTZ and DAZ disabled; any other
FP environment returns `ET_MOD5_ADAMW_ERR_DETERMINISM` (`-10`) before mutating state.
No silent precision, device, or fallback paths exist.

### 2.1 Global L2 Gradient-Norm Clipping

Given parameter count $N$ and finite binary32 gradients $g_0, \dots, g_{N-1}$, the
global L2 norm is accumulated in IEEE-754 binary64:

$$\|g\|_2 = \sqrt{\sum_{i=0}^{N-1} (\mathrm{f64}(g_i))^2}$$

If `max_grad_norm > 0` and $\|g\|_2 > \mathrm{f64}(\text{max\_grad\_norm})$, then
$\text{clip\_scale} = \mathrm{f64}(\text{max\_grad\_norm}) / \|g\|_2$ and each effective
gradient is $g'_i = \mathrm{f32}(\mathrm{f64}(g_i) \cdot \text{clip\_scale})$.
At or below the boundary ($\|g\|_2 \le \text{max\_grad\_norm}$), gradients are unchanged
($\text{clip\_scale} = 1.0$, `clipped = 0`).

### 2.2 Learning-Rate Schedules

For 1-based update step $t = \text{step} + 1 \ge 1$ with finite `0 <= min_lr <= base_lr`:

- `ET_MOD5_LR_CONSTANT` (`0`): $\eta(t) = \text{base\_lr}$.
- `ET_MOD5_LR_LINEAR_WARMUP_LINEAR_DECAY` (`1`) and `ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY` (`2`)
  (requiring `total_steps > 0` and `warmup_steps <= total_steps`):
  - Warmup ($W > 0$ and $t \le W$): $\eta(t) = \text{base\_lr} \cdot (t / W)$.
  - Clamped tail ($t \ge T$): $\eta(t) = \text{min\_lr}$.
  - Decay ($W < t < T$), with $p = (t - W) / (T - W) \in (0, 1)$:
    - Linear decay: $\eta(t) = \text{min\_lr} + (\text{base\_lr} - \text{min\_lr}) \cdot (1 - p)$.
    - Cosine decay: $\eta(t) = \text{min\_lr} + (\text{base\_lr} - \text{min\_lr}) \cdot \tfrac{1}{2}(1 + \cos(\pi p))$.

### 2.3 CPU-f32 AdamW Update and Atomicity

For update index $t = \text{step} + 1$, exact bias corrections are evaluated in binary64
using exponentiation by squaring (`ipow_f64`) and rounded once to binary32:

$$\hat{b}_1 = \mathrm{f32}(1 - \beta_1^t), \qquad \hat{b}_2 = \mathrm{f32}(1 - \beta_2^t), \qquad \eta_{32} = \mathrm{f32}(\eta(t))$$

With decoupled weight decay $\lambda$ and $\text{decay} = \mathrm{f32}(1 - \eta_{32} \cdot \lambda)$,
each element $i \in [0, N)$ computes in binary32 (contraction disabled):

$$m_i^{(t)} = \beta_1 m_i^{(t-1)} + (1 - \beta_1) g'_i$$
$$v_i^{(t)} = \beta_2 v_i^{(t-1)} + (1 - \beta_2) (g'_i)^2$$
$$\hat{m}_i = m_i^{(t)} / \hat{b}_1, \qquad \hat{v}_i = v_i^{(t)} / \hat{b}_2$$
$$p_i^{(t)} = p_i^{(t-1)} \cdot \text{decay} - \eta_{32} \cdot \frac{\hat{m}_i}{\sqrt{\hat{v}_i} + \epsilon}$$

`et_mod5_adamw_step` executes three zero-allocation passes: (1) validate all live
parameters, moments, and gradients are finite and accumulate $\|g\|_2$ in binary64,
(2) preflight every element's $m_i^{(t)}, v_i^{(t)}, p_i^{(t)}$ for finiteness and
$v_i^{(t)} \ge 0$, and (3) commit in place and increment `step` and `cumulative_tokens`.
Any non-finite input, intermediate, or output leaves live trainer state bitwise unchanged.

## 3. Binary Training-State Snapshot Format (`ETM5TRN1` v1.0)

Snapshots are non-executable (`0600`), versioned, little-endian, and SHA-256 checksummed
over all preceding bytes. No executable code or native pointers are stored or loaded.

| Byte Offset | Size (bytes) | Field | Notes |
|---:|---:|---|---|
| `0` | 8 | `magic` | Exact ASCII `"ETM5TRN1"` |
| `8` | 2 | `version_major` | `u16` LE (`1`) |
| `10` | 2 | `version_minor` | `u16` LE (`0`) |
| `12` | 4 | `flags` | `u32` LE (must be `0`) |
| `16` | 8 | `param_count` | `u64` LE (`1 .. 16,777,216`) |
| `24` | 8 | `step` | `u64` LE completed AdamW steps |
| `32` | 8 | `cumulative_tokens` | `u64` LE total processed tokens |
| `40` | 8 | `cursor.shard_index` | `u64` LE D1 shard index |
| `48` | 8 | `cursor.token_offset` | `u64` LE token offset within shard |
| `56` | 8 | `cursor.epoch` | `u64` LE dataset epoch |
| `64` | 32 | `rng_state[4]` | `4 x u64` LE non-zero xoshiro256** state |
| `96` | 20 | `beta1, beta2, eps, weight_decay, max_grad_norm` | `5 x f32` LE |
| `116` | 4 | `schedule.kind` | `u32` LE (`0`, `1`, or `2`) |
| `120` | 16 | `schedule.base_lr, schedule.min_lr` | `2 x f64` LE |
| `136` | 16 | `schedule.warmup_steps, schedule.total_steps` | `2 x u64` LE |
| `152` | 32 | `model_config_hash` | SHA-256 of canonical model configuration |
| `184` | 32 | `full_config_hash` | SHA-256 over `"ETM5CFG1"` + model hash + optimizer + schedule + tokenizer |
| `216` | 4 | `tokenizer_id_len` | `u32` LE (`L`, `1 .. 128`) |
| `220` | 4 | `reserved` | `u32` LE (must be `0`) |
| `224` | `L` | `tokenizer_id` | UTF-8 bytes (no embedded NUL) |
| `224 + L` | `4 * N` | `params[N]` | Finite binary32 parameter weights |
| `224 + L + 4N` | `4 * N` | `m[N]` | Finite binary32 AdamW first moments |
| `224 + L + 8N` | `4 * N` | `v[N]` | Finite nonnegative binary32 AdamW second moments |
| `224 + L + 12N` | 32 | `sha256` | SHA-256 over bytes `[0 .. 224 + L + 12N)` |

- **Atomic write**: refuses existing symlink or non-regular destinations via
  `fstatat(..., AT_SYMLINK_NOFOLLOW)`, writes a `0600` random temporary file in the parent
  directory (`O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC`), calls `fsync` on the file,
  atomically replaces via `renameat`, and calls `fsync` on the parent directory.
- **Checked read**: opens parent and target with `O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK`,
  verifies `S_ISREG`, exact byte size, unchanged `(st_dev, st_ino, st_size)`, magic,
  version `1.0`, constant-time SHA-256 trailer, `full_config_hash` and complete config
  equality against the live trainer, and finite-only `params`/`m`/`v` (`v >= 0`) into
  staged buffers before committing atomically to live state.

## 4. Measured Validation Results

Validated in the pinned Ubuntu 22.04 container (`eshkol-r0-debug:20260928`,
`clang-21` / `clang++-21` 21.1.8, Eshkol `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`,
`--network none`):

### 4.1 Independent Python Oracle Tolerances

| Check | Scope | Measured Result |
|---|---|---|
| LR schedules (`CONSTANT`, `LINEAR`, `COSINE`) | 20 steps x 3 schedules | `max_abs_err = 0.000e+00` (exact) |
| Global L2 grad norm | 16 steps (4 clipped, 12 unclipped) | `max_abs_err = 0.000e+00` (exact) |
| AdamW vs exact f32 Python contract | 16 steps x 16 params (`p`, `m`, `v`) | `0 ULP` (bitwise identical) |
| AdamW vs binary64 mathematical oracle | 16 steps x 16 params | `max |dp| = 2.413e-07`, `max |dm| = 1.603e-08`, `max |dv| = 1.863e-09` |

### 4.2 Compiled Eshkol AOT 16-Step Training Trace (MOD5 1-Block GQA Decoder)

Model: 12 parameter tensors (2,864 f32 parameters), 4Q/2KV GQA attention, RMSNorm,
SwiGLU dispatched through MOD5 K1 kernels; 2 SHA-256-verified D1 v1.0 shards (27 tokens
each, 3 windows/shard, vocab 16, tokenizer `mod5-adamw-token-v1`); AdamW with
`beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.01`, `max_grad_norm=0.5`,
cosine schedule `base_lr=0.05`, `min_lr=0.005`, `warmup_steps=4`, `total_steps=16`.
Step-1 central-difference numerical gradient check passed on all 12 parameter tensors.

| Step | Mean CE Loss | LR | Global Grad Norm | Clipped | Cursor (`shard:off:epoch`) | Tokens |
|---:|---:|---:|---:|:---:|:---:|---:|
| 1 | `2.7694186496899564` | `0.012500000000000001` | `0.50857929787481437` | 1 | `0:9:0` | 8 |
| 2 | `2.6578562817218079` | `0.025000000000000001` | `0.52353094454950477` | 1 | `0:18:0` | 16 |
| 3 | `2.3998353918853339` | `0.037500000000000006` | `0.68548685799803011` | 1 | `1:0:0` | 24 |
| 4 | `1.9783986571832175` | `0.050000000000000003` | `1.412204465333329` | 1 | `1:9:0` | 32 |
| 5 | `1.5308405659807014` | `0.049233331091504039` | `1.5246992912204336` | 1 | `1:18:0` | 40 |
| 6 | `1.0880862578627464` | `0.046985571585149877` | `2.1790038318087084` | 1 | `0:0:1` | 48 |
| 7 | `1.0623072879426809` | `0.043409902576697322` | `1.1410035839723873` | 1 | `0:9:1` | 56 |
| 8 | `0.93056263446908294` | `0.03875` | `1.0246402627031581` | 1 | `0:18:1` | 64 |
| 9 | `0.74060957337257283` | `0.03332342851480672` | `1.1426663232374126` | 1 | `1:0:1` | 72 |
| 10 | `0.67916157940751631` | `0.027500000000000004` | `1.4735817640064892` | 1 | `1:9:1` | 80 |
| 11 | `0.63163320269508505` | `0.021676571485193284` | `1.2406711808677748` | 1 | `1:18:1` | 88 |
| 12 | `0.56229729734667844` | `0.016250000000000007` | `0.52568382857396301` | 1 | `0:0:2` | 96 |
| 13 | `0.51009999215666846` | `0.011590097423302682` | `0.41749560801955976` | 0 | `0:9:2` | 104 |
| 14 | `0.47187111266745929` | `0.0080144284148501305` | `0.36900825217046701` | 0 | `0:18:2` | 112 |
| 15 | `0.44576378205375222` | `0.0057666689084959659` | `0.39057736570348262` | 0 | `1:0:2` | 120 |
| 16 | `0.42371086992396334` | `0.0050000000000000001` | `0.39556392060135398` | 0 | `1:9:2` | 128 |
| Eval | `0.40158723838804949` | — | — | — | `1:9:2` | 128 |

Loss decreased strictly on every step from `2.7694186496899564` (`0x1.627c4f70ace87p+1`)
to `0.40158723838804949` (`0x1.9b39af5d78739p-2`, 85.5% reduction).

### 4.3 Exact Bitwise Fresh-Process Resume & Artifact Digests

Process 1 trained steps 1–8 and exited after saving the midpoint snapshot. Process 2
(a fresh OS process) restored the midpoint snapshot, verified rejection of both a 1-bit
corrupt snapshot (`-5`) and a config-mismatch snapshot (`-7`) with live state SHA-256
unchanged, then trained steps 9–16. Step 9–16 traces (`loss_hex`, `lr_hex`, `grad_norm`,
`clipped`, `cursor`, `tokens`, `rng`), final evaluation loss, final live state SHA-256,
and final snapshot bytes were **bitwise identical** between uninterrupted 16-step and
8+8 fresh-process resumed runs:

| Artifact | SHA-256 |
|---|---|
| Both fresh AOT ELF builds (`mod5-adamw-aot`) | `af2fff3aa11f03c85e3afe3f7bb1e1d3d4fe663d00945c63e3c6381b3b89672a` |
| Step-8 midpoint snapshot (`mid.etsnap`) | `236c832ee7e3e3918388f26bf65dd4f3be71cd5026fecfacbded9b391a8d5830` |
| Step-8 live state digest | `e50b37745141dde014f5871ae7bdced824ce0eb5b06fee2636bb0afa6a3b4951` |
| Uninterrupted step-16 snapshot (`full.etsnap`) | `a12c6db64164a4813a4393029f5d4b5cad4825d41002134220e0288340d57547` |
| Fresh-process resumed step-16 snapshot (`resumed.etsnap`) | `a12c6db64164a4813a4393029f5d4b5cad4825d41002134220e0288340d57547` |
| Final live state digest (both runs) | `a99d0ebff8114ec700f45971015acd435eedd36728d2645ac681bf12b7b1551f` |

## 5. Measured Limitations

- CPU-f32 serial execution on Linux x86-64 only; GPU, distributed reduction, and mixed
  precision master/narrow weights are outside this module's scope.
- Single global hyperparameter group per `et_mod5_adamw_state`; per-group decay exclusions
  require separate parameter groups or an extension to format version 2.
- Cosine schedule calls glibc `cos` on the pinned Ubuntu 22.04 x86-64 toolchain; cross-OS
  or cross-libm bit-level equivalence of transcendental `cos` is not claimed.
- Maximum snapshot parameter count is bounded to `16,777,216` f32 elements (~201 MB).
