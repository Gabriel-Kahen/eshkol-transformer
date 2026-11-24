# FSDP / ZeRO-1, ZeRO-2, and ZeRO-3 Sharded Training Engine

Standalone Eshkol-native C11/C++17 sharded optimizer, gradient, and parameter
training engine across `W in {1, 2, 4}` real multi-process OS ranks over
`AF_UNIX` `SOCK_STREAM` collectives, plus a compiled Eshkol AOT driver.

- Public C11/C++17 ABI: `include/eshkol_transformer/fsdp_zero_abi.h`
- Core implementation: `native/fsdp_zero.c` -> deterministic archive
  `build/fsdp_zero/libeshkol_transformer_fsdp_zero.a`
- Eshkol AOT bridge: `native/fsdp_zero_eshkol_bridge.{h,c}` -> deterministic
  archive `build/fsdp_zero/libeshkol_transformer_fsdp_zero_aot.a`
- Compiled Eshkol AOT driver: `tests/fsdp_zero/test_fsdp_zero_aot.esk`
- Build & test scripts: `scripts/build-fsdp-zero.sh`, `scripts/test-fsdp-zero.sh`

## Sharding Stages and Persistent Memory Contract

For `P = num_params` binary32 parameters and world size `W in {1, 2, 4}`, flat
parameters are deterministically padded to:

- `shard_len = ceil(P / W)` (`S`)
- `padded_len = shard_len * W` (`P_pad`)

Trailing pad slots `[P .. P_pad)` are strictly zeroed on load, zeroed on
gradients prior to reduction, skipped during decoupled AdamW updates, and
verified `0.0f` on checkpoint load so padding never affects global gradient
norms, moments, or parameter updates.

Each stage allocates **only** the persistent buffers mandated by its sharding
level (Stage 3 literally holds `params_full == NULL` and `grads_full == NULL`):

| Stage | Persistent Params / Rank | Persistent Grads / Rank | Persistent Optimizer (`m`, `v`) / Rank | Total Persistent / Rank | Optimizer Reduction | Total Persistent Reduction |
|---|---:|---:|---:|---:|---:|---:|
| Unsharded baseline (`W=1` padded) | `4 * P_pad` B | `4 * P_pad` B | `8 * P_pad` B | `16 * P_pad` B | `1.0x` | `1.0x` |
| `ET_ZERO_STAGE_1` (ZeRO-1) | `4 * P_pad` B | `4 * P_pad` B | `8 * S` B | `(8 * P_pad + 8 * S)` B | **`W`x (`1/W`)** | `2W / (W + 1)` |
| `ET_ZERO_STAGE_2` (ZeRO-2) | `4 * P_pad` B | `4 * S` B | `8 * S` B | `(4 * P_pad + 12 * S)` B | **`W`x (`1/W`)** | `4W / (W + 3)` |
| `ET_ZERO_STAGE_3_FSDP` (ZeRO-3 / FSDP) | `4 * S` B | `4 * S` B | `8 * S` B | `16 * S` B | **`W`x (`1/W`)** | **`W`x (`1/W`)** |

### Concrete Measured Persistent Memory (`P = 243`, 2-layer tanh MLP `8 -> 17 -> 5`)

| Configuration | `P_pad` | `S` (`shard_len`) | Param Bytes | Grad Bytes | Opt (`m+v`) Bytes | Total Persistent Bytes | Unsharded Baseline | Opt Ratio | Total Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 1, `W=1` | 243 | 243 | 972 B | 972 B | 1,944 B | 3,888 B | 3,888 B | 1.0x | 1.0x |
| Stage 1, `W=2` | 244 | 122 | 976 B | 976 B | 976 B | 2,928 B | 3,904 B | 2.0x | 1.33x |
| Stage 1, `W=4` | 244 | 61 | 976 B | 976 B | 488 B | 2,440 B | 3,904 B | 4.0x | 1.60x |
| Stage 2, `W=1` | 243 | 243 | 972 B | 972 B | 1,944 B | 3,888 B | 3,888 B | 1.0x | 1.0x |
| Stage 2, `W=2` | 244 | 122 | 976 B | 488 B | 976 B | 2,440 B | 3,904 B | 2.0x | 1.60x |
| Stage 2, `W=4` | 244 | 61 | 976 B | 244 B | 488 B | 1,708 B | 3,904 B | 4.0x | 2.29x |
| Stage 3 FSDP, `W=1` | 243 | 243 | 972 B | 972 B | 1,944 B | 3,888 B | 3,888 B | 1.0x | 1.0x |
| Stage 3 FSDP, `W=2` | 244 | 122 | 488 B | 488 B | 976 B | 1,952 B | 3,904 B | 2.0x | **2.0x** |
| Stage 3 FSDP, `W=4` | 244 | 61 | 244 B | 244 B | 488 B | 976 B | 3,904 B | 4.0x | **4.0x** |

## Real Multi-Process `AF_UNIX` Collectives and Communication Volume

Collectives operate over pre-connected `AF_UNIX` `SOCK_STREAM` socket pairs
(`et_fsdp_comm_create_fds`) or a `<sock_dir>/fsdp_r0.sock` rendezvous handshake
(`et_fsdp_comm_open_unix`) with zero shared memory and zero filesystem sockets
left behind after rendezvous.

- **`et_fsdp_allgather_f32`**: gathers `W` shards `[S]` in canonical ascending
  rank order `0..W-1` into `[P_pad]`.
- **`et_fsdp_reduce_scatter_avg_f32`**: preflights all inputs for finiteness,
  reduces in canonical ascending rank order `0..W-1` in IEEE-754 `binary64`,
  divides by `W`, and scatters only rank `r`'s owned slice `[S]`.
- **`et_fsdp_allreduce_avg_f32`**: ZeRO-1 full-gradient mean using the identical
  canonical ascending `binary64` sum divided by `W`.
- **`et_fsdp_allreduce_sum_f64`**: exact `binary64` sum across ranks for exact
  global L2 gradient norm clipping (`sum_r ||g_shard_r||_2^2`) and exact global
  mean loss.

Per-step communication volume per worker rank (star coordinator rank 0 exchanges
`(W - 1)` peer payloads per collective):

| Stage | Collectives per Training Step | Worker Payload Sent / Step | Worker Payload Recv / Step |
|---|---|---:|---:|
| ZeRO-1 | `AllReduceAvg(P_pad)` + `AllReduceSum64(1)` + `AllGather(S)` | `4 * P_pad + 4 * S + 8` B | `8 * P_pad + 8` B |
| ZeRO-2 | `ReduceScatterAvg(P_pad -> S)` + `AllReduceSum64(1)` + `AllGather(S)` | `4 * P_pad + 4 * S + 8` B | `4 * P_pad + 4 * S + 8` B |
| ZeRO-3 FSDP | `AllGather(S -> P_pad)` + `ReduceScatterAvg(P_pad -> S)` + `AllReduceSum64(1)` | `4 * P_pad + 4 * S + 8` B | `4 * P_pad + 4 * S + 8` B |

Measured rank-0 cumulative coordinator payload over 12 Stage-3 FSDP steps plus
final parameter gather (`P = 243`, `P_pad = 244`, including 12 `AllReduceSum64`
loss reductions):

- `W = 1`: `0 B` sent, `0 B` recv (zero sockets opened)
- `W = 2`: `18,736 B` sent, `18,248 B` recv
- `W = 4`: `47,424 B` sent, `45,228 B` recv

## Stage-3 FSDP Step Lifecycle & Sharded Checkpoint Format (`ETFS1000` v1.0)

1. `et_fsdp_engine_gather_params`: `AllGather(params_shard)` into an ephemeral
   full parameter buffer; after forward+backward, the ephemeral buffer is
   explicitly poisoned with quiet-NaN bit pattern `0x7fc0dead` and released.
2. `et_fsdp_engine_step`: zero-pads trailing slots to `0.0f`, runs
   `ReduceScatterAvg` into `grad_shard[S]`, computes exact global L2 norm via
   `AllReduceSum_f64`, applies global clipping (`max_grad_norm = 0.5`), and runs
   an atomic shard-local decoupled AdamW update on `(params_shard, m_shard, v_shard)`.
3. **Sharded Checkpoint (`ETFS1000` v1.0)**:
   - Fixed 128-byte little-endian header + per-rank binary32 payload + 32-byte
     SHA-256 checksum over header + payload.
   - Stage 3 persists only `params_shard + m_shard + v_shard` (`3 * S * 4`
     payload bytes per rank).
   - Atomic `0600` save via `openat(O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC)`
     + `fsync` + `renameat` + parent `fsync`.
   - Checked load via `open(O_RDONLY | O_NOFOLLOW | O_CLOEXEC)` + regular-file
     `fstat` + magic/version/length/SHA-256/config/finiteness/pad-zero checks;
     atomic on any rejection.

## Verification Evidence

Verified inside `eshkol-r0-debug:20260928` (`x86_64`, Ubuntu 22.04, `clang-21` /
`clang++-21` 21.1.8, pinned Eshkol toolchain) via `bash scripts/test-fsdp-zero.sh`:

- **Deterministic Archives & Symbol Audit**:
  - `build/fsdp_zero/libeshkol_transformer_fsdp_zero.a`:
    `8151276ce4b78f6987830987f89163662a4a4da1121d894a8bdc7514a784854c`
  - `build/fsdp_zero/libeshkol_transformer_fsdp_zero_aot.a`:
    `cd4291db9763924cc83ef6d2c3c1213b634eff0af571f2c48f328cdb55d33148`
- **Dual Fresh Eshkol AOT Compilations (`--strict-types --optimize 0 --no-stdlib`)**:
  - Byte-identical ELF SHA-256:
    `f75ca17fa36f7552e42fc876bad51223bcd39e23e4b89e59f1c634d762eeea2b`
- **C++17 Header & Linkage + Normal `-O2` + ASan/UBSan/LSan (`detect_leaks=1:halt_on_error=1`)**: PASS
- **Central Finite-Difference Gradient Check (all 243 MLP parameters)**:
  `max |analytical - fd| = 1.773e-08` (threshold `< 1e-4`).

### 12-Step AdamW Training Parity (`lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, max_grad_norm=0.5`)

Global L2 clipping actively fired on step 2 (`grad_norm = 0.50617034 > 0.5`).
Loss decreased strictly on every step from `1.58441094` (step 0) to
`0.01455209` (step 11). For each fixed `W`, Stages 1, 2, and 3 matched
**bitwise**.

| World Size (`W`) | Stage 3 Persistent Total | Max Weight Delta vs `W=1` | Max Loss Delta vs `W=1` | Max Weight Delta vs Python `binary64` Oracle | 6+6 Fresh-Process Resume | Final Weights SHA-256 (972 B) |
|---|---:|---:|---:|---:|---|---|
| `W = 1` | 3,888 B (`1.0x`) | `0.000e+00` | `0.000e+00` | `2.838e-07` | Bitwise exact | `205a79d87af1efb254cd12f5bc88ec7bf2c349fe8e6af1eecf81127fabe7306d` |
| `W = 2` | 1,952 B (`2.0x`) | `1.043e-07` | `3.405e-09` | `2.838e-07` | Bitwise exact | `ec48ecb4ff166b6c6009248501ffd70b4265abf8d658d517071b4007c961a0e8` |
| `W = 4` | 976 B (`4.0x`) | `1.490e-07` | `5.314e-09` | `1.348e-07` | Bitwise exact | `84a2413481e80f35b83c505ff85c5166a3ccb97cb5b094cd24f49d5da4c07d51` |

### Fault, Non-Finite Abort, and Negative Verification

- **Atomic Non-Finite Abort (`W=4`, Stage 3 FSDP)**: injecting `qNaN`
  (`0x7fc00000`) or `+Inf` (`0x7f800000`) on a single rank returns
  `ET_FSDP_ERR_NONFINITE` (`-5`) on all 4 ranks, preserves every rank's
  `params_shard`, `m_shard`, `v_shard`, and `step` bitwise, and recovers
  cleanly on the next valid step.
- **Negative & Fault Tests (zero FD leaks via `/proc/self/fd`)**: bad magic,
  version mismatch, tampered payload checksum mismatch, truncated/trailing
  checkpoint bytes, symlink refusal on both load and save, stage/world/config
  mismatch, re-signed negative `v_shard`, sticky collective tag mismatch
  (`ET_FSDP_ERR_MISMATCH`), peer disconnect (`ET_FSDP_ERR_PEER_DISCONNECTED`),
  monotonic deadline timeout (`ET_FSDP_ERR_TIMEOUT`), and duplicate-rank
  handshake rejection.

## Measured Limitations and Unsupported Cases

- Supported world sizes are explicitly `W in {1, 2, 4}`; other values fail
  closed with `ET_FSDP_ERR_INVALID_ARG`.
- Transport is single-host Linux `AF_UNIX` `SOCK_STREAM` with a star
  coordinator on rank 0; multi-node networking, RDMA, NCCL/RCCL, GPU tensors,
  and per-layer asynchronous communication overlap are outside this module.
- Maximum parameter count is `ET_FSDP_MAX_PARAMS = 4,194,304` binary32 elements
  and maximum collective payload span is `64 MiB`.
- Because each rank quantizes its local microbatch mean gradient to `binary32`
  before `ReduceScatterAvg`, cross-`W` trajectories agree to `<= 1.49e-07`
  rather than bitwise identity (whereas Stages 1/2/3 at fixed `W` and 6+6
  fresh-process resume at fixed `W` are bitwise identical).
