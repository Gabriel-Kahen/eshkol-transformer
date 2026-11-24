# 2D Hybrid Parallelism (`DP × TP` Device Mesh, Orthogonal `AF_UNIX` Subgroups)

This document specifies the standalone Eshkol-native **2D Hybrid Parallelism (`DP × TP`)** training engine and compiled Eshkol AOT driver implemented in:

- Public C11/C++17 ABI: `include/eshkol_transformer/hybrid_mesh_abi.h`
- Native C11 core: `native/hybrid_mesh.c`
- Eshkol AOT bridge: `native/hybrid_mesh_eshkol_bridge.h`, `native/hybrid_mesh_eshkol_bridge.c`
- Compiled Eshkol AOT driver: `tests/hybrid_mesh/test_hybrid_mesh_aot.esk`
- Verification & oracle suite: `tests/hybrid_mesh/header_mesh_cpp.cpp`, `tests/hybrid_mesh/test_hybrid_mesh_native.c`, `tests/hybrid_mesh/test_hybrid_mesh_oracle.py`, `scripts/build-hybrid-mesh.sh`, `scripts/test-hybrid-mesh.sh`

## 1. 2D Device Mesh Topology & Orthogonal `AF_UNIX` Subgroups

Supported factorizations satisfy `dp_size, tp_size ∈ {1, 2, 4}` and `world_size = dp_size * tp_size <= 4`:

- `(dp=1, tp=1)` — single-process dense baseline (`world=1`)
- `(dp=2, tp=1)` — 2-way Data Parallel (`world=2`)
- `(dp=1, tp=2)` — 2-way Tensor Parallel (`world=2`)
- **`(dp=2, tp=2)` — true 4-process 2D Hybrid Mesh (`world=4`)**
- `(dp=4, tp=1)` — 4-way Data Parallel (`world=4`)
- `(dp=1, tp=4)` — 4-way Tensor Parallel (`world=4`)

Deterministic coordinate mapping for `global_rank ∈ [0, world_size)`:

```text
dp_rank = global_rank / tp_size
tp_rank = global_rank % tp_size
global_rank = dp_rank * tp_size + tp_rank
```

`et_mesh_open_unix` establishes two orthogonal `AF_UNIX` `SOCK_STREAM` star-topology subgroup communicators per rank under one shared `CLOCK_MONOTONIC` deadline:

1. **`tp_group` (size `tp_size`)**: connects ranks sharing `dp_rank` via `<sock_dir>/mesh_tp_dp<dp_rank>.sock`. For `(2×2)`, the orthogonal TP groups are `{0, 1}` (`dp_rank=0`) and `{2, 3}` (`dp_rank=1`).
2. **`dp_group` (size `dp_size`)**: connects ranks sharing `tp_rank` via `<sock_dir>/mesh_dp_tp<tp_rank>.sock`. For `(2×2)`, the orthogonal DP groups are `{0, 2}` (`tp_rank=0`) and `{1, 3}` (`tp_rank=1`).

Unit-size axes (`dp_size == 1` or `tp_size == 1`) open zero sockets and emit zero IPC bytes. All reductions accumulate in canonical ascending subgroup-rank order (`0 .. sub_size - 1`) in IEEE-754 `binary64`.

## 2. Model Architecture, Sharding Contract & Communication Table

Model: **RMSNorm -> TP-Sharded SwiGLU FFN (+ Residual) -> Replicated Linear Head -> Cross-Entropy Loss**

Constraints: `d_ff % tp_size == 0`, `global_batch % dp_size == 0`, `d_ff_local = d_ff / tp_size`, `local_batch = global_batch / dp_size`.

### Weight Sharding Contract

| Parameter | Full Shape | Per-Rank Shape | Sharding |
| :--- | :--- | :--- | :--- |
| `gamma` | `[d_model]` | `[d_model]` | Replicated across all ranks |
| `b_down` | `[d_model]` | `[d_model]` | Replicated; added **once** post-TP-AllReduce |
| `W_head` | `[num_classes, d_model]` | `[num_classes, d_model]` | Replicated across all ranks |
| `b_head` | `[num_classes]` | `[num_classes]` | Replicated across all ranks |
| `W_gate` | `[d_ff, d_model]` | `[d_ff / tp_size, d_model]` | Column-Parallel (`1 / tp_size`) |
| `W_up` | `[d_ff, d_model]` | `[d_ff / tp_size, d_model]` | Column-Parallel (`1 / tp_size`) |
| `W_down` | `[d_model, d_ff]` | `[d_model, d_ff / tp_size]` | Row-Parallel (`1 / tp_size`) |

### Per-Step Communication Table

| Phase | `tp_group` Collective | `dp_group` Collective | Notes |
| :--- | :--- | :--- | :--- |
| Forward | `1 × TP AllReduceSum_f32` (`Z [B_loc, D]`) | **0 (zero DP comm)** | `b_down` added once after TP sum |
| Backward | `1 × TP AllReduceSum_f32` (`dU [B_loc, D]`) | **0 (zero DP comm)** | Analytic local grads in `binary64` |
| Non-Finite Consensus | `1 × ConsensusOr_u64` (8 B) | `1 × ConsensusOr_u64` (8 B) | Propagates any single-rank NaN/Inf to all `dp*tp` ranks |
| Gradient & Loss Sync | `0` | `1 × DP AllReduceAvg_f32` (all local grads) + `1 × DP AllReduceAvg_f64` (loss) | Bitwise-identical grads across matching `tp_rank` DP replicas |
| Global L2 Norm | `1 × TP AllReduceSum_f64` (1 scalar) | `0` | Sums sharded sq-norms across `tp_group`, adds replicated sq-norm once |
| AdamW Update | `2 × ConsensusOr_u64` (pre/post-commit guard) | `2 × ConsensusOr_u64` | Atomic decoupled AdamW; zero split-brain |
| Final Weight Gather | `1 × TP AllGather_f32` (on demand) | `0` | Reconstructs full dense weights |

Over 10 training steps (+ 1 final gather), active axes observe exact deterministic counts:
- `tp_allreduce_calls = 30` when `tp_size > 1` (`0` when `tp_size == 1`)
- `dp_allreduce_calls = 20` when `dp_size > 1` (`0` when `dp_size == 1`)
- `tp_allgather_calls = 1` on final gather when `tp_size > 1` (`0` when `tp_size == 1`)

## 3. Versioned Checkpoints (`ET2D1000` v1.0)

Each rank persists `(dp_size, tp_size, dp_rank, tp_rank, step, model_config, optim_config, local_params, m, v)` in a non-executable little-endian binary format:

- `160`-byte fixed header: magic `"ET2D1000"`, version `1.0`, 2D mesh coordinates, model & optimizer config bits, `step`, `local_param_count`, `payload_bytes`, and `32`-byte FIPS 180-4 SHA-256 over `header[0..128) || payload`.
- Atomic save: refuses existing symlinks/non-regular targets, writes a `0600` sibling temporary file (`O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC`), `fsync`s file, `rename`s atomically, and `fsync`s parent directory.
- Safe load: opens with `O_RDONLY | O_NOFOLLOW | O_CLOEXEC`, verifies regular file, exact byte length, magic, version `1.0`, SHA-256, exact 2D mesh coordinates (`ET_MESH_ERR_MESH_MISMATCH`), exact config (`ET_MESH_ERR_CONFIG_MISMATCH`), finiteness, and `v >= 0.0f` before atomically committing live state.

## 4. Pinned-Container Verification Evidence (`eshkol-r0-debug:20260928`)

Workload: `d_model=8`, `d_ff=16`, `num_classes=6`, `global_batch=8`, `rms_eps=1e-5`, `seed=20261124`, `lr=0.05`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.01`, `max_grad_norm=0.5`, `steps=10`.
Parameters: `454` total (`70` replicated = `280 B`, `384` full sharded = `1536 B`).

### Deterministic Artifact SHA-256 Hashes

| Artifact | SHA-256 | Determinism Gate |
| :--- | :--- | :--- |
| `build/hybrid_mesh/libeshkol_transformer_hybrid_mesh.a` | `58f634dc2027d61097c090c8da74dce67c6f2dec25e609e449211356cdb4cc58` | Dual fresh builds byte-identical (`cmp`) + symbol audit |
| `build/hybrid_mesh/libeshkol_transformer_hybrid_mesh_aot.a` | `20f6790e7e45ac5911054001f6e23aec0f69e3cfd41790c7c14054b21022e0bc` | Dual fresh builds byte-identical (`cmp`) + symbol audit |
| Compiled Eshkol AOT ELF (`test_hybrid_mesh_aot.esk`) | `a3f2516e25ce6dd2713fb2dd725f04a52af29fcbbbe3e9b737a6d2e28513f4b4` | Dual fresh `--strict-types --optimize 0 --no-stdlib` ELFs byte-identical (`cmp`) |

### Measured Numerical & Multi-Process Results

- **Central Finite-Difference Gradient Check** (`h = 1/512`):
  - All `454` parameters: `max_param_err = 1.393e-07` (`< 1e-5` gate)
  - All `64` input elements (`dX`): `max_dx_err = 3.398e-08` (`< 1e-5` gate)
- **Independent Python `binary64` Oracle**:
  - Step 0: `loss_err = 3.44e-11`, `grad_err = 6.43e-09`, `dx_err = 1.65e-09`
  - All 6 native topologies (`1×1, 2×1, 1×2, 2×2, 4×1, 1×4`): `max_loss_err = 1.70e-08`, `max_weight_err = 2.98e-07`
  - Compiled Eshkol AOT (`1×1, 2×1, 1×2, 2×2`): `max_vs_1x1 = 1.49e-07`, `max_vs_f64_ref = 2.98e-07`
- **10-Step AdamW Training (Compiled Eshkol AOT)**:

| Mesh `(dp×tp)` | Init Loss | Final Loss | Sharded Bytes / Rank | `tp_ar` | `dp_ar` | `tp_ag` | `tp_bytes` | `dp_bytes` | Final Gathered `weights_sha256` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `(1×1)` | `1.720022` | `0.000228` | `1536` | `0` | `0` | `0` | `0` | `0` | `8c45b0a23ea9f2ea1e77f930c419df275a7594ec3dbee55b0b5b9e532a8a0e26` |
| `(2×1)` | `1.720022` | `0.000228` | `1536` | `0` | `20` | `0` | `0` | `18960` | `1795afb69853af0286822b2f52cdf3b15d2599b7ad10e2d48a60cbd7380b4bfe` |
| `(1×2)` | `1.720022` | `0.000228` | `768` | `30` | `0` | `1` | `5920` | `0` | `eb7bf8b9587f804b121aa3b6be679b523021e168efc4922af454db386b126722` |
| `(2×2)` | `1.720022` | `0.000228` | `768` | `30` | `20` | `1` | `3360` | `11280` | `8eb4a50066f79ceed9e0a36eddf4c935342b5ab052f0ae3af6a61a82692e4d9b` |

- **Bitwise DP Replica Identity (`2×2`)**: `rank 0` and `rank 2` share identical `local_sha256` (`cc89f507b46261085f162824597f17bd28fce37100ebebe9d7e31ad0a953f042`); `rank 1` and `rank 3` share identical `local_sha256` (distinct from `tp_rank=0`).
- **5+5 Fresh-Process `ET2D1000` Checkpoint Resume (`2×2`)**: bitwise-identical `weights_sha256` (`8eb4a50066f79ceed9e0a36eddf4c935342b5ab052f0ae3af6a61a82692e4d9b`) and `local_sha256` across all 4 ranks.
- **Single-Rank Non-Finite Atomic Abort (`2×2`)**: injecting `qNaN` (and `+Inf` / forward `x` `qNaN`) on rank 3 aborts all 4 ranks with `ET_MESH_ERR_NONFINITE (-7)`, preserves all 4 shard digests bitwise, and recovers cleanly on the next step.
- **Sanitizers & Resources**: ASan/UBSan/LSan (`detect_leaks=1:halt_on_error=1`) PASS; open FD count `4 -> 4` (zero FD leaks).

## 5. Measured Limitations & Unsupported Cases

- **CPU-only `AF_UNIX` transport**: single-host multi-process IPC over `AF_UNIX` `SOCK_STREAM` (no GPU/NCCL/RCCL or TCP/RDMA transport; no silent GPU claim).
- **World size cap**: `dp_size, tp_size ∈ {1, 2, 4}` with `dp_size * tp_size <= 4`; other mesh factorizations fail explicitly with `ET_MESH_ERR_INVALID_ARG`.
- **Divisibility constraints**: `d_ff % tp_size != 0` or `global_batch % dp_size != 0` fails explicitly with `ET_MESH_ERR_DIVISIBILITY`.
- **Single-block FFN + Head scope**: covers RMSNorm + TP-SwiGLU FFN + Residual + Replicated Linear Head; multi-layer attention + pipeline parallelism (`3D` mesh) is out of scope for this standalone module.
- **Buffer cap**: collective operands are capped at `ET_MESH_MAX_BUFFER_BYTES` (`64 MiB`).
