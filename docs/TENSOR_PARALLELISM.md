# Megatron-Style 1D Tensor Parallelism Engine

Standalone, Eshkol-native 1D Tensor Parallelism (TP) engine supporting
`W in {1, 2, 4}` real multi-process OS ranks over `AF_UNIX` `SOCK_STREAM`
collectives, with exact analytic backward passes, C11 + C++17 ABI linkage, and
a compiled Eshkol AOT driver.

## Artifacts & Build

- Public C11/C++17 ABI: `include/eshkol_transformer/tensor_parallel_abi.h`
- Core native implementation: `native/tensor_parallel.c` -> deterministic static
  archive `build/tensor_parallel/libeshkol_transformer_tensor_parallel.a`
- Eshkol AOT bridge: `native/tensor_parallel_eshkol_bridge.{h,c}` ->
  deterministic static archive
  `build/tensor_parallel/libeshkol_transformer_tensor_parallel_aot.a`
- Build & symbol audit: `bash scripts/build-tensor-parallel.sh`
- Full verification (C++17 linkage, native `-O2`, Python binary64 oracle + f64
  FD, ASan/UBSan/LSan with `detect_leaks=1:halt_on_error=1`, dual fresh Eshkol
  AOT ELFs): `bash scripts/test-tensor-parallel.sh`

## Megatron-LM 1D Communication Topology

Let `f` denote the identity in forward and `AllReduce_SUM` in backward, and let
`g` denote `AllReduce_SUM` in forward and the identity in backward.

```text
Replicated Input X [B, T, D]
        |
   +----v----+
   | RMSNorm |  (replicated gamma1 [D], 0 comm)
   +----+----+
        |
        v  f (fwd: identity | bwd: 1 fused AllReduce_SUM on dX_local = dQ@Wq + dK@Wk + dV@Wv)
   +---------------------------------------------------------------+
   | Head-Sharded Causal GQA Attention (H_q % W == 0, H_kv % W == 0)
   |   Rank r owns:                                                |
   |     W_q_r [H_q/W * D_h, D], W_k_r, W_v_r [H_kv/W * D_h, D]    |
   |     Local Causal GQA SDPA (H_q/W query heads, H_kv/W KV heads)|
   |     RowParallel W_o_r [D, H_q/W * D_h]                        |
   +-------------------------------+-------------------------------+
        |
        v  g (fwd: 1 AllReduce_SUM on Y_partial_r | bwd: identity)
   +----v----+
   | Residual|  h1 = X + attn_out (replicated identically on all ranks)
   +----+----+
        |
   +----v----+
   | RMSNorm |  (replicated gamma2 [D], 0 comm)
   +----+----+
        |
        v  f (fwd: identity | bwd: 1 fused AllReduce_SUM on dX_local = dGate@Wgate + dUp@Wup)
   +---------------------------------------------------------------+
   | TP-Sharded SwiGLU MLP (d_ff % W == 0)                         |
   |   Rank r owns:                                                |
   |     ColumnParallel W_gate_r, W_up_r [d_ff/W, D]               |
   |     Local SiLU(gate_r) * up_r                                 |
   |     RowParallel W_down_r [D, d_ff/W]                          |
   +-------------------------------+-------------------------------+
        |
        v  g (fwd: 1 AllReduce_SUM on Y_partial_r | bwd: identity)
   +----v----+
   | Residual|  Y = h1 + mlp_out (replicated identically on all ranks)
   +---------+
```

## Collective Counts & Sharded Weight Memory

| Primitive / Block | Sharding Rule | Forward AllReduces | Backward AllReduces |
|---|---|:---:|:---:|
| `ColumnParallelLinear` | `out_features % W == 0` (row-sharded `W_r`) | `0` | `1` |
| `RowParallelLinear` | `in_features % W == 0` (col-sharded `W_r`, unsharded `b` added once post-AllReduce) | `1` | `0` |
| `TP-Sharded SwiGLU MLP` | `d_ff % W == 0` (`W_gate, W_up` col-parallel; `W_down` row-parallel) | `1` | `1` |
| `TP-Sharded Causal GQA` | `H_q % W == 0`, `H_kv % W == 0`, `H_q % H_kv == 0`, `D == H_q * D_h` | `1` | `1` |
| `Full TP Transformer Block` | RMSNorm + TP-GQA + Res + RMSNorm + TP-SwiGLU + Res | `2` | `2` |

### AF_UNIX Collective Transport (`et_tp_group_t`)

- Uses real connected `AF_UNIX` `SOCK_STREAM` socket descriptors duplicated with
  `F_DUPFD_CLOEXEC` (no shared memory, no threads).
- `et_tp_allreduce_sum_f32`: canonical ascending rank-order (`0 .. W-1`)
  binary64 (`double`) accumulation on rank 0 before narrowing once to binary32
  and broadcasting exact result bytes to all peers, guaranteeing bitwise
  identical outputs across all ranks regardless of OS scheduling order.
- `et_tp_allgather_f32`: ascending rank-order `[r=0 .. W-1]` concatenation into
  a disjoint destination buffer.
- Call-wide monotonic deadlines (`CLOCK_MONOTONIC`) return `ET_TP_ERR_TIMEOUT`
  on stalls; closed peer sockets return `ET_TP_ERR_PEER_DISCONNECTED`;
  non-finite operands trigger a coordinated `ET_TP_ERR_NONFINITE` abort leaving
  destination buffers unmodified and the group reusable. All failure paths close
  duplicated descriptors with zero FD leaks.

### Per-Rank Sharded Weight Memory (Block Geometry `B=2, T=4, D=32, F=32, H_q=8, H_kv=4, D_h=4`)

| World Size `W` | Sharded Weight Bytes / Rank | Ratio vs `W=1` | Replicated Gamma Bytes |
|:---:|---:|:---:|---:|
| `W = 1` | `24576` B | `1 / 1` | `256` B |
| `W = 2` | `12288` B | `1 / 2` | `256` B |
| `W = 4` | `6144` B | `1 / 4` | `256` B |

## Measured Verification Results (`eshkol-r0-debug:20260928`, `clang-21`)

- Deterministic static archives (`cmp` byte-identical across fresh builds):
  - `libeshkol_transformer_tensor_parallel.a`:
    `3869e220bf0a60599805241369be6a516260308aa52bd83fb8583408df14ff9b`
  - `libeshkol_transformer_tensor_parallel_aot.a`:
    `e1ffc662f27ab0743ba0b6621517318606d1e859e38bd0a219603947b158d9c4`
- Dual fresh Eshkol AOT compilations (`--strict-types --optimize 0 --no-stdlib`),
  byte-identical ELF and stdout:
  - `test_tensor_parallel_aot` ELF SHA-256:
    `8833a90f08d8632a97d17187be79a749a4d9cc16cef5e8e046929f714376eb13`
- Independent Python binary64 dense reference + f64 central FD oracle across
  ColumnParallelLinear, RowParallelLinear, TP-SwiGLU, TP-GQA, and Full TP
  Transformer Block (`W = 1, 2, 4`):
  - `max_ref_err = 5.261e-08` (`<= 1e-6`)
  - `max_w24_vs_w1_err = 5.960e-08` (`<= 1e-6`)
  - `max_f64_fd_err = 2.029e-10` (`<= 1e-7`)
- Compiled Eshkol AOT 10-step SGD training (`W = 1, 2, 4`):
  - Loss decreased strictly every step: `0.12772027242453057` ->
    `0.060476215104417606`
  - Max output parity error across all 10 steps: `1.192e-07` (`<= 1e-6`)
  - Max gathered gradient parity error (`W=2` and `W=4` vs `W=1`):
    `1.863e-09` (`<= 1e-6`)
  - Max AllGathered final weight parity error after 10 SGD steps:
    `W=2`: `2.980e-08`, `W=4`: `1.490e-08` (`<= 1e-6`)
  - Exact IPC collective counts (`W=2, 4`): `40` IPC AllReduces (`2` fwd + `2`
    bwd x `10` steps) and `21` IPC AllGathers (`7` sharded weights x 2 grad
    gathers + 1 final weight gather); `W=1` performs `0` IPC calls.
  - Eshkol AOT central FD gradient checks across all 9 block parameter tensors
    + input `X` (`10` checks): `max_err = 2.746e-07` (`<= 5e-4`).
- Fail-closed negative checks (`12` Eshkol AOT cases + additional native tests):
  non-divisible `Out % W`, `In % W`, `d_ff % W`, `H_q % W`, `H_kv % W`,
  `H_q % H_kv`, unsupported `W=3`, non-finite inputs, misaligned pointers,
  overlapping/aliased buffers, monotonic timeout (`ET_TP_ERR_TIMEOUT`), peer
  disconnect (`ET_TP_ERR_PEER_DISCONNECTED`), duplicate socket FDs, tag
  mismatch (`ET_TP_ERR_MISMATCH`), and zero FD leaks under `/proc/self/fd`.
- Sanitizers: C11 native suite and C++17 linkage binary pass under
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

## Measured Limitations & Unsupported Cases

- **CPU-only single-host execution**: All computation executes on CPU in IEEE
  binary32 storage with binary64 inner accumulation; GPU execution, mixed
  precision (fp16/bf16), RDMA, NCCL/RCCL, and multi-node TCP/IB transports are
  not implemented and are not claimed.
- **Supported TP world sizes**: Only `W in {1, 2, 4}` is admitted by the ABI
  (`ET_TP_MAX_WORLD_SIZE = 4`); other world sizes fail closed with
  `ET_TP_ERR_INVALID_ARG`.
- **Star-topology coordinator**: Rank 0 gathers payloads and broadcasts reduced
  results sequentially over `AF_UNIX` `SOCK_STREAM` sockets; this guarantees
  canonical ascending rank-order determinism at the cost of a rank-0 serial
  bottleneck for large buffers (capped at `64 MiB` per collective).
- **No sequence parallelism or 2D/3D parallelism**: RMSNorm activations and
  residuals are replicated across TP ranks; sequence parallelism, pipeline
  parallelism, and ZeRO/DP composition are out of scope for this standalone
  engine.
- **Unoptimized reference GEMM/SDPA**: Matrix multiplies and causal GQA SDPA
  use straightforward deterministic C11 loops with binary64 accumulation rather
  than BLAS/SIMD-tuned kernels; throughput benchmarking at production scale is
  not claimed.
