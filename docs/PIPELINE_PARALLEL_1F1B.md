# 1F1B Pipeline Parallelism Engine (`P = 2, 4` Multi-Process Stages)

This document specifies the standalone, Eshkol-native **1F1B (One-Forward-One-Backward)
Pipeline Parallelism Engine**, its C11/C++17 public ABI, deterministic 1F1B and
GPipe schedules, real multi-process `AF_UNIX` point-to-point transport,
differentiable multi-layer Pre-RMSNorm + SwiGLU residual pipeline stages,
compiled Eshkol AOT driver, and measured verification evidence.

## Artifacts and public headers

- Public C11/C++17 ABI: `include/eshkol_transformer/pipeline_1f1b_abi.h`
- Core engine static library: `native/pipeline_1f1b.c` -> `build/pipeline_1f1b/libeshkol_transformer_pipeline_1f1b.a`
- Eshkol AOT bridge: `native/pipeline_1f1b_eshkol_bridge.h`, `native/pipeline_1f1b_eshkol_bridge.c` -> `build/pipeline_1f1b/libeshkol_transformer_pipeline_1f1b_aot.a`
- C++17 linkage test: `tests/pipeline_1f1b/header_pp_cpp.cpp`
- Native C11 verification & oracle-dump driver: `tests/pipeline_1f1b/test_pipeline_1f1b_native.c`
- Independent Python binary64 schedule & numerical oracle: `tests/pipeline_1f1b/test_pipeline_1f1b_oracle.py`
- Compiled Eshkol AOT driver: `tests/pipeline_1f1b/test_pipeline_1f1b_aot.esk`
- Build and test scripts: `scripts/build-pipeline-1f1b.sh`, `scripts/test-pipeline-1f1b.sh`

## Deterministic 1F1B schedule and timeline

For `num_stages` $P \in \{2, 4\}$, stage index $s \in \{0, \dots, P-1\}$, and
microbatch count $M \in [P, 32]$, `et_pp_build_schedule_v1` emits exactly $2M$
ordered steps per stage:

- **Warmup phase**: $w_s = P - 1 - s$ forward steps `F(0), ..., F(w_s - 1)`.
  Stage `0` executes $P - 1$ warmups; the last stage $P - 1$ executes `0` warmups.
- **Steady-state phase**: $M - w_s$ interleaved `(F(w_s + i), B(i))` pairs for
  $i = 0, \dots, M - w_s - 1$. Every forward step is paired immediately with
  the backward step of the oldest in-flight microbatch `i`.
- **Cooldown phase**: $w_s$ remaining backward steps `B(M - w_s), ..., B(M - 1)`.
- **Pipeline bubble fraction** (identical for 1F1B and GPipe):
  $$\text{bubble}(P, M) = \frac{P - 1}{M + P - 1}$$
  - $P = 2, M = 4$: $1/5 = 0.200000$
  - $P = 2, M = 8$: $1/9 \approx 0.111111$
  - $P = 4, M = 4$: $3/7 \approx 0.428571$
  - $P = 4, M = 8$: $3/11 \approx 0.272727$

### Exact 1F1B schedule timeline (`P = 4, M = 8`)

```text
Stage 0 (w=3, peak=4): F0  F1  F2  | F3 B0 | F4 B1 | F5 B2 | F6 B3 | F7 B4 | B5  B6  B7
Stage 1 (w=2, peak=3):     F0  F1  | F2 B0 | F3 B1 | F4 B2 | F5 B3 | F6 B4 | F7 B5 | B6  B7
Stage 2 (w=1, peak=2):         F0  | F1 B0 | F2 B1 | F3 B2 | F4 B3 | F5 B4 | F6 B5 | F7 B6 | B7
Stage 3 (w=0, peak=1):             | F0 B0 | F1 B1 | F2 B2 | F3 B3 | F4 B4 | F5 B5 | F6 B6 | F7 B7
```

Because every stage runs backward steps in exact ascending microbatch order
`B(0), B(1), ..., B(M-1)`, parameter gradient accumulation order matches
single-process unpipelined execution bit-for-bit.

## Peak in-flight activation memory: 1F1B vs GPipe

Each stage allocates an activation ring-buffer pool of exact capacity
`slot_cap = sched.peak_live_activations` (not $M$ slots) and verifies at runtime
that every forward fits within `slot_cap`, every backward releases the oldest
live microbatch slot, and `final_live == 0` at step completion.

| Config | Stage `s` | GPipe Peak Slots (`M`) | 1F1B Peak Slots (`P - s`) | Activation Memory Reduction |
|---|---:|---:|---:|---:|
| `P = 2, M = 4` | `0` | `4` | `2` | `50.0%` |
| `P = 2, M = 4` | `1` | `4` | `1` | `75.0%` |
| `P = 2, M = 8` | `0` | `8` | `2` | `75.0%` |
| `P = 2, M = 8` | `1` | `8` | `1` | `87.5%` |
| `P = 4, M = 8` | `0` | `8` | `4` | `50.0%` |
| `P = 4, M = 8` | `1` | `8` | `3` | `62.5%` |
| `P = 4, M = 8` | `2` | `8` | `2` | `75.0%` |
| `P = 4, M = 8` | `3` | `8` | `1` | `87.5%` |

## Real multi-process `AF_UNIX` transport and stage model

- **No shared memory**: `native/pipeline_1f1b.c` imports zero shared-memory
  (`shm_*`, `mmap`) or threading symbols (enforced by `nm -u` audit in
  `scripts/build-pipeline-1f1b.sh`).
- **Process topology**: `et_pp_run_multiprocess_step_v1` and
  `et_pp_train_multiprocess_sgd_v1` spawn `P` distinct OS child processes via
  `fork()`. Adjacent stage pairs `(s, s+1)` communicate exclusively over
  dedicated `AF_UNIX` `SOCK_STREAM | SOCK_CLOEXEC` `socketpair`s.
- **Wire framing (`et_pp_frame_header_t`, 32 bytes)**:
  `(magic=0x4554505031463142, msg_type, microbatch_id, src_stage, dst_stage, payload_bytes)`
  followed by exact payload bytes.
  - Forward activations: `ET_PP_MSG_FORWARD_ACT` (`s -> s+1`)
  - Backward gradients: `ET_PP_MSG_BACKWARD_GRAD` (`s+1 -> s`)
  - Coordinator stage result: `ET_PP_MSG_STAGE_RESULT` (`s -> coordinator`)
- **Transport safety**: `poll()` with `CLOCK_MONOTONIC` deadline timeout
  (`ET_PP_ERR_TIMEOUT`), `MSG_DONTWAIT | MSG_NOSIGNAL`, `POLLIN` drained before
  `POLLHUP` reporting (`ET_PP_ERR_PEER_DISCONNECTED`), strict header matching
  (`ET_PP_ERR_PROTOCOL`), and finite IEEE-754 binary64 payload validation
  (`ET_PP_ERR_NONFINITE`). Every child closes unused FDs immediately after
  `fork()` and the coordinator reaps all children via `waitpid()`, guaranteeing
  zero leaked FDs and zero zombie processes.
- **Differentiable stage layers**: Each stage owns `layers_per_stage`
  Pre-RMSNorm + SwiGLU MLP residual blocks (`x + W_down (SiLU(W_gate RMSNorm(x)) * (W_up RMSNorm(x)))`).
  Stage `P - 1` additionally owns the linear output head `(W_head, b_head)` and
  computes exact Cross-Entropy or MSE loss scaled by `1 / M` for exact gradient
  accumulation across all `M` microbatches.

## Measured verification results

Verified inside the pinned Ubuntu 22.04 container (`eshkol-r0-debug:20260928`,
Clang 21.1.8, pinned Eshkol `fe9dfd52`) via `bash scripts/test-pipeline-1f1b.sh`,
`python3 -m unittest -v tests/q0/test_python_isolation.py`, and
`bash scripts/check-ci-topology.sh`:

| Check | Measured result |
|---|---|
| Core deterministic archive (`libeshkol_transformer_pipeline_1f1b.a`) | SHA-256 `3dd0fb6f807ae7133be03071666340c99fe12da21709ef55798659af24d528ef` |
| Bridge deterministic archive (`libeshkol_transformer_pipeline_1f1b_aot.a`) | SHA-256 `7cb53205802ff1ba8f70cb50f289947d7828f9f05d92dfc2e73a29f43f4dd1ab` |
| Compiled Eshkol AOT ELF (`test_pipeline_1f1b_aot`, dual fresh builds) | SHA-256 `5d8ad3fe948c2f26cb8e8c96feb538db0c0ee63b22bbd41c952ec6f396275eb8` (byte-identical ELF & stdout) |
| C++17 header & archive linkage (`clang++-21 -std=c++17`) | PASS (normal `-O2` + ASan/UBSan/LSan) |
| Native C11 central FD checks (`eps = 1e-6`) | `layer_x = 8.565e-11`, `layer_p = 1.739e-10`, `full_ce = 1.954e-10`, `full_mse = 1.583e-11` |
| Native C11 multi-process parity (`P = 2, 4`, `M = 4, 8`, CE & MSE, 1F1B & GPipe vs `P = 1`) | `max_loss_err = 0.000e+00`, `max_grad_err = 0.000e+00` (`<= 1e-6` gate) |
| Native C11 10-step SGD parity (`P = 1, 2, 4`, `M = 8`, `lr = 0.08`) | Strictly decreasing loss `1.793446 -> 1.740036`, `sgd_w_err = 0.000e+00` |
| Independent Python binary64 oracle (`P = 2, 4`, `M = 8`, CE, MSE, FD, 10-step SGD) | `P=2_err = 1.388e-17`, `P=4_ce_err = 1.388e-17`, `P=4_mse_err = 1.735e-18`, `fd_err = 1.816e-10`, `sgd10_loss_err = 4.441e-16`, `sgd10_w_err = 2.776e-17` |
| Eshkol AOT driver (`test_pipeline_1f1b_aot.esk`) | `FD checked=84` (`max_dx_err = 7.823e-11`, `max_dp_err = 1.713e-10`); `M=8` `P=2,4` CE & MSE `gerr = 0`; `SGD10` `2.018090 -> 1.794433` (`max_lerr = 0`, `w2_err = 0`, `w4_err = 0`); `peaks = 4,3,2,1` vs GPipe `8,8,8,8`; `9` negative checks PASS |
| Native negative tests & FD hygiene | `23` checks PASS, `/proc/self/fd` delta `= 0` (normal + ASan/UBSan/LSan `detect_leaks=1:halt_on_error=1`) |
| Python isolation & CI topology | `tests/q0/test_python_isolation.py` (`5/5 OK`), `scripts/check-ci-topology.sh` (`23 shared suites PASS`) |

## Limitations and unsupported cases

- **Stage count**: Pipeline schedules and multi-process execution support only
  `P in {2, 4}` (all other stage counts fail closed with `ET_PP_ERR_INVALID_STAGES`).
- **Microbatch count**: Requires `P <= M <= 32` (`ET_PP_ERR_INVALID_MICROBATCHES` otherwise).
- **Dimensions**: Bounded compile-time maxima (`layers_per_stage <= 8`,
  `total_layers <= 32`, `tokens_per_mb <= 32`, `d_model <= 64`, `d_ff <= 128`,
  `out_dim <= 64`, `num_steps <= 128`, `timeout_ms <= 60000`).
- **Execution backend & precision**: Single-host Linux multi-process CPU execution
  over `AF_UNIX` `SOCK_STREAM` sockets in IEEE-754 binary64 (`double`). No GPU
  execution, TCP/RDMA multi-node transport, interleaved virtual stages, or
  mixed precision is claimed or implemented.
- **Standalone integration**: This workstream adds standalone artifacts only and
  is not wired into `Makefile` or CI workflows.
