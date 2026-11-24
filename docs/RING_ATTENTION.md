# Ring Attention / Sequence-Parallel Causal Attention (Forward + Analytic Backward)

This module implements a standalone, Eshkol-native **Ring Attention** engine
(Liu et al. 2023) with exact online-softmax forward, exact analytic ring
backward, Grouped-Query Attention (`H % Hkv == 0`) support, causal block
skipping, real multi-process `AF_UNIX` ring IPC across `W in {1, 2, 4}`
sequence shards, and a compiled Eshkol AOT driver.

## Artifacts and public contracts

- Header: `include/eshkol_transformer/ring_attn_abi.h` (`extern "C"`, C11 and
  C++17 clean, ABI `1.0`)
- Core archive: `build/ring_attn/libeshkol_transformer_ring_attn.a`
  (from `native/ring_attn.c`)
  - Deterministic archive SHA-256:
    `037d8bf64db7c02e00793e3b447ac7f62d5dcc5d3b470a7dbfc0f678754dcf89`
  - Exported symbols:
    - `et_ring_attn_validate_config_v1`
    - `et_ring_attn_comm_create_v1`, `et_ring_attn_comm_destroy_v1`,
      `et_ring_attn_ring_exchange_v1`
    - `et_ring_attn_fwd_block_step_v1`, `et_ring_attn_fwd_finalize_v1`,
      `et_ring_attn_bwd_block_step_v1`
    - `et_ring_attn_forward_v1`, `et_ring_attn_backward_v1`
    - `et_ring_attn_multiprocess_forward_v1`,
      `et_ring_attn_multiprocess_backward_v1`
    - `et_ring_attn_dense_ref_forward_v1`,
      `et_ring_attn_dense_ref_backward_v1`
- Eshkol AOT bridge archive:
  `build/ring_attn/libeshkol_transformer_ring_attn_aot.a`
  (from `native/ring_attn_eshkol_bridge.{h,c}`)
  - Deterministic archive SHA-256:
    `65f1eb365a6bbfc3359abf09432c91a0ff68bb2b250b1820f9f30249d60f7629`
  - Exports only `et_ring_attn_aot_*` transport, stats accessor, and scalar
    marshaling symbols with zero allocator imports.
- Compiled Eshkol AOT driver: `tests/ring_attn/test_ring_attn_aot.esk`
  - Compiled via `eshkol-run --strict-types --optimize 0 --no-stdlib`
  - Byte-identical dual fresh compilation ELF SHA-256:
    `104ca447d94131a00c420e333c37b300ccba0b384745d32b4da21ec3ec15057c`
- Build & verification scripts:
  - `scripts/build-ring-attn.sh`
  - `scripts/test-ring-attn.sh`
- Test suites:
  - `tests/ring_attn/header_ring_attn_cpp.cpp`
  - `tests/ring_attn/test_ring_attn_native.c`
  - `tests/ring_attn/test_ring_attn_oracle.py`

## Tensor shape & precision contract

Admitted domain:

- `world_size` (`W`): `W in {1, 2, 4}`
- `batch` (`N`): `1 .. 4`
- `seq_len` (`T`): `1 .. 1024` with `T % W == 0` (`T_local = T / W`)
- `num_heads` (`H`): `1 .. 16`
- `num_kv_heads` (`Hkv`): `1 .. H` with `H % Hkv == 0` (supports MHA `Hkv = H`,
  GQA, and MQA `Hkv = 1`; head group mapping `hkv = h / (H / Hkv)`)
- `head_dim` (`Dh`): `1 .. 128`
- `causal`: `0` (non-causal) or `1` (causal)
- `scale`: `0.0` selects `1.0 / sqrt((double)Dh)`; otherwise finite `> 0.0`
- `timeout_ms`: `1 .. 600000`

Dense row-major layouts:

| Tensor | Global Shape | Local Shard Shape (Rank `r`) | Storage Type |
|---|---|---|---|
| `Q`, `O`, `dO`, `dQ` | `[N, T, H, Dh]` | `[N, T_local, H, Dh]` | IEEE-754 `binary32` (`float`) |
| `K`, `V`, `dK`, `dV` | `[N, T, Hkv, Dh]` | `[N, T_local, Hkv, Dh]` | IEEE-754 `binary32` (`float`) |
| `LSE` | `[N, T, H]` | `[N, T_local, H]` | IEEE-754 `binary64` (`double`) |

All dot products, online softmax accumulators `(m_i, l_i, O_i)`, backward `D_i`
projections, and circulating gradient accumulators `(dK_acc, dV_acc)` execute
in IEEE-754 `binary64` (`double`) under `FE_TONEAREST` with `-ffp-contract=off
-fexcess-precision=standard -fno-fast-math`. All outputs remain byte-for-byte
untouched on any error.

## Exact Ring Attention equations

### Forward pass (Online Softmax + Causal Block Skip)

Sequence length `T` is partitioned evenly across `W` ranks (`T_local = T / W`).
Rank `r` holds `Q_r` locally and initializes per-query `binary64` state:

```text
m_i = -infinity,    l_i = 0.0,    O_acc[i, 0..Dh-1] = 0.0
```

Over `W` steps `s = 0 .. W - 1`, rank `r` processes incoming KV shard
`(K_blk, V_blk)` originating from `kv_rank = (r - s + W) % W`:

1. **Causal block skip**: if `causal == 1 && kv_rank > r`, skip computation
   completely and increment `skipped_causal_blocks`.
2. **Block range**:
   - If `causal == 1 && kv_rank == r`, apply strict intra-block lower-triangular
     mask `0 <= local_j <= local_i` (`j_end = local_i + 1`).
   - Otherwise (`causal == 0 || kv_rank < r`), attend to all `0 <= local_j < T_local`.
3. **Online log-sum-exp rescaling** (for each `(n, local_i, h)` with `hkv = h / (H / Hkv)`):

```text
s[j]      = (Q_r[n, i, h] . K_blk[n, j, hkv]) * scale     for 0 <= j < j_end
m_blk     = max_{0 <= j < j_end} s[j]
m_new     = (l_i == 0) ? m_blk : max(m_i, m_blk)
alpha     = (l_i == 0) ? 0.0   : exp(m_i - m_new)
w[j]      = exp(s[j] - m_new)
l_new     = l_i * alpha + sum_{0 <= j < j_end} w[j]
O_acc[i]  = O_acc[i] * alpha + sum_{0 <= j < j_end} w[j] * V_blk[n, j, hkv]
(m_i, l_i) <- (m_new, l_new)
```

4. **Ring rotation** (for `W > 1` and `s < W - 1`, `W - 1` rounds total):
   rank `r` sends `(K_blk, V_blk)` to `(r + 1) % W` while simultaneously
   receiving the next KV shard from `(r + W - 1) % W` over connected `AF_UNIX`
   `SOCK_STREAM` sockets.

After step `s = W - 1`:

```text
O_r[n, i, h]   = (float)(O_acc[n, i, h] / l_i)
LSE_r[n, i, h] = m_i + log(l_i)
```

### Backward pass (Circulating `(K, V, dK_acc, dV_acc)` Ring)

Before the ring loop, rank `r` precomputes the FlashAttention/RingAttention
row projection in `binary64`:

```text
D_i = sum_{d=0}^{Dh-1} dO_r[n, i, h, d] * O_r[n, i, h, d]
```

Rank `r` initializes `dQ_acc = 0.0` and circulating packet
`(dK_acc = 0.0, dV_acc = 0.0, K_blk = K_r, V_blk = V_r)` (`dK_acc, dV_acc`
transmitted in `binary64` to eliminate per-hop `binary32` drift).

Over `W` steps `s = 0 .. W - 1` with `kv_rank = (r - s + W) % W`:

1. If `causal == 1 && kv_rank > r`, skip compute completely
   (`skipped_causal_blocks++`).
2. Otherwise, for each `(n, local_i, h)` and `0 <= j < j_end`:

```text
S_{ij}    = (Q_r[n, i, h] . K_blk[n, j, hkv]) * scale
P_{ij}    = exp(S_{ij} - LSE_r[n, i, h])
dP_{ij}   = dO_r[n, i, h] . V_blk[n, j, hkv]
dS_{ij}   = P_{ij} * (dP_{ij} - D_i)
dQ_acc[i, h]    += (dS_{ij} * scale) * K_blk[j, hkv]
dK_acc[j, hkv]  += (dS_{ij} * scale) * Q_r[i, h]
dV_acc[j, hkv]  += P_{ij} * dO_r[i, h]
```

3. **Ring rotation** (for `W > 1`, `W` rounds total):
   - For `s < W - 1`, rotate `(dK_acc, dV_acc, K_blk, V_blk)` to `(r + 1) % W`.
   - For `s == W - 1`, rotate `(dK_acc, dV_acc)` one final hop to `(r + 1) % W`
     so every rank receives its own completely accumulated `(dK_r, dV_r)`
     (`cur_kv_rank == r`).

### Multi-process `AF_UNIX` ring transport

- Topology: `W` independent `AF_UNIX` `SOCK_STREAM` `socketpair` ring edges
  with `SOCK_CLOEXEC` (`edge[r][0]` = rank `r` `send_fd`, `edge[(r+W-1)%W][1]`
  = rank `r` `recv_fd`). Zero shared memory is used.
- Deadlock-free duplex I/O: simultaneous non-blocking `poll(POLLOUT | POLLIN)`
  on `(send_fd, recv_fd)` with `MSG_DONTWAIT | MSG_NOSIGNAL`, verified with a
  `512 KiB x 4-rank` exchange exceeding Linux default socket buffer sizes.
- Shared monotonic deadline (`CLOCK_MONOTONIC`) enforces `ET_RING_ATTN_ERR_TIMEOUT`.
- Wire framing validates `(magic, phase, round, kv_rank, sender_rank, payload_bytes)`
  and reports `ET_RING_ATTN_ERR_PROTOCOL` on mismatch or
  `ET_RING_ATTN_ERR_PEER_DISCONNECTED` on hangup/EOF.

## Causal block skip counts

For causal attention across `W` ranks, rank `r` skips all `kv_rank > r`
(`W - 1 - r` blocks), yielding `W * (W - 1) / 2` skipped rank-blocks out of
`W * W` total in both forward and backward:

| World size `W` | Causal | Total blocks (`W^2`) | Executed blocks | Skipped blocks | Forward rounds | Backward rounds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | `0 / 1` (0.0%) | 0 | 0 |
| 2 | 1 | 4 | 3 | `1 / 4` (25.0%) | 1 | 2 |
| 4 | 1 | 16 | 10 | `6 / 16` (37.5%) | 3 | 4 |
| 2 | 0 | 4 | 4 | `0 / 4` (0.0%) | 1 | 2 |
| 4 | 0 | 16 | 16 | `0 / 16` (0.0%) | 3 | 4 |

## Measured verification results

Verified inside `eshkol-r0-debug:20260928` (`clang-21`, `clang++-21`,
`eshkol-run --strict-types --optimize 0 --no-stdlib`):

| Gate | Measured Result |
|---|---|
| Deterministic archives (`ar rcsD`) | Byte-identical across two fresh builds; strict `nm` export/import audit PASS |
| C++17 header & linkage (`-std=c++17`) | PASS (normal `-O2` and ASan/UBSan/LSan) |
| Native `W = 1, 2, 4` vs `dense_ref` parity (MHA, GQA 2:1, MQA 4:1, custom scale, `T_local=1`, causal & non-causal) | `max_err`: `O = 0.000e+00`, `LSE = 4.441e-16`, `dQ = 0.000e+00`, `dK = 0.000e+00`, `dV = 0.000e+00` (`<= 1e-6`) |
| Native exhaustive central FD gradients (`Q/K/V`, `W=1/2/4`, causal & non-causal) | `checked = 384`, `max_abs_err = 1.198e-05` |
| Independent Python `binary64` oracle (`test_ring_attn_oracle.py`, 9 cases, 4,552 values) | `max_err`: `O = 2.980e-08`, `LSE = 4.441e-16`, `dQ = 1.439e-08`, `dK = 1.432e-08`, `dV = 9.359e-08`, `py_fd_err = 7.553e-10` |
| Deadlock-free large-payload ring exchange | `512 KiB x 4-rank` `AF_UNIX` ring PASS |
| Negative rejections, peer disconnect, timeout, protocol & FD leaks | All fail closed with untouched sentinels; open FDs `4 == 4` |
| ASan + UBSan + LSan (`detect_leaks=1:halt_on_error=1`) | PASS (native C11 + C++17) |
| Compiled Eshkol AOT (`test_ring_attn_aot.esk`, `N=1, T=8, H=4, Hkv=2, Dh=4`) | Dual fresh compiles produce byte-identical ELF (`104ca447...15057c`); `W=4` causal skips `6/16` fwd & bwd; parity `err_o=0`, `err_lse=4.441e-16` (causal) / `8.882e-16` (non-causal), `err_dq=err_dk=err_dv=0`; `W=4` FD gradients (`18` causal `max_abs_err=1.941e-05`, `18` non-causal `max_abs_err=1.346e-05`) |
| `tests/q0/test_python_isolation.py` & `scripts/check-ci-topology.sh` | PASS |

## Measured limitations and unsupported cases

- **CPU-only execution**: executes on CPU in IEEE-754 `binary64` internal
  accumulation with `binary32` tensor I/O; no GPU/ROCm/CUDA offload is claimed.
- **Supported world sizes**: strictly `W in {1, 2, 4}`; other values fail
  explicitly with `ET_RING_ATTN_ERR_UNSUPPORTED`.
- **Even sequence partitioning**: requires `T % W == 0` and `H % Hkv == 0`;
  uneven shards or non-divisible GQA head counts fail explicitly with
  `ET_RING_ATTN_ERR_SHAPE_MISMATCH`.
- **Single-host `AF_UNIX` transport**: multi-process IPC uses local `AF_UNIX`
  stream sockets; multi-node TCP/RDMA transport is out of scope.
- **No attention dropout or arbitrary 2D attention mask**: supports standard
  causal (`causal=1`) and full unmasked (`causal=0`) attention only.
- **Bounded dimensions**: `N <= 4`, `T <= 1024`, `H <= 16`, `Dh <= 128`, and
  `timeout_ms <= 600000` are enforced explicitly to prevent integer overflow.
