# RadixAttention Prefix KV Cache Tree (SGLang-Style LRU Radix Tree)

This document specifies the standalone, Eshkol-native RadixAttention Prefix KV
Cache Tree engine, its C11/C++17 ABI, structural & accounting invariants,
`ETRX1000` v1.0 SHA-256 checksummed snapshot format, compiled Eshkol AOT
bridge/driver, and measured verification results.

## Artifacts and public headers

- Public C11/C++17 ABI: `include/eshkol_transformer/radix_kv_abi.h`
- Core implementation: `native/radix_kv.c` -> `build/radix_kv/libeshkol_transformer_radix_kv.a`
- Eshkol AOT bridge: `native/radix_kv_eshkol_bridge.h`, `native/radix_kv_eshkol_bridge.c` -> `build/radix_kv/libeshkol_transformer_radix_kv_aot.a`
- Native C11 & C++17 tests: `tests/radix_kv/test_radix_kv_native.c`, `tests/radix_kv/header_radix_cpp.cpp`
- Development reference oracle: `tests/radix_kv/test_radix_kv_oracle.py`
- Compiled Eshkol AOT driver: `tests/radix_kv/test_radix_kv_aot.esk`
- Build & verification scripts: `scripts/build-radix-kv.sh`, `scripts/test-radix-kv.sh`

## Physical KV Pool Geometry and Bounded Limits

The physical KV pool is a single pre-allocated contiguous row-major IEEE-754
binary32 tensor owned by `et_radix_kv_tree`:

```text
[max_kv_tokens, n_layers, 2 (K=0, V=1), n_kv_heads, head_dim]
```

Every physical KV token slot `s in [0, max_kv_tokens)` is linked into an
intrusive per-edge slot chain (`slot_next[s]`, `slot_token_id[s]`,
`slot_owner_node[s]`). Edge splits relink slot chains in place with **zero KV
tensor copies** and **zero heap allocations** after `et_radix_kv_create`.

| Parameter | Constraint |
|---|---|
| `max_kv_tokens` | `1 <= max_kv_tokens <= 4096` |
| `max_nodes` | `2 <= max_nodes <= 4096` (node `0` is always root) |
| `n_layers` | `1 <= n_layers <= 64` |
| `n_kv_heads` | `1 <= n_kv_heads <= 64` |
| `n_q_heads` | `1 <= n_q_heads <= 64`, `n_q_heads % n_kv_heads == 0` |
| `head_dim` | `1 <= head_dim <= 256` |
| Total pool floats | `<= 67,108,864` (256 MiB cap, `size_t` overflow checked) |
| `cfg.reserved` | Must be `0` |

### Per-operation tensor shapes

| Operation | Tensor | Shape (row-major `float`) |
|---|---|---|
| `et_radix_kv_insert` | `k_data`, `v_data` | `[seq_len, n_layers, n_kv_heads, head_dim]` |
| `et_radix_kv_insert_suffix` | `suffix_k`, `suffix_v` | `[suffix_len, n_layers, n_kv_heads, head_dim]` |
| `et_radix_kv_read_kv` | `out_k`, `out_v` | `[path_len, n_kv_heads, head_dim]` |
| `et_radix_kv_attend_gqa_f32` | `q`, `out` | `[q_len, n_q_heads, head_dim]` |
| `et_radix_kv_attend_gqa_f32` | `suffix_k`, `suffix_v` | `[suffix_len, n_kv_heads, head_dim]` (`NULL` iff `suffix_len == 0`) |

## Core Contracts and Structural Invariants

All invariants below are audited on demand by `et_radix_kv_verify_invariants`:

1. **Generational handles (`et_radix_kv_handle`)**:
   - Handles pack `(generation << 32) | node_idx` (`generation >= 1`).
   - Evicting a node increments its generation so stale handles are rejected
     with `ET_RADIX_ERR_INVALID_HANDLE`.
   - Splitting an interior edge `N` at offset `k` allocates a new upper parent
     node `P` for tokens `[0..k-1]` and trims `N` in place to `[k..edge_len-1]`,
     preserving existing handles to `N` (which still represent the exact same
     full sequence from root), KV slot ownership, and pin counts.
2. **Exact token & node conservation**:
   - `used_tokens + free_tokens == max_kv_tokens`
   - `evictable_tokens + pinned_tokens == used_tokens`
   - `active_nodes + free_nodes == max_nodes` (`active_nodes >= 1`)
   - Every physical slot `0 .. max_kv_tokens - 1` appears either once in the
     free-slot stack or once in an alive non-root node's edge chain.
   - Sibling children of every node have strictly unique `first_token` keys.
3. **Ancestor pin protection & refcount monotonicity**:
   - Every node tracks both `self_pin_count` (direct `pin` calls on this node)
     and `pin_count = self_pin_count + sum_{c in children} c.pin_count`.
   - `et_radix_kv_pin` increments `self_pin_count` on the target and `pin_count`
     along `[target .. root]`. Any non-root node with `pin_count > 0` counts
     towards `pinned_tokens` and is never evicted.
   - `et_radix_kv_unpin` requires `self_pin_count > 0`; unpinning an unpinned
     node or an ancestor pinned only by descendants returns
     `ET_RADIX_ERR_DOUBLE_UNPIN` with zero mutation.
4. **Deterministic logical LRU eviction & cascading**:
   - A monotonic `uint64_t lru_tick` (zero wall-clock nondeterminism) increments
     on `et_radix_kv_match_prefix` and `et_radix_kv_insert`, stamping
     `last_access_tick` along `[matched_or_inserted_node .. root]`.
   - Invariant: `parent.last_access_tick >= child.last_access_tick` for all
     non-root parents.
   - Eviction always selects the unpinned leaf (`num_children == 0 && pin_count == 0`)
     outside the active protected path with minimum `(last_access_tick ASC, node_idx ASC)`.
   - Removing a leaf decrements its parent's `num_children`; if an unpinned
     interior parent reaches `num_children == 0`, it becomes an eligible leaf
     in the next eviction iteration (cascading LRU leaf eviction).
5. **Atomic OOM preflight**:
   - Before performing any split, eviction, or slot write, `et_radix_kv_insert`,
     `et_radix_kv_insert_suffix`, and `et_radix_kv_match_prefix` compute the
     exact available free + evictable tokens and nodes outside the protected
     prefix path.
   - If insufficient capacity exists because active nodes are pinned, the call
     returns `ET_RADIX_ERR_OOM` with **zero tree/slot/tick mutation**.
6. **Exact RadixAttention GQA attention parity (`et_radix_kv_attend_gqa_f32`)**:
   - Evaluates causal GQA attention for query positions `qi in [0, q_len)` at
     absolute positions `abs_pos = (prefix_len + suffix_len - q_len) + qi`
     using single-pass binary64 online softmax (`#pragma STDC FENV_ACCESS ON`,
     `#pragma STDC FP_CONTRACT OFF`, `-ffp-contract=off -fexcess-precision=standard
     -fno-fast-math`), yielding **bitwise `0.0` parity** between cached-prefix
     RadixAttention decode/prefill and full uncached from-scratch prefill.

## Versioned `ETRX1000` v1.0 SHA-256 Checksummed Snapshots

- **Layout (little-endian, byte-wise encoded)**:
  - 64-byte header: `"ETRX1000"` magic, version `(1, 0)`, config
    `(max_kv_tokens, max_nodes, n_layers, n_kv_heads, head_dim)`,
    `active_nodes`, `used_tokens`, `reserved = 0`, `lru_tick`, `payload_bytes`.
  - Canonical pre-order DFS payload (children sorted by `first_token ASC`),
    storing `(parent_dfs_idx, edge_len, flags=0, reserved=0, last_access_tick)`,
    `edge_tokens[edge_len]`, and raw finite IEEE-754 binary32 KV payloads.
  - 32-byte trailing SHA-256 digest over `header || payload`.
- **Save security**: validates invariants and finiteness, writes to a `0600`
  sibling temp file via `mkstemp` + `fchmod(0600)` + `fsync` + atomic `rename`.
- **Load security**: opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`
  (rejecting symlinks with `ET_RADIX_ERR_SYMLINK`), verifies regular file size,
  magic, version, constant-time SHA-256 digest (`ET_RADIX_ERR_CHECKSUM`),
  truncation (`ET_RADIX_ERR_TRUNCATED`), finiteness, and structural invariants
  before returning a fresh tree. Transient request pins are reset to `0` on load.

## Measured Verification Results (`eshkol-r0-debug:20260928`)

Verified inside pinned container `eshkol-r0-debug:20260928` (`clang-21`,
`clang++-21`, `--network none`):

- Deterministic core archive `libeshkol_transformer_radix_kv.a` SHA-256:
  `6a531152bd60ef5ad75cb4dcd748027c76d8faa5533cc733979b6742a453e1a0`
- Deterministic bridge archive `libeshkol_transformer_radix_kv_aot.a` SHA-256:
  `96c049ac177659b3286d6c0776c0f5f997f9a2befafff7068c1359745636a699`
- Dual fresh Eshkol AOT ELF `test_radix_kv_aot` SHA-256 (byte-identical `cmp`):
  `6d772afc9828faf724c9bb0507d17466498d922a56196378a9112df158010779`
- AOT shared-system-prompt workload (4 requests, 96 logical tokens):
  `physical=40`, `reused=56` (`58.33%` KV reuse rate), `splits=3`, zero
  duplicate prefix allocations.
- AOT RadixAttention GQA parity (`prefix=20, suffix=4`): `max_diff=0` (bitwise
  exact match against uncached 24-token prefill).
- Native C11 suite (`-O2` + ASan/UBSan/LSan `detect_leaks=1:halt_on_error=1`):
  edge splits, multi-way branching, pin/unpin, cascading LRU leaf eviction,
  atomic OOM rollback, bitwise GQA parity, `ETRX1000` snapshot round-trip +
  tamper/truncation/symlink rejections, and 500-op randomized fuzz with full
  invariant verification after every operation: **PASS**.
- Independent Python reference RadixTree + LRU + binary64 GQA oracle:
  **PASS** (`50` trace ops exact state/eviction/hit match, `3` GQA checks,
  `max_abs_err=2.970e-08`).

## Known Limitations and Unsupported Cases

- **CPU-only binary32 storage / binary64 accumulation**: GPU device-memory
  pools, FP16/BF16 KV quantization, and multi-threaded concurrent tree access
  are not implemented (caller must synchronize externally if shared across
  threads).
- **Linear child scan & LRU victim scan**: Designed for deterministic bounded
  pools (`max_nodes <= 4096`); very large pools (> 4096 nodes/tokens) require
  replacing the linear victim scan with an indexed min-heap and hash-indexed
  child maps.
- **No automatic edge re-coalescing after eviction**: When one sibling of a
  two-child node is evicted, the surviving interior node remains explicit to
  preserve handle identity and `last_access_tick` semantics; if its remaining
  child is later evicted, the parent cascades cleanly as an unpinned leaf.
- **Transient pins are not persisted across `ETRX1000` snapshots**: Reloaded
  warm-start caches begin with `pinned_tokens == 0`.
