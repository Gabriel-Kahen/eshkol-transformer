# PagedAttention / Block-Table KV-Cache Manager with Copy-on-Write Prefix Sharing

This document specifies the standalone, Eshkol-native PagedAttention block-table
KV-cache manager, Copy-on-Write (CoW) prefix sharing, canonical SHA-256 prefix
block deduplication, exact online-softmax GQA causal decode attention, and
compiled Eshkol AOT verification.

## Artifacts and public headers

- Public C11/C++17 ABI: `include/eshkol_transformer/paged_kv_abi.h`
- Core implementation: `native/paged_kv.c` -> `build/paged_kv/libeshkol_transformer_paged_kv.a`
- Eshkol AOT bridge: `native/paged_kv_eshkol_bridge.h`, `native/paged_kv_eshkol_bridge.c` -> `build/paged_kv/libeshkol_transformer_paged_kv_aot.a`
- Native & C++17 tests: `tests/paged_kv/test_paged_kv_native.c`, `tests/paged_kv/header_paged_kv_cpp.cpp`
- Development oracle: `tests/paged_kv/test_paged_kv_oracle.py`
- Compiled Eshkol AOT verifier: `tests/paged_kv/test_paged_kv_aot.esk`
- Build and test scripts: `scripts/build-paged-kv.sh`, `scripts/test-paged-kv.sh`

## Block geometry and tensor contracts

### Physical KV pool layout

The physical KV pool is a single pre-allocated contiguous row-major IEEE-754
binary32 tensor (either caller-provided via `external_pool` or pool-owned):

```text
[num_blocks, num_layers, 2 (K=0, V=1), num_kv_heads, block_size, head_dim]
```

Exact element offset for physical block `b`, layer `l`, side `is_v` (`0` for K,
`1` for V), KV head `h`, intra-block slot `s`, and head coordinate `d`:

```text
(((((b * num_layers + l) * 2 + is_v) * num_kv_heads + h) * block_size + s) * head_dim + d)
```

### Bounded dimension limits

| Parameter | Constraint |
|---|---|
| `num_blocks` | `1 <= num_blocks <= 4096` |
| `num_layers` | `1 <= num_layers <= 64` |
| `num_kv_heads` | `1 <= num_kv_heads <= 64` |
| `num_q_heads` | `1 <= num_q_heads <= 64`, `num_q_heads % num_kv_heads == 0` |
| `block_size` | `1 <= block_size <= 64` |
| `head_dim` | `1 <= head_dim <= 256` |
| `max_seqs` | `1 <= max_seqs <= 256` |
| `max_blocks_per_seq` | `1 <= max_blocks_per_seq <= min(1024, num_blocks)` |
| Total pool floats | `<= 67,108,864` (256 MiB cap, checked against overflow) |
| `cfg.reserved` | Must be `0` |

### Per-operation tensor shapes

| Operation | Tensor | Shape (row-major `float`) | Mutability |
|---|---|---|---|
| `et_paged_kv_append_token` | `k_token`, `v_token` | `[num_layers, num_kv_heads, head_dim]` | Read-only (`const float *`) |
| `et_paged_kv_append_tokens` | `k_tokens`, `v_tokens` | `[num_tokens, num_layers, num_kv_heads, head_dim]` | Read-only (`const float *`) |
| `et_paged_kv_read_token` | `out_k`, `out_v` | `[head_dim]` | Output (`float *`) |
| `et_paged_kv_decode_attention_v1` | `q` | `[num_q_heads, head_dim]` | Read-only (`const float *`) |
| `et_paged_kv_decode_attention_v1` | `out` | `[num_q_heads, head_dim]` | Output (`float *`) |

## Copy-on-Write (CoW), prefix sharing, and rollback invariants

1. **Deterministic free list**: The free-block stack initializes to
   `[num_blocks - 1, ..., 1, 0]` so successive pops deterministically yield
   `0, 1, 2, ...`. Every pool operation after `et_paged_kv_create` performs
   **zero heap allocations** on the hot path.
2. **Zero-copy fork (`et_paged_kv_seq_fork`)**: Copies `seq_len` and
   `block_table[0..num_logical_blocks - 1]` from parent to child and increments
   `ref_count[phys]++` for every mapped physical block (`cow_copies` unchanged).
3. **Atomic Copy-on-Write (`et_paged_kv_append_token` / `et_paged_kv_append_tokens`)**:
   - Before mutating any pool, refcount, free-list, or sequence state, the
     manager validates FP environment, sequence liveness, capacity
     (`ET_PAGED_KV_ERR_SEQ_FULL`), 4-byte alignment, pairwise buffer
     disjointness (`ET_PAGED_KV_ERR_ALIAS`), finiteness of every K/V element
     (`ET_PAGED_KV_ERR_NONFINITE`), and exact required free blocks:
     `extra_blocks + ((cur_len % block_size != 0 && ref_count[tail_phys] > 1) ? 1 : 0)`.
   - If `free_count < total_needed_blocks`, returns `ET_PAGED_KV_ERR_OOM_BLOCKS`
     with **zero state mutation** (siblings remain bitwise untouched, zero blocks
     leaked).
   - When writing at `slot = pos % block_size > 0` into a shared block
     (`ref_count[old_blk] > 1`), pops `new_blk`, copies prefix tokens
     `0..slot - 1` across all layers/sides/heads, updates only the writing
     sequence's `block_table[logical_block] = new_blk`, decrements
     `ref_count[old_blk]--`, increments `cow_copies++`, and writes slot `slot`.
4. **Speculative decoding / beam rollback (`et_paged_kv_seq_truncate`)**:
   - Rejects `new_len > seq_len` with `ET_PAGED_KV_ERR_INVALID_RANGE`.
   - Releases unused tail blocks in reverse logical order; any physical block
     whose `ref_count` drops to `0` is evicted from the prefix cache and pushed
     onto the free stack.
5. **Canonical SHA-256 prefix-block deduplication**:
   - `et_paged_kv_hash_token_block` computes:
     `SHA256("ETPKV1\0\0" || (prev_hash ? prev_hash[32] : zeros[32]) || LE32(block_size) || LE32(token_ids[0..block_size-1]))`.
     Chaining `prev_hash` incorporates the token IDs of the preceding prompt.
   - Only full blocks (`logical_block < seq_len / block_size`) may be registered
     via `et_paged_kv_prefix_cache_register`.
   - `et_paged_kv_prefix_cache_attach` attaches a cached physical block at a
     block-aligned position (`seq_len % block_size == 0`) with `ref_count++`,
     `seq_len += block_size`, and `prefix_cache_hits++` (zero physical copies).
   - **Immutability safety**: A prefix-indexed block is never mutated in place.
     Writing to a shared block triggers CoW; truncating or writing to an
     exclusively owned block automatically evicts its prefix-cache entry via
     open-addressing backward-shift deletion before any write occurs.
6. **Zero-leak cleanup (`et_paged_kv_seq_free`)**: Freeing all active sequences
   returns `active_seqs == 0`, `used_blocks == 0`, `shared_blocks == 0`,
   `prefix_cached_blocks == 0`, and `free_blocks == total_blocks`.

## Context-bound prefix cache (opt-in C ABI 1.1)

Create a pool with `et_paged_kv_create_with_prefix_context_v1` to bind it to
one KV producer. The constructor copies three caller-supplied 32-byte
fingerprints for the loaded model including weights and adapters, tokenizer
including vocabulary and token-ID mapping, and the complete KV-producing
configuration. Include every configuration input that can change KV, such as
position handling, attention and mask policy, or inference numerics.
Destroy and recreate the pool when the producer changes. Reset clears the
cache and retains the context.

Use `et_paged_kv_hash_token_block_with_context_v1` to make typed keys for full
blocks. Every digest uses the following canonical preimage, with fixed field
widths and little-endian 32-bit integers.

~~~text
SHA256("ETPKV2\0\0" || model[32] || tokenizer[32] || producer_config[32] ||
       predecessor_present[1] || predecessor_digest_or_zeros[32] ||
       LE32(block_size) || LE32(token_ids[0..block_size-1]))
~~~

The predecessor flag is zero for a root and one for a continuation. The
hash helper rejects a predecessor with a different context or block size.
Register and attach with the corresponding `_with_context_v1` functions.
These validate all three fingerprints and the block size before using the
pool's existing prefix index. Context mismatch returns
`ET_PAGED_KV_ERR_CONTEXT` and preserves the sequence, index, refcounts,
physical KV, counters, and caller's hit output.

Each pool accepts the prefix API family selected at creation. The original
constructors accept the raw token hash APIs with their original hash bytes
and behavior. Context-bound pools accept typed keys with the same full-block,
copy-on-write, and eviction rules. A key is caller-controlled data and can
be constructed or copied. The caller is responsible for using authentic
fingerprints, the correct chain for the logical position, and the actual KV
produced from that context and prefix. The digest covers the context and
token chain. The manager compares the supplied fingerprints and relies on
the caller for fingerprint provenance and the association between the key
and the stored KV.

## Exact Paged GQA Causal Decode Attention (`et_paged_kv_decode_attention_v1`)

For each query head `qh` in `[0, num_q_heads)` mapped to KV head
`kv_h = qh / (num_q_heads / num_kv_heads)`, the kernel streams once over
logical blocks `0 .. num_logical_blocks - 1` (resolving each through
`seq->block_table[lb]` and clamping the final block to `seq_len % block_size`)
using single-pass online softmax in IEEE-754 binary64 accumulators (`running_max`,
`running_sum`, `acc[head_dim]`) with `#pragma STDC FENV_ACCESS ON`,
`#pragma STDC FP_CONTRACT OFF`, `-ffp-contract=off -fexcess-precision=standard -fno-fast-math`,
and **zero heap allocations**. Output is staged in a bounded stack buffer and
committed to `out` only after all heads and finite checks pass.

## Historical experiment evidence

The original `cd714a38b` experiment was verified inside the pinned Ubuntu 22.04 container (`eshkol-r0-debug:20260928`,
Clang 21.1.8, pinned Eshkol `fe9dfd52`) via `bash scripts/test-paged-kv.sh`,
`python3 -m unittest -v tests/q0/test_python_isolation.py`, and
`bash scripts/check-ci-topology.sh`. The following hashes and results describe
that historical source tree, not this isolated extraction with ownership fixes:

| Check | Measured result |
|---|---|
| Core deterministic archive (`libeshkol_transformer_paged_kv.a`) | SHA-256 `485cd23caf2db2d391f958c5f0ea46cd1ca551d710be91262c3e44ac0605a65d` |
| Bridge deterministic archive (`libeshkol_transformer_paged_kv_aot.a`) | SHA-256 `1eb51f1ee15ced7e44ef1eade7b2702cfa08f61a77568af1ad16c54f0ee7e090` |
| Symbol audit (`nm -g` exports + `nm -u` imports) | PASS (zero threading/dlopen/exec/IO dependencies) |
| C++17 header & archive linkage (`clang++-21 -std=c++17`) | PASS (normal + ASan/UBSan/LSan) |
| Native C11 suite (CoW, prefix cache, truncate, OOM rollback, negatives, FP-env) | PASS (`max_attn_abs_err = 0.000e+00` vs C binary64 dense reference across 36 GQA checks) |
| Python double-precision dense GQA oracle (`tests/paged_kv/test_paged_kv_oracle.py`) | PASS (8 cases, 984 output values, `max_abs_err = 5.579e-08 <= 1e-6`) |
| Sanitizers (`ASan`, `UBSan`, `LSan detect_leaks=1:halt_on_error=1`) | PASS (C11 + C++17 suites, zero leaks) |
| Compiled Eshkol AOT (`tests/paged_kv/test_paged_kv_aot.esk`, `--strict-types --optimize 0 --no-stdlib`) | Byte-identical ELF & stdout across 2 fresh builds; ELF SHA-256 `9d6df6c14badec39e0fa0572985b330f779fb531163d2980320ec4d8814389b1` |
| AOT CoW / prefix / GQA parity / truncate / OOM / cleanup | `forks=3 cow_copies=3 shared_blocks=1 prefix_hits=1`, 8 GQA parity checks `max_abs_err = 3.173e-08 <= 1e-6`, truncate `13 -> 9` bitwise restore PASS, OOM & negatives PASS, cleanup `free_blocks=10/10` PASS |
| Python isolation & CI topology gates | PASS (`5/5` unittest OK; `23` shared suites, `38`-command coverage) |

## Isolated candidate verification (2026-11-25)

On the extracted main-based candidate, the pinned Ubuntu 22.04/Clang 21.1.8/
Eshkol `fe9dfd52` toolchain verification passed. The deterministic archive and
symbol audit passed with core SHA-256 `545b06bf3502080537e3dba7c77f54d39c0e38595d928b2597b756679640f22a`
and bridge SHA-256 `16a889e17f91034b1217edab2089db10ab4473ea9b2f9f9dc6e304c0c2bba351`.
Native C11 tests and the eight-case/984-value Python f64 oracle passed; the
maximum oracle absolute error was `5.579e-08`. Every create-time `calloc`
cut was forced in pool-owned, external-pool, and AOT-bridge creation, with
unchanged caller outputs, unchanged external storage, and sanitizer-clean
cleanup. The native C11, allocation-cut, and C++17 header/linkage suites
passed ASan/UBSan/LSan. The pinned rerun on clean commit `ceeb74f3` and tree
`94b8db33` also passed both strict AOT builds, which produced identical ELF
and stdout. The ELF SHA-256 is
`0d4f5c5333bef13bb445757fc4656d82bc4c5397c59e5796951bf3322a87648d`.
The fixtures covered CoW, prefix sharing, eight GQA parity checks, bitwise
truncate restore, exhaustion, invalid inputs, and full pool cleanup.
The recorded maximum GQA absolute error was `3.174e-08` rounded up.
The gate seal is
`/home/gabe/.codex/evidence/eshkol-transformer/paged-kv-exact-ceeb74f3.NVUTKL/SHA256SUMS`
with SHA-256 `b79616cc9898ffe89ef3c029025bd7a25c521223579950b68d1bcd63afed5bda`.

## Fixed model attention witness

`make test-paged-kv-model-attention` runs the private G3-C4 fixed full-prefix
model twice on each of two four-token prompts. Both routes use the same seeded
model owner and the same forward schedule. A test-only link wrapper selects
dense attention for the reference and paged attention for the comparison.
The paged route consumes the model's real Q, K and V views of shape
`[1, 2, 4, 2]` and checks contiguous positions and the triangular causal mask.
It appends one K/V row at a time to two physical blocks, decodes at the
current prefix length, and passes the resulting attention through the model's
unchanged output projection, feed-forward block, normalization and tied head.

Optimized and ASan/UBSan/LSan runs on Ubuntu 22.04 and Clang 21.1.8 matched
stdout. The gate compared 32 attention values and 2,048 logits. Maximum absolute
errors against the dense route were `2^-28` and `2^-26`, respectively.
Changing the second block of tokens preserved all 512 prefix logits bit for
bit, and an out-of-vocabulary token left the destination intact before
attention. Each run released both paged blocks.

The witness covers one layer with two equal query and KV head counts.
Persistent generation cache ownership, batching, grouped-query model
dispatch, and serving performance require separate qualification.

## Limitations and scope

- Bounded CPU single-threaded binary32 KV storage with binary64 online-softmax
  accumulation (`num_blocks <= 4096`, `num_layers <= 64`, `num_kv_heads <= 64`,
  `num_q_heads <= 64`, `block_size <= 64`, `head_dim <= 256`, `max_seqs <= 256`,
  pool <= 256 MiB).
- Each pool is scoped to one KV producer identity. Legacy pools rely solely
  on caller scoping. Context-bound pools check caller-supplied fingerprints
  without attesting to the producer or stored KV.
- This manager and AOT verifier are opt-in. They are not connected to default
  model dispatch, and this gate does not measure live serving performance.
- GPU paged-attention kernels, quantized KV caches (FP8/INT8/INT4), sliding
  window eviction, and multi-threaded concurrent mutation are outside this scope.
- `tests/paged_kv/test_paged_kv_oracle.py` is a development-only verification
  oracle; all delivered KV-cache management, CoW sharing, prefix hashing,
  decode attention, and AOT execution paths are Eshkol/C-native.
