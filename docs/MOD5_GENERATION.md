# MOD5 Autoregressive Generation, KV-Cache, and Deterministic Sampling

This document specifies the MOD5 autoregressive generation runtime
(`include/eshkol_transformer/mod5_generate_abi.h`, `native/mod5_generate.c`),
its flat-scalar Eshkol AOT bridge, and its verified pinned-container evidence.
The archive `build/mod5_gen/libeshkol_transformer_mod5_generate.a` is C11 and
C++17 compatible (`extern "C"`), links `build/mod5/libeshkol_transformer_mod5.a`
(`et_mod5_kernel_provider_v1()`), and uses zero silent fallbacks.

## Bounded Shape and Attention Invariants

| Dimension | Field | Supported Range | Invariant |
|---|---|---|---|
| Vocabulary size | `vocab_size` ($V$) | `1 .. 1024` | `ET_MOD5_GEN_MAX_VOCAB` |
| Maximum sequence length | `max_seq` ($S$) | `1 .. 64` | `ET_MOD5_GEN_MAX_SEQ` ($\le$ `ET_MOD5_MAX_EXACT_POSITION`) |
| Decoder layers | `num_layers` ($L$) | `1 .. 8` | `ET_MOD5_GEN_MAX_LAYERS` |
| Hidden width | `hidden_dim` ($D$) | `1 .. 32` | `ET_MOD5_GEN_MAX_DIM`, $D \bmod H_q = 0$ |
| SwiGLU FFN width | `ffn_dim` ($F$) | `1 .. 64` | `ET_MOD5_GEN_MAX_FFN` |
| Query heads | `num_heads` ($H_q$) | `1 .. 16` | `ET_MOD5_GEN_MAX_HEADS`, $D_h = D / H_q \ge 1$ |
| Key/Value heads | `num_kv_heads` ($H_{kv}$) | `1 .. H_q` | $H_q \bmod H_{kv} = 0$ (MHA: $H_{kv}=H_q$, GQA: $1<H_{kv}<H_q$, MQA: $H_{kv}=1$) |
| Sliding window | `window_size` | `0` or $\ge 1$ | `0` = full causal; $> 0$ admits key $k$ iff $0 \le \text{qpos} - \text{kpos} < \text{window\_size}$ |
| ALiBi slopes | `use_alibi`, `alibi_slopes[H_q]` | `0` or `1` | When `1`, all $H_q$ slopes must be finite; bias is `slopes[h] * (float)(kpos - qpos)` |
| RMSNorm epsilon | `rms_epsilon` | finite $> 0.0\text{f}$ | Rejects `0.0f`, `-0.0f`, negative, `NaN`, and `Inf` |

All model weights follow canonical MOD5 row-major `[out, in]` layout:
`token_embedding[V, D]`, `norm_final[D]`, `head[V, D]` (may alias
`token_embedding` for tied checkpoints), and per-layer `norm1[D]`, `wq[D, D]`,
`wk[Hkv*Dh, D]`, `wv[Hkv*Dh, D]`, `wo[D, D]`, `norm2[D]`, `wgate[F, D]`,
`wup[F, D]`, `wdown[D, F]`.

## KV-Cache Layout and Failure Atomicity

`et_mod5_gen_kv_cache` owns two dense row-major binary32 buffers (`k_data`,
`v_data`) with physical layout `[L, Hkv, max_seq, Dh]`:

```text
offset(layer, kv_head, pos, d) = (((layer * Hkv + kv_head) * max_seq) + pos) * Dh + d
```

- `seq_len` tracks committed prefix length (`0 <= seq_len <= max_seq`).
- `et_mod5_gen_kv_cache_read_slot` exposes committed `(k, v)` vectors at
  `(layer, kv_head, pos < seq_len)` and rejects out-of-range indices with
  `ET_MOD5_GEN_ERR_BOUNDS`.
- Prefill (`et_mod5_gen_prefill`) and incremental decode
  (`et_mod5_gen_decode_step`) stage new K/V projections and layer activations in
  temporary buffers and commit to `cache` and caller outputs only after all $L$
  layers, final RMSNorm, and head projection produce strictly finite binary32
  results. Any overflow (`ET_MOD5_GEN_ERR_CACHE_FULL`), out-of-bounds token
  (`ET_MOD5_GEN_ERR_BOUNDS`), non-finite intermediate (`ET_MOD5_GEN_ERR_NONFINITE`),
  misaligned pointer (`ET_MOD5_GEN_ERR_UNALIGNED`), or unsupported x86-64
  x87/MXCSR floating-point environment (`ET_MOD5_GEN_ERR_FP_ENV`) leaves the
  cache and caller buffers unmodified.

## MOD5 K1 Provider Linkage and Serial Binary32 Semantics

`et_mod5_gen_engine_create` discovers `et_mod5_kernel_provider_v1()` via
`et_kernel_runtime_discover` and retains the K1 runtime for the engine lifetime.
When a full-prefix forward pass matches an exact verified MOD5 capability row
(e.g., RMSNorm `[1,8,16]`/`[1,16,32]`, SwiGLU `[1,8,32]`/`[1,16,64]`, Variant
Attention `[1,4,1|2,8,8,4]`/`[1,4,1|2,16,16,8]`), the engine dispatches directly
through `et_kernel_runtime_dispatch`. General shapes (including single-token
decode, MHA, and sequences up to `max_seq = 64`) execute the identical serial
binary32 order with `-ffp-contract=off -fexcess-precision=standard
-frounding-math -fno-fast-math`. `et_mod5_gen_verify_provider` explicitly
cross-checks both paths across RMSNorm, SwiGLU, MQA, and GQA with ALiBi and
sliding windows and requires exact `0.0` maximum absolute difference.

## Deterministic Sampling and Counter-Based RNG

### Counter-Based SplitMix64 RNG (`et_mod5_gen_rng`)

The RNG state is an observable two-word struct `{ uint64_t key; uint64_t counter; }`:

1. `et_mod5_gen_rng_init(rng, seed)` sets `key = seed`, `counter = 0`.
2. Each categorical draw evaluates
   `splitmix64_mix(splitmix64_mix(key ^ 0x243f6a8885a308d3) + (counter + 1) * 0x9e3779b97f4a7c15)`
   and increments `counter` by `1`.
3. Uniform binary64 variates use exact 53-bit scaling:
   `u = (double)(raw >> 11) * (1.0 / 9007199254740992.0)` in `[0.0, 1.0)`.
4. `counter == UINT64_MAX` is an exhausted sentinel that rejects further draws
   with `ET_MOD5_GEN_ERR_OVERFLOW` without mutating state or outputs.
5. Capturing `{key, counter}` at any step and restoring it replays the exact
   remaining token stream bit-for-bit.

### Sampling Semantics (`et_mod5_gen_sample` / `et_mod5_gen_generate`)

All logits must be finite binary32 values (`ET_MOD5_GEN_ERR_NONFINITE`
otherwise).

- **Greedy (`ET_MOD5_GEN_SAMPLE_GREEDY`)**: selects the argmax using strict `>`
  comparison in ascending token ID order, so exact ties (including `-0.0f` vs
  `+0.0f`) deterministically choose the **lowest token ID**. Greedy mode never
  advances `rng->counter` and accepts `rng == NULL`.
- **Categorical (`ET_MOD5_GEN_SAMPLE_CATEGORICAL`)**:
  1. Validates `temperature > 0.0f` (finite), `top_k <= vocab_size` (`0` means
     all $V$), `0.0f < top_p <= 1.0f` (finite), and `eos_token_id` in
     `{-1} U [0, V-1]`.
  2. Computes binary64 shifted softmax `exp(((double)logit[i] - (double)max) / (double)temperature)`
     with explicit finite checks on every intermediate.
  3. Sorts `(prob, id)` using an allocation-free deterministic in-place heapsort
     under the strict total order: descending `prob`, breaking ties by ascending
     `id`.
  4. Applies top-k truncation to `k = (top_k == 0 ? V : top_k)` and renormalizes.
  5. Applies top-p (nucleus) filtering by retaining the shortest prefix of
     length $m \ge 1$ whose cumulative probability reaches `top_p`.
  6. Draws one RNG block and selects the first kept candidate whose cumulative
     mass exceeds `u * sum_p`.
- **EOS early stop**: in `et_mod5_gen_generate`, if `eos_token_id >= 0` and the
  sampled token equals `eos_token_id`, generation records that token, sets
  `stopped_by_eos = 1`, and halts immediately without decoding past EOS.

## Measured Parity and Reproducibility Evidence

Validated inside the pinned container `eshkol-r0-debug:20260928` (`clang-21` /
`clang++-21` 21.1.8, Eshkol toolchain `/tc`, `--network none`) via:

```bash
bash scripts/test-mod5-generate.sh
python3 -m unittest tests/q0/test_python_isolation.py
bash scripts/check-ci-topology.sh
```

### Cached Incremental Prefill+Decode vs Full-Context Recompute Parity

All native cases passed under both normal (`-O2`) and ASan/UBSan/LSan
(`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1`)
modes, alongside the compiled Eshkol AOT executable:

| Suite / Case | Shape (`L, D, F, Hq, Hkv`) | Attention | Split | Measured `max_abs_diff` | 1-by-1 `max_abs_diff` |
|---|---|---|---|---:|---:|
| Native K1 provider parity | MOD5 verified rows | RMSNorm, SwiGLU, MQA/GQA+ALiBi+window | Direct vs serial | `0.000e+00` | — |
| Native `MHA_full_causal` | `2, 16, 32, 4, 4` | MHA, full causal | `5/14` | `0.000e+00` | `0.000e+00` |
| Native `GQA_Hkv2_K1_T8` | `2, 16, 32, 4, 2` | GQA (`Hkv=2`), K1 exact `T=8` | `3/8` | `0.000e+00` | `0.000e+00` |
| Native `GQA_Hkv2_full` | `2, 16, 32, 4, 2` | GQA (`Hkv=2`), full causal | `6/16` | `0.000e+00` | `0.000e+00` |
| Native `MQA_Hkv1_K1_T8` | `2, 16, 32, 4, 1` | MQA (`Hkv=1`), K1 exact `T=8` | `4/8` | `0.000e+00` | `0.000e+00` |
| Native `MQA_Hkv1_full` | `2, 16, 32, 4, 1` | MQA (`Hkv=1`), full causal | `5/15` | `0.000e+00` | `0.000e+00` |
| Native `GQA_ALiBi` | `2, 16, 32, 4, 2` | GQA + ALiBi slopes | `7/18` | `0.000e+00` | `0.000e+00` |
| Native `MHA_ALiBi` | `2, 16, 32, 4, 4` | MHA + ALiBi slopes | `4/16` | `0.000e+00` | `0.000e+00` |
| Native `GQA_ALiBi_window1` | `2, 16, 32, 4, 2` | GQA + ALiBi + `window=1` | `5/12` | `0.000e+00` | `0.000e+00` |
| Native `GQA_ALiBi_window3` | `2, 16, 32, 4, 2` | GQA + ALiBi + `window=3` | `4/16` | `0.000e+00` | `0.000e+00` |
| Native `GQA_window5` | `2, 16, 32, 4, 2` | GQA + `window=5` | `6/19` | `0.000e+00` | `0.000e+00` |
| Native `MAX_L8_D32_F64_S64_win16` | `8, 32, 64, 4, 2` | Max bounds + ALiBi + `window=16` | `16/64` | `0.000e+00` | `0.000e+00` |
| Eshkol AOT `PARITY_T8` | `1, 16, 32, 4, 2` | Trained C1 reload, K1 `T=8` | `4/8` | `0.000e+00` | — |
| Eshkol AOT `PARITY_GEN8` | `1, 16, 32, 4, 2` | Trained C1 reload, 8 decode steps | `4/11` | `0.000e+00` | — |

### Artifact and Fresh-Process Generation Digests

| Artifact / Output | Details | SHA-256 |
|---|---|---|
| `build/mod5_gen/libeshkol_transformer_mod5_generate.a` | Deterministic normal archive (two fresh builds `cmp` identical) | `3994db6859c05c77fac47f6ebd4fa3979a8c05fd141c6649a1f22d8b15d5a576` |
| Compiled AOT ELF (`tests/mod5/test_mod5_generate_aot.esk`) | `--strict-types --optimize 0 --no-stdlib`, two fresh dirs `cmp` identical | `282785c7960fa0699973f5fa68a06457bc5ad82ef50f0b5f78f88914d8832630` |
| Trained `eshkol-mod5-v1` C1 checkpoint | 12 SGD steps on D1 fixture (`loss 2.7694 -> 1.5492`) | `362ad110daf3856879bdab074dc6b257771040b04a93901a05c95e435836bc87` |
| AOT Greedy generation (`tokens=9,5,9,5,9,5,9,5`) | Identical across fresh processes and seeds (`rng_counter=0`) | `22d7b2fd84589829944c981ebb732f709ae5653fba3ce582a25e4dabae099fcc` |
| AOT Sampled (`seed=42`, `T=0.85, top_k=6, top_p=0.90`) | `tokens=3,2,11,8,5,15,11,14`, `rng_counter=8`, identical across 2 fresh processes | `4267de9852cee4721e0e1960b09822e344a5ce270dfb605dcc7b4a7431766fe2` |
| AOT Sampled (`seed=1337`, `T=0.85, top_k=6, top_p=0.90`) | `tokens=10,12,14,6,2,6,4,1`, `rng_counter=8`, identical across 2 fresh processes | `9ebcbb7d7c40b1778f557fa118d0d1b3c8cec82ee099ee9a69b3ae883aa2d719` |

Corrupting a single byte of the C1 checkpoint causes fresh-process reload to
fail closed (`rc=2`, `MOD5 GEN ERROR corrupt-data checkpoint checksum mismatch`)
without emitting any generated tokens.

## Scope and Measured Limitations

- Supported execution is single-batch (`N=1`), serial CPU binary32 on the
  reviewed x86-64 IEEE-754 lane (`FE_TONEAREST`, masked exceptions, no FTZ/DAZ).
- Bounds are enforced explicitly up to `max_seq <= 64`, `D <= 32`, `F <= 64`,
  `L <= 8`, `Hq <= 16`, `V <= 1024`; larger shapes reject with
  `ET_MOD5_GEN_ERR_CONFIG`.
- GPU acceleration, batched KV-caches (`N > 1`), beam search, quantized weights,
  and throughput benchmarks are outside this scope.
