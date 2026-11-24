# Unigram LM / SentencePiece-Style Viterbi Lattice Tokenizer & Forward-Backward EM Trainer

## Overview

This module provides a standalone, deterministic, C11/C++17-compatible byte-complete Unigram Language Model (SentencePiece / Kudo 2018) tokenizer engine, exact Viterbi lattice DP encoder, exact log-sum-exp Forward-Backward marginal likelihood and expected-count engine, deterministic Kudo (2018) Forward-Filtering Backward-Sampling (FFBS) subword regularizer, EM vocabulary trainer with per-piece pruning loss, versioned SHA-256 checksummed binary artifact format (`ETUG1000` v1.0), and compiled Eshkol AOT bridge (`--strict-types --optimize 0 --no-stdlib`).

Delivered files:
- Public C11/C++17 ABI: `include/eshkol_transformer/unigram_tok_abi.h`
- Core implementation: `native/unigram_tok.c`
- Eshkol AOT bridge: `native/unigram_tok_eshkol_bridge.h`, `native/unigram_tok_eshkol_bridge.c`
- Native C11, C++17, Python binary64 oracle, and compiled Eshkol AOT test suites:
  - `tests/unigram_tok/header_unigram_cpp.cpp`
  - `tests/unigram_tok/test_unigram_tok_native.c`
  - `tests/unigram_tok/test_unigram_tok_oracle.py`
  - `tests/unigram_tok/test_unigram_tok_aot.esk`
- Build and verification scripts: `scripts/build-unigram-tok.sh`, `scripts/test-unigram-tok.sh`

## Floating-Point Determinism & Environment Contract

- Compiled with `-std=c11 -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard` and `#pragma STDC FENV_ACCESS ON` + `#pragma STDC FP_CONTRACT OFF`.
- Compile-time `_Static_assert` requires `CHAR_BIT == 8`, IEEE-754 `binary64` (`DBL_MANT_DIG == 53`, `DBL_MAX_EXP == 1024`), and `FLT_EVAL_METHOD == 0`.
- Every floating-point entry point checks `FE_TONEAREST` and x87/MXCSR default control masks on x86_64 (`(x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u`), returning `ET_UNIGRAM_ERR_FP_ENV` (`-13`) on mismatch with zero silent fallbacks.
- All scores and DP accumulators reject NaN and non-finite results explicitly (`ET_UNIGRAM_ERR_NONFINITE`, `-12`); piece log-probabilities must satisfy `isfinite(log_prob) && log_prob <= 0.0`.

## Byte-Complete Lossless Vocabulary Contract

- **Base byte fallback pieces (`0 .. 255`)**: Every raw byte value `b` permanently occupies token ID `b` with 1-byte spelling `[b]`. Single-byte fallback pieces are **never pruned** and are strictly validated on model creation and deserialization.
- **Learned multi-byte pieces (`256 .. active_count - 1`)**: Unique raw byte sequences of length in `[2, 256]`. After training, learned pieces are canonically ordered by descending `log_prob`, tie-broken by ascending `(piece_len, raw_bytes)`.
- **Optional special tokens (`active_count .. vocab_size - 1`)**: Configured via `special_mask` (`ET_UNIGRAM_SPECIAL_BOS`, `EOS`, `PAD`, `UNK`) in canonical contiguous order `(<bos>, <eos>, <pad>, <unk>)`; absent specials report `ET_UNIGRAM_TOKEN_NONE` (`UINT32_MAX`, `-1` in the `int64_t` AOT bridge). Special tokens decode to 0 bytes.
- **Lossless guarantee**: Because all 256 single bytes are always present in the byte-trie, every arbitrary byte string $X \in \{0, \dots, 255\}^N$ (empty input, embedded NULs `0x00`, all 256 byte values, valid multibyte UTF-8, invalid UTF-8 sequences, and arbitrary binary blobs) has a connected segmentation lattice and decodes byte-for-byte identically to $X$.

## Mathematical Specification

Let active pieces be $V = \{0, \dots, V_{\text{active}} - 1\}$ with byte spellings $s_i$ (length $\ell_i \ge 1$) and log-probabilities $\theta_i = \log P(i) \le 0$.

### 1. Exact Viterbi Lattice DP (`et_unigram_encode_v1`)

For input bytes $X[0..N-1]$, initialize $B[0] = 0.0$ and $B[1..N] = -\infty$. For each position $p = 0, \dots, N-1$ and every trie-matched piece $i$ with $X[p..p+\ell_i-1] = s_i$:

$$B[p + \ell_i] \leftarrow \max\big(B[p + \ell_i],\; B[p] + \theta_i\big)$$

**Strict Deterministic Total-Order Tie-Breaking** at end position $p + \ell_i$:
1. Strictly higher cumulative log-probability score $B[p] + \theta_i$,
2. Longer piece length $\ell_i$ (fewer tokens),
3. Smaller token ID $i$.

Backtracking from $N$ to $0$ yields the globally optimal segmentation $x^* = \arg\max_{x \in \mathcal{S}(X)} \sum_j \theta_{x_j}$.

### 2. Exact Forward-Backward Marginal DP (`et_unigram_forward_backward_v1`)

Using numerically stable pairwise `log_add_exp(a, b) = max(a, b) + log1p(exp(-|a - b|))`:

- **Forward pass**: $\alpha[0] = 0.0$, $\alpha[p + \ell_i] \leftarrow \text{log\_add\_exp}(\alpha[p + \ell_i], \alpha[p] + \theta_i)$ for $p = 0 \dots N-1$. Marginal log-likelihood: $\log Z = \alpha[N]$.
- **Backward pass**: $\beta[N] = 0.0$, $\beta[p] \leftarrow \text{log\_add\_exp}(\beta[p], \theta_i + \beta[p + \ell_i])$ for $p = N-1 \dots 0$.
- **Exact expected counts**:

$$\mathbb{E}[c_i] = \sum_{\substack{0 \le p < N \\ X[p..p+\ell_i-1] = s_i}} \exp\big(\alpha[p] + \theta_i + \beta[p + \ell_i] - \log Z\big) = \frac{\partial \log Z}{\partial \theta_i}$$

Conservation identity: $\sum_{i=0}^{V_{\text{active}}-1} \mathbb{E}[c_i] \cdot \ell_i = N$.

### 3. EM Vocabulary Trainer & Iterative Pruning (`et_unigram_train_v1`)

1. **Seed extraction**: Count raw byte frequencies and all contiguous multi-byte substrings of length $L \in [2, \text{max\_piece\_bytes}]$ across corpus documents (never crossing document boundaries) with `count >= min_frequency`. Sort candidates by descending `(count * len, count)`, ascending `(len, raw_bytes)`, keeping up to `seed_vocab_size - 256` multi-byte pieces. Initialize $\theta_i = \log(\text{count}_i + \alpha) - \log\sum_j(\text{count}_j + \alpha)$ and log-sum-exp normalize.
2. **M-step modes** (with Dirichlet smoothing pseudocount $\alpha$):
   - `ET_UNIGRAM_MSTEP_MLE` ($\alpha \in [10^{-6}, 10.0]$):
     $$u_i = \log(\mathbb{E}[c_i] + \alpha) - \log \sum_j (\mathbb{E}[c_j] + \alpha), \qquad \theta_i = u_i - \text{logsumexp}(u)$$
   - `ET_UNIGRAM_MSTEP_DIGAMMA_VB` ($\alpha \in [0.1, 10.0]$, using exact binary64 recurrence + 7-term Bernoulli series for $\psi(x)$):
     $$u_i = \psi(\mathbb{E}[c_i] + \alpha) - \psi\!\left(\sum_j (\mathbb{E}[c_j] + \alpha)\right), \qquad \theta_i = u_i - \text{logsumexp}(u)$$
3. **Per-piece pruning loss ($\Delta \mathcal{L}_i$)**: For each non-byte piece $i \ge 256$, compute its best alternative Viterbi segmentation score $\text{alt\_score}[i]$ on $s_i$ with piece $i$ excluded from the lattice:
   $$\Delta \mathcal{L}_i = \mathbb{E}[c_i] \cdot \big(\theta_i - \text{alt\_score}[i]\big)$$
   Retain all 256 single-byte fallbacks plus the top $\max(\text{target\_extra}, \min(\text{cur\_extra} - 1, \lfloor \text{cur\_extra} \cdot \text{shrinking\_factor} \rfloor))$ non-byte pieces ordered by descending $(\Delta \mathcal{L}_i, \mathbb{E}[c_i])$, ascending $(\ell_i, s_i)$.
4. **Final EM polish**: Run `final_em_iters` EM updates on the pruned vocabulary, sort learned pieces canonically by descending $\theta_i$ then ascending $(\ell_i, s_i)$, log-sum-exp normalize so $\sum_i \exp(\theta_i) = 1.0$, and evaluate final corpus marginal log-likelihood.

### 4. Kudo (2018) Subword Regularization (`et_unigram_sample_v1`)

Runs temperature-scaled forward filtering ($\alpha_{\text{temp}} \cdot \theta_i$, $\alpha_{\text{temp}} \in (0, 1000]$) and backward sampling from $N$ to $0$ using deterministic SplitMix64 PRNG seeded by `seed`, always emitting a valid lossless lattice segmentation.

## Binary Artifact Format (`ETUG1000` v1.0)

All multi-byte fields are unsigned little-endian (including IEEE-754 `binary64` bit patterns) serialized via explicit byte shifts without pointer aliasing or struct padding.

| Offset | Size | Field | Invariant |
| --- | ---: | --- | --- |
| `0` | 8 | `magic` | Exact ASCII `"ETUG1000"` (`ET_UNIGRAM_ERR_MAGIC`) |
| `8` | 2 | `version_major` | `1` (`ET_UNIGRAM_ERR_VERSION`) |
| `10` | 2 | `version_minor` | `0` (`ET_UNIGRAM_ERR_VERSION`) |
| `12` | 4 | `flags` | Must be `0` in v1.0 (`ET_UNIGRAM_ERR_FORMAT`) |
| `16` | 4 | `vocab_size` | `active_count + popcount(special_mask)`, in `[256, 4096]` |
| `20` | 4 | `active_count` | `256 + learned_count`, in `[256, 4096]` |
| `24` | 4 | `special_mask` | Subset of `0x0F` |
| `28` | 4 | `bos_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `32` | 4 | `eos_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `36` | 4 | `pad_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `40` | 4 | `unk_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `44` | 4 | `pieces_blob_bytes` | Exact byte length of serialized piece records |
| `48` | `pieces_blob_bytes` | `pieces[active_count]` | Per piece: `u64` LE binary64 `log_prob` (finite, `<= 0.0`) + `u16` LE `piece_len` + raw bytes. `0..255` must be `[0x00]..[0xFF]`; `>= 256` length `[2, 256]`; strictly unique. |
| `48 + pieces_blob_bytes` | 32 | `sha256` | Raw SHA-256 over bytes `[0 .. 48 + pieces_blob_bytes - 1]` |

### Atomic Persistence & Hardened Reads

- **Save (`et_unigram_save_file_v1`)**: Rejects existing symlink or non-regular targets via `lstat`; writes a non-executable (`0600`) sibling temp file via `mkstemp`, calls `fsync`, re-checks `lstat`, atomically `rename`s into place, and `fsync`s the parent directory.
- **Load (`et_unigram_load_file_v1`)**: Opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`, verifies `S_ISREG` and bounded file size before allocation, validates exact size, magic, version, constant-time SHA-256, canonical special IDs, byte-fallback completeness, piece uniqueness, and finite non-positive log-probabilities before publishing the handle.

## Measured Verification Evidence

Validated inside the pinned container `eshkol-r0-debug:20260928` (`clang-21` / `clang++-21`, pinned Eshkol toolchain `/tc/build/eshkol-run`):

| Check | Result | Measured Value |
| --- | --- | --- |
| Core archive (`normal`, byte-identical across 2 builds) | PASS | `94b35109d920c171513f049a85eab532c4a8f0b1978bf6b48adb45f7434a64c3` |
| Bridge archive (`normal`, byte-identical across 2 builds) | PASS | `ee347edd04e80f65f48e3ecbcf33867a561d957dcdd7c17635069e72b0a4bc74` |
| Core archive (`sanitize`, byte-identical across 2 builds) | PASS | `947e73b9c499128d7ad04a8a3732a32327aa80fc4ac1064f62026f3015dc337e` |
| Bridge archive (`sanitize`, byte-identical across 2 builds) | PASS | `65e43513aafa90534160fac6ff9660f3457f1b6d66e52fadabe9948a668e7394` |
| C++17 header & archive linkage (`clang++-21 -std=c++17`) | PASS | `header_unigram_cpp` exit 0 |
| Native C11 normal + ASan/UBSan/LSan (`detect_leaks=1`) | PASS | Brute-force Viterbi/FB parity, central FD `< 1e-6`, MLE & Digamma VB EM, FFBS, all-256/NUL/UTF-8, tamper/re-signed semantic/symlink/FP-env traps |
| Independent Python binary64 oracle (`test_unigram_tok_oracle.py`) | PASS | 4/4 suites: bitwise-identical piece `log_prob`, Viterbi IDs/score, FB `log Z` & `E[c_i]`, and `ETUG1000` bytes/SHA-256 |
| Eshkol AOT binary (`--strict-types --optimize 0 --no-stdlib`, 2 fresh builds) | PASS | Byte-identical ELF SHA-256: `83e02b5fecfc783e3101177a852f1ae11773e0716483c7bab101a20c128790ab` |
| AOT training & EM convergence | PASS | `vocab=320` (`256` bytes + `64` learned), `seed_active=1024`, `prune_rounds=9`, `em_iters=23`, `em_start_ll=-127.75980544487277` -> `em_end_ll=-127.54224242913848` |
| AOT Viterbi compression & fresh-process reload parity | PASS | `80` raw bytes -> `5` tokens (**16.00x** compression); `vscore=-14.711950066356186`, `fb_log_z=-14.711949966397963`, `tokens=256,263,257,260,258` identical across train & reload |
| AOT Forward-Backward central FD gradient check | PASS | `fd_max_err = 2.5247643975490064e-09` (`< 1e-6`) |
| AOT trained `ETUG1000` SHA-256 (train == reload) | PASS | `4719b7fc3b4b73fd2cd4252951ea3a5177b243ff6b0cc782961fc25883543618` |
| AOT lossless all-256 + NUL + invalid/valid UTF-8 round-trip | PASS | `277` bytes decoded byte-for-byte identically |
| AOT negative & fresh-process artifact rejection checks | PASS | `vocab<256` (`-1`), OOB decode (`-10`), non-finite (`-12`), FP-env (`-13`), checksum (`-8`), truncated (`-7`), bad magic (`-5`), bad version (`-6`), symlink (`-11`) |
| `tests/q0/test_python_isolation.py` & `scripts/check-ci-topology.sh` | PASS | 5/5 isolation tests, 23 shared CI suites / 38-command coverage |

## Measured Limitations & Explicit Boundaries

- CPU-only implementation (no GPU execution claimed).
- Vocabulary ceiling is `4096` tokens (`256` base bytes + learned pieces + up to `4` special tokens); seed vocabulary ceiling is `16384`; individual piece spellings are capped at `256` bytes (`64` bytes during seed extraction); single-call input and training corpus sizes are bounded at `1 MiB` (`1,048,576` bytes).
- Byte-level segmentation operates directly on raw bytes without Unicode NFKC normalization or whitespace metacharacter rewriting.
