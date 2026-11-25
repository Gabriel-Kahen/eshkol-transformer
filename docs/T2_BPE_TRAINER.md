# T2 Deterministic Byte-Level BPE Trainer & Eshkol AOT Bridge

## Overview

This module provides a deterministic, C11/C++17-compatible byte-level Byte-Pair Encoding (BPE) tokenizer trainer, exact lossless encode/decode engine, versioned SHA-256 checksummed binary artifact serializer (`ETBPE100` v1.0), and a compiled Eshkol AOT bridge (`--strict-types --optimize 0 --no-stdlib`).

This is an opt-in native/AOT codec, separate from the accepted Wave-2 T2
tokenizer. For raw byte documents within the accepted T2 limits (at most 256
merges, 65,536 training bytes, 4,096 documents/chunks, and 65,536 encoded
input bytes) and native `uint32_t` minimum frequencies, the merge IDs and
ordinary encoded token IDs follow the same training and rank-order rules. The
oracle test compares them directly against `tests/t2/reference.py`.
`ETBPE100` is a binary format; the
accepted `eshkol-bpe-tokenizer` v1.0 artifact is canonical ASCII TSV. Neither
loader accepts the other's artifact, and the two SHA-256 identities have
different domains. There is no conversion, public tokenizer registration, D1
composition, or streaming bridge. Native BOS/EOS/PAD/UNK are fixed optional
omitted specials; accepted T2 uses named specials with `omit` or `error` decode
policy and optional prefix/suffix lists. The native path has no strict UTF-8
mode and allows up to 3,840 merges, beyond the accepted T2 ceiling.

Delivered files:
- Public C11/C++17 ABI: `include/eshkol_transformer/t2_bpe_trainer_abi.h`
- Core implementation: `native/t2_bpe_trainer.c`
- Eshkol AOT bridge: `native/t2_bpe_trainer_eshkol_bridge.h`, `native/t2_bpe_trainer_eshkol_bridge.c`
- Native C11, C++17, Python oracle, and Eshkol AOT test suites:
  - `tests/t2_trainer/test_t2_trainer_native.c`
  - `tests/t2_trainer/header_trainer_cpp.cpp`
  - `tests/t2_trainer/test_t2_trainer_oracle.py`
  - `tests/t2_trainer/test_t2_trainer_aot.esk`
- Build and validation scripts: `scripts/build-t2-trainer.sh`, `scripts/test-t2-trainer.sh`

## Vocabulary & Deterministic Training Algorithm

### Token ID Layout
- **Base byte vocabulary (`0..255`)**: Every raw byte value `b` has fixed token ID `b` and 1-byte spelling `[b]`.
- **Learned merges (`256 .. 256 + M - 1`)**: Merge rank `r` (`0 <= r < M`) merges operands `(left, right)` with `left < 256 + r` and `right < 256 + r` into `new_id = 256 + r`.
- **Optional special tokens (`256 + M .. 256 + M + S - 1`)**: Configured via `special_mask` (`ET_T2_BPE_SPECIAL_BOS`, `EOS`, `PAD`, `UNK`). Enabled specials receive contiguous IDs immediately after learned merges in canonical order `(<bos>, <eos>, <pad>, <unk>)`; absent specials have sentinel ID `UINT32_MAX` (`-1` in the `int64_t` AOT bridge).
- **Vocabulary bounds**: `target_vocab_size` must satisfy `256 + popcount(special_mask) <= target_vocab_size <= 4096`; thus learned merges plus enabled specials cannot exceed 3,840. `min_frequency` is a `uint32_t` and must be at least 1.

### Training Loop & Canonical Tie-Breaking
Training accepts `doc_count` raw byte documents. Each document is initialized to its raw byte IDs `0..255`. Document boundaries never contribute adjacent pairs; empty documents are valid.

For each merge rank `r = 0 .. (target_vocab_size - 256 - S) - 1`:
1. Count every adjacent token-ID pair `(s[i], s[i+1])` within each document using checked `uint64_t` counters in a generation-stamped open-addressed hash table.
2. Filter pairs requiring `count >= min_frequency` and `decoded_len(left) + decoded_len(right) <= 256` (`ET_T2_BPE_MAX_TOKEN_BYTES`, preventing exponential token-length blowup).
3. Select the winning pair maximizing `count`, breaking ties deterministically by **lowest unsigned lexicographic `(left_id, right_id)`** (smallest `left_id`, then smallest `right_id`). Because every active pair is compared under this total order, hash-table iteration order never affects selection.
4. If no eligible pair exists, training stops early cleanly with `vocab_size = 256 + M + S`.
5. Otherwise assign `new_id = 256 + r`, record `(left, right, new_id)`, and replace all non-overlapping occurrences of `(left, right)` left-to-right in place within every document.

## Exact Encode, Decode, and Lossless Round-Trip Proof

### Encode Semantics
`et_t2_bpe_encode` converts input bytes `[b_0, ..., b_{n-1}]` to initial IDs `[b_0, ..., b_{n-1}]` and repeatedly locates the lowest-rank applicable merge present among adjacent pairs (via an `O(1)` pair-to-rank hash table), replacing all non-overlapping left-to-right occurrences of that merge pair.

**Equivalence to rank-order sweep**: Every merge `r` produces `new_id = 256 + r`, which is strictly greater than both operands (`left, right < 256 + r`) and greater than all earlier merge outputs. Therefore, applying merge `r` can never create a pair matching any earlier merge `r' < r`. Applying the minimum-rank present pair at each step produces the exact same token sequence as sweeping `r = 0 .. M - 1` sequentially, while requiring at most `min(len - 1, M)` iterations.

### Lossless Round-Trip Proof
Let `S(id)` denote the decoded byte string of token `id`:
- Base case (`0 <= id < 256`): `S(id) = [id]`, length `1`.
- Inductive step (`id = 256 + r` from `(left, right)`): `S(id) = S(left) || S(right)`, precomputed flat into `vocab_bytes` during `build_model` (zero recursion during decode).
- Initial state of `encode(X)`: `concat_i S(t_i) = [X_0] || ... || [X_{n-1}] = X`.
- Merge invariant: Replacing any adjacent pair `(left, right)` with `new_id` substitutes `S(left) || S(right)` with `S(new_id) = S(left) || S(right)`, leaving `concat_i S(t_i) = X` invariant at every step.
- `decode(encode(X))` concatenates `S(t_i)` for every emitted token `t_i`, yielding `X` byte-for-byte for **all** arbitrary byte sequences `X` (empty input, NUL `0x00`, all 256 byte values `0x00..0xFF`, multi-byte UTF-8, and invalid UTF-8 sequences).

Special tokens (`256 + M <= id < vocab_size`) have 0 decoded bytes and are omitted by `decode`; any `id >= vocab_size` fails explicitly with `ET_T2_BPE_ERR_TOKEN_ID` without modifying the output buffer.

## Binary Artifact Format (`ETBPE100` v1.0)

All multi-byte integers are unsigned little-endian, encoded and decoded via explicit byte shifts (zero misaligned-pointer casts, zero struct-padding dependence). Total size is strictly `76 + 12 * merge_count` bytes (`<= 46,156` bytes).

| Offset | Size | Field | Invariant |
| --- | ---: | --- | --- |
| `0` | 8 | `magic` | Exact ASCII `"ETBPE100"` (`ET_T2_BPE_ERR_MAGIC` on mismatch) |
| `8` | 2 | `version_major` | `1` (`ET_T2_BPE_ERR_VERSION` on mismatch) |
| `10` | 2 | `version_minor` | `0` (`ET_T2_BPE_ERR_VERSION` on mismatch) |
| `12` | 4 | `flags` | Must be `0` in v1.0 |
| `16` | 4 | `vocab_size` | `256 + merge_count + popcount(special_mask)`, in `[256, 4096]` |
| `20` | 4 | `merge_count` | `<= 3840`, file size must equal `76 + 12 * merge_count` |
| `24` | 4 | `special_mask` | Subset of `0x0F` |
| `28` | 4 | `bos_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `32` | 4 | `eos_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `36` | 4 | `pad_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `40` | 4 | `unk_id` | Canonical contiguous ID or `0xFFFFFFFF` |
| `44` | `12 * M` | `merges[M]` | Each `(left, right, new_id)` with `new_id == 256 + r`, `left < new_id`, `right < new_id`, unique `(left, right)`, reconstructed length `<= 256` |
| `44 + 12 * M` | 32 | `sha256` | Raw SHA-256 digest over bytes `[0 .. 44 + 12 * M - 1]` |

### Atomic Persistence & Hardened Reads
- **Save (`et_t2_bpe_save_file`)**: Rejects existing symlink or non-regular destinations via `lstat`; writes a non-executable (`0600`) sibling temp file via `mkstemp`, `fsync`s the file descriptor, re-checks destination `lstat`, atomically `rename`s into place, and `fsync`s the parent directory. A pre-rename I/O failure returns `ET_T2_BPE_ERR_IO`; a parent-directory open/fsync/close failure after rename returns `ET_T2_BPE_ERR_PUBLISHED_UNSYNCED` (`-12`). The latter means the new destination is visible but crash durability is unproven; callers must not retry assuming no publication.
- **Load (`et_t2_bpe_load_file`)**: Opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`, verifies `S_ISREG` and bounded file size before allocation, validates exact length, magic, version, SHA-256 checksum, canonical special IDs, and acyclic/unique/bounded merge references before publishing the model handle.

## Measured Verification Evidence

The table below records the predecessor `23e8fa35` validation inside the
pinned container `eshkol-r0-debug:20260928` (`clang-21` / `clang++-21`, pinned
Eshkol toolchain `/tc/build/eshkol-run`). Its hashes are not successor hashes.
On the successor tree, the focused pinned Clang 21 native regression gate
passes after the durability and overlap fixes. The full successor gate is
queued behind the shared compiler lock.

| Check | Result | Measured Value |
| --- | --- | --- |
| Core archive (`normal`, byte-identical across 2 builds) | PASS | `7ce105cd418a8bf70d6e95919d6bee1ab4da8461ba85a8aa0dd0671d31d01b56` |
| Bridge archive (`normal`, byte-identical across 2 builds) | PASS | `bb04c36c0d47f1ab94505ae3ccaa40c6f862dc4ed7dc9dc4a00a2a93f34d444c` |
| Core archive (`sanitize`, byte-identical across 2 builds) | PASS | `503f0e5c32ac59da43ee19a121a56760f58572b055a48ce37a5a35445297cae0` |
| Bridge archive (`sanitize`, byte-identical across 2 builds) | PASS | `c319a7731f3bb0c000aa9f9c68ba5f008510a91f36b6a826963234c4f7c13d3a` |
| C++17 header & archive linkage (`clang++-21 -std=c++17`) | PASS | `header_trainer_cpp` exit 0 |
| Native C11 normal + ASan/UBSan/LSan (`detect_leaks=1`) | PASS | All determinism, tie-break, round-trip, tamper, re-signed semantic, and symlink checks |
| Independent Python oracle (`test_t2_trainer_oracle.py`) | PASS | Predecessor 4/4 suites: exact merge-by-merge table, token ID sequence, lossless decode, and `ETBPE100` bytes/SHA-256 |
| Eshkol AOT binary (`--strict-types --optimize 0 --no-stdlib`, 2 fresh builds) | PASS | Byte-identical ELF SHA-256: `4f580f65f6221cdcbc34f2066adb0352549dcbdd7d4b5024a85e547f156239fa` |
| AOT train + fresh-process reload compression | PASS | `vocab=320`, `merges=60`, `81` raw bytes -> `18` tokens (**4.50x** compression, 77.78% reduction) |
| AOT lossless all-256 + NUL round-trip | PASS | `264` bytes (NULs + `0x00..0xFF` + invalid UTF-8) decoded byte-for-byte identically |
| AOT trained artifact SHA-256 (train == reload) | PASS | `a10bf0892d61535389fa5fbb80fec5455ea029b03eb4d4c7ee22e170659ec76a` |
| AOT fresh-process rejection checks | PASS | Corrupt checksum (`-8`), truncated (`-7`), bad magic (`-5`), version mismatch (`-6`), symlink (`-11`) |
| `tests/q0/test_python_isolation.py` & `scripts/check-ci-topology.sh` | PASS | 5/5 isolation tests, 23 shared CI suites verified |

## Measured Limitations & Explicit Boundaries

- CPU-only implementation (no GPU execution claimed).
- Vocabulary ceiling is `4096` tokens (`256` base bytes plus at most `3840` combined merges and special tokens); individual merged token spellings are capped at `256` bytes. The extended ceiling above the accepted T2 256-merge limit has not been independently measured by the cross-T2 oracle cases.
- Encoding operates on whole byte buffers in memory (streaming incremental BPE transducers remain in the existing Wave-2 T2 Eshkol contract documented in `docs/BPE_TOKENIZER_FORMAT.md`).
- Decode rejects overlapping token-ID input and byte output; output-count storage must be separate from both buffers.
- Atomic rename durability assumes POSIX same-directory rename semantics without concurrent hostile directory replacement.
