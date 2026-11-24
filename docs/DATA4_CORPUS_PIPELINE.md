# DATA4 Corpus Quality-Filtering, Deduplication, and Provenance Pipeline

Status: implemented on `codex/data4-dedup-quality-pipeline` with native C11/C++17,
ASan/UBSan/LSan, independent Python oracle, and compiled Eshkol AOT evidence in the
pinned `eshkol-r0-debug:20260928` container. This module is additive and does not
modify `Makefile`, `docs/ROADMAP.md`, or existing DATA4/CLI3 files.

## Public C11/C++17 ABI and Eshkol AOT Bridge

- Core ABI: `include/eshkol_transformer/data4_pipeline_abi.h` and `native/data4_pipeline.c`
- Eshkol AOT bridge: `native/data4_pipeline_eshkol_bridge.h` and `native/data4_pipeline_eshkol_bridge.c`
- Static archive: `build/data4_pipeline/libeshkol_transformer_data4_pipeline.a` (26 exported `et_data4_*` symbols; deterministic `ar rcsD` archive verified byte-identical across two fresh builds)

All entry points enforce explicit error codes (`et_data4_status`), IEEE-754 binary64
round-to-nearest floating-point checks, integer-overflow checked arithmetic, pointer
alignment validation, and input/output aliasing rejection (`ET_DATA4_ERR_ALIASING`)
with zero silent fallbacks.

## Pipeline Stage Semantics

Documents are supplied with unique ASCII `source_id` metadata (`[A-Za-z0-9._-]`,
length `1..64`) and raw byte spans. Invalid or duplicate source IDs fail upfront
(`ET_DATA4_ERR_INVALID_SOURCE_ID`, `ET_DATA4_ERR_DUPLICATE_SOURCE_ID`). Empty
corpora (`doc_count == 0`) succeed with zeroed counts and a verifiable report.

Documents are evaluated strictly in input order. Rejected documents never enter
the exact-deduplication or LSH near-deduplication indices, so a rejected document
cannot suppress a later passing document.

1. **Strict UTF-8 validation (`rejected_utf8`)**: RFC 3629 byte decoding rejecting
   overlong 2/3/4-byte encodings (`0xC0..0xC1`, `0xE0 0x80..0x9F`, `0xF0 0x80..0x8F`),
   UTF-16 surrogates (`U+D800..U+DFFF`), codepoints above `U+10FFFF`, stray
   continuations, and truncated sequences. Counts total codepoints, printable
   codepoints, and alphanumeric codepoints deterministically without libc locale
   dependence:
   - Printable: `\t`, `\n`, `\r`, `0x20..0x7E`, and `cp >= 0xA0` excluding
     noncharacters (`U+FDD0..U+FDEF`, `U+xxFFFE`, `U+xxFFFF`).
   - Alphanumeric: ASCII `[A-Za-z0-9]` plus deterministic Latin-1/Extended, Greek,
     Cyrillic, Kana, CJK Unified Ideographs, and Hangul Syllable ranges.
2. **Length bounds (`rejected_length`)**: Inclusive bounds on both raw byte length
   (`[min_bytes, max_bytes]`) and decoded Unicode scalar count
   (`[min_codepoints, max_codepoints]`).
3. **Quality ratios (`rejected_quality`)**: Rejects when
   `printable_count / codepoint_count < min_printable_ratio` or
   `alnum_count / codepoint_count < min_alnum_ratio` (with `0.0` for zero codepoints).
4. **Repetition ratios (`rejected_repetition`)**:
   - **Duplicate-line ratio**: POSIX `\n`-delimited lines (a single trailing `\n`
     does not create a phantom empty line); `(total_lines - unique_lines) / total_lines`.
   - **Repeated n-gram ratio**: sliding byte n-grams of order `rep_ngram_size`
     (`1..64`); `(total_windows - unique_windows) / total_windows`.
   Rejects if either exceeds `max_dup_line_ratio` or `max_repeated_ngram_ratio`.
5. **Exact deduplication (`duplicate_exact`)**: Open-addressed hash table keyed by
   canonical FIPS 180-4 SHA-256 over unmodified raw bytes, confirmed by exact byte
   length and `memcmp`. First-seen kept document is retained as canonical.
6. **Near-deduplication (`duplicate_near`)**: Deterministic seeded MinHash + LSH
   banding over canonical sorted unique byte `dedup_ngram_size`-grams:
   - Base 64-bit shingle hash `h0`: FNV-1a 64-bit followed by SplitMix64 mix.
   - `K` permutations (`1..256`) generated from `minhash_seed` as bijective affine
     maps `(a_k * h0 + b_k) mod 2^64` (odd `a_k`) followed by SplitMix64 mix.
   - `B` LSH bands (`1 <= B <= K`, `K % B == 0`, `R = K / B` rows per band) with
     exact `R`-word slice comparison on bucket collisions.
   - All distinct LSH candidates across bands are sorted ascending by original
     document ordinal and verified with **exact set Jaccard** (`|A ∩ B| / |A ∪ B|`)
     via linear two-pointer merge over sorted shingle sets. Because every candidate
     undergoes exact Jaccard verification, LSH hash collisions never cause
     false-positive drops. The earliest kept document meeting
     `jaccard_num / jaccard_den >= near_dup_jaccard_threshold` is recorded with
     its canonical `source_id` and exact `jaccard_num` / `jaccard_den`.

## Canonical Provenance Report Format

`et_data4_report_format` emits a versioned, non-executable, LF-terminated ASCII TSV
report verified by `et_data4_report_verify`:

- Fixed header order: `format\teshkol-data4-corpus-pipeline-report`,
  `version\t1\t0`, all configuration fields (floating-point thresholds encoded as
  exact 16-char lowercase IEEE-754 binary64 hex), and aggregate document/byte counts.
- One `document` record per input document in ordinal order `0 .. input-documents - 1`:
  `document\t<ordinal>\t<source_id>\t<byte_count>\t<codepoint_count>\t<sha256_hex>\t<disposition>\t<matched_source_id_or_empty>\t<jaccard_num>\t<jaccard_den>`
- Trailing checksum records:
  - `checksum-algorithm\tsha256`
  - `checksum\t<64 lowercase hex>` covering all bytes preceding `checksum\t`.

`et_data4_report_verify` rejects non-ASCII/CRLF/NUL bytes, wrong format/version
(`ET_DATA4_ERR_VERSION`), tampered checksums (`ET_DATA4_ERR_CHECKSUM`), and
resealed semantic mismatches (count/byte drift, duplicate source IDs, non-kept or
forward canonical references, or below-threshold Jaccard ratios) with
`ET_DATA4_ERR_FORMAT`.

## Measured Validation Results

Validated offline in `eshkol-r0-debug:20260928` (`clang-21`, `clang++-21`, pinned
Eshkol toolchain `/tc`):

- `scripts/build-data4-pipeline.sh` (normal): deterministic archive SHA-256
  `283b70ffe455b72720a948dc32178fac6692c4a1f197a0be4899262b6f63bcd6`, 26/26 exact
  symbols, C++17 header check PASS.
- `scripts/build-data4-pipeline.sh` (sanitize): deterministic archive SHA-256
  `5dac328ad428f03f450b89ef8b521f9732f6352b5e5f6a6c5a79dfb0b470b2a8`, symbol audit PASS.
- `tests/data4/header_pipeline_cpp.cpp`: C++17 static layout/signature assertions
  and runtime linkage PASS.
- `tests/data4/test_data4_pipeline_native.c`: normal (`-O2`) and ASan/UBSan/LSan
  (`detect_leaks=1:halt_on_error=1`) runs produced byte-identical stdout,
  `report.tsv` (1981 bytes), and `minhash.txt` (`input=10 kept=3 utf8=1 len=1
  qual=1 rep=2 exact=1 near=1 near_j=111/113`).
- `tests/data4/test_data4_pipeline_oracle.py`: 3/3 tests PASS (byte-exact report
  parity, bit-exact 64-permutation MinHash parity, tampered and resealed invalid
  report rejection).
- `tests/data4/test_data4_pipeline_aot.esk`: compiled twice from fresh directories
  with `--strict-types --optimize 0 --no-stdlib`; byte-identical ELFs and stdout:
  - `DATA4 AOT CORPUS input=9 kept=2 utf8=1 length=2 quality=1 repetition=1 exact=1 near=1 jaccard=111/113 PASS`
  - `DATA4 AOT REPORT bytes=1857 sha256=dc2dd6feade38c52f0b62fc306f4c3653efc4ceb140af7640fd57d1854ae0b4c tamper=REJECTED PASS`
  - Compiled ELF SHA-256: `59285ff1ac97d080570604eeea1c46b67f012efa77738e191da32456c3e0d3e1`
- `python3 -m unittest tests/q0/test_python_isolation.py`: 5/5 PASS.
- `bash scripts/check-ci-topology.sh`: PASS (23 shared suites, 38 commands).

## Measured Limitations

- CPU-only in-memory corpus preparation; streaming/external-sort sharding over
  multi-terabyte corpora is outside this slice.
- MinHash/LSH recall depends on the configured `(minhash_num_perms, minhash_bands)`
  S-curve; false negatives remain possible when `bands < num_perms` for pairs near
  the threshold, while false positives are strictly eliminated by exact Jaccard
  verification.
