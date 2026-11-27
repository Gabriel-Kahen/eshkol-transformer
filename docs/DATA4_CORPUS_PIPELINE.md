# DATA4 Corpus Quality-Filtering, Deduplication, and Provenance Pipeline

The native library, Eshkol AOT bridge, and same-aggregate T2/D1 adapter
are opt-in build artifacts. The installed CLI3 corpus commands use the
separate [DATA4-A/C report](DATA4_INSTALLED_REPORT.md).

## Same-aggregate T2/D1 publication

Load `t2_wave2_root.esk`, `data4_corpus.esk`, and
`data4_quality_corpus.esk` in that order in a trusted Eshkol aggregate
linked to the DATA4-Q archive and the existing T2/D1 dependencies:

```scheme
(data4-quality-corpus-write! tokenizer directory shard-token-limit documents config)
```

Supply a registered tokenizer from the same aggregate, an existing D1
output directory, and an integer shard limit from 1 to 65,536. The input is a
proper list of 0 to 128 documents of the form `(source-id bytevector)`
totaling at most 65,536 raw bytes. D1 remains the owner of target exclusion
and manifest publication. Staging allows up to 65,536 token IDs, including
any tokenizer-added special tokens.

Pass `#f` for the native configuration defaults or supply
`#(min-bytes max-bytes min-codepoints max-codepoints min-printable-ratio
min-alnum-ratio max-dup-line-ratio max-repeated-ngram-ratio rep-ngram-size near)`.
Explicit byte and codepoint bounds are exact integers from 0 to 65,536, and
the four ratios use the native binary64 constraints documented below. Set
`near` to `#t` to keep native near-dedup defaults, `#f` to disable near
deduplication, or `#(ngram-size permutations bands seed jaccard-threshold)`
to configure it. The bridge takes a signed i64 seed carrying the seed's
64-bit pattern. Construct a custom configuration with `vector` so exact
integers retain their types in the nested vector on the pinned compiler.

The adapter snapshots input IDs and bytevectors, submits them in ordinal
order, and stages exactly the original bytes of each document whose
bridge disposition is `kept`. It verifies the copied quality report and
checks its checksum against the same session's digest. Native session and
buffer ownership ends before token encoding. T2 staging and D1's token
validation complete before D1 publishes any shard or manifest.

Success returns `#(d1-summary quality-report-bytes kept-source-ids)`. Empty
and all-rejected batches produce a manifest with zero shards and preserve their
distinct report rows. D1 stores one flat token stream. The in-memory quality
report and ID list carry source provenance, and the quality report checksum
covers that report alone. The caller owns any optional persistence of this
receipt. The adapter has no installed command or public ABI.

Use `make test-data4-quality-t2-d1` to exercise the adapter in a genuine
same-aggregate AOT executable. The gate verifies first-seen selection and
near/exact duplicate provenance across reordered batches, raw multibyte and
line-ending preservation through T2/D1, empty and all-filtered inputs,
preexisting-target preservation, and release of the native session when
invalid inputs or injected report faults fail before publication.

## Public C11/C++17 ABI and Eshkol AOT Bridge

- Core ABI: `include/eshkol_transformer/data4_pipeline_abi.h` and `native/data4_pipeline.c`
- Eshkol AOT bridge: `native/data4_pipeline_eshkol_bridge.h` and `native/data4_pipeline_eshkol_bridge.c`
- Static archive: `build/data4_pipeline/libeshkol_transformer_data4_pipeline.a` (26 exported `et_data4_*` symbols; deterministic `ar rcsD` archive verified byte-identical across two fresh builds)

The core ABI returns explicit `et_data4_status` codes, checks the binary64
round-to-nearest environment where floating-point decisions occur, checks
arithmetic and typed pointer alignment, and rejects overlapping declared
input/output spans. The AOT bridge owns copies of input documents and uses
opaque sessions and caller-provided byte buffers; those buffers must be valid
for their supplied capacity.

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
`ET_DATA4_ERR_FORMAT`. Verification checks the report's internal consistency.
It cannot independently prove UTF-8, quality, repetition, or near-dedup
decisions without the original raw documents. The checksum detects accidental
change, not adversarial re-signing or a swap with another valid report.

## Measured Validation Results

The development-only DATA4-Q -> T2/D1 witness passed in the pinned
`eshkol-r0-debug:20260928` Ubuntu 22.04 / LLVM 21.1.8 environment with the
`fe9dfd52` toolchain. The quality run kept `alpha`, `bravo`, `charlie` in
ordinal order, rejected invalid UTF-8, low quality and an exact duplicate,
and yielded the expected 20 T2 token IDs. D1 validated and T2 read back all
20 IDs in order. A duplicate source ID failed before a D1 target appeared.
The retained local receipt is
`/home/gabe/.cache/eshkol-data4-q-t2-d1-witness-20261125-r5/SHA256SUMS`
(8/8 hashes verified). Independent source-and-test delta review found no
blocker. This is local integration evidence, not hosted acceptance.

Historical `60093c6e` evidence was recorded offline in
`eshkol-r0-debug:20260928` (`clang-21`, `clang++-21`, pinned Eshkol toolchain
`/tc`). Current extracted-tree validation is tracked separately:

On current-main extraction `fe4c0aa60`/tree `c0d6d9d4`,
`scripts/test-data4-pipeline.sh` passed in the pinned offline Ubuntu 22.04,
LLVM 21.1.8 container with the verified `fe9dfd52` Eshkol compiler. The gate
passed deterministic normal and sanitized archives, C++17 linkage, normal and
ASan/UBSan/LSan native parity, 3/3 independent Python oracle checks, and two
fresh byte-identical Eshkol AOT ELFs with matching output. The normal and
sanitized archive SHA-256 values were `8e840fa5ef93d1d6` and
`7a7e9355074eafe3` (prefixes); the AOT ELF SHA-256 was
`7685b2bfd974aa2db44310c5f6c822936b4110b54a2d595b678594772b5411f9`.
This is local candidate evidence, not hosted CI or integration acceptance.

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
