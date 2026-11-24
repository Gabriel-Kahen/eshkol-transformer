# B0 Wave 5 Extended Native Benchmark Suite (`wave5-bench-v1`)

Status: **Implemented and verified inside the pinned `eshkol-r0-debug:20260928` / Clang 21.1.8 container**.

## Overview

The B0 Wave 5 Extended Benchmark Suite (`native/b0_wave5_bench.c` ->
`build/b0_wave5/b0_wave5_bench`) links all 11 real Wave 5 native static archives
(plus `libeshkol_transformer_k1.a` for K1 ABI 1.0 provider dispatch) into a
single deterministic C11 executable with zero mocks, stubs, or silent fallbacks:

| # | Group Key | Engine | Static Archive | Build Script |
|---:|---|---|---|---|
| 1 | `mod5_rope` | MOD5 RoPE | `build/mod5_rope/libeshkol_transformer_mod5_rope.a` | `scripts/build-mod5-rope.sh` |
| 2 | `mod5_moe` | MOD5 MoE | `build/mod5_moe/libeshkol_transformer_mod5_moe.a` | `scripts/build-mod5-moe.sh` |
| 3 | `mod5_mla` | MOD5 MLA | `build/mod5_mla/libeshkol_transformer_mod5_mla.a` | `scripts/build-mod5-mla.sh` |
| 4 | `mod5_adamw` | MOD5 AdamW | `build/mod5_adamw/libeshkol_transformer_mod5_adamw.a` | `scripts/build-mod5-adamw.sh` |
| 5 | `o2_muon` | O2 Muon+WSD | `build/o2_muon/libeshkol_transformer_o2_muon.a` | `scripts/build-o2-muon.sh` |
| 6 | `lora` | LoRA | `build/lora/libeshkol_transformer_lora.a` | `scripts/build-lora.sh` |
| 7 | `dpo` | DPO/cDPO | `build/dpo/libeshkol_transformer_dpo.a` | `scripts/build-dpo.sh` |
| 8 | `spec_decode` | Speculative Decoding | `build/spec_decode/libeshkol_transformer_spec_decode.a` | `scripts/build-spec-decode.sh` |
| 9 | `t2_trainer` | T2 BPE Trainer | `build/t2_trainer/libeshkol_transformer_t2_trainer.a` | `scripts/build-t2-trainer.sh` |
| 10 | `data4_pipeline` | DATA4 Pipeline | `build/data4_pipeline/libeshkol_transformer_data4_pipeline.a` | `scripts/build-data4-pipeline.sh` |
| 11 | `io5_quant` | IO5 Quant Codec | `build/io5_quant/libeshkol_transformer_io5_quant.a` | `scripts/build-io5-quant.sh` |

Shared K1 runtime dependency: `build/k1/libeshkol_transformer_k1.a` (used by
`MOD5 RoPE`, `MOD5 MoE`, and `MOD5 MLA` K1 providers).

## Canonical JSON Schema (`wave5-bench-v1`)

`b0_wave5_bench` writes a canonical UTF-8 JSON artifact atomically
(`<output>.tmp.<pid>` with `O_CREAT | O_EXCL`, `fsync(2)`, then `rename(2)`).
Every JSON object uses strict ASCII lexicographical key ordering, compact
separators (`,` and `:`), finite numbers only (`NaN` / `Infinity` forbidden),
no duplicate keys, no UTF-8 BOM, and a single trailing `\n`.

```text
schema          = "wave5-bench-v1"
version         = 1
artifact_sha256 = SHA-256(canonical UTF-8 JSON of the top-level document excluding "artifact_sha256")
environment     = {architecture, cpu_model, kernel, logical_cpus, observed_at_utc, os_release}
groups          = {<group_key>: {archive, cases: [...], engine}, ...}
measurement     = {backend, clock: "clock_monotonic_raw", elapsed_unit: "nanosecond",
                   limitations: [...], memory_source: "linux_getrusage_rusage_self",
                   peak_rss_unit: "kibibyte", selected_group, smoke_mode}
process         = {self_peak_rss_kib}
toolchain       = {archives: {<11 group_keys>: {filename, sha256}}, build_mode, cflags,
                   compiler, git_commit, git_dirty, k1_archive: {filename, sha256}}
```

Each benchmark case records:

- `name`, `operation`, `shape`
- `warmup`, `samples`, `iterations`, `work_units_per_iter`
- `min_ns`, `median_ns`, `p95_ns`, `max_ns`, `mean_ns`, `median_ns_per_iter`
  (measured with `CLOCK_MONOTONIC_RAW`, satisfying `min_ns <= median_ns <= p95_ns <= max_ns` and `min_ns <= mean_ns <= max_ns`)
- `throughput` and `throughput_unit` (`tokens_per_sec`, `elems_per_sec`, `docs_per_sec`, or `steps_per_sec`)
- `output_sha256` (full 64-hex SHA-256 digest of the verified numerical output bytes)
- `checksum` (deterministic non-zero 62-bit positive integer derived from `output_sha256`, re-checked on every timed sample and verified identical across independent runs and sanitizer builds)

### CLI Interface

```text
b0_wave5_bench --output <path> [--smoke | --full] [--group <name|all>]
               [--archive-dir <dir>] [--compiler-id <str>] [--cflags <str>]
               [--build-mode normal|sanitize] [--git-commit <sha>] [--git-dirty 0|1]
```

Invalid flags, conflicting `--smoke --full`, unknown `--group`, missing
`--output`, invalid `--build-mode`/`--git-dirty`, or unreadable archive paths
fail closed with non-zero exit status and create no output file.

## Methodology & Verification

- **No unverified or dead-code-eliminated timing**: every case executes an
  untimed witness pass first, validates finiteness and domain invariants (e.g.
  sparse MoE active expert count `N*T*K`, MLA `16:1` KV compression ratio,
  Muon NS5 orthogonality improvement, lossless T2 BPE round-trip, exact DATA4
  dispositions and report checksum verification), records the witness SHA-256,
  folds live output words into a `volatile` sink during timed iterations, and
  re-verifies exact witness SHA-256 equality after every sample batch.
- **Deterministic build & C++17 cleanliness**: `scripts/build-b0-wave5.sh`
  syntax-checks all 11 public ABI headers plus `kernel_abi.h` together under
  `-std=c++17 -Wall -Wextra -Werror -Wpedantic`, builds all 11 prerequisite
  static archives with symbol audits, builds `b0_wave5_bench` twice in isolated
  directories with `-Wl,--build-id=none` and `cmp`-verifies byte-for-byte
  binary determinism, and builds a full ASan/UBSan/LSan sanitized runner
  (`b0_wave5_bench_san`) linking sanitized archives of all 12 native C sources.
- **Test suite (`scripts/test-b0-wave5.sh` + `tests/b0/test_b0_wave5.py`)**:
  verifies two-run numerical checksum invariance, normal-vs-sanitize checksum
  invariance under `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, single-group filtering
  (`--group lora`), on-disk archive SHA-256 matching, and negative tests
  (single-byte tamper, wrong schema, missing group, timing order violation,
  zero checksum, non-finite JSON constants, non-canonical whitespace, and
  invalid CLI flags).

## Measured Container Results (`--full`, `eshkol-r0-debug:20260928`)

Measured inside the pinned `eshkol-r0-debug:20260928` container in `--full`
mode (`21` measured batches per case, `CLOCK_MONOTONIC_RAW`):

- **OS / Kernel / Arch**: Ubuntu 22.04.5 LTS, Linux `7.2.7-1-cachyos`, `x86_64`
- **CPU**: AMD Ryzen 7 3700X 8-Core Processor (16 logical CPUs)
- **Compiler**: `Ubuntu clang version 21.1.8 (++20251221032842+2078da43e25a-1~exp1~20251221153008.77)`
- **Flags (`normal`)**: `-std=c11 -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math`
- **Process Peak RSS (`--full`, `normal`)**: `3,928 KiB`
- **Observed UTC**: `2026-11-24T04:34:26Z`
- **Artifact SHA-256 (`--full`)**: `cfa5fce499d3f4a2cb83d3527bca849a7a03da7251171c234467f64844374617`

### Linked Archive SHA-256 Provenance (`normal`, `-O2`)

| Group | Archive | SHA-256 |
|---|---|---|
| `data4_pipeline` | `libeshkol_transformer_data4_pipeline.a` | `283b70ffe455b72720a948dc32178fac6692c4a1f197a0be4899262b6f63bcd6` |
| `dpo` | `libeshkol_transformer_dpo.a` | `fe7851cf56eeb2dc3c17a0b949f85f34b5b75face2268106e343abd6edf75560` |
| `io5_quant` | `libeshkol_transformer_io5_quant.a` | `f288e25b379dec6508ec5d7fa4bdf95113ef82b30a9abf5e38fe711f96f8bb65` |
| `lora` | `libeshkol_transformer_lora.a` | `4e99c47b6e88230748fca7b548a6d76117146a0d121497ecae5df4677ee9c5ec` |
| `mod5_adamw` | `libeshkol_transformer_mod5_adamw.a` | `2fe22d74cdd460bdbe9c01eae375910b03451695a9eab94ab3ffdcf0bc957c8b` |
| `mod5_mla` | `libeshkol_transformer_mod5_mla.a` | `2c94ffeba653d11d2d1cf64065602c6492c363c5f2914119c4723612fb48a17e` |
| `mod5_moe` | `libeshkol_transformer_mod5_moe.a` | `c0461398d88d5d83a813774e40f6af5c96c9a9d3cd78f27dec8e832b85d3066e` |
| `mod5_rope` | `libeshkol_transformer_mod5_rope.a` | `0f9c52496647eb5f262fc38d47db6436228d14abb9ae2d7979101877f696f1be` |
| `o2_muon` | `libeshkol_transformer_o2_muon.a` | `071843e3119641ed76ab526d94498151c1afbad9435fd450a487d7794fa57959` |
| `spec_decode` | `libeshkol_transformer_spec_decode.a` | `c2816a3b2dea6754d93ec98dd9769dd677dc50e40d6ce883952ab2f8191a76af` |
| `t2_trainer` | `libeshkol_transformer_t2_trainer.a` | `7ce105cd418a8bf70d6e95919d6bee1ab4da8461ba85a8aa0dd0671d31d01b56` |
| `k1` (shared dep) | `libeshkol_transformer_k1.a` | `017b1c38b7e1aa4e59bfe5eb1c2111a43c56bc5f9b7c78c663461236a515b32d` |

### All 11 Engine Groups (25 Measured `--full` Cases)

Batch timings (`min_ns`, `median_ns`, `p95_ns`, `max_ns`, `mean_ns`) are over
`samples = 21` batches of `iters` iterations each; `Med/iter` is
`median_ns / iters`.

| Group | Case | Shape | Iters | Batch Min / Med / P95 / Max (ns) | Med/iter (ns) | Throughput | Checksum |
|---|---|---|---:|---|---:|---:|---:|
| `data4_pipeline` | `data4_pipeline_run_32docs` | `docs=32,perms=64,bands=16,ngram=5` | 5 | 3,293,287 / 3,304,809 / 3,331,829 / 3,369,260 | 660,961.80 | 48,414.30 `docs_per_sec` | `2567976085114291461` |
| `data4_pipeline` | `data4_report_format_verify_32docs` | `docs=32` | 15 | 808,296 / 834,725 / 846,317 / 848,702 | 55,648.33 | 575,039.68 `docs_per_sec` | `3948035584075723578` |
| `dpo` | `dpo_forward_4x32x128` | `B=4,T=32,V=128,beta=0.1` | 12 | 3,902,540 / 3,921,695 / 3,940,260 / 3,949,026 | 326,807.92 | 783,334.76 `tokens_per_sec` | `2961980085568121753` |
| `dpo` | `cdpo_forward_backward_4x32x128` | `B=4,T=32,V=128,beta=0.1,ls=0.1` | 12 | 8,267,125 / 8,337,287 / 8,699,907 / 9,198,302 | 694,773.92 | 368,465.19 `tokens_per_sec` | `4457806139069666729` |
| `io5_quant` | `io5_f16_roundtrip_4096` | `elems=4096` | 80 | 1,365,742 / 1,374,157 / 1,385,639 / 1,386,641 | 17,176.96 | 238,458,924.27 `elems_per_sec` | `1077915332878361864` |
| `io5_quant` | `io5_bf16_roundtrip_4096` | `elems=4096` | 80 | 682,030 / 725,361 / 734,347 / 735,479 | 9,067.01 | 451,747,474.71 `elems_per_sec` | `4248473230282104784` |
| `io5_quant` | `io5_q8_0_roundtrip_4096` | `elems=4096,blocks=128` | 80 | 1,287,064 / 1,295,419 / 1,309,176 / 1,333,732 | 16,192.74 | 252,952,905.58 `elems_per_sec` | `3100945267975099916` |
| `lora` | `lora_forward_32x64x64_r8` | `M=32,K=64,N=64,r=8` | 15 | 1,559,535 / 1,605,240 / 1,719,495 / 1,758,338 | 107,016.00 | 19,137,325.26 `elems_per_sec` | `4063495575623326999` |
| `lora` | `lora_backward_32x64x64_r8` | `M=32,K=64,N=64,r=8` | 15 | 2,149,101 / 2,178,165 / 2,320,503 / 2,449,404 | 145,211.00 | 14,103,614.74 `elems_per_sec` | `1549790080720268081` |
| `lora` | `lora_merge_unmerge_64x64_r8` | `K=64,N=64,r=8` | 15 | 783,460 / 807,184 / 817,603 / 823,524 | 53,812.27 | 76,116,474.06 `elems_per_sec` | `4165463906230784745` |
| `mod5_adamw` | `adamw_step_4096_cosine` | `params=4096,clip=1.0` | 15 | 791,375 / 794,931 / 801,473 / 803,457 | 52,995.40 | 77,289,727.03 `elems_per_sec` | `3635353181431765735` |
| `mod5_adamw` | `adamw_step_16384_linear` | `params=16384,clip=1.0` | 15 | 3,179,804 / 3,196,535 / 3,214,629 / 3,215,000 | 213,102.33 | 76,883,250.14 `elems_per_sec` | `4320949008288623268` |
| `mod5_mla` | `mla_forward_2x16x32_h4d16_dc8` | `N=2,T=16,D=32,H=4,Dh=16,dc=8` | 6 | 1,480,967 / 1,492,579 / 1,504,381 / 1,512,306 | 248,763.17 | 128,636.41 `tokens_per_sec` | `4499374356370172551` |
| `mod5_mla` | `mla_backward_2x16x32_h4d16_dc8` | `N=2,T=16,D=32,H=4,Dh=16,dc=8` | 3 | 2,036,580 / 2,061,326 / 2,069,883 / 2,070,012 | 687,108.67 | 46,571.96 `tokens_per_sec` | `1736038749182037492` |
| `mod5_mla` | `mla_cache_prefill_decode_1x16_dc8` | `N=1,prefill=8,decode=8,D=32,dc=8` | 6 | 1,529,358 / 1,541,481 / 1,551,991 / 1,553,644 | 256,913.50 | 62,277.77 `tokens_per_sec` | `1049498930473115113` |
| `mod5_moe` | `moe_forward_2x16x32x64_e8k2` | `N=2,T=16,D=32,F=64,E=8,K=2` | 5 | 3,761,475 / 3,775,161 / 3,805,878 / 3,811,749 | 755,032.20 | 42,382.30 `tokens_per_sec` | `1472746388688267101` |
| `mod5_moe` | `moe_backward_2x16x32x64_e8k2` | `N=2,T=16,D=32,F=64,E=8,K=2` | 5 | 10,615,931 / 10,709,146 / 11,319,401 / 13,024,558 | 2,141,829.20 | 14,940.50 `tokens_per_sec` | `3018570694690426137` |
| `mod5_rope` | `rope_forward_2x4x64x16` | `N=2,H=4,T=64,Dh=16` | 20 | 1,338,791 / 1,350,123 / 1,368,046 / 1,373,035 | 67,506.15 | 121,351,906.46 `elems_per_sec` | `1855922434858089051` |
| `mod5_rope` | `rope_backward_2x4x64x16` | `N=2,H=4,T=64,Dh=16` | 20 | 1,339,332 / 1,356,745 / 1,417,238 / 1,525,220 | 67,837.25 | 120,759,612.16 `elems_per_sec` | `776724453759144663` |
| `o2_muon` | `muon_ns5_32x32` | `rows=32,cols=32,ns_steps=5` | 10 | 2,235,693 / 2,239,160 / 2,253,307 / 2,254,419 | 223,916.00 | 4,573,143.50 `elems_per_sec` | `4218031192922239048` |
| `o2_muon` | `muon_hybrid_wsd_step` | `groups=32x32+32x16+256x1,ns=5` | 6 | 1,692,685 / 1,706,441 / 1,725,676 / 1,745,684 | 284,406.83 | 3,516.09 `steps_per_sec` | `2914260679533232094` |
| `spec_decode` | `spec_verify_greedy_k8_v1024` | `K=8,V=1024` | 80 | 480,541 / 512,050 / 522,701 / 531,157 | 6,400.62 | 1,249,877.94 `tokens_per_sec` | `2237054244374633305` |
| `spec_decode` | `spec_verify_sample_k8_v1024` | `K=8,V=1024,top_k=64,top_p=0.95` | 15 | 13,382,560 / 13,489,130 / 13,877,949 / 14,615,532 | 899,275.33 | 8,896.05 `tokens_per_sec` | `2610309907381510405` |
| `t2_trainer` | `t2_bpe_train_v384` | `docs=16,target_vocab=384,min_freq=2` | 1 | 1,115,111 / 1,125,711 / 1,137,834 / 1,139,257 | 1,125,711.00 | 14,213.24 `docs_per_sec` | `1023000832233701247` |
| `t2_trainer` | `t2_bpe_encode_decode_corpus` | `docs=16,vocab=384` | 12 | 9,530,906 / 9,639,108 / 9,799,750 / 10,003,993 | 803,259.00 | 1,567,364.95 `tokens_per_sec` | `3325833360535438414` |

## Explicit Limitations

- **CPU-only serial execution**: Every engine executes single-threaded on host
  x86-64 CPU (`measurement.backend.acceleration = false`). No GPU, HIP/ROCm,
  CUDA, or multi-node acceleration claims are made.
- **Bounded verified ABI shapes**: All 11 native Wave 5 engines enforce strict
  bounded CPU ABI dimensions (`MOD5 RoPE`, `MOD5 MoE`, `MOD5 MLA`, `O2 Muon`,
  `LoRA`, `DPO/cDPO`, `Speculative Decoding`, `T2 BPE Trainer`, `DATA4 Pipeline`,
  `IO5 Quant Codec`). Measured throughput reflects these bounded shapes and is
  not extrapolated to large-scale production training or inference workloads.
- **Strict IEEE floating-point discipline**: Compiled with `-ffp-contract=off
  -fexcess-precision=standard -fno-fast-math` for deterministic reproducibility
  rather than aggressive auto-vectorized fast-math peak throughput.
- **Host variability**: Timings reflect a single unpinned host (`AMD Ryzen 7
  3700X`) without CPU core isolation or frequency locking. Process RSS
  (`ru_maxrss`) is the whole-process Linux high-water mark, not per-engine
  allocation attribution.

## Reproduction

Run the full validation suite (deterministic build, two-run `--smoke` checksum
identity, `--group lora` filter, ASan/UBSan/LSan `--smoke` with leak detection,
Python schema/tamper/CLI negative tests, Python isolation, and CI topology)
inside the pinned container:

```bash
docker run --rm --network none \
  -v /home/gabe/.cache/eshkol-fe9dfd52-ubuntu22:/tc:ro \
  -v "$PWD":"$PWD" \
  -v /home/gabe/eshkol-transformer/.git:/home/gabe/eshkol-transformer/.git:ro \
  -w "$PWD" -e PYTHONDONTWRITEBYTECODE=1 \
  eshkol-r0-debug:20260928 \
  bash -ec 'git config --global --add safe.directory "*"
    export CC=clang-21 CXX=clang++-21 ESHKOL_SOURCE_DIR=/tc/src ESHKOL_BUILD_DIR=/tc/build
    bash scripts/test-b0-wave5.sh
    python3 -m unittest -v tests/q0/test_python_isolation.py
    bash scripts/check-ci-topology.sh
    rm -rf build .tmp'
```

Run a standalone `--full` measurement report:

```bash
CC=clang-21 CXX=clang++-21 bash scripts/benchmark-wave5.sh --full
```
