# B0 Wave 4/5 Unified Performance & Memory Benchmarks (`wave45-bench-v1`)

Status: **Implemented and verified on the pinned `eshkol-r0-debug:20260928` / Clang 21.1.8 container**.

## Overview

The Wave 4/5 benchmark harness (`tests/b0/bench_wave45.c`) links all six native
Wave 4/5 static libraries directly in a single pure C11 executable with zero
silent fallbacks:

1. `build/k1/libeshkol_transformer_k1.a` — K1 ABI 1.0 provider discovery and dispatch
2. `build/mod5/libeshkol_transformer_mod5.a` — MOD5 RMSNorm, SwiGLU, and MQA/GQA variant attention
3. `build/pa4_wider/libeshkol_transformer_pa4_wider.a` — PA4-Wider reference and online-softmax causal attention
4. `build/mp4/libeshkol_transformer_mp4.a` — MP4 `f16`/`bf16` casts, gradient unscaling, and loss scaler
5. `build/ac4/libeshkol_ac4.a` — AC4 activation checkpointing (`NONE`, `EVERY_LAYER`, `SQRT_SEGMENTS`, `EVERY_K`)
6. `build/dist5/libeshkol_dist5.a` — DIST5 multi-process data-parallel collectives over `AF_UNIX` sockets

This harness is development benchmark tooling independent of the delivered
Eshkol-native training path. It reports host CPU latency, live AC4 arena peak
activation bytes, exact MP4 storage ratios, and process RSS without making
accelerator, multi-node, or admission claims.

## Report Format (`wave45-bench-v1`)

`bench_wave45` writes a canonical UTF-8 JSON report atomically (write to
`<output>.tmp.<pid>` with `O_CREAT | O_EXCL`, `fsync(2)`, then `rename(2)`).
The document has sorted keys at every object level, compact separators (`,` and
`:`), finite JSON numbers only (`NaN`/`Infinity` forbidden), no duplicate keys,
no UTF-8 BOM, and a single trailing `\n`.

```text
format             = "wave45-bench-v1"
version            = 1
checksum.algorithm = "sha256"
checksum.digest    = SHA-256(canonical JSON of {"format":..., "payload":..., "version":1})
payload            = {ac4, dist5, environment, measurement, mod5, mp4, pa4_wider, process, toolchain}
```

`tests/b0/test_wave45_benchmark.py` enforces:

- exact byte-for-byte canonical serialization (`raw == canonical_json(envelope) + b"\n"`);
- constant-time SHA-256 verification over the canonical integrity object;
- strict schema key matching (missing or unknown keys rejected);
- exact format `"wave45-bench-v1"` and integer version `1`;
- strictly positive finite latencies (`min_ns <= median_ns <= max_ns`, `median_ns_per_iter > 0`);
- exact 2:1 MP4 narrow-to-wide storage byte ratio (`wide_bytes == 2 * narrow_bytes`, `storage_byte_ratio == 2.0`);
- strict AC4 peak activation memory reduction at `L=4, 8, 16` (`EVERY_LAYER`, `SQRT_SEGMENTS`, `EVERY_K` strictly less than `NONE`) and `live_activation_peak_bytes == planned_activation_peak_bytes`;
- negative test coverage for tampered metrics, missing fields, wrong version/format, non-finite JSON constants and Python floats, non-positive latency, broken MP4 ratios, non-reducing AC4 policies, and non-canonical whitespace.

## Measurement Methodology

- **Clock**: `CLOCK_MONOTONIC` in integer nanoseconds (`min_ns`, `median_ns`,
  `max_ns` over batch samples, lower median `(N - 1) / 2`, plus derived
  `median_ns_per_iter`).
- **Process RSS**: `getrusage(RUSAGE_SELF)` (`self_peak_rss_kib`) and
  `getrusage(RUSAGE_CHILDREN)` (`children_peak_rss_kib`), in Linux kibibytes.
- **No unverified timing**:
  - **MOD5**: runs an untimed forward/backward K1 dispatch first, verifies
    finite non-zero outputs, hashes output bytes with SHA-256, and re-verifies
    the exact output digest after every timed sample batch (7 samples x 200
    iterations for RMSNorm/SwiGLU, 7 samples x 80 iterations for MQA/GQA).
  - **PA4-Wider**: runs untimed reference and online dispatches first, checks
    finiteness and numerical parity within `3e-7 + 3e-6 * max(|ref|, |online|)`
    for `O`, `dQ`, `dK`, and `dV`, records witness SHA-256 digests, and runs 8
    alternating AB/BA pairs (4 AB, 4 BA) at `T=16` (30 iters), `T=32` (20
    iters), and `T=64` (10 iters), verifying the exact witness digest on every
    timed arm.
  - **MP4**: verifies `et_mp4_loss_scaler_init` / `et_mp4_loss_scaler_update`
    state transitions, checks exact `2:1` byte lengths (`[2, 8, 16]` = 512 B :
    1,024 B and `[256, 4]` = 2,048 B : 4,096 B), confirms `overflow == 0` on
    unscale, and times 7 samples x 500 iterations with SHA-256 digest checks.
  - **AC4**: checks `et_ac4_plan_v1`, enforces `activation_limit_bytes =
    plan.activation_peak_bytes` on every run, verifies bitwise parity of `loss`,
    `output`, `d_input`, and `d_parameters` across all four policies against
    `NONE` at `L=4, 8, 16`, and times 7 interleaved samples x 60 iterations.
  - **DIST5**: forks `W=1, 2, 4` isolated worker processes connected via
    pre-created `AF_UNIX` `SOCK_STREAM` socket pairs, synchronizes with
    `et_dist5_barrier`, verifies exact rank-ordered bitwise output parity on
    every rank after warmup and every timed batch (7 samples x 40 iterations at
    256 and 4,096 f32 elements), and reports rank-0 monotonic latencies plus
    output SHA-256.
- **Sanitizer verification**: `scripts/test-b0-wave45.sh` compiles all six
  native `.c` sources together with `bench_wave45.c` under `-O1 -g
  -fsanitize=address,undefined -fno-omit-frame-pointer
  -fno-sanitize-recover=all` and runs `--smoke` with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, validating the emitted
  sanitizer smoke report through the same verifier.

## Measured Results (`eshkol-r0-debug:20260928`)

Measured inside the pinned `eshkol-r0-debug:20260928` container:

- **OS / Kernel / Arch**: Ubuntu 22.04.5 LTS, Linux `7.2.7-1-cachyos`, `x86_64`
- **CPU**: AMD Ryzen 7 3700X 8-Core Processor (16 logical CPUs)
- **Compiler**: `Ubuntu clang version 21.1.8 (++20251221032842+2078da43e25a-1~exp1~20251221153008.77)`
- **Flags (`normal`)**: `-std=c11 -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math`
- **Process Peak RSS (`normal`)**: `self_peak_rss_kib = 2,640 KiB`, `children_peak_rss_kib = 2,236 KiB`
- **Report SHA-256 (`normal`, `2026-11-23T19:50:58Z`)**: `e77ebd3bff6cf07fb9f100567095aca267ef1d83af07fd12cd3f2f3a669ef4e6`

### 1. MOD5 Kernels (K1 dispatch, median ns/iter)

| Kernel | Shape | Forward (ns/iter) | Backward (ns/iter) | Forward+Backward (ns/iter) |
|---|---|---:|---:|---:|
| RMSNorm | `[1, 2, 4]` | 1,094.23 | 1,444.85 | 2,591.03 |
| RMSNorm | `[2, 8, 16]` | 2,402.77 | 3,543.19 | 6,020.19 |
| RMSNorm | `[1, 16, 32]` | 3,612.72 | 5,411.34 | 9,035.99 |
| SwiGLU | `[1, 2, 4]` | 1,033.21 | 1,401.16 | 2,485.18 |
| SwiGLU | `[2, 8, 32]` | 6,473.76 | 9,652.95 | 16,746.15 |
| SwiGLU | `[1, 16, 64]` | 11,956.03 | 18,026.63 | 30,084.26 |
| Variant Attention (MQA) | `[1, 4, 1, 8, 8, 4]` | 4,526.49 | 7,485.99 | 11,966.39 |
| Variant Attention (MQA) | `[1, 4, 1, 16, 16, 8]` | 10,265.65 | 20,925.92 | 31,325.70 |
| Variant Attention (GQA) | `[1, 4, 2, 8, 8, 4]` | 4,627.55 | 7,621.00 | 12,653.56 |
| Variant Attention (GQA) | `[1, 4, 2, 16, 16, 8]` | 10,650.50 | 21,071.70 | 32,858.74 |

### 2. PA4-Wider Reference vs. Online-Softmax (AB/BA dispatch, median ns/iter)

| Shape `[N,Hq,Hkv,Tq,Tk,Dh]` | Operation | Ref (ns/iter) | Online (ns/iter) | Online/Ref | Speedup | Max Abs Diff |
|---|---|---:|---:|---:|---:|---:|
| `[1, 4, 2, 16, 16, 16]` | forward | 25,902.60 | 15,043.27 | 0.580763 | 41.92% | `1.192093e-07` |
| `[1, 4, 2, 16, 16, 16]` | backward | 41,996.20 | 21,218.37 | 0.505245 | 49.48% | `0.0` |
| `[1, 4, 2, 16, 16, 16]` | joint | 70,294.03 | 38,322.57 | 0.545175 | 45.48% | `1.192093e-07` |
| `[1, 4, 2, 32, 32, 16]` | forward | 97,957.80 | 57,646.65 | 0.588485 | 41.15% | `1.192093e-07` |
| `[1, 4, 2, 32, 32, 16]` | backward | 156,452.50 | 77,031.00 | 0.492360 | 50.76% | `0.0` |
| `[1, 4, 2, 32, 32, 16]` | joint | 258,131.40 | 131,668.40 | 0.510083 | 48.99% | `1.192093e-07` |
| `[1, 4, 2, 64, 64, 16]` | forward | 370,680.90 | 206,610.90 | 0.557382 | 44.26% | `1.192093e-07` |
| `[1, 4, 2, 64, 64, 16]` | backward | 595,429.40 | 290,597.70 | 0.488047 | 51.20% | `0.0` |
| `[1, 4, 2, 64, 64, 16]` | joint | 956,974.00 | 495,270.90 | 0.517539 | 48.25% | `1.192093e-07` |

### 3. MP4 Mixed-Precision Casts & Unscale (exact 2:1 storage ratio)

| Operation | Shape | Elements | Storage (Narrow:Wide) | Median (ns/iter) | Throughput (Melem/s) |
|---|---|---:|---|---:|---:|
| `cast.f32-to-f16` | `[2, 8, 16]` | 256 | 512 B : 1,024 B (2:1) | 1,571.73 | 162.88 |
| `cast.f16-to-f32` | `[2, 8, 16]` | 256 | 512 B : 1,024 B (2:1) | 1,375.01 | 186.18 |
| `grad.unscale-f16-to-f32` | `[2, 8, 16]` | 256 | 512 B : 1,024 B (2:1) | 1,715.90 | 149.19 |
| `cast.f32-to-bf16` | `[2, 8, 16]` | 256 | 512 B : 1,024 B (2:1) | 1,327.85 | 192.79 |
| `cast.bf16-to-f32` | `[2, 8, 16]` | 256 | 512 B : 1,024 B (2:1) | 1,213.63 | 210.94 |
| `grad.unscale-bf16-to-f32` | `[2, 8, 16]` | 256 | 512 B : 1,024 B (2:1) | 1,548.93 | 165.28 |
| `cast.f32-to-f16` | `[256, 4]` | 1,024 | 2,048 B : 4,096 B (2:1) | 3,436.70 | 297.96 |
| `cast.f16-to-f32` | `[256, 4]` | 1,024 | 2,048 B : 4,096 B (2:1) | 2,660.76 | 384.85 |
| `grad.unscale-f16-to-f32` | `[256, 4]` | 1,024 | 2,048 B : 4,096 B (2:1) | 2,994.19 | 342.00 |
| `cast.f32-to-bf16` | `[256, 4]` | 1,024 | 2,048 B : 4,096 B (2:1) | 2,460.06 | 416.25 |
| `cast.bf16-to-f32` | `[256, 4]` | 1,024 | 2,048 B : 4,096 B (2:1) | 1,965.94 | 520.87 |
| `grad.unscale-bf16-to-f32` | `[256, 4]` | 1,024 | 2,048 B : 4,096 B (2:1) | 2,323.80 | 440.66 |

### 4. AC4 Activation Checkpointing (`(N,T,Hq,Hkv,Dh)=(2,3,4,2,2)`, dropout `0.2`)

| Layers | Policy | `k` | Live Peak Act. (B) | Memory Saved | Workspace (B) | Replays | Median (µs/iter) | Time vs. `NONE` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | `NONE` | 0 | 8,832 | 0.00% | 15,232 | 0 | 63.719 | 1.000× |
| 4 | `EVERY_LAYER` | 0 | 3,648 | 58.70% | 10,048 | 4 | 88.688 | 1.392× |
| 4 | `SQRT_SEGMENTS` | 0 | 3,456 | 60.87% | 9,856 | 6 | 100.932 | 1.584× |
| 4 | `EVERY_K` | 4 | 3,264 | 63.04% | 9,664 | 10 | 125.707 | 1.973× |
| 8 | `NONE` | 0 | 16,512 | 0.00% | 29,312 | 0 | 125.990 | 1.000× |
| 8 | `EVERY_LAYER` | 0 | 4,416 | 73.26% | 17,216 | 8 | 176.143 | 1.398× |
| 8 | `SQRT_SEGMENTS` | 0 | 3,840 | 76.74% | 16,640 | 13 | 206.422 | 1.638× |
| 8 | `EVERY_K` | 8 | 3,264 | 80.23% | 16,064 | 36 | 348.371 | 2.765× |
| 16 | `NONE` | 0 | 31,872 | 0.00% | 57,472 | 0 | 256.223 | 1.000× |
| 16 | `EVERY_LAYER` | 0 | 5,952 | 81.33% | 31,552 | 16 | 355.343 | 1.387× |
| 16 | `SQRT_SEGMENTS` | 0 | 4,224 | 86.75% | 29,824 | 28 | 429.211 | 1.675× |
| 16 | `EVERY_K` | 16 | 3,264 | 89.76% | 28,864 | 136 | 1,094.695 | 4.272× |

All four policies produced bitwise-identical `loss`, `output`, `d_input`, and
`d_parameters` at each depth `L=4, 8, 16`.

### 5. DIST5 Multi-Process Collectives (`AF_UNIX` `SOCK_STREAM`, forked ranks)

| World Size (`W`) | Operation | Elements | Payload (B) | Median (µs/iter) |
|---:|---|---:|---:|---:|
| 1 | `allreduce_f32` | 256 | 1,024 | 0.410 |
| 1 | `sync_gradients_f32` | 256 | 1,024 | 0.755 |
| 1 | `allreduce_f32` | 4,096 | 16,384 | 2.702 |
| 1 | `sync_gradients_f32` | 4,096 | 16,384 | 8.314 |
| 2 | `allreduce_f32` | 256 | 1,024 | 31.860 |
| 2 | `sync_gradients_f32` | 256 | 1,024 | 31.820 |
| 2 | `allreduce_f32` | 4,096 | 16,384 | 47.465 |
| 2 | `sync_gradients_f32` | 4,096 | 16,384 | 50.610 |
| 4 | `allreduce_f32` | 256 | 1,024 | 81.402 |
| 4 | `sync_gradients_f32` | 256 | 1,024 | 75.792 |
| 4 | `allreduce_f32` | 4,096 | 16,384 | 97.669 |
| 4 | `sync_gradients_f32` | 4,096 | 16,384 | 117.061 |

## Explicit Limitations

- **CPU-only serial execution**: All measured kernels execute serially on host
  x86-64 CPU (`backend.acceleration = false`). No GPU, HIP/ROCm, CUDA, or SIMD
  accelerator claims are made.
- **Single-host UDS transport for DIST5**: DIST5 benchmarks measure local
  forked processes communicating over `AF_UNIX` `SOCK_STREAM` socket pairs with
  a serial rank-0 coordinator. They do not measure multi-node networking, RDMA,
  or NCCL/RCCL.
- **Bounded verified ABI shapes**: MOD5, PA4-Wider, MP4, and AC4 execute only
  their verified CPU binary32/storage ABI shapes; timings include K1 validation
  and dispatch overhead on small shapes and are not extrapolated to larger
  models or production training throughput.
- **Host variability**: Measurements reflect one machine (`AMD Ryzen 7 3700X`)
  without CPU frequency locking or sibling-thread isolation. Process RSS
  (`ru_maxrss`) is the Linux high-water mark for the whole runner process, not
  an isolated per-kernel allocation count.

## Reproduction

Run the standalone `-O2` benchmark and report verifier:

```bash
CC=clang-21 CXX=clang++-21 bash scripts/benchmark-wave45.sh
```

Run the full validation suite (normal `-O2` report + ASan/UBSan/LSan smoke run
+ Python schema/checksum/negative tests) inside the pinned container:

```bash
docker run --rm --network none \
  -v /home/gabe/.cache/eshkol-fe9dfd52-ubuntu22:/tc:ro \
  -v "$PWD":"$PWD" \
  -w "$PWD" -e PYTHONDONTWRITEBYTECODE=1 \
  eshkol-r0-debug:20260928 \
  bash -ec 'git config --global --add safe.directory "*"
    export CC=clang-21 CXX=clang++-21 ESHKOL_SOURCE_DIR=/tc/src ESHKOL_BUILD_DIR=/tc/build
    bash scripts/test-b0-wave45.sh
    python3 -m unittest tests/q0/test_python_isolation.py
    bash scripts/check-ci-topology.sh
    rm -rf build .tmp'
```

When running from a linked Git worktree, also bind-mount the shared `.git`
directory read-only (for example,
`-v /home/gabe/eshkol-transformer/.git:/home/gabe/eshkol-transformer/.git:ro`).
