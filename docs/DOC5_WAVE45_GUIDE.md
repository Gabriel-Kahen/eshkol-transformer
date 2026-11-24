# Wave 4/5 Operator Guide & Subsystem Index

Source baseline `319c8822a78a5ce4811893a773681055f7a3b37e`. This guide indexes
every Wave 4/5 native C11/C++17 subsystem, compiled Eshkol AOT bridge,
benchmark suite, and end-to-end lifecycle test integrated on `HEAD`, with exact
`Makefile` targets and links to each subsystem's specification.

All runtime execution described here runs on the pinned `x86_64` CPU lane
(`eshkol-r0-debug:20260928`, Ubuntu 22.04, LLVM/Clang 21.1.8, Eshkol commit
`fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`). No GPU execution, multi-node
networking, or production-scale throughput claims are made.

## Pinned Container Quick Start

Run any Wave 4/5 gate or the combined Wave 4/5 documentation and live
walkthrough inside the pinned toolchain container:

```bash
export CC=clang-21 CXX=clang++-21 ESHKOL_SOURCE_DIR=/tc/src ESHKOL_BUILD_DIR=/tc/build
bash scripts/test-doc5-wave45-walkthrough.sh
```

To also verify the baseline DOC5 corpus-to-model walkthrough:

```bash
make test-doc5
```

---

## 1. Data & Tokenizer

| Subsystem | Public ABI / Bridge Header | Verification Targets | Specification |
| --- | --- | --- | --- |
| DATA4 Corpus Slice & Installed Report | `internal/data4/lib/data4_corpus.esk`, `internal/data4/lib/data4_report_reader.esk` | `make test-ci-data4-after-build`, `bash scripts/test-data4-source.sh`, `bash scripts/test-data4-native.sh` | [DATA4_CORPUS_SLICE.md](DATA4_CORPUS_SLICE.md), [DATA4_INSTALLED_REPORT.md](DATA4_INSTALLED_REPORT.md) |
| DATA4 Quality, Dedup & Provenance Pipeline | `include/eshkol_transformer/data4_pipeline_abi.h`, `native/data4_pipeline_eshkol_bridge.h` | `make test-data4-pipeline` | [DATA4_CORPUS_PIPELINE.md](DATA4_CORPUS_PIPELINE.md) |
| T2 Deterministic Byte-Level BPE Trainer | `include/eshkol_transformer/t2_bpe_trainer_abi.h`, `native/t2_bpe_trainer_eshkol_bridge.h` | `make test-t2-trainer` | [T2_BPE_TRAINER.md](T2_BPE_TRAINER.md) |

- **DATA4-Pipeline** validates UTF-8, filters by byte length and quality metrics, removes exact SHA-256 and MinHash LSH near-duplicates, and writes/verifies checksummed provenance reports.
- **T2-Trainer** deterministically trains byte-level BPE merges, encodes/decodes losslessly, and serializes SHA-256 checksummed `ETBPE100` v1.0 artifacts.

## 2. Model Architecture & Attention

| Subsystem | Public ABI / Bridge Header | Verification Targets | Specification |
| --- | --- | --- | --- |
| MOD5 Variants (RMSNorm, SwiGLU, MQA/GQA) | `include/eshkol_transformer/mod5_primitives_abi.h`, `native/mod5_eshkol_bridge.h` | `make test-mod5`, `make test-mod5-aot`, `make test-mod5-e2e` | [MOD5_VARIANTS.md](MOD5_VARIANTS.md), [MOD5_E2E_C1_IO5.md](MOD5_E2E_C1_IO5.md) |
| MOD5 RoPE & 2-Layer Llama-Style Decoder | `include/eshkol_transformer/mod5_rope_abi.h`, `native/mod5_rope_eshkol_bridge.h` | `make test-mod5-rope` | [MOD5_ROPE_MULTILAYER.md](MOD5_ROPE_MULTILAYER.md) |
| MOD5 Sparse Top-K MoE Router & SwiGLU | `include/eshkol_transformer/mod5_moe_abi.h`, `native/mod5_moe_eshkol_bridge.h` | `make test-mod5-moe` | [MOD5_MOE_ROUTER.md](MOD5_MOE_ROUTER.md) |
| MOD5 Multi-Head Latent Attention (MLA) | `include/eshkol_transformer/mod5_mla_abi.h`, `native/mod5_mla_eshkol_bridge.h` | `make test-mod5-mla` | [MOD5_MLA_ATTENTION.md](MOD5_MLA_ATTENTION.md) |
| PA4 & PA4-Wider Causal Attention | `include/eshkol_transformer/pa4_attention_candidate.h`, `include/eshkol_transformer/pa4_wider_attention_abi.h`, `native/pa4_wider_eshkol_bridge.h` | `bash scripts/test-pa4.sh`, `make test-pa4-wider`, `make test-pa4-wider-aot` | [PA4_ATTENTION.md](PA4_ATTENTION.md), [PA4_WIDER_ATTENTION.md](PA4_WIDER_ATTENTION.md), [PA4_WIDER_ESHKOL_AOT.md](PA4_WIDER_ESHKOL_AOT.md) |
| RoPE Long-Context Extension (PI / NTK / YaRN) | `include/eshkol_transformer/rope_yarn_abi.h`, `native/rope_yarn_eshkol_bridge.h` | `make test-rope-yarn` | [ROPE_YARN_LONGCTX.md](ROPE_YARN_LONGCTX.md) |
| AC4 Activation Checkpointing | `include/eshkol_transformer/ac4_checkpointing_abi.h`, `native/ac4_eshkol_bridge.h` | `make test-ac4`, `make test-ac4-aot` | [AC4_CHECKPOINTING.md](AC4_CHECKPOINTING.md) |

## 3. Optimizers & Composed Training

| Subsystem | Public ABI / Bridge Header | Verification Targets | Specification |
| --- | --- | --- | --- |
| MOD5 AdamW, LR Schedules & Fresh-Process Resume | `include/eshkol_transformer/mod5_adamw_abi.h` | `make test-mod5-adamw` | [MOD5_ADAMW_RESUME.md](MOD5_ADAMW_RESUME.md) |
| O2 Muon (Newton-Schulz5) + Hybrid AdamW + WSD | `include/eshkol_transformer/o2_muon_abi.h`, `native/o2_muon_eshkol_bridge.h` | `make test-o2-muon` | [O2_MUON_WSD.md](O2_MUON_WSD.md) |
| Wave 4/5 Composed Trainer (MP4 + AC4 + MOD5 + DIST5 + EVAL5) | `include/eshkol_transformer/wave4_composed_trainer_abi.h`, `native/wave4_composed_eshkol_bridge.h` | `make test-wave4-composed`, `make test-wave4-composed-aot` | [WAVE4_COMPOSED_TRAINING.md](WAVE4_COMPOSED_TRAINING.md) |

## 4. Fine-Tuning, Alignment & Evaluation

| Subsystem | Public ABI / Bridge Header | Verification Targets | Specification |
| --- | --- | --- | --- |
| LoRA Adapter Fine-Tuning (`ETLR1000`) | `include/eshkol_transformer/lora_adapter_abi.h`, `native/lora_eshkol_bridge.h` | `make test-lora` | [LORA_ADAPTER_FINETUNE.md](LORA_ADAPTER_FINETUNE.md) |
| Direct Preference Optimization (DPO / cDPO) | `include/eshkol_transformer/dpo_alignment_abi.h`, `native/dpo_eshkol_bridge.h` | `make test-dpo` | [DPO_ALIGNMENT.md](DPO_ALIGNMENT.md) |
| GRPO / RLVR Alignment Engine | `include/eshkol_transformer/grpo_abi.h`, `native/grpo_eshkol_bridge.h` | `make test-grpo` | [GRPO_RLVR_ALIGNMENT.md](GRPO_RLVR_ALIGNMENT.md) |
| EVAL5 Contamination-Gated Benchmarks | `include/eshkol_transformer/eval5_benchmark_abi.h`, `native/eval5_eshkol_bridge.h` | `make test-eval5`, `make test-eval5-aot` | [EVAL5_BENCHMARKS_CONTAMINATION.md](EVAL5_BENCHMARKS_CONTAMINATION.md) |

## 5. Quantization, Mixed Precision & Weight Interchange

| Subsystem | Public ABI / Bridge Header | Verification Targets | Specification |
| --- | --- | --- | --- |
| MP4 Mixed Precision (`f16`/`bf16` Storage & Loss Scaling) | `include/eshkol_transformer/mp4_mixed_precision_abi.h`, `native/mp4_eshkol_bridge.h` | `make test-mp4`, `make test-mp4-aot` | [MP4_MIXED_PRECISION.md](MP4_MIXED_PRECISION.md) |
| IO5 Safetensors, GGUF, Mapping & Live C1 | `tools/io5/safetensors_interop.py`, `tools/io5/gguf_interop.py`, `tools/io5/model_mapping.py` | `make test-io5`, `make test-io5-gguf`, `make test-io5-live-c1` | [IO5_SAFETENSORS.md](IO5_SAFETENSORS.md), [IO5_GGUF.md](IO5_GGUF.md), [IO5_MODEL_MAPPING.md](IO5_MODEL_MAPPING.md), [IO5_LIVE_C1_INTEROP.md](IO5_LIVE_C1_INTEROP.md) |
| IO5 Native `F16`/`BF16`/`Q8_0` Quantized Interchange | `include/eshkol_transformer/io5_quant_abi.h`, `tools/io5/quantized_interchange.py` | `make test-io5-quant` | [IO5_QUANTIZED_INTERCHANGE.md](IO5_QUANTIZED_INTERCHANGE.md) |
| W4A32 INT4 Group-Wise Quantization & Fused GEMM | `include/eshkol_transformer/w4a32_quant_abi.h`, `native/w4a32_quant_eshkol_bridge.h` | `make test-w4a32-quant` | [W4A32_INT4_QUANTIZATION.md](W4A32_INT4_QUANTIZATION.md) |
| OCP FP8 (`E4M3FN`/`E5M2`) & Block-Scaled `W8A8` GEMM | `include/eshkol_transformer/fp8_gemm_abi.h`, `native/fp8_gemm_eshkol_bridge.h` | `make test-fp8-gemm` | [FP8_E4M3_E5M2_GEMM.md](FP8_E4M3_E5M2_GEMM.md) |

## 6. Distributed & Hardware Observability

| Subsystem | Public ABI / Bridge Header | Verification Targets | Specification |
| --- | --- | --- | --- |
| DIST5 Multi-Process Data-Parallel Collectives | `include/eshkol_transformer/dist5_collectives_abi.h`, `native/dist5_eshkol_bridge.h` | `make test-dist5`, `make test-dist5-aot` | [DIST5_DATA_PARALLEL.md](DIST5_DATA_PARALLEL.md) |
| Ring Attention (Sequence-Parallel Causal Attention) | `include/eshkol_transformer/ring_attn_abi.h`, `native/ring_attn_eshkol_bridge.h` | `make test-ring-attn` | [RING_ATTENTION.md](RING_ATTENTION.md) |
| AMD4 HIP / `/dev/kfd` Observability Probe | `include/eshkol_transformer/amd4_hip_abi.h`, `native/amd4_eshkol_bridge.h` | `make test-amd4`, `make test-amd4-aot` | [AMD4_HIP_OBSERVABILITY.md](AMD4_HIP_OBSERVABILITY.md) |

## 7. Serving & Decoding

| Subsystem | Public ABI / Bridge Header | Verification Targets | Specification |
| --- | --- | --- | --- |
| MOD5 Autoregressive Generation & KV-Cache | `include/eshkol_transformer/mod5_generate_abi.h` | `make test-mod5-generate` | [MOD5_GENERATION.md](MOD5_GENERATION.md) |
| Exact Lossless Speculative Decoding | `include/eshkol_transformer/spec_decode_abi.h`, `native/spec_decode_eshkol_bridge.h` | `make test-spec-decode` | [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) |
| PagedAttention / Block-Table KV-Cache & CoW Sharing | `include/eshkol_transformer/paged_kv_abi.h`, `native/paged_kv_eshkol_bridge.h` | `make test-paged-kv` | [PAGED_KV_CACHE.md](PAGED_KV_CACHE.md) |
| Continuous Batching & Chunked-Prefill Scheduler | `include/eshkol_transformer/continuous_batch_abi.h`, `native/continuous_batch_eshkol_bridge.h` | `make test-continuous-batch` | [CONTINUOUS_BATCHING.md](CONTINUOUS_BATCHING.md) |
| Constrained Decoding (DFA + JSON-Object Schema FSA) | `include/eshkol_transformer/constrained_decode_abi.h`, `native/constrained_decode_eshkol_bridge.h` | `make test-constrained-decode` | [CONSTRAINED_DECODING_FSA.md](CONSTRAINED_DECODING_FSA.md) |

## 8. Benchmarks & Verification Gates

| Suite | Entry Script | Verification Targets | Specification |
| --- | --- | --- | --- |
| B0 Wave 4/5 Unified Benchmarks (`wave45-bench-v1`) | `scripts/test-b0-wave45.sh`, `scripts/benchmark-wave45.sh` | `make test-b0-wave45`, `make benchmark-wave45` | [WAVE45_BENCHMARKS.md](WAVE45_BENCHMARKS.md) |
| B0 Wave 5 11-Engine Benchmarks (`wave5-bench-v1`) | `scripts/test-b0-wave5.sh`, `scripts/benchmark-wave5.sh` | `make test-b0-wave5`, `make benchmark-wave5` | [B0_WAVE5_BENCHMARKS.md](B0_WAVE5_BENCHMARKS.md) |
| DOC5 Baseline Walkthrough | `scripts/test-doc5-walkthrough.sh` | `make test-doc5` | [DOC5_CORPUS_TO_MODEL_TUTORIAL.md](DOC5_CORPUS_TO_MODEL_TUTORIAL.md), [DOC5_API_REFERENCE.md](DOC5_API_REFERENCE.md) |
| DOC5 Wave 4/5 Verification Walkthrough | `scripts/test-doc5-wave45-walkthrough.sh` | `bash scripts/test-doc5-wave45-walkthrough.sh` | This guide |
| Q0 Python Isolation & CI Topology | `tests/q0/test_python_isolation.py`, `scripts/check-ci-topology.sh` | `make test-python-isolation`, `make test-ci-topology` | [Q0_VALIDATION.md](Q0_VALIDATION.md), [CI_EFFICIENCY.md](CI_EFFICIENCY.md) |

## 9. Full 8-Stage E2E LLM Lifecycle (`make test-e2e-lifecycle`)

The standalone compiled Eshkol AOT suite (`include/eshkol_transformer/e2e_lifecycle_abi.h`,
`native/e2e_lifecycle_eshkol_bridge.h`, `scripts/test-e2e-lifecycle.sh`) links
nine Wave 4/5 static archives plus K1 into a single deterministic executable
(`--strict-types --optimize 0 --no-stdlib`, zero dynamic project `.so`
dependencies) and runs all eight stages across five distinct OS process
boundaries:

1. **Stage 1 (`DATA4`)**: Filters 11 raw documents to 6 clean documents (rejecting exact duplicate, MinHash near-duplicate, invalid UTF-8, short length, and low quality) with a SHA-256 report.
2. **Stage 2 (`T2 BPE` + `D1 .ets`)**: Trains a byte-level BPE tokenizer (`ETBPE100`, `vocab=360`, `102` merges), saves atomically, reloads in a fresh OS process, verifies lossless decode, and writes a D1 `.ets` token shard (`766` tokens).
3. **Stage 3 (`MOD5 RoPE` + `MOD5 AdamW`)**: Pretrains a 1-block rotary transformer (`D=16, H=2, Dh=8, D_ff=32, T=8, 13,568` parameters, FD gradient max abs err `<= 2.0e-3`), rejects a corrupted snapshot, and proves bitwise weight and `ETM5TRN1` SHA-256 parity between an uninterrupted 8-step run and a 4+4 fresh-process resume.
4. **Stage 4 (`LoRA`)**: Fine-tunes a `rank=4` adapter on `W_ff1` for 10 steps with frozen-base SHA-256 invariance, reversible merge/unmerge (`<= 1e-6`), wrong-base-hash rejection, and fresh-process `ETLR1000` reload.
5. **Stage 5 (`DPO`)**: Runs 12 DPO alignment steps (`beta=0.25`) with prompt masking, frozen reference SHA-256 invariance, decreasing loss, increasing margin, and `acc=1.0`.
6. **Stage 6 (`EVAL5`)**: Rejects a contaminated eval split fail-closed (`status=-7`, no manifest written), scores a clean held-out split (LM NLL/PPL + MC accuracy), writes an atomic manifest, and verifies SHA-256 and metric parity in a fresh OS process.
7. **Stage 7 (`Spec-Decode`)**: Runs exact lossless speculative decoding (Stage 3 base draft vs. Stage 5 aligned target) with 100% greedy match, seeded stochastic replay parity, and EOS stopping.
8. **Stage 8 (`IO5-Quant`)**: Quantizes and dequantizes all `13,568` trained weights through `F16`, `BF16`, and `Q8_0` (`14,416` bytes) within verified error bounds.

Run the complete lifecycle gate directly:

```bash
make test-e2e-lifecycle
```

See [E2E_LLM_LIFECYCLE.md](E2E_LLM_LIFECYCLE.md) for the stage dataflow diagram,
tensor-shape contract, measured SHA-256 evidence, and explicit limitations.
