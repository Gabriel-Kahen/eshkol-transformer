# Eshkol Transformer API reference

Integration baseline `8d185344f3abaacb2fdd5ff75743f0ff02a92ab2`, with
the DATA4-C and DOC5 candidate changes. Paths and commands are relative to
the checkout root. Eshkol facades need the corresponding
linked package. Native providers advertise supported operations and shapes
through K1. Check the capability before dispatch and follow each contract's
ownership and release rules.

The source column identifies the public facade or header where available.
Rows marked private, diagnostic, or opt-in describe source and test entry
points with those boundaries. The gate column gives the existing Make target
or shell command for the indicated component.

## Waves 0–1

| Component | Source | Key symbols or commands | Gate | Scope |
| --- | --- | --- | --- | --- |
| A0 | `lib/transformer/api_contract.esk` | `transformer-api-contract-version`, `transformer-api-modules`, `transformer-api-operations` | `make test-a0` | Draft API inventory and contract fixtures. Availability depends on the linked component. |
| E1 | `lib/transformer/error_public.esk` | `transformer-error?`, `transformer-error-category`, `transformer-error-operation`, `transformer-error-message`, `transformer-error-details`, `transformer-error-cause` | `make test-e1` | Structured error accessors. |
| E1 | `lib/transformer/error_internal.esk` | `transformer-error-make`, `transformer-error-raise`, `transformer-error-wrap-foreign` | `make test-e1` | Private construction and foreign-error wrapping. |
| E1B | `lib/transformer/error_consumer.esk` | `transformer-error?`, `transformer-error-category` | `make test-e1b` | Accessors for separately compiled consumers. |
| E1B | `native/e1b_error_consumer_private.esk` | `et-e1b-private-raise` | `make test-e1b` | Fixed-arity private raise boundary. |
| P1 / P1L | `lib/transformer/module.esk` | `module-parameters`, `module-buffers`, `module-state-dict`, `module-load-state-dict!`, `state-dict-release!` | `make test-p1`, `make test-p1-native` | Named trees, buffers, tied parameters and owned state dictionaries. |
| P1 / P1L | `internal/p1/lib/transformer/module.esk` | `module-create-with-provider-internal`, `tensor-provider-register-internal!` | `make test-p1` | Trusted module construction and provider registration. |
| T1 | `lib/transformer/tokenizer.esk` | `tokenizer-byte`, `tokenizer-load`, `tokenizer-save!`, `tokenizer-encode`, `tokenizer-decode`, `tokenizer-fingerprint`, `tokenizer-special-token-id` | `make test-t1` | Canonical byte tokenizer with configured special tokens. |
| D1 | `lib/transformer/data.esk` | `token-corpus-write!`, `token-corpus-validate`, `token-corpus-summary-total-tokens` | `make test-d1` | Versioned, checksummed manifest and token shards. A successful write publishes a flat token stream. |
| K1 | `include/eshkol_transformer/kernel_abi.h` | `et_kernel_provider_v1`, `et_kernel_runtime_discover`, `et_kernel_runtime_dispatch`, `et_kernel_runtime_destroy` | `make test-k1` | Native ABI 1.0, explicit discovery and checked dispatch. Unverified capabilities are unavailable. |
| I1 | `include/eshkol_transformer/i64_tensor.h` | `et_i64_tensor_create_v1`, `et_i64_tensor_provider_v1` | `make test-i1` | Dense exact signed-i64 CPU storage and bounded storage-copy provider. |
| C1 | `internal/c1/lib/transformer/checkpoint_internal.esk` | `c1-checkpoint-save-state-internal!`, `c1-checkpoint-load-state-internal`, `c1-checkpoint-inspect-internal` | `make test-c1` | Private C1 container for logical P1 state, with checksummed payloads and checked atomic file I/O. See C2 for the public persistence facade. |
| X1 | `lib/transformer/config.esk` | `config-parse`, `config-resolve`, `config-validate`, `config-canonical`, `config-fingerprint`, `config-ref` | `make test-x1` | Declarative config and canonical resolved-run identity. |

## Waves 2–3

| Component | Source | Key symbols or commands | Gate | Scope |
| --- | --- | --- | --- | --- |
| I2 | `include/eshkol_transformer/f32_tensor.h` | `et_f32_tensor_create_v1`, `et_f32_tensor_provider_v1` | `make test-i2`, `make test-i2-native` | Exact-bit dense CPU-f32 storage, borrowed K1 views and P1 gradient/mutation substrate. |
| K2 | `lib/transformer/capabilities.esk` | `capability-discover`, `capability-request`, `capability-verified?`, `capability-require` | `make test-k2` | Process-local capability facade over the audited K1/I2 discovery table. |
| N2 | `include/eshkol_transformer/n2_primitives_abi.h` | `et_n2_kernel_provider_v1` | `make test-n2` | Exact CPU-f32 rows for embedding, linear, normalization, activation, dropout and residual forward/VJP operations. |
| N3K | `include/eshkol_transformer/n3k_primitives_abi.h` | `et_n3k_kernel_provider_v1` | `make test-n3k` | Exact diagnostic CPU-f32 primitive rows and initializer. |
| A2 | `include/eshkol_transformer/a2_attention_abi.h` | `et_a2_kernel_provider_v1` | `make test-a2` | Bounded CPU-f32 attention and RoPE with forward/backward coverage. |
| A2 | `include/eshkol_transformer/a2_kv_cache.h` | `et_a2_kv_cache_create_v1`, `et_a2_kv_cache_transaction_begin_v1`, `et_a2_kv_cache_transaction_commit_v1` | `make test-a2` | Owned cache with staged updates and scoped views. |
| L2 | `include/eshkol_transformer/indexed_cross_entropy.h` | `et_l2_indexed_cross_entropy_provider_v1` | `make test-l2` | Bounded indexed CPU-f32 token loss and backward. |
| L3S | `include/eshkol_transformer/l3s_masked_objective_abi.h` | `et_l3s_kernel_provider_v1` | `make test-l3s` | Masked reduction and explicit gradient seeds for the fixed CPU-f32 objective contract. |
| O2 | `lib/transformer/optim.esk` | `optimizer-create`, `optimizer-step!`, `optimizer-zero-grad!`, `optimizer-state`, `optimizer-load-state!`, `optimizer-state-release!` | `make test-o2` | Bounded AdamW, groups, clipping, accumulation, schedules and state continuation. |
| D2 | `lib/transformer/data.esk` | `token-dataset-open`, `token-dataset-next-batch`, `token-dataset-cursor`, `token-dataset-seek!`, `token-dataset-close!`, `token-batch-loss-mask`, `token-batch-release!` | `make test-d2` | Bounded shard loading, packing, shuffle, loss masks and cursor restore. |
| T2 | `internal/t2/lib/t2_bpe_core.esk` | `t2-bpe-train-core`, `t2-stream-encoder-open`, `t2-stream-encoder-push!`, `t2-stream-encoder-finish!`, `t2-stream-decoder-push!` | `make test-t2` | Private deterministic BPE training and streaming. Use the T1-compatible tokenizer facade for ordinary artifact operations and CLI3 for training a BPE artifact. |
| C2 | `lib/transformer/persistence.esk` | `persistence-policy`, `checkpoint-inspect`, `checkpoint-metadata-ref`, `checkpoint-load`, `checkpoint-save!` | `make test-c2`, `make test-c2-public`, `make test-c2-checkpoint-operational` | Detached training-state schema, validated load and atomic save. Continuation of an entire live trainer has separate gates. |
| M3 | `lib/transformer/model.esk` | `model-forward`, `model-output-logits`, `model-output-release!`, `diagnostic-output-vjp!` | `make test-m3` | Fixed-profile decoder composition and explicitly released outputs. |
| M3T | `lib/transformer/diagnostic_transport.esk` | `diagnostic-model-create`, `diagnostic-input-create`, `diagnostic-workspace-begin!`, `diagnostic-workspace-vjp-begin!`, `diagnostic-workspace-release!` | `make test-m3t` | Bounded Eshkol/native diagnostics with scoped model, input, logits and workspace lifetimes. |
| TR3 | `lib/transformer/trainer.esk` | `trainer-create`, `trainer-step!`, `trainer-state`, `trainer-load-state!`, `trainer-release!` | `make test-tr3b`, `bash scripts/test-tr3-fresh-process-resume.sh` | Trainer facade and bounded objective/resume witnesses. Public admission has separate gates. |
| E3 | `lib/transformer/evaluation.esk` | `diagnostic-evaluate-fixed!`, `diagnostic-evaluation-f32-bits`, `diagnostic-evaluation-count` | `make test-e3-native-parity`, `make test-e3-d2` | Fixed-profile evaluation and dataset/metric witnesses. |
| G3-N | `include/eshkol_transformer/g3n_primitives_abi.h` | `et_g3n_kernel_provider_v1` | `make test-g3n` | Bounded generation-side numerical primitives. |
| G3-S | `include/eshkol_transformer/g3s_sampling_abi.h` | `et_g3s_kernel_provider_v1` | `make test-g3s` | Bounded native sampling primitives. |
| G3-C4 | `include/eshkol_transformer/g3c4_primitives_abi.h` | `et_g3c4_kernel_provider_v1` | `make test-g3c4` | Bounded native generation primitives. |
| CLI3 | `src/eshkol_transformer/cli.esk` | `et_e1b_public_cli3_dispatch_v1`, `cli3-corpus-validate-command` | `make test-ci-cli3-after-build` | Eshkol executable entry for the installed command set. |
| CLI3 | `native/cli3_extension.esk` | `tokenizer byte`, `tokenizer train-bpe`, `tokenizer inspect`, `corpus build`, `corpus inspect`, `checkpoint inspect`, `pretrain`, `evaluate` | `make test-ci-cli3-after-build` | Installed commands. Pretraining is a fixed CPU-f32 profile with 1–4 updates per invocation. |
| CLI3 | `lib/transformer/cli3_corpus_validate.esk` | `corpus validate`, `cli3-corpus-validate-command` | `make test-ci-cli3-after-build` | D1 validation with explicit byte/token limits. |
| CLI3 | `native/cli3_generate_extension.esk` | `generate`, `--prompt-hex`, `--sampling` | `make test-ci-cli3-after-build` | Fixed-profile C2 restore and one-byte generation. |

## Bounded DATA4 installed slice

| Component | Source | Key symbols or commands | Gate | Scope |
| --- | --- | --- | --- | --- |
| DATA4 | `internal/data4/lib/data4_corpus.esk` | `data4-corpus-prepare`, `data4-corpus-write!` | `make test-ci-data4-after-build`, `bash scripts/test-data4-source.sh`, `bash scripts/test-data4-native.sh` | Private bounded byte filtering, exact deduplication and document provenance over T2/D1. |
| DATA4 | `internal/data4/lib/data4_report_reader.esk` | `data4-report-validate` | `make test-ci-data4-after-build` | Canonical v2.0 report validation against the D1 summary and exact manifest bytes; v1.0 is rejected. |
| DATA4 / CLI3 | `native/cli3_data4_extension.esk` | `corpus prepare`, `corpus report-validate`, `data4-report.tsv` | `make test-ci-cli3-after-build`, `bash scripts/test-cli3-data4-installed.sh` | Installed authoring and validation. D1 corpus and report sidecar publish separately. |

## Build and validation

| Component | Source | Key symbols or commands | Gate | Scope |
| --- | --- | --- | --- | --- |
| F0 / R0 | `toolchain/eshkol.lock` | `eshkol_commit`, `supported_os`, `clang_version` | `make toolchain`, `make configure`, `make build` | Pinned compiler identity and build entry points. |
| Q0 | `tests/q0/test_python_isolation.py` | — | `make test-q0`, `make test-python-isolation` | Validation harness and Python development/interop boundary. |
| B0 | `scripts/test-b0.sh` | — | `make test-b0`, `make benchmark-attention-supported` | Benchmark evidence for the specifically measured host, build and shapes. |

See the [public API contract](PUBLIC_API_CONTRACT.md) for signatures and
ownership conventions, the [roadmap](ROADMAP.md) for admission state, and the
[walkthrough](DOC5_CORPUS_TO_MODEL_TUTORIAL.md) for a reproducible test run.
