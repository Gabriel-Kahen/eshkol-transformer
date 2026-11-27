# MOD5 training, C1 checkpoint, and IO5 round trips

The test executable writes a D1 1.0 manifest and one checksummed
`shard-0000000000000000.ets` with vocabulary size 16 and fixture tokenizer
identity `mod5-e2e-token-v1`. Each training or evaluation process uses the
trusted D1 validators to read the manifest and the shard via bounded native
I/O. It consumes the validated shard bytes directly. The sequence
`1 4 1 5 9 2 6 5 3` gives eight input tokens and their eight next-token
targets. The repeated input token also exercises accumulation in the
embedding gradient.

The Eshkol AOT decoder uses one bias-free pre-norm block with residuals,
four query heads, two key/value heads and a head width of four. It uses
causal attention across the full eight-token context. MOD5 RMSNorm,
SwiGLU and GQA forward and backward dispatch through the existing native
K1 provider. Eshkol tensor operations implement the embedding lookup,
projections, rowwise log-sum-exp mean cross-entropy, gradient composition
and SGD with learning rate 0.25. RMSNorm uses epsilon 0.125. Each update
rounds the weights to binary32 before the next forward pass. The tied
variant uses one shared live tensor and sums the
embedding and output-projection gradients before its update.

## C1 contract

| Canonical path | Shape |
|---|---|
| `blocks.0.attention.key.weight` | `[8,16]` |
| `blocks.0.attention.output.weight` | `[16,16]` |
| `blocks.0.attention.query.weight` | `[16,16]` |
| `blocks.0.attention.value.weight` | `[8,16]` |
| `blocks.0.ffn.down.weight` | `[16,32]` |
| `blocks.0.ffn.gate.weight` | `[32,16]` |
| `blocks.0.ffn.up.weight` | `[32,16]` |
| `blocks.0.norm1.weight` | `[16]` |
| `blocks.0.norm2.weight` | `[16]` |
| `head.weight` | `[16,16]` |
| `norm_final.weight` | `[16]` |
| `token_embedding.weight` | `[16,16]` |

Matrices use `[output,input]` row-major order. C1 stores CPU F32 parameters
through the trusted P1 state and production `i2-dense-cpu-f32-v1` codec.
The tied checkpoint declares a C1 head/embedding alias group and stores
matching payloads at both paths. The loader checks the ordered paths, kind,
shape, dtype, device, layout, finite values, and permitted alias topology.
For the tied variant, it also checks the exact bits of both payloads before
restoring the shared live tensor. The canonical model profile supplies the
parameter names and topology. The C1 file carries its own format version
and provider identity.

## IO5 gate

| Route | Import selection |
|---|---|
| Mapped HF safetensors | Embedded `eshkol-mod5-v1` metadata |
| Plain HF safetensors | Explicit HF scheme, MOD5 profile, provider and tie if applicable |
| Sharded HF safetensors | `model.safetensors.index.json` with multiple shards bounded at 3,000 bytes |
| Mapped GGUF v3 | Embedded `eshkol-mod5-v1` metadata |
| Plain GGUF v3 | Explicit GGUF scheme, MOD5 profile, provider and tie if applicable |

For both variants, all five restored C1 files must match the trained
checkpoint byte for byte. Each restore runs in a fresh AOT process and
must reproduce the live training process's final binary64 loss bits and
SHA-256 over the row-major little-endian binary32 logits. A correctly
checksummed model with a different head weight must produce a different
evaluation. The gate also checks D1 corruption and out-of-range token IDs,
IO5 corruption and invalid rosters, and live C1 checksum, roster and
shape rejection. Plain external formats rely on explicit import metadata
and validate structure. Checksum rejection applies where an expected
checksum is available.

The two tagged MOD5 safetensors routes cast the trained weights to F16 or BF16
and restore F32 C1 for evaluation. They cover both tied and untied variants.
The gate compares every restored path, shape and alias against the source.
Each restored model produces a changed logits digest, repeatable in a second
fresh process. Test limits for the maximum absolute weight error and absolute
mean evaluation loss change are 0.001 and 0.002 for F16, and 0.008 and 0.010
for BF16. See the [wire contract](IO5_QUANTIZED_INTERCHANGE.md#tagged-mod5-safetensors).

## Running and measured results

Run `bash scripts/test-mod5-e2e-io5.sh` with the pinned toolchain on Ubuntu
22.04 x86-64. The script builds twice with `--strict-types --optimize 0
--no-stdlib` and fresh build and cache directories, compares the ELF bytes,
and records the SHA-256 values and training traces.

The replay on main `d0b1adb228842ee2cd3c018f3374fedb8aa902bb` used
`eshkol-r0-debug:20260928` with network disabled, Ubuntu 22.04,
Clang/LLVM 21.1.8, and Eshkol
`fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`. The untied model passed
12 numerical gradient samples and the tied model passed 11, one selected
coordinate per distinct live parameter. Both loss traces decreased on every
SGD step. All four end-to-end tests passed, including the ten IO5 round trips.

| SGD step | Untied mean loss | Tied mean loss |
|---:|---:|---:|
| 0 | 2.7694186496899564 | 2.855940798976941 |
| 1 | 2.7055240010175532 | 2.7925579130843405 |
| 2 | 2.6426960779905033 | 2.7404191715469755 |
| 3 | 2.5774006572314039 | 2.6901400330723062 |
| 4 | 2.5065263322588809 | 2.6354853481382041 |
| 5 | 2.4271715651356338 | 2.572202382986108 |
| 6 | 2.3366342245822134 | 2.4982455354921957 |
| 7 | 2.2326951496564007 | 2.4144605056984192 |
| 8 | 2.1143023941731909 | 2.3239122359837445 |
| 9 | 1.9824969699220847 | 2.2288863223227557 |
| 10 | 1.8408352730164956 | 2.1292195400083935 |
| 11 | 1.6945738379979245 | 2.0254713480357975 |
| 12 | 1.5492397748056204 | 1.923272295153039 |

| Artifact or payload | SHA-256 |
|---|---|
| Both AOT ELF builds | `ae7a1912c69354e3a0b672b9eb9313ea514078c7ac734f854b7e8661edcee4e4` |
| Untied C1, including all five restores | `362ad110daf3856879bdab074dc6b257771040b04a93901a05c95e435836bc87` |
| Tied C1, including all five restores | `44148061ce9129e8931a346a6604c12165b8186e597a19faed37cd33b8f01a41` |
| Untied evaluation logits | `48fea377772ead7ca60fa5deea440a209a894af54cb59e3a55ae89b880c08171` |
| Tied evaluation logits | `5db767f0ab9f65c3122fda29db9e5109d281c8f66ed8916bb161acc1b76feb83` |

The exact binary64 final losses are `0x1.8c9afa5673d9fp+0` (untied) and
`0x1.ec5b92b8fc339p+0` (tied).

To use an executable produced with the same flags, run these commands from
a working directory where it can write and read the fixture:

```sh
mkdir corpus
./mod5-e2e --mode write-corpus corpus
./mod5-e2e --mode train-save trained.etcp
./mod5-e2e --mode load-eval trained.etcp
```

Pass `--tied` after the output path to train with shared weights. Evaluation
selects the variant from the checkpoint alias group.

This gate covers a single deterministic D1 window and this decoder shape on
the pinned CPU toolchain. The checkpoint persists model weights and logical
ties. Training resume, optimizer state, alternate devices or dtypes,
quantized GGUF, and execution by HF or GGML model runtimes require separate
validation.

## Width-16 C2 storage prerequisite

The native opt-in proof covers the fixture's 16-token vocabulary, width 16,
FFN width 32 and key/value width 8. All 12 logical C1 paths use `[16]`,
`[8,16]`, `[16,16]`, `[16,32]` or `[32,16]`. The rank-one row already covers
`[16]`. The private I2 1.2 descriptor with evidence
`I2:bounded-exact-f32-storage.copy-v3` appends the four exact matrix rows to
I2 1.1. It advertises deterministic CPU-f32 `storage.copy` through K1 ABI
1.0. Dispatch uses the existing dense, zero-offset, non-overlapping copy
preflight and copies the bytes without reinterpretation.

Run `scripts/test-mod5-width16-storage-copy.sh [evidence-directory]` for the
native and sanitizer proof. It checks the merged fixture's roster, both exact
descriptors and K1 reports, legacy and width-16 copies, and unchanged buffers
on unlisted shapes and an equal-byte mismatched view. The opt-in build exposes
one hidden accessor. The canonical accessor still returns I2 1.1 and the
existing K2 audit rejects both I2 1.2 and attempts to relabel its descriptor.
Default K2 rejects all four new matrix requests at the capability check used
by C2 before reconstruction.

The next gate is an independently reviewed K2 profile with fixed provider
selection and an exact native descriptor, live K1 row and Eshkol report audit.
The report must authenticate its process, live generation and provider profile
when the existing public C2 LOAD path requires each authenticated staged model
and optimizer descriptor. The supported C2 witness must prove tied and untied
LOAD, byte-identical SAVE, owner release, and early rejection of a forged or
stale report, a report from the default profile, and unlisted shapes. Native
storage admission alone does not establish optimizer or training resume.
