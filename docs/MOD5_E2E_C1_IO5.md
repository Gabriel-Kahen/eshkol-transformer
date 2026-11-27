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

## Proposed width-16 C2 gate

[PR #266](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/266)
merged the V=16 native storage proof as `7f8933034`. It admits the private
I2 1.2 descriptor and its `[8,16]`, `[16,16]`, `[16,32]` and `[32,16]`
copy rows. [PR #265](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/265)
merged the separate width-four C2 witness as `e34a6a8e3`, exercising nine
parameters and 18 moments through the standard K2 profile. The
[V=256 storage-copy candidate](#v256-storage-copy-candidate) passed focused
native tests. Opt-in K2 work follows its admission.

K2 admission alone cannot qualify a coherent C2 image for the V=16 decoder.
[`c2-d2-cursor-pair-internal`](../native/c2_d2_cursor_pair_extension.esk)
requires a canonical T1 or T2 fingerprint and a vocabulary of at least 256.
The [native C2 parser](../native/c2_checkpoint_format.c) recognizes those
fingerprint families and requires the cursor vocabulary to equal canonical X1.
The V=16 D1 fixture uses `mod5-e2e-token-v1`. C2 qualification of that exact
fixture would require a separately accepted tokenizer and cursor contract.

Use the existing [V=256 scoring geometry and T1/D1 binding](EVAL5_BENCHMARKS_CONTAMINATION.md#measured-v256-compiled-scoring-witness)
for the proposed CPU-f32 C2 fixture. That witness establishes C1 evaluation
on the geometry. Keep the block and final norm roster above. The head and
embedding each become `[256,16]`. Use a genuine T1 byte
tokenizer, a D1 corpus bound to its owner fingerprint, and current and epoch-start
cursors from the accepted D2 reader. Canonical X1 should declare vocabulary 256,
context 8, hidden width 16, one layer, four query heads, two KV heads, head
width 4, deterministic CPU f32, accumulation one, and the Philox key as
`run.seed`. Check those values and the model profile's fixed FFN width 32,
exact paths, shapes and permitted tie explicitly. Generic C2 checks cursor
vocabulary and the Philox key against X1, and checks O2 against C1. The MOD5
geometry-to-roster comparison belongs in the explicit profile gate because
X1 1.0 lacks FFN and tie fields and generic C2 does not enforce a model-specific
roster.

Both variants have 12 logical C1 entries. The untied variant has 12 unique
parameters and 24 O2 moments, for 36 staged descriptors. The tied variant has
equal-bit payloads at both paths and one logical `head.weight` /
`token_embedding.weight` alias group. `head.weight` is the canonical O2 path.
It has 11 unique parameters and 22 moments, for 34 staged descriptors. C1 LOAD
adopts distinct detached carriers per entry plus logical alias metadata. A
live tied receiver would use one shared parameter and sum both gradient
contributions into it. Each canonical parameter needs one O2 group assignment
and two same-shape moments. The accepted I2 decode and SAVE copy seams and
O2 reconstruction builder take bounded ranks and extents. This source supports
reusing the detached owner path with the new shapes. Exact width-16 owner
transfer, moment bytes, group options, schedule and release still need runtime
evidence.

### Order of admission and evidence

1. After #266 is admitted, extend the opt-in native proof with the exact
   `[256,16]` CPU-f32 `storage.copy` row and an audited provider identity and
   report for the V=256 fixture. Match the roster above with those two path
   substitutions. Check byte-exact copies, `[16,256]` and other unlisted
   shapes, same-byte mismatched views, and untouched output on failure. Pin
   the report bytes and hidden symbol inventory. Keep ordinary I2 1.1 and
   its existing report and dispatch.
2. After that native proof, admit explicit K2 profile selection through one
   reviewed aggregate with a fixed private resolver. Audit the descriptor,
   all eleven live K1 rows, and the Eshkol report and entries before publication.
   Keep `(capability-discover)` on the default profile. Bind an opt-in report
   to its exact factory, process, live generation and profile, and require
   that native K1 matching and returned entry snapshots use the same profile.
   Preserve the authenticated factory and generation checks on requests and
   entries. In particular, the current `capability-request` constructor
   ensures the default runtime, so profile admission must prove that request
   creation cannot silently switch the selected report's runtime. Profile
   choice comes from the trusted opt-in path. The checkpoint's persisted I2
   provider ID remains data. Each fresh process must discover and audit its own
   runtime. Under the existing serialized fork contract an inherited parent
   report is invalid. A failed audit publishes no candidate runtime. Pin the
   opt-in source closure, defined and undefined symbols, exports and strings
   separately from the
   ordinary aggregate. Keep provider accessors and resolver authority private.
3. After the #265 and #266 predecessors and both profile steps are admitted,
   exercise real tied and untied V=256 images with
   `(checkpoint-load path policy capability-report)`,
   `(checkpoint-save! state path policy options)` and
   `(trainer-state-release! state)`. Start with a genuine MOD5 decoder roster
   and an O2 update-boundary snapshot with nonzero moments. In a fresh OS
   process, match the authenticated report to the explicit profile and check
   the validated staged roster and controls before I2/O2 reconstruction.
   Preserve the C2 loop that calls `(capability-request 'storage.copy 'f32
   'cpu shape #t)` and `(capability-require report 'tensor.f32 request)` for
   each model and moment descriptor.
   Use the existing C2 owner composition, verify all component bits and
   alias topology, re-save byte-identically and verify idempotent release and
   rejection of released-owner use. SAVE has no report argument. Its accepted
   authority is the exact idle C2 owner and its scoped component borrows.
4. Require pre-reconstruction rejection of forged, stale and wrong-profile
   reports, unlisted shapes and valid-checksum mutations of the roster, tie,
   moment shape, optimizer group coverage, X1 geometry and tokenizer/cursors.
   Preserve staged C2 errors such as `corrupt-data` for inconsistent alias
   payloads and `shape-mismatch` for inconsistent moment extents. Preserve
   `invalid-argument` for forged reports, `invalid-state` for stale generations
   or origins, and `unsupported` for genuine K1 no-matches. Reject a genuine
   wrong-profile report as `unsupported` before shape matching. Exercise the
   profile check with an overlapping shape such as `[16]`. Test the model
   roster check with a structurally valid alternate path or tie whose O2
   records still agree, so generic C2 validation cannot mask that check.
   Keep staging cleanup and zero reconstruction on these paths. Check partial
   reconstruction rollback, no publication on rejected SAVE, and the default
   K2/C2 package gates alongside the opt-in closure. Seal the supported receipt
   against the exact source and artifact. A subsequent continuation gate can
   qualify a restored live receiver and a matched training suffix.

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
ties. Training resume, optimizer state, alternate devices, and execution by
HF or GGML model runtimes require separate validation.

## Trained Q8_0 GGUF scoring

The same gate exports both trained variants as tagged `eshkol-decoder-v1`
GGUF with explicit F32 fallback, then imports to C1 and scores the D1 window
in fresh CPU F32 processes. The `[16,32]` FFN down projection occupies 16
Q8_0 blocks (544 bytes). The eight other matrices and three RMSNorm vectors
retain their exact F32 bits. Tensor payloads total 9,952 bytes versus 11,456
bytes for F32. The importer restores each variant's C1 alias topology,
including both head and embedding paths of the tied model.

The test independently inspects the wire descriptors, scales and signed
quants, then checks the decoded weights, repeatable scores, changed logits
digests and mean-loss drift. A corrupt quant byte must fail import before
creating a C1 file or replacing an existing one.

Ubuntu 22.04 with Clang/LLVM 21.1.8 produced byte-identical AOT executables
from two fresh builds of source tree `02964ade`. All six MOD5 end-to-end
tests passed, including the F16 and BF16 safetensors routes and both Q8_0
scoring cases. The Q8_0 gate requires nonzero absolute mean-loss drift of
at most 0.01 for both variants.

| Variant | Imported mean loss | Absolute mean-loss drift | Maximum absolute weight delta |
|---|---:|---:|---:|
| Untied | 1.5492229154686037 | 1.686e-5 | 5.924e-4 |
| Tied | 1.9232759077819499 | 3.613e-6 | 5.590e-4 |

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

### V=256 storage-copy candidate

The [V=256 scoring fixture](EVAL5_BENCHMARKS_CONTAMINATION.md#measured-v256-compiled-scoring-witness)
keeps this decoder's width-16 block and uses the canonical T1 byte tokenizer.
Its head and embedding both have shape `[256,16]`. The private flag
`ET_I2_PRIVATE_MOD5_WIDTH16_V256_STORAGE_COPY` adds that exact row through
`et_i2_private_mod5_width16_v256_storage_provider_v1()`. The hidden accessor
returns `eshkol-transformer-f32` version `1.3`, evidence
`I2:mod5-v256-exact-f32-storage.copy-v1`, and K1 ABI 1.0. Its deterministic
`tensor.f32/storage.copy` row includes the complete I2 1.2 shape set plus
`[256,16]`. The I2 1.1 and V=16 I2 1.2 descriptors both reject the added
shape, including when built in the same object as I2 1.3.

Run `CC=clang-21 bash scripts/test-mod5-width16-storage-copy.sh --v256
[evidence-directory]` on the supported Ubuntu 22.04 x86-64 toolchain. The gate
checks the generated V=256 roster, all admitted rows, byte-exact copies and
untouched buffers on transposed, neighboring and equal-byte mismatched shapes.
It pins standalone object symbols and rejects both opt-in descriptors at the
default K2 audit, including relabeled variants. The optimized, repeated and
ASan/UBSan/LSan V=256 K1 reports are 3,225 bytes with SHA-256
`50e09d8e0bf757492e1a0c10aacaf58ba124d18f6e48b65cf3f21f41dd288ab6`.
The combined build also checks the original I2 1.1 and I2 1.2 report hashes.
Explicit K2 profile admission and tied/untied C2 LOAD/SAVE remain separate gates.
