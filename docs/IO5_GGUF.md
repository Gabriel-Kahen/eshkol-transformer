# IO5 bounded GGUF interoperability 1.0

This standalone interoperability tool translates C1 1.0 CPU dense contiguous f32
logical-state bytes to and from a conservative GGUF subset. It reuses the reviewed
[safetensors slice's C1 primitives and file boundary](IO5_SAFETENSORS.md), without
changing that implementation. It performs no provider lookup, P1 registration,
model construction, live-state ingress or training. The delivered training path
remains Eshkol-native. GGUF metadata cannot select executable behavior.

The safetensors predecessor is merged through PR #217. This GGUF slice has
independent source/test approval and a required supported CI lane; integration
and exact hosted evidence are tracked in [PR #222](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/222).
IO5 remains active; model-name mapping and live model/state integration are later
gates. Neither format slice establishes full trainer-state or resume equivalence.

## Commands

```sh
python3 tools/io5/gguf_interop.py c1-to-gguf input.etcp output.gguf
python3 tools/io5/gguf_interop.py gguf-to-c1 output.gguf restored.etcp
# Explicit external tensor semantics, with inert provider spelling:
python3 tools/io5/gguf_interop.py gguf-to-c1 external.gguf output.etcp \
  --plain-provider i2-dense-cpu-f32-v1
make test-io5-gguf
```

Default import requires the IO5 manifest. `--plain-provider` explicitly treats
each complete external tensor name as **one** literal P1 path segment, every tensor
as a parameter, and the state as having no aliases. Dots, slashes and Unicode are
not split or normalized. External metadata is discarded; no tied-value inference
or module compatibility is claimed. The flag rejects any IO5 metadata rather
than overriding incomplete, unknown or corrupted semantic metadata.

## Accepted wire subset

The primary [GGUF specification](https://github.com/ggml-org/ggml/blob/417a024b4fb6bc3116d41af7c067166a039bdf12/docs/gguf.md)
and [native reader implementation](https://github.com/ggml-org/ggml/blob/417a024b4fb6bc3116d41af7c067166a039bdf12/src/gguf.cpp)
were inspected for the format and stricter native admission rules. This tool
accepts only:

- GGUF version **3**, little endian, magic bytes `GGUF`.
- Tensor type **0 (F32)**, rank **1–4**, positive extents through signed-i64 maximum,
  with checked u64 products and bounded exact payload size. Scalar and zero-extent
  tensors reject as unsupported by this slice; this does not label all upstream
  representations invalid.
- Nonempty UTF-8 tensor names of at most **63 bytes**, without NUL. Names are unique.
- **32-byte alignment**: `general.alignment` is UINT32 value 32, or absent with
  default 32. Other alignments reject even when valid for another GGUF consumer.
- Tensor offsets relative to the data section, contiguous in **descriptor order**
  after rounding each payload size up to 32. All prefix/inter-tensor/final padding
  is zero. The final tensor's padding is required and ends exactly at EOF. Empty
  tensor sets are accepted with an aligned prefix and no data.
- STRING metadata, plus the one UINT32 alignment field. Keys are unique, at most
  128 bytes, and match `[a-z0-9_]+(\.[a-z0-9_]+)*`. UTF-8 strings validate strictly.
  `general.architecture` is required, matching `[a-z0-9]+`. Export uses `eshkol`
  solely as an **inert data-profile label**, without an executable architecture,
  architecture-specific name mapping or inference compatibility claim.

GGML stores its fastest-varying dimension first. C1 stores the fastest-varying
dimension last. Export therefore **reverses the dimension list and leaves every
payload byte unchanged**; import reverses the list back. The independent upstream
Python writer/reader test proves this convention for non-square ranks 1–4 using
distinct indexed bit patterns, rather than relying on a self-round-trip alone.

Big endian, other versions, quantization, other tensor types, scalar/empty tensors,
strided representations, other metadata types (including arrays and numeric model
configuration), `general.quantization_version`, any `split.*` metadata, shard/index
composition, descriptor reordering, holes, overlaps, nonzero padding, truncation
and trailing bytes reject explicitly. This is a data interchange subset, not a
general GGUF model loader or converter for arbitrary inference files.

## Versioned semantic metadata

Exports use synthetic tensor names `t000000`, `t000001`, etc. in canonical C1
entry order. Three STRING metadata fields retain logical-state information:

| Key | Value |
|---|---|
| `eshkol.io5.gguf_version` | Exactly `1.0` |
| `eshkol.io5.manifest` | JSON object string with exactly `provider`, `entries`, `aliases` |
| `eshkol.io5.c1_sha256` | Lowercase hexadecimal SHA-256 of the complete canonical C1 file |

The manifest schema and strict JSON/alias/path rules match the
[safetensors semantic contract](IO5_SAFETENSORS.md): provider spelling is inert;
entries contain exactly `name`, segmented `path`, and `kind`; paths are in strict
P1 UTF-8 lexical order. Every GGUF tensor is referenced exactly once. Alias groups
are disjoint, sorted, parameter-only, and require equal shapes and exact bits.
Aliased payloads are duplicated in GGUF, while metadata preserves logical ties.
Unknown `eshkol.io5.*` keys reject. Other supported STRING metadata is ignored.
IO5 metadata requires architecture label `eshkol` and cannot be plain-imported.

Import validates metadata, descriptors, all bounds/padding, and the reconstructed
C1 checksum before publication. A supported C1→GGUF→C1 round trip returns **the
identical complete C1 bytes**, preserving provider text, segmented paths, kinds,
alias topology, shapes and every f32 representation, including signed zero,
subnormals, signaling/quiet NaN payloads and infinities. There is no float cast,
transpose, arithmetic, quantization, dtype conversion or approximate comparison.
C1 input retains the predecessor's strict container/tensor checksum and state
validation; bool/i64 buffers and unsupported versions/devices/layouts reject.

GGUF itself provides no integrity checksum. IO5's checksum is **unkeyed**, binds
the reconstructed C1 semantics, and does not authenticate a sender or bind ignored
metadata, wire formatting or padding independently of structural validation. An
attacker can recompute it. Plain imports cannot detect otherwise valid f32 bit
flips. Dropping IO5 metadata prevents default logical-state reconstruction.

## Bounds and file publication

The tool shares the predecessor's lowering-only `--max-file`, `--max-metadata`,
`--max-tensor`, and `--max-entries` policies: maxima 64 MiB file, 1 MiB metadata,
32 MiB per tensor, and 4096 tensors. For GGUF, the metadata ceiling covers the
**complete prefix including fixed header, metadata, tensor descriptors and prefix
padding**; manifest JSON is independently bounded by the same ceiling. At most
128 metadata pairs are admitted. C1 path/provider/alias bounds remain unchanged.
All wire integers are unsigned, sizes/products/rounding must fit u64, and signed-i64
extent admission occurs before payload slicing. Counts and prefix spans are
checked before iteration, string copying, rank allocation or tensor extraction.

Conversion is non-streaming with bounded copies; memory high-water marks and
performance are unmeasured. The CLI uses the predecessor's regular non-symlink
bounded reads, same-directory mode-0600 temporary, no-clobber atomic publication
or explicit `--overwrite`, and file/directory fsync. Invalid input leaves an
existing destination unchanged. Directory fsync failure may report an error
after publication; crashes can leave untrusted temporaries. The predecessor's
trusted-parent-directory and concurrent same-size-input limitations apply. This
slice adds no native filesystem ABI or native fault-injection claim.

## Focused evidence and next gates

Install the reference only into a development interpreter:

```sh
python -m pip install -r tests/io5/requirements-gguf-reference.lock
IO5_GGUF_REFERENCE_PYTHON=/absolute/path/to/python make test-io5-gguf
```

The configured gate **requires** `gguf==0.19.0`, `numpy==2.5.4`, and the existing
`safetensors==0.6.2` reference before running tests. Missing or wrong packages
fail the gate. Default stdlib-only execution explicitly skips unavailable upstream
references and supplies no independent compatibility claim. Scratch resides in
the worktree's `.tmp/scratch` on `/home`; no Eshkol build prerequisite is invoked.

Ten GGUF unittest groups generate deterministic fixtures and malformed negatives,
including integer/count/length overflow, endian/version/dtype/alignment rejection,
dimension ordering, exact f32 bits, logical ties and metadata corruption, padding,
descriptor contiguity, exact/one-over policies, 256 seeded byte mutations and CLI
failure/no-clobber behavior. The existing independent C1 builder/parser validates
the C1 fixtures. Actual `gguf` reader checks every exported shape and exact payload;
actual writer repacks IO5 metadata/tensors for exact C1 restoration and independently
generates plain rank 1–4 fixtures. NumPy is confined to this development reference.
The entire eleven-group safetensors regression suite and five Q0 Python isolation
groups also run, keeping the existing tool source unchanged. A further native
witness is explicitly skipped unless its existing-library prefix is configured.

### Required hosted coverage

The shared full-coverage engine adds `Supported suite / io5-gguf` on Ubuntu 22.04
with pinned Python 3.14.6. It installs the combined reference lock and requires
`IO5_GGUF_REFERENCE_PYTHON` when running `make test-io5-gguf`, so missing references
cannot yield a successful skipped compatibility gate. All 21 compiled matrix
suites and their prerequisites remain unchanged. Suite evidence requires both
the compiled matrix and IO5 job to succeed; the 22-suite inventory rejects old
evidence or successful job lists missing GGUF, including exact-tree and docs-base
reuse. The existing fail-closed change selector remains unchanged, with explicit
tests that IO5 tool, test, lock and script changes require full coverage.

### Existing native reader/writer witness

An already installed GGML runtime supplies the local native witness; no native
dependency is downloaded or built. Configure its prefix explicitly:

```sh
IO5_GGUF_NATIVE_PREFIX=/absolute/path/to/existing/runtime make test-io5-gguf-native
```

The target fails when the prefix is absent or unusable. The development Python
test compiles only a small C caller against that prefix's headers and library,
records their SHA-256 identities and the compiler version, and checks runtime
library resolution. The native reader observes each generated tensor's dimensions
and exact bytes; a fresh native writer copies metadata and adds the live allocated
tensors, then produces a GGUF file that restores the identical original C1 bytes.
The fixture covers ranks 1–4, non-square shapes, special f32 bits, buffer/parameter
kinds and a logical alias. Its dimensions are non-unit: native writers may
canonicalize unit-dimension ranks, so this witness does not establish arbitrary
shape-preserving native reserialization. Dropping such rank information causes
the IO5 checksum to reject rather than silently returning a different state.

Local validation used the existing `t3-local-models/runtime` GGML 0.27.0 library
whose installed headers match upstream llama.cpp commit
[`1d88cb05f1a257f90741538ca3aca6c1b015c6d3`](https://github.com/ggml-org/llama.cpp/tree/1d88cb05f1a257f90741538ca3aca6c1b015c6d3/ggml).
The library SHA-256 is
`371a0a0a3de3d2250d032fc37484c87ddf86174732b3e7613bbf718f110a65e7`;
the small caller used host GCC 16.2.1. This is bounded local cross-reader/writer
evidence, not supported native CI or formal source/binary correspondence. Hosted
CI does not install this native dependency and reports the witness as skipped.

Local configured reference validation with the native prefix passed **22 IO5 +
5 isolation groups**; 112 CI unit/mutation tests and the topology check also pass.
Independent source/test review approved this GGUF slice after the descriptor-order
acceptance fix and its regression. The CI/native follow-up is independently approved;
the exact hosted result is recorded in PR #222, rather than inferred from local tests.
This evidence establishes no live M3 snapshot/load, forward output, provider
ownership, optimizer/RNG/tokenizer state, resume equivalence, GPU execution,
mixed precision or performance.
