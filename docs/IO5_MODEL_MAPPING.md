# IO5 model-name mapping and sharded safetensors 1.0

The standalone `tools/io5/model_mapping.py` tool converts data-only
[C1 1.0](CHECKPOINT_FORMAT.md) CPU dense contiguous F32 files. It uses the
[safetensors](IO5_SAFETENSORS.md) and [GGUF](IO5_GGUF.md) wire readers and writers.
Provider spellings are inert UTF-8 strings. The tool uses the Python standard
library and requires a Linux filesystem with the file operations documented below.

## Profiles and names

C1 paths are arrays of segments. The dots below show segment boundaries in the
C1 column. The external columns are literal tensor names. `i` is a decimal layer
index without a leading zero unless it is zero itself. Layer indices must form
a contiguous interval starting at zero. Each selected profile and scheme fixes a
bijective mapping. Unknown names, wrong segment boundaries and collisions fail.

| C1 path stem | `huggingface` stem | `gguf` stem | M3 weight shape |
|---|---|---|---|
| `token_embedding` | `model.embed_tokens` | `token_embd` | [256,4] |
| `position_embedding` | `model.embed_positions` | `position_embd` | [2,4] |
| `blocks.i.norm1` | `model.layers.i.input_layernorm` | `blk.i.attn_norm` | [4] |
| `blocks.i.attention.query` | `model.layers.i.self_attn.q_proj` | `blk.i.attn_q` | [4,4] |
| `blocks.i.attention.key` | `model.layers.i.self_attn.k_proj` | `blk.i.attn_k` | [4,4] |
| `blocks.i.attention.value` | `model.layers.i.self_attn.v_proj` | `blk.i.attn_v` | [4,4] |
| `blocks.i.attention.output` | `model.layers.i.self_attn.o_proj` | `blk.i.attn_output` | [4,4] |
| `blocks.i.norm2` | `model.layers.i.post_attention_layernorm` | `blk.i.ffn_norm` | [4] |
| `blocks.i.ffn.up` | `model.layers.i.mlp.up_proj` | `blk.i.ffn_up` | [8,4] |
| `blocks.i.ffn.down` | `model.layers.i.mlp.down_proj` | `blk.i.ffn_down` | [4,8] |
| `blocks.i.ffn.gate` | `model.layers.i.mlp.gate_proj` | `blk.i.ffn_gate` | Absent |
| `norm_final` | `model.norm` | `output_norm` | [4] |
| `head` | `lm_head` | `output` | [256,4] |

- **`eshkol-m3-v1`** requires all M3 weights in the table, exactly layer zero,
  and a `bias` of shape [4] for each of `norm1`, `norm2` and `norm_final`.
  The 15 paths represent 14 unique parameter tensors matching the canonical
  [M3 reference manifest](../tests/m3/reference_manifest.json) and
  [M3T transport roster](M3T_TRANSPORT_PROPOSAL.md). C1 must contain exactly
  one alias group linking `head.weight` to `token_embedding.weight`.
- **`eshkol-decoder-v1`** requires `weight` on every layer's norm, attention,
  up and down stems. It also requires token embedding, final norm and head
  weights. Zero layers are allowed. Position embedding and the SwiGLU gate
  weight are optional. Norm and projection biases are optional, including
  gate bias when gate weight exists. Norm weights and biases are rank one.
  Embeddings, projections and the head are rank two. Extents must be positive.
  The profile validates names and ranks independently of architecture-specific
  hidden dimensions, head counts and projection layouts.
- **`eshkol-mod5-v1`** requires one or more layers and exactly the
  `token_embedding.weight`, `norm_final.weight` and `head.weight` roots.
  Every layer requires both RMSNorm weights, the query, key, value and output
  projection weights, and all three SwiGLU projection weights (`up`, `down`
  and `gate`). It rejects biases and all learned `position_embedding` paths.
  Norm weights are rank one. Embedding, head and projection weights are rank
  two. All extents must be positive. The profile accommodates MQA, GQA and
  MHA projection shapes. It validates names and ranks and leaves validation
  of head counts, inter-tensor dimensions, and ALiBi/RoPE configuration to
  the model runtime.
- All profiles accept parameters only. A head/embedding tie requires both
  entries and identical shapes and F32 bytes. Both decoder profiles retain a C1 tie
  when present and preserves distinct entries when C1 has no alias, including
  distinct entries with equal bytes. Other alias groups fail. Wire payloads
  include both tensors even when C1 marks a tie.

## Mapped files and external import

Mapped safetensors and GGUF store string metadata with these exact IO5 keys:

| Key | Value |
|---|---|
| `eshkol.io5.mapping_version` | `1.0` |
| `eshkol.io5.profile` | Selected profile |
| `eshkol.io5.scheme` | Selected naming scheme |
| `eshkol.io5.manifest` | JSON string with exactly `provider`, `entries`, `aliases` |
| `eshkol.io5.c1-sha256` (safetensors) or `eshkol.io5.c1_sha256` (GGUF) | Lowercase hex SHA-256 of the complete C1 file |

The manifest has one entry per tensor in canonical C1 path order. Each entry
has exactly `name`, `path` and `kind`. Aliases are arrays of C1 entry indices.
The importer checks the name corresponding to each individual path in the chosen
profile and scheme, reconstructs C1, and verifies its checksum. All F32 bits
survive, including NaN payloads, infinities, subnormals and signed zero. GGUF
exports use the inert `general.architecture` label `eshkol`; the wire codec
reverses the dimension list for GGML and keeps payload bytes in place.

Plain external import requires `--external`, `--profile`, `--scheme` and
`--provider`. Files carrying any `eshkol.io5.*` metadata fail external import.
Names must exactly match the selected scheme. The importer creates a C1
parameter for each tensor, discards other supported metadata, and uses the
supplied provider spelling. In both decoder profiles,
`--tie-head-to-embedding` requests a tie and checks shape and exact bit
equality. The M3 profile always performs that check and creates its required
tie. Inputs omitting one of the two tensors fail. The plain import has no
preexisting semantic checksum for comparison.

The external wire domains are the predecessor subsets. Safetensors requires F32
and string metadata. GGUF requires little-endian v3, F32, alignment 32,
contiguous descriptors and zero padding. GGUF metadata must be STRING except
for the UINT32 alignment field. Real model files with array or numeric model
configuration, split GGUF, quantized tensors and other dtypes are outside the
admitted domain. Naming does not add model configuration, tokenizer, runtime
loading, weight permutation, model execution, or training-state restore.

## Sharded index

Export and import take an explicit `model.safetensors.index.json` path in an
existing directory. Export writes canonical UTF-8 JSON with sorted keys and
compact separators. The top-level object has exactly `metadata` and
`weight_map`. `weight_map` maps each external tensor name to one shard
basename. Export groups tensors greedily in C1 order and names files
`model-00001-of-0000N.safetensors`, starting at one. A shard is a standalone
F32 safetensors file. The index carries the IO5 metadata for the entire bundle.

| Metadata key | Value |
|---|---|
| `total_size` | Unsigned sum of tensor payload bytes, counting both copies of a tie and excluding headers |
| `eshkol.io5.sharded_version` | `1.0` |
| `eshkol.io5.manifest` | Manifest JSON string in the schema above |
| `eshkol.io5.c1-sha256` | Lowercase hex SHA-256 of the complete C1 file |
| `eshkol.io5.shard_sha256` | JSON object mapping every referenced shard basename to its lowercase full-file SHA-256 |
| `eshkol.io5.mapping_version`, `eshkol.io5.profile`, `eshkol.io5.scheme` | Present together for mapped exports |

For a mapped index supply both `--profile` and `--scheme` on export.
Without that pair, names are synthetic `t000000`, `t000001`, etc. in C1
order and the bounded C1 domain (including buffers, empty tensors and empty
states) is supported subject to the serialized index, metadata and shard byte
ceilings below (for example, JSON escaping of non-ASCII paths can exceed the
1 MiB index ceiling before the C1 file ceiling is reached). Empty states have
zero shards. On import, optional profile and scheme arguments assert existing
metadata values.

The reader enforces exact index key sets, unique keys at every JSON object
depth, valid UTF-8 and maximum depth 16. It accepts ASCII shard basenames
ending in `.safetensors`, with a maximum of 255 bytes and no `..`
substring, slash, backslash or NUL. The checksum map must cover exactly
the referenced shards. Each shard must contain exactly the tensor names
assigned to it. The reader verifies every file checksum, the exact total
payload size and the reconstructed C1 checksum before writing C1. Hashes are
unkeyed integrity checks. They do not authenticate a publisher.

## Bounds and publication

`--max-file`, `--max-metadata`, `--max-tensor` and `--max-entries` only lower
64 MiB per file, 1 MiB per metadata region, 32 MiB per tensor, and 4,096
tensors. The index is limited by both the file and metadata ceiling. Shard
file bytes, summed across the bundle and excluding the index, are capped at
the file ceiling plus the metadata ceiling. `--max-shards` only lowers the
4,096-shard ceiling. External and mapped standalone imports inherit the
predecessor limits for each wire format.

`--max-shard-bytes` is an export target for the **complete serialized shard**
including the 8-byte prefix, padded header and payload. It defaults to the
file ceiling. A tensor that cannot fit with its own header fails export.
`--allow-oversize-single-tensor` permits it in a shard by itself. All hard
file, metadata, tensor, count and aggregate ceilings still apply. Conversion
retains bounded copies in memory. No streaming, peak memory or throughput
claim is made.

Inputs must be regular files. Sharded import opens the index directory once,
rejects a symlink at the directory leaf, and opens the index and shard
basenames relative to the directory descriptor with `O_NOFOLLOW`. Directory
ancestors are trusted. Shard outputs and the index are staged as mode-0600
files, fsynced, and published atomically per file. Default no-clobber
publication uses hard links. `--overwrite` allows replacement of existing
regular shard and index files. Cooperating readers and writers share a
directory advisory lock. The writer publishes the index last and restores
the previous affected files on an ordinary publication failure before that
point. An overwrite with a different shard count can leave unreferenced old
files, which the reader ignores.

A multi-file overwrite is not crash-atomic. A crash or rollback failure can
leave temporary files, backups or a partial bundle. A reader rejects a bundle
whose index and shards disagree. Noncooperating filesystem writers and
adversarial replacement of directory ancestors are outside this boundary.
A directory fsync failure after publication can report an error with new files
already visible. Single-file outputs use the predecessor's atomic publication.

## Commands and tests

```sh
python3 tools/io5/model_mapping.py inspect-mapping input.c1 \
  --profile eshkol-m3-v1 --scheme huggingface
python3 tools/io5/model_mapping.py c1-to-mapped-safetensors input.c1 model.safetensors \
  --profile eshkol-m3-v1
python3 tools/io5/model_mapping.py mapped-safetensors-to-c1 model.safetensors restored.c1
python3 tools/io5/model_mapping.py c1-to-mapped-gguf input.c1 model.gguf \
  --profile eshkol-m3-v1
python3 tools/io5/model_mapping.py mapped-gguf-to-c1 model.gguf restored.c1
python3 tools/io5/model_mapping.py mapped-gguf-to-c1 external.gguf imported.c1 \
  --external --profile eshkol-mod5-v1 --scheme gguf --provider i2-dense-cpu-f32-v1 \
  --tie-head-to-embedding
python3 tools/io5/model_mapping.py c1-to-sharded-safetensors input.c1 output/model.safetensors.index.json \
  --profile eshkol-m3-v1 --scheme huggingface --max-shard-bytes 4200
python3 tools/io5/model_mapping.py sharded-safetensors-to-c1 output/model.safetensors.index.json restored.c1
python3 -m unittest -v tests.io5.test_model_mapping
```

Single-file exports default to their format's naming scheme. Either explicit
scheme is supported for either wire format. `inspect-mapping` validates the
C1 input and writes the selected mapping metadata and a manifest object to
stdout. All conversion commands use exit 2 for rejected arguments, invalid
files and filesystem errors.

The existing IO5 test harness discovers the generated-fixture mapping tests.
`IO5_REFERENCE_PYTHON` selects the pinned safetensors reference interpreter for
`make test-io5`. `IO5_GGUF_REFERENCE_PYTHON` selects the pinned safetensors,
GGUF and NumPy reference interpreter for `make test-io5-gguf`. The upstream
checks skip explicitly when the corresponding package is absent. The external
packages are used only in these development tests.

The [live C1 gate](IO5_LIVE_C1_INTEROP.md) covers compiled write and reload
through the fixed M3 profile and the unmapped and sharded formats.
