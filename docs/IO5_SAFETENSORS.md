# IO5 bounded safetensors interoperability 1.0

This first IO5 slice is a standalone Python interoperability tool, permitted by
the project engineering rules. It translates data-only [C1 1.0](CHECKPOINT_FORMAT.md)
CPU dense contiguous f32 logical-state bytes to and from safetensors. It does not
register a provider, construct a model, call P1, expose a trainer API, or participate
in the delivered Eshkol training path. Provider text remains inert. C1/M3 source,
installed facades, and production dependency manifests are unchanged.

IO5 remains **active**: this bounded slice has independent source/test approval;
integration and supported CI remain gates. GGUF and live state/model ingress are
unimplemented. The M3 fixture uses
the accepted 15 logical paths, 14 unique parameter tensors and tied head/embedding from its
development manifest, with deterministic generated f32 bytes. This is format
compatibility evidence, not a real M3 snapshot, model load or forward-pass test.

## Commands and admission

```sh
python3 tools/io5/safetensors_interop.py c1-to-safetensors input.etcp output.safetensors
python3 tools/io5/safetensors_interop.py safetensors-to-c1 output.safetensors restored.etcp
# Explicit external tensor mapping; provider is an inert spelling, not a lookup:
python3 tools/io5/safetensors_interop.py safetensors-to-c1 external.safetensors output.etcp \
  --plain-provider i2-dense-cpu-f32-v1
make test-io5
```

The import command requires the IO5 metadata below by default. A normal external
file requires `--plain-provider`: each complete safetensors name becomes **one**
P1 path segment, every entry is a parameter, and there are no aliases. Dots,
slashes and Unicode in names are literal; no heuristic path splitting or tied-value
inference occurs. External arbitrary string metadata is discarded. This explicit
mapping supplies no evidence that the resulting logical state matches any module.
The plain flag cannot override any IO5 metadata, including an incomplete or unknown
IO5 version. Empty tensor names are unsupported because P1 segments are nonempty.

The tool implements the [upstream safetensors byte format](https://github.com/safetensors/safetensors/blob/main/README.md#format):
u64 little-endian header length, UTF-8 JSON starting with `{`, followed by the
exact data buffer. Data offsets are relative to that buffer. Header key order and
trailing JSON whitespace are accepted. Duplicate keys anywhere, non-string metadata,
non-UTF-8 strings (including escaped lone surrogates), malformed JSON, holes,
overlaps, hidden trailing data, invalid descriptors and invalid tensor spans reject.
Descriptors require exactly `dtype`, `shape`, and `data_offsets`; unknown descriptor
fields reject. Only `F32`, CPU interpretation, C-order dense contiguous storage
are supported. No cast, byteswap, approximate comparison or executable decoder is
used. All IEEE-754 bits survive, including signed zero, subnormals, infinities and
signaling/quiet NaN payloads. Shape `[]` is one scalar; zero extents produce empty
payloads. Empty states and tensors at shared contiguous boundaries are accepted.

## IO5 metadata contract

Exports contain ordinary F32 tensor descriptors named `t000000`, `t000001`, etc.
in C1 entry order and three string-valued `__metadata__` fields:

| Key | Value |
|---|---|
| `eshkol.io5.version` | Exactly `1.0` |
| `eshkol.io5.manifest` | A JSON object string with exactly `provider`, `entries`, `aliases` |
| `eshkol.io5.c1-sha256` | Lowercase hex SHA-256 of the complete canonical C1 file |

The manifest's provider is the exact inert UTF-8 spelling from C1. Entries are an
array in canonical P1 segment-by-segment UTF-8 byte lexical order, each with exactly
`name` (the matching safetensors key), `path` (an array of separate strings), and
`kind` (`parameter` or `buffer`). Every safetensors tensor is referenced exactly
once. Aliases are an array of disjoint groups of at least two increasing entry
indices, ordered by first index; all members must be parameters with identical
shapes and exact bytes. The manifest retains C1's alias topology without claiming
shared storage in safetensors. Payloads for tied entries are duplicated.

Import validates every descriptor and the entire manifest before publishing output.
It reconstructs canonical C1 bytes and checks their SHA-256 against the metadata.
Thus an IO5 C1→safetensors→C1 round trip returns **identical complete bytes**,
including segmented paths, kinds, aliases, provider spelling and f32 payloads.
Other string metadata is allowed and ignored; unknown `eshkol.io5.*` fields reject.

C1 input validates all fixed header fields, versions, reserved bytes, exact section
ends, strict paths/order, checked rank/shape/byte products, payload offsets, both
tensor and container SHA-256 checksums, and alias semantics. Supported state and
provider-interface versions remain C1 1.0 and P1 provider 2.0. Bool/i64 buffers,
other dtype/device/layout/version/features reject explicitly; no bytes are converted.

Safetensors has no native integrity checksum. IO5 metadata adds an **unkeyed**
semantic C1 checksum, covering reconstructed paths/kinds/shapes/aliases/provider/data;
it is not authentication and does not bind ignored metadata or JSON formatting.
An attacker can recompute it. Plain external imports validate structure but cannot
detect arbitrary bit flips in otherwise valid f32 payloads. Repacking with a tool
that drops IO5 metadata removes lossless logical-state reconstruction; default
import then rejects rather than guessing lost state semantics.

## Resource and file boundary

| Limit | Default and maximum |
|---|---:|
| Input/output file | 64 MiB |
| C1 metadata / safetensors padded JSON header / manifest JSON | 1 MiB each |
| One tensor payload | 32 MiB |
| Tensors / alias members | 4096 |
| Rank / path segments | 64 |
| One UTF-8 path segment or tensor name | 65536 bytes |
| Provider spelling | 127 bytes |
| JSON container depth | 16 |

`--max-file`, `--max-metadata`, `--max-tensor` and `--max-entries` only lower
their corresponding ceilings. These are interoperability-tool policy bounds,
not new C1 hard limits or demonstrated operational memory ceilings. Non-streaming
conversion retains several bounded copies; peak memory/performance is unmeasured.
Integer fields must be actual unsigned integers through u64, never booleans,
floating point or negative. All intermediate shape products and f32 byte sizes
must fit u64. JSON depth is checked before decoding; header size and file/count/span
limits are checked before payload extraction or tensor interpretation. Native
address-sized memory mapping is not used. No input-controlled code is loaded.

On Linux the CLI opens regular non-symlink inputs with `O_NOFOLLOW`, checks size
before bounded reading, and rejects changed lengths. Conversion completes before
output creation. Output uses a same-directory mode-0600 temporary, file fsync,
atomic hard-link publication without overwrite by default, or atomic replacement
with `--overwrite`, then directory fsync. Invalid input leaves an existing destination
unchanged. Directory fsync failure can report an error **after publication**; the
diagnostic does not promise that every filesystem error preserved the previous
destination. Crashes may leave `.io5-*` temporaries; the tool never sweeps or reuses
them. Parent directories are trusted; hostile concurrent directory replacement and
concurrent same-size input rewrites are outside this CLI boundary. This is separate
from the accepted native C1 I/O ABI and supplies no new native failpoint evidence.

## Evidence and remaining gates

`scripts/test-io5.sh` runs 11 focused unittest groups and the five Q0 Python
isolation groups without Eshkol build prerequisites. Tests generate all fixtures:
all-bit-pattern representatives, Unicode/ambiguous segmented paths, parameter and
buffer kinds, ties, empty/scalar tensors, M3 shape roster, exact/one-over policies,
malformed JSON/headers/offsets/shapes/checksums/aliases, 256 seeded byte mutations,
and CLI no-clobber/failure behavior. The independent C1 parser validates generated
C1 bytes. No generated fixture is committed.

For the independent upstream serializer/deserializer test, install
`python -m pip install -r tests/io5/requirements-reference.lock` in a development interpreter and set
`IO5_REFERENCE_PYTHON=/absolute/path/to/python make test-io5`. The actual Rust-backed
package checks exported f32 shapes and bits and generates external scalar/empty/
nonfinite fixtures for import, without NumPy or PyTorch. The test is explicitly
skipped in the stdlib-only run if that package is unavailable. Setting
`IO5_REFERENCE_PYTHON` makes the package mandatory: the gate fails before running
tests if the package is absent or differs from 0.6.2, pending an explicit review.
Local final validation used that exact package: 11/11 IO5 and 5/5 isolation tests
passed. Independent review approved the source and tests after lone-surrogate and
lexical negative-zero parser fixes, both protected by regressions. This does not
establish supported CI, native live-state ownership, GGUF,
model-name remapping, sharded indexes, optimizer/RNG/tokenizer state, exact resume,
GPU execution, mixed precision, or performance. Those remain downstream work.
