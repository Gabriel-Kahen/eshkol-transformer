# X1 configuration and resolved-run manifest contract

Status: **X1 review candidate**. Issues #1 and #20 accepted the schema and
canonicalization direction with required clarifications, and merged E1B issue #33
now supplies the separately compiled raise-only boundary used here.

## Scope and public operations

X1 implements the six configuration operations fixed by A0:

- `config-parse text`
- `config-resolve config overrides`
- `config-validate resolved`
- `config-canonical resolved`
- `config-fingerprint resolved`
- `config-ref resolved key`

Configuration is CPU control-plane data. Parsing never performs file I/O, follows an
include, reads an environment variable, expands a template, evaluates code, imports a
module, constructs an arbitrary object, or supplies a default. The caller supplies the
complete input text. Resolution and validation have no training, experiment-tracking,
checkpoint, tensor, device-transfer, or capability-discovery side effects.

The source schema, resolved-manifest format, canonicalization, and fingerprint are
independent version domains. Version 1 uses config schema `[1,0]`, resolved-manifest
format `[1,0]`, canonicalization identifier `eshkol-config-json-v1`, and fingerprint
identifier `sha256:eshkol-config-json-v1`.

Failures use the shared E1 registry through E1B. Installed `transformer.config`
requires only `transformer.error_consumer`; a never-installed trusted X1 root
requires `e1b_error_consumer_private` and passes one of `invalid-argument`,
`version-mismatch`, or `unsupported`, the failing public operation symbol, a bounded
message, data-only E1 details, and cause `#f` to the fixed five-value
`et-e1b-private-raise` seam. The combined object localizes that seam, E1 constructors,
core dispatcher, implementation helpers, and compiler companions before application
source is compiled. It globally exports only the six A0 error accessors and six
package-specific X1 wrappers. When X1 has positional failure context, details are the
proper E1 map `((context (<bounded-scalar-summary> ...)))`; strings longer than 256
bytes and non-schema symbols or opaque values are replaced by fixed inert symbols.
When no context exists details are `()`. No raw error representation, executable
detail value, E1 constructor, private bridge, core dispatcher, or X1 implementation
helper is present in the installed source closure. Configuration identities are native
condition shells backed by the one combined artifact's process-local registry; all
metadata reads are defensive copies, and adversarial same-message conditions cannot
forge an identity.

The pinned compiler crashes when a direct public function contains a lexical
`extern`. As in the reviewed E1B facade, each public fixed-arity X1 function therefore
calls one source-reachable safe-only closure alias. Each alias conveys only the exact
same narrow configuration operation; it provides no E1 construction, dispatch,
representation, or additional configuration capability and is not an A0 export.

## Source configuration grammar

The source is one flat JSON object. It is UTF-8 without a BOM, but every accepted
version-1 byte is in the ASCII subset. A parser rejects non-ASCII bytes rather than
normalizing or transliterating them. The only permitted whitespace is JSON's ASCII
space, tab, carriage return, and line feed. It may occur before or after the object
and between tokens. A source document has no required final-newline convention.

Object keys are unescaped ASCII JSON strings. Version 1 permits no escape sequence in
any string, including `\u` escapes. Values are JSON integers, booleans, or the exact
ASCII enum strings listed below. Objects, arrays, `null`, floating-point numbers,
exponents, non-standard numeric constants, and trailing non-whitespace bytes are
rejected. Integer tokens use an optional minus followed by JSON decimal digits, have
at most 19 digits excluding the sign, have no leading zero except the token `0`, and
must fit signed 64-bit range. The otherwise valid JSON token `-0` is rejected to
preserve one integer spelling. Schema validation then rejects negative values where
the field requires a nonnegative or positive integer. JSON booleans are not integers.

The parser enforces these hard limits before returning a configuration:

| Limit | Version-1 value |
|---|---:|
| UTF-8 input size | 16,384 bytes |
| Root keys | 14 |
| Object nesting depth | 1, the root object only |
| Integer digits | 19, excluding an optional minus sign |

Duplicate keys are rejected even when their values are equal. Unknown keys are
rejected. Key order and permitted insignificant whitespace do not affect the parsed
value.

## Version-1 source schema

The root object accepts exactly the following keys. “Required” means present in the
source document; an override does not satisfy a missing required source key.

| Key | Source requirement | Resolved rule |
|---|---|---|
| `config-schema-major` | Required integer `1` | Must remain `1`; not overridable |
| `config-schema-minor` | Required integer `0` | Must remain `0`; not overridable |
| `model.vocabulary-size` | Required positive i64 | `V >= 1` |
| `model.context-length` | Required positive i64 | Context capacity is at least one token |
| `model.hidden-size` | Required positive i64 | Model hidden width `D` |
| `model.layer-count` | Required positive i64 | Decoder layer count `L` |
| `model.query-head-count` | Required positive i64 | Query-head count `Hq` |
| `model.kv-head-count` | Optional positive i64 | Defaults to final `Hq`; must divide `Hq` |
| `model.head-size` | Optional positive i64; source-only/read-only to overrides | If absent, derive `D / Hq`; requires exact division |
| `model.dtype` | Optional string `"f32"` | Defaults to `"f32"`; no precision conversion |
| `model.device` | Optional string `"cpu"` | Defaults to `"cpu"`; no device fallback |
| `run.seed` | Required nonnegative i64 | Explicit reproducibility seed |
| `run.deterministic` | Optional boolean | Defaults to `true`; version 1 rejects `false` |
| `training.accumulation-steps` | Optional positive i64 | Defaults to `1` |

The source contains no secret-bearing or free-form fields. In particular, paths,
credentials, callbacks, commands, code, host identity, timestamps, and current
capability reports are not schema values.

An unsupported schema version is a `version-mismatch`; malformed syntax, unknown or
duplicate keys, missing required keys, and ill-typed values are `invalid-argument`.
An unavailable well-formed dtype, device, or digest implementation is `unsupported`.

## Override and resolution contract

`overrides` is an alternating flat option list of schema key and value, using quoted
colon symbols for keys, for example:

```scheme
(list ':model.hidden-size 768 ':run.seed 42)
```

Each symbol after the leading colon spells the corresponding JSON key exactly.
Integer override fields take native exact integers, `run.deterministic` takes a native
boolean, and the `model.dtype` and `model.device` enum fields take the quoted native
symbols `'f32` and `'cpu`, not JSON strings. Source JSON continues to use the strings
`"f32"` and `"cpu"`.
An unknown key, a duplicate override key, an odd-length list, a missing value, an
ill-typed value, or an attempt to override either schema-version key or the
source-only/derived `model.head-size` key is `invalid-argument`. Duplicate overrides
are rejected even when their values agree.

Semantic precedence is `defaults < input < explicit overrides`; derivation applies
only to a field absent after that overlay. Resolution evaluates it deterministically:

1. Verify the required source schema version and required source keys.
2. Validate every explicitly present source leaf's intrinsic type, range, enum, and
   policy in fixed schema-key order.
3. Copy the source values and mark their provenance `input`.
4. Apply the unique explicit overrides and mark them `override`.
5. Fill independent absent optional fields from versioned defaults.
6. Fill `model.kv-head-count`, when absent, from the final query-head count and mark
   it `default`.
7. Derive `model.head-size`, when absent, from the final hidden size and query-head
   count, but only when the division is exact.
8. Validate every resolved type, range, and cross-field invariant.

Defaults and derivation are evaluated against the final overlaid inputs, but they
never overwrite a present source or override value. Override list order therefore
cannot affect the result because repeated paths are invalid. An override equal to the
source value still has provenance `override`; explicitly supplying a default value
still has provenance `input`.

An override can repair a cross-field relationship among individually valid source
leaves, because compatibility is checked on the final overlaid values. It cannot
sanitize an explicitly supplied source leaf with an invalid type, range, enum, or
policy: source admission happens first and fails even when the same key has an
otherwise valid override.

The resolved value has all 14 schema keys. `config-ref` accepts the corresponding
bare quoted symbol, for example `'model.hidden-size`, and returns a new immutable
scalar. Colon-prefixed option symbols belong only to the flat override list. Any
other key is `invalid-argument`.

## Validation and incompatible combinations

Validation is complete before a resolved value is exposed. It requires:

- all integer fields to fit signed i64 and meet the positive or nonnegative range in
  the schema;
- `model.hidden-size = model.query-head-count * model.head-size`, with no integer
  overflow;
- `model.query-head-count mod model.kv-head-count = 0`;
- the exact dtype `f32` and device `cpu`;
- `run.deterministic = true` and an explicit nonnegative seed; and
- a positive accumulation-step count.

An explicitly supplied head size must satisfy the same equality as a derived head
size. A hidden size that is not divisible by the query-head count cannot produce a
derived head size and is rejected. Validation never rounds a quotient, repairs a
conflict, narrows a dtype, transfers a device, changes precision, disables a
determinism request, or selects a CPU/scalar/Python substitute.

`config-validate` returns `#t` only for a resolved X1 value. It does not turn the
static `cpu`/`f32` selection into runtime capability evidence; downstream operations
must still require their independently verified capabilities.

## Canonical resolved-run manifest

`config-canonical` emits an ASCII-only subset of UTF-8 JSON with exactly one final LF
and no following bytes. Every object key is sorted lexicographically by its ASCII
bytes, separators are `,` and `:` with no whitespace, integers use their shortest
decimal spelling, booleans are `true` or `false`, and arrays preserve their declared
order. Reserializing the same resolved value always produces identical bytes.

The root object has exactly these fields:

```text
canonicalization: "eshkol-config-json-v1"
checksum-algorithm: "sha256"
checksum-coverage: "whole-document-including-final-lf"
config-schema-version: [1,0]
format: "eshkol-resolved-run"
format-version: [1,0]
limits: {
  integer-digits: 19,
  max-input-bytes: 16384,
  max-input-keys: 14,
  max-input-nesting-depth: 1
}
provenance: <object described below>
required-features: []
resolved: <object containing all 14 schema keys>
```

Version 1 has no required feature and rejects a nonempty `required-features` array.
Unknown format or schema major/minor versions are unsupported; readers do not infer a
migration. Unknown or duplicate manifest fields are malformed.

There is deliberately no checksum digest field in the manifest. The digest covers
the whole manifest, so embedding it would create a circular definition. Integrity is
carried by `config-fingerprint` or by a downstream envelope that records that
fingerprint. `checksum-algorithm` and `checksum-coverage` make this external integrity
contract explicit without claiming that the manifest authenticates itself.

For example, this minimal source:

```json
{"config-schema-major":1,"config-schema-minor":0,"model.context-length":128,"model.hidden-size":64,"model.layer-count":2,"model.query-head-count":4,"model.vocabulary-size":256,"run.seed":1729}
```

resolves to the following exact canonical line, followed by one LF:

```json
{"canonicalization":"eshkol-config-json-v1","checksum-algorithm":"sha256","checksum-coverage":"whole-document-including-final-lf","config-schema-version":[1,0],"format":"eshkol-resolved-run","format-version":[1,0],"limits":{"integer-digits":19,"max-input-bytes":16384,"max-input-keys":14,"max-input-nesting-depth":1},"provenance":{"config-schema-major":"input","config-schema-minor":"input","model.context-length":"input","model.device":"default","model.dtype":"default","model.head-size":"derived","model.hidden-size":"input","model.kv-head-count":"default","model.layer-count":"input","model.query-head-count":"input","model.vocabulary-size":"input","run.deterministic":"default","run.seed":"input","training.accumulation-steps":"default"},"required-features":[],"resolved":{"config-schema-major":1,"config-schema-minor":0,"model.context-length":128,"model.device":"cpu","model.dtype":"f32","model.head-size":16,"model.hidden-size":64,"model.kv-head-count":4,"model.layer-count":2,"model.query-head-count":4,"model.vocabulary-size":256,"run.deterministic":true,"run.seed":1729,"training.accumulation-steps":1}}
```

### Provenance is part of identity

`provenance` has exactly the same 14 keys as `resolved`, in canonical key order.
Each value is exactly one of `input`, `override`, `default`, or `derived` and states
the final source of that resolved leaf. Required schema-version keys always have
`input` provenance. `model.kv-head-count` is `default` when copied from the final
query-head count; `model.head-size` is `derived` when calculated from hidden size and
query-head count.

Provenance is covered by the fingerprint. Two invocations with equal resolved scalar
values but different final provenance intentionally have different canonical bytes
and different fingerprints. The manifest does not preserve shadowed input/default
values or override ordering because duplicate override paths are forbidden and those
details cannot change the final resolution.

Provenance records declared configuration origin, not execution evidence. It must not
contain a timestamp, hostname, absolute path, project working-tree state, compiler
binary hash, current capability report, environment variable, or secret.

## Fingerprint

`config-fingerprint` returns:

```text
sha256:eshkol-config-json-v1:<64 lowercase hexadecimal digits>
```

The hexadecimal component is SHA-256 over every byte returned by
`config-canonical`, including the metadata, provenance, resolved values, and final
LF. There is no Unicode normalization, newline substitution, or checksum-field
exclusion step. The canonicalization identifier occurs both in the bytes being
hashed and in the fingerprint prefix. A digest implementation that is unavailable
raises `unsupported`; no weaker digest is substituted.

The exact example manifest above has fingerprint:

```text
sha256:eshkol-config-json-v1:a27ed00686df9879a06d4cb177c65807b3443a41463e8a397f8f1672562242b0
```

Downstream artifacts may store this complete fingerprint and the exact canonical
manifest. They must recompute and compare it before treating the manifest as the run
configuration. The fingerprint proves byte identity and corruption detection, not
authenticity or runtime capability.

## Limitations and evolution

- Version 1 is flat and intentionally small. It does not configure tokenizer or
  dataset artifacts, optimizer algorithms or floating hyperparameters, schedulers,
  generation, experiment tracking, orchestration, or checkpoint storage.
- Version 1 supports only declared `f32`/`cpu`/deterministic configuration. This is
  not evidence that the current Eshkol runtime provides f32 tensor storage or any
  numerical operation.
- Source JSON permits no floats, arrays, nested objects, escapes, Unicode text, file
  references, interpolation, or arbitrary extension keys.
- The resolved manifest is canonical output and a downstream embedding format. A0
  defines no public operation that loads it back into a resolved object; silently
  making `config-parse` do so would change the public contract.
- The manifest has no embedded digest and is not self-authenticating. Consumers that
  need corruption detection retain the external fingerprint.
- Opaque configuration and E1 identity metadata is retained in the completed
  process-local combined artifact. Exactly one such registry-owning artifact is
  supported per process. X1 has no persistence/load API for these identities and
  makes no verified concurrency or thread-safety claim on the pinned runtime.
- The public boundary is canonical-pin, x86-64 LP64 SysV, supported-lane AOT only.
  No JIT/bitcode artifact is published. The checked-in X1 object has an exact
  repository-owned runtime-undefined-symbol manifest; arbitrary native-object
  injection into the trusted partial link is outside scope.
- Changing a field, default, derivation, range, provenance rule, byte rule, format
  identifier, version, limit, checksum coverage, or fingerprint prefix is a public
  schema/format/canonicalization decision. It requires an explicit version or
  migration decision through issue #1; readers never guess.
