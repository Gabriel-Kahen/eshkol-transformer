# Frozen oracle fixture format

## Scope

Q0 fixtures are development/test data used to compare compiled Eshkol output with
independent reference implementations. They are never a production checkpoint or
runtime serialization format. A reader must treat them as inert data: it must not
evaluate code, import the named generator, unpickle objects, call `torch.load`, or
perform any other executable deserialization.

Version 1 is named `eshkol-oracle`. Changing any rule below requires a version or
migration decision coordinated through issue #1.

## Canonical bytes and integrity

A fixture is strict UTF-8 JSON without a BOM and with exactly one final LF. Object
keys are sorted lexicographically, arrays preserve their declared order, separators
are `,` and `:` with no whitespace, strings are Unicode NFC, and no bytes follow
the final LF. Readers reserialize and byte-compare.

JSON floating-point numbers are forbidden. JSON integers are signed 64-bit values;
schema fields further restrict versions, seeds, ranks, and dimensions to
non-negative values. Floating values, including NaN, infinities, and signed zero,
are represented by fixed-width lowercase hexadecimal bit patterns.

The root object has exactly:

```text
checksum: {algorithm: "sha256", digest: <64 lowercase hex>}
format: "eshkol-oracle"
payload: <version-1 payload>
version: 1
```

The digest is SHA-256 over canonical JSON without a final LF for the object
`{format, payload, version}`. It therefore covers the format, version, generator
metadata, cases, tensor metadata, and tensor data. The checksum field is excluded.
Readers enforce a 16 MiB file limit, bounded JSON depth, strict duplicate/unknown
key rejection, canonical byte equality, checksum verification, then schema checks.

## Payload

The payload has exactly `cases`, `generator`, and `tensors`.

Generator metadata records a stable name/version, fixed seed, framework name and
exact full runtime version (including build suffixes such as `+cpu`), and SHA-256 values for the generator source and development-only
dependency lock. It contains no timestamp, hostname, absolute path, or temporary
directory.

Tensor records are sorted by unique name and have exactly:

- `name`: stable ASCII identifier.
- `role`: `input`, `expected`, or `analytic_gradient`.
- `dtype`: `bool`, `int64`, `float32`, or `float64`.
- `shape`: row-major dimensions; `[]` is a scalar and a zero dimension is empty.
- `device`: `cpu` in version 1. No fallback or implicit transfer occurs.
- `layout`: `row_major`.
- `encoding`: `bool01`, `twos-complement-hex-be`, or
  `ieee754-hex-be`, as required by the dtype.
- `data`: flat row-major encoded elements whose count matches the shape.

Cases are sorted by unique name. Supported kinds are `known_value`, `parity`,
`gradient`, `repeated_input`, `special_value`, `boundary`, and `malformed`.
A successful expectation names at least one output; an error expectation has a
non-empty stable error contract and no outputs.

Every case input must reference a tensor with role `input`. A successful gradient
case must reference only `analytic_gradient` outputs; every other successful case
must reference only `expected` outputs. Role mismatches are malformed fixtures.

Absolute and relative tolerances are non-negative finite float64 bit patterns.
Comparison uses `abs(actual-expected) <= absolute + relative*abs(expected)`.
Same-sign infinities compare equal. NaNs compare equal only when `equal_nan` is
true. Integer and boolean comparison requires zero tolerance.

## Development isolation

All Python, the PyTorch generator, its dependency lock, and fixture readers live
under `tests/q0`. Neither `eshkol.toml` dependencies nor production Eshkol source
may import or invoke them. The first fixture pins PyTorch exactly and uses literal
CPU float64 scalars, a fixed seed, deterministic-algorithm mode, and one thread.

The parity test invokes real `eshkol-run` (or the explicit `ESHKOL_RUN` path).
If unavailable, it reports a blocked test and never substitutes Python output.
The runner must identify as Eshkol 1.3.4 or newer, compile the probe with `-o`, and
then execute the emitted binary. JIT run mode is not acceptance evidence. The probe
must emit exactly one scalar line; empty, annotated, or multi-value output is
rejected.

## F0 integration contract

Q0 does not replace the repository-wide F0 entrypoint. Its focused canonical
entrypoint is `scripts/test-q0.sh`; F0 can call that script after installing the
development-only lock. Production packaging must exclude Python files and fixtures.
