# Shared structured-error contract

E1 implements the A0 error model once for all subsystems.  It adds no public API:
the public surface remains the six unary `transformer-error-*` accessors and the
twelve categories in `PUBLIC_API_CONTRACT.md`.

## Internal module

`transformer.error_internal` exports the six public accessors plus exactly these
implementation helpers:

```scheme
(transformer-error-make category operation message details cause)
(transformer-error-raise category operation message details cause)
(transformer-error-wrap-foreign category operation message details
                                source-domain source-code source-message cause)
```

`make` returns an immutable CPU control-plane error.  `raise` constructs and raises
that same validated value.  `wrap-foreign` uses the caller-supplied A0 category and
adds `source-domain`, `source-code`, and `source-message` to details.  It rejects a
caller detail with one of those reserved keys rather than overwriting it.  The caller
owns the reviewed source-status-to-A0-category mapping: unknown statuses map to
`internal`, while a known specific category must not be concealed as `internal`.
`source-domain` is a symbol, `source-code` is a representation-real exact integer or
symbol, and `source-message` is a UTF-8 string that may be empty.

The public A0 facades load only `transformer.error_public`, whose declared surface
contains the six accessors.  The three helpers are absent from the declared surfaces
of `transformer.public` and `transformer.capabilities`.

Public packages that cannot safely flatten either internal module use the reviewed
[E1B separately compiled consumer boundary](E1B_CONSUMER_BOUNDARY.md). Their
installed source closure contains only the six accessors and narrow package-specific
stubs; E1 construction and identity remain inside one prelocalized trusted artifact.

## Values and ownership

Categories are exactly `invalid-argument`, `shape-mismatch`, `dtype-mismatch`,
`device-mismatch`, `noncontiguous`, `unsupported`, `invalid-state`, `io`,
`corrupt-data`, `version-mismatch`, `determinism-unavailable`, and `internal`.
Operation is a nonempty symbol.  Message is a nonempty UTF-8 string.  Cause is `#f`
or another valid transformer error.

Details use this grammar:

```text
details = () | (entry ...)
entry   = (symbol value)
value   = () | boolean | character | exact-integer | symbol | string | (value ...)
```

Each entry is a proper two-element list, not a dotted pair.  Top-level keys are
unique by `symbol->string` spelling and returned in lexicographic spelling order;
fresh and interned symbols with the same spelling collide.  Returned keys are
canonical `string->symbol` values.  Nested proper lists are values, not maps.
Procedures, ports, vectors, bytevectors, hash tables, records, other opaque values,
inexact/complex numbers, improper lists, cycles, and shared pairs are rejected.
On the pinned runtime, `(make-rectangular 1 0)` produces a tagged complex value for
which `complex?` is true and `integer?`, `exact?`, and `real?` are false. E1 requires
all three of `integer?`, `exact?`, and `real?` for accepted numeric values, so tagged
complex values are rejected without converting ordinary exact integers or bignums
through an inexact `real-part` result. Example input
`((z-context 1024) (path "run.toml")
(expected (cpu f32)))` is returned as `((expected (cpu f32)) (path "run.toml")
(z-context 1024))`.

Construction copies all strings and data, canonicalizes details, and recursively
copies causes.  Accessors validate their receiver and return fresh messages, details,
and causes; symbols and exact scalar values are immutable immediates.  Mutation of
constructor inputs or one accessor result cannot affect the stored value or a later
accessor result.

The implementation uses native Eshkol condition objects as process-local identity
tokens because the pinned compiler exposes `define-record-type` values as mutable
vectors.  Every token has the constant generic condition message
`transformer.error_internal:v1` and no irritants.  A lexically hidden append-only
registry owns the copied metadata; only a registered token identity satisfies
`transformer-error?`.  An unrelated or caller-forged condition with identical text
does not.  Predicate lookup compares identities only: it does not invoke candidate
procedures or inspect candidate messages or irritants.

Metadata for a registered identity is never mutated.  Accessors return detached,
deep-owned snapshots.  Eshkol strings and lists in those snapshots remain mutable by
their caller, but changing them cannot change the registry or a later snapshot.

On pinned v1.3.4-evolve, `provide` is informational and required source modules are
flattened into one compilation unit.  `transformer.error_core` therefore declares
the six accessors plus one technically name-reachable `e1-internal-dispatch` bridge
so public and internal facades can share the lexical registry and compile from a
fresh cache.  The bridge validates a fixed opcode, exact proper argument list, and
all values before dispatch.  It is an unsupported implementation detail: it is not
in the `transformer.error_public`, `transformer.public`, or
`transformer.capabilities` provide list, the README API, or the SemVer contract.  The
supported
construction boundary remains the three named helpers in
`transformer.error_internal`.  Retest and remove this workaround when upstream gains
module-private bindings or a suitable opaque record facility.

Identity tokens are not serializable or valid across processes.  The registry is a
strong process-lifetime list with linear lookup and monotonically growing index and
metadata cost.  The pinned runtime has neither usable weak keys nor tracing
collection for these native conditions; native error tokens already have arena
lifetime.  Errors are exceptional control-plane values, so E1 makes no bounded-memory
claim.  Captured-list mutation is not synchronized, and concurrent construction or
access is unverified and unsupported; no thread-safety claim is made.

## Validation limits and bootstrap failures

- main message: 4 KiB UTF-8;
- source message: 4 KiB UTF-8;
- details: 64 entries, 512 value/list nodes, 16 levels, and 16 KiB cumulative
  UTF-8 bytes;
- cause chain: 16 levels.

All malformed or over-limit construction is `invalid-argument`; there is no empty
result, partial value, fallback, or executable payload.  `transformer-error?` is the
only non-raising validator and returns `#f` for foreign values.  The other five
accessors raise `invalid-argument` on a foreign or malformed value.

Construction and accessor failures use a private bootstrap path that creates the
same registered E1 representation from compile-time-safe constants.  The error operation
names the failing helper/accessor, details are empty, and cause is `#f`.  This path
does not re-enter ordinary validation, preventing recursive failure loops.

Run the focused executable contract with:

```sh
/usr/bin/bash -c 'make test-e1'
```

Run the separately compiled public-consumer boundary gate with:

```sh
/usr/bin/bash -c 'make test-e1b'
```
