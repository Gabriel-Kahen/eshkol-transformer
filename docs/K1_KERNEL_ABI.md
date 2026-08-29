# K1 native-kernel ABI and capability report

## Scope and evidence boundary

K1 defines an independent C11 ABI version domain for explicitly linked native-kernel
providers. ABI version 1.0 is declared in
`include/eshkol_transformer/kernel_abi.h`. It does not implement a tensor backend,
numerical kernel, gradient, dtype conversion, device transfer, or fallback. It does
not change any A0 Eshkol public name or arity.

The default runtime report is grounded only in R0 evidence merged through PR #15.
It contains all required A0 capability symbols in bytewise name order with status
`unverified`, empty constraints, and no determinism claim. In particular, successful
host execution does not prove CPU backend identity, and compile-time feature or dtype
advertising does not prove storage or execution. GPU, f16, bf16, f32, i64, bool, and
all target kernel requests therefore fail `capability-require` unless a separately
reviewed provider supplies exact `verified` evidence. K1 ships no provider.

The test-only provider under `tests/k1` uses names and evidence IDs beginning with
`test` or `TEST-ONLY`. It is compiled directly into the conformance test, is absent
from production sources and default discovery, and is not capability evidence.

## Version and discovery contract

The provider symbol is `eshkol_transformer_kernel_provider_v1`: the symbol encodes
only ABI major 1. The provider descriptor carries `abi_major`, `abi_minor`,
`struct_size`, and `required_features`.

- A major other than 1 is `version-mismatch`.
- A descriptor smaller than the complete 1.0 prefix is `version-mismatch`.
- A larger descriptor and a provider minor newer than 0 are accepted by reading only
  the 1.0 prefix, provided every required-feature bit is known. The tests exercise a
  compatible 1.1 provider.
- An unknown required-feature bit is `version-mismatch`; it is never ignored.
- Capability, request, tensor-view, and call descriptors are also size-tagged and
  reject an undersized prefix before reading later fields.

On the supported x86-64 ABI, the frozen 1.0 prefixes are 80 bytes for a provider,
120 for a capability, 72 for a tensor view, 56 for a request, and 104 for a call.
Those explicit prefix constants, rather than a future header's total structure size,
govern compatible-minor discovery. Minor extensions append fields and preserve every
embedded v1 layout and offset.

Discovery is explicit and process-local. The caller supplies a resolver callback;
K1 asks it for the one major-versioned symbol. The runtime does not call `dlopen`,
inspect an environment variable, search a directory, read a configuration path, or
load a persisted library. A missing explicitly resolved symbol is an `unsupported`
error. Calling `et_kernel_runtime_baseline` is the only provider-free discovery
path, and it returns the deterministic unverified baseline rather than asserting a
verified absence.

Providers are trusted native code already linked or registered by the embedding
process. Discovery validates and deep-copies all capability metadata plus the callback
addresses into a new runtime-owned immutable snapshot and sorts it canonically. The
descriptor and metadata may be released after discovery; the linked callback code
must remain valid for the runtime's lifetime.
Capability accessors return borrowed read-only pointers valid until
`et_kernel_runtime_destroy`. Destroying a null runtime is safe.

Capability names, implementations, operations, dtypes, and device descriptors use
lowercase ASCII symbols. A verified entry must include at least one operation, dtype,
device, and shape-range alternative. Matching requires exact symbol membership,
exact rank, every extent within its inclusive range, and a deterministic entry when
requested. Missing, unsupported, unverified, or nonmatching entries all cause
`capability-require` to return `unsupported`; they never select a substitute.
ABI v1 bounds a provider to 4096 capabilities, each symbol list and shape-range list
to 4096 entries, rank to 64, and each call to 1024 inputs and 1024 outputs. Exceeding
a bound is an explicit malformed-call error before iteration or allocation.

## Call, tensor, and failure-atomicity contract

An ABI tensor view is borrowed storage plus exact byte length, dtype, device, layout,
zero byte offset, rank, and concrete nonnegative dimensions. ABI v1 recognizes the
storage widths of `bool`, `f16`, `bf16`, `f32`, and `i64` only so it can validate byte
counts; recognizing a code is not a capability claim. The call boundary requires:

- dense row-major layout and zero offset;
- exact nonoverflowing byte length from dtype and shape;
- nonnull storage for a nonempty tensor;
- every tensor on the request device;
- output ranges disjoint from every input and other output; and
- an exactly matching verified capability before provider validation.

Inputs remain caller-owned and read-only. Outputs remain caller-owned. Discovery may
allocate only its CPU control-plane snapshot. Dispatch performs no allocation, copy,
cast, materialization, transfer, scalar implementation, device substitution,
finite-difference calculation, approximate gradient, or fallback of any kind.
The request dtype names the capability's compute dtype; individual operand dtypes may
differ (for example, integer indices with floating storage) and remain subject to the
provider's operation-specific validation. K1 does not invent those downstream tensor
schemas. Caller-owned `et_kernel_error` storage is CPU control data and must not alias
a call descriptor or any tensor storage.

Dispatch is two-phase. K1 performs all generic checks, then calls the provider's
`validate_call`, which must be non-mutating and complete every provider-specific
fallible check. Only after both passes succeed does K1 call the provider's `void`
`invoke_call` commit function. The commit function must be non-failing and fully
write every declared output. Consequently every recoverable error leaves all inputs
and outputs byte-for-byte unchanged. A provider that mutates during validation or
raises a recoverable failure during commit violates ABI conformance. Process
termination, hardware loss, and memory corruption are outside in-process recovery.

## ABI-local errors and the Eshkol boundary

Native functions return zero on success and an A0-aligned ABI-local error category on
failure. `et_kernel_error` is caller-owned fixed storage containing a category, a
stable reason code, operation, and diagnostic. It is not the public Eshkol error
representation and does not add an A0 public surface.

E1 issue #23 owns the shared `transformer.error_internal` implementation. A future
Eshkol-facing K1 adapter must depend on reviewed E1, preserve K1 categories and
causes through that boundary, and use fixed-arity scalar/opaque-pointer externs with
caller-provided buffers. K1 does not duplicate E1 here. The current Eshkol fixture
only AOT-links and reads the fixed-width ABI major/minor functions; it proves linkage,
not a kernel, dtype, device, or error-mapping capability.

## Canonical machine-readable report

`et_kernel_runtime_report_json` writes into caller-provided storage and first supports
a size query. Required capacity includes the terminating NUL; the NUL is not report
data. An undersized buffer is left unchanged.

Report version 1 is strict UTF-8 JSON with these canonical rules:

- root keys are `abi`, `entries`, `format`, `process_local`, `provider_abi`, and
  `version`, in that order;
- entry keys are `constraints`, `deterministic`, `evidence`, `implementation`,
  `name`, `status`, and `version`, in that order;
- constraint keys are `devices`, `dtypes`, `operations`, and `shape_ranges`, in
  that order;
- entries and string lists are sorted by raw UTF-8 bytes; shape alternatives are
  sorted by rank then dimension ranges;
- JSON integers are unsigned base-10 with no leading zero; floating JSON numbers are
  never emitted; unbounded maxima are `null`;
- strings must be valid UTF-8; quote and backslash use two-byte escapes, U+0000
  through U+001F use lowercase `\u00xx`, and other UTF-8 bytes are not escaped;
- there is no insignificant whitespace, and exactly one LF terminates the report.

`process_local` is always true. `provider_abi` is null for the baseline or contains
the resolved provider's major/minor. A report is a current-process observation and
must not be persisted as capability proof for another host or process.

The version-1 baseline has SHA-256
`7e14cc845902b6a37f9946a163d355085ea704c043863f7e37481b4ab0deec59`.
The K1 gate emits it in both `C` and `C.UTF-8` locales, byte-compares the results,
checks that hash, strict-parses and canonically reserializes it, and exercises provider
metadata in different source orders to prove canonical round-trip identity.

## Verification

Run `make test-k1` after configuring the exact F0 toolchain. The gate uses the C/C++
compilers recorded in F0 provenance, enables warnings-as-errors, checks C and C++
headers, executes ABI and malformed-call conformance twice, verifies canonical JSON,
runs address/undefined-behavior sanitizers, and AOT-compiles/links/runs the fixed-width
Eshkol version probe twice with compiler `1.3.4-evolve`.

Local CachyOS/LLVM 22 results are compatibility evidence only. Acceptance still
requires the supported Ubuntu 22.04 x86-64, LLVM/Clang 21.1.8 lane. LeakSanitizer is
disabled by default because it is unavailable under the local traced executor;
setting `K1_ASAN_DETECT_LEAKS=1` enables it where supported.
