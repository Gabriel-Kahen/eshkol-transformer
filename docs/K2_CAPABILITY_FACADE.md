# K2 process-local capability facade

Status: **active**. K2 implements the twelve existing A0 capability operations as
an E1B-compatible production facade. It does not add or rename an A0 operation,
execute a kernel, compose arbitrary providers, widen K1/I2 evidence, define the K1
canonical provider symbol, or add a serialization format.

The accepted implementation contract is issue #1 comment `5628507469`, refined by
the final bounded audit in comment `5628517368`. The corrected determinism and
constraints evidence is comment `5628492425`; the private diagnostic protocol is
comment `5628542429`.

## Public surface and evidence boundary

`transformer.capabilities` provides the six accepted E1 error accessors and exactly
these A0 operations:

```scheme
(capability-discover)
(capability-request operation dtype device shape deterministic?)
(capability-verified? report capability request)
(capability-require report capability request)
(capability-report-entries report)
(capability-entry-name entry)
(capability-entry-status entry)
(capability-entry-implementation entry)
(capability-entry-version entry)
(capability-entry-constraints entry)
(capability-entry-deterministic? entry)
(capability-entry-evidence entry)
```

The facade discovers exactly the explicit unchanged I2 provider through a genuine
K1 runtime. The fixed private resolver recognizes only
`ET_KERNEL_PROVIDER_SYMBOL_V1`, requires its private static context sentinel, and
returns only `et_f32_tensor_provider_v1()`. No resolver, provider/context/callback
pointer, environment/path lookup, dynamic loader, caller provider, or provider
composition authority is public or source-reachable.

Before publishing a report, K2 exact-audits the I2 descriptor, calls
`et_kernel_runtime_discover`, and exact-audits all eleven sorted K1 entries. The
golden report is evidence for this audit, never a substitute for live discovery.
The report contains these rows:

| Capability | Status |
|---|---|
| `autodiff.reverse` | `unverified` |
| `kernel.activation` | `unverified` |
| `kernel.causal-attention` | `unverified` |
| `kernel.embedding-backward` | `unverified` |
| `kernel.indexed-cross-entropy` | `unverified` |
| `kernel.matmul` | `unverified` |
| `kernel.norm` | `unverified` |
| `tensor.bool` | `unverified` |
| `tensor.contiguous` | `unverified` |
| `tensor.f32` | `verified` |
| `tensor.i64` | `unverified` |

Every unverified row retains implementation `eshkol-core`, version
`"1.3.4-evolve"`, evidence
`"R0-merged-through-PR15:untested-with-reason"`, determinism `#f`, and this fresh
four-key constraint carrier:

```scheme
((devices ()) (dtypes ()) (operations ()) (shape-ranges ()))
```

The sole verified row is implementation `eshkol-transformer-f32`, version `"1.0"`,
evidence `"I2:bounded-exact-f32-storage.copy-v1"`, determinism `#t`, and:

```scheme
((devices (cpu))
 (dtypes (f32))
 (operations (storage.copy))
 (shape-ranges (() ((0 4611686018427387903)))))
```

Both `deterministic? #t` and `#f` requests match this deterministic row. False means
the caller does not require proof of determinism; it does not request a
nondeterministic implementation. Rank zero and rank one extents through
`4611686018427387903` match. Rank two or greater, a one-over extent, another
operation/dtype/device/capability, and each unverified row do not match. K2 calls
K1's exact `et_kernel_runtime_capability_require` for every ABI-representable
decision and never substitutes a source-only matcher.

The exact canonical K1/I2 report is 3,059 bytes with SHA-256
`742800ea988627d9093f8fe394c8ad2be801e35413e20bbcad816f2431be86cb`.
The K2 gate discovers the genuine fixed I2 provider, emits this K1 report twice,
compares the bytes, and asserts both the byte count and digest.

## Logical request domain

`capability-request` requires operation, dtype, and device to be symbols, shape to
be a proper acyclic list of concrete nonnegative exact integers, and
determinism to be exactly `#t` or `#f`. Only symbolic device descriptors are admitted
until an opaque device producer/type contract is accepted. The constructor deep
copies the five logical fields and does not impose K1 ABI limits as new A0 limits.

Before a native call, all four symbol positions—capability selector, operation,
dtype, and device—must satisfy K1's complete symbol grammar: 1 through 127 ASCII
bytes, first byte `[a-z]`, and remaining bytes `[a-z0-9._:-]`. Rank must be at most
64 and each extent must fit unsigned 64-bit. A syntactically valid A0 request or
capability outside that native domain is classified without a K1 call: verified is
`#f`, while require raises `unsupported`. It is never truncated, cast, flattened,
or sent to K1 so that K1's lower-level malformed-request error could contradict the
logical A0 contract.

The pinned compiler's exact-integer/bignum reachability is tested directly. Its
quoted-list materialization does not preserve a bignum list element, so the
unsigned-64 one-over probe constructs the shape dynamically with `list`; that value
remains an exact nonnegative integer through the real compiled facade and is
classified without a K1 call. K2 documents only values and construction paths
proven under that compiler; it does not infer unavailable arbitrary-precision
behavior from source syntax.

## Snapshot identity, ownership, and lifetime

Report, request, and entry values are opaque caller-region-owned closures created by
three distinct private compiled factories. Native code registers and authenticates
the exact three factory code addresses through separate kind-specific functions.
The addresses must be pairwise distinct at runtime; same-address repeat registration
is idempotent, while a collision or different re-registration fails before public
publication. Repeated strict AOT evidence guards against compiler/linker folding.

Validation checks `procedure?`, native factory identity, then uses a guarded hidden
query and validates exact tag, self, kind, origin PID, generation, and lifecycle.
Foreign/relay/query-capture/wrong-kind/unregistered values raise `invalid-argument`
before hidden payload disclosure. Authentic wrong-origin or stale-generation values
raise `invalid-state`. An ordinary alias of one in-region shell remains valid.

Published snapshots have no public release or death transition. Recognized dead
values exist only in a private test-only injected publication-failure path: an
authentic candidate is invalidated before test-only exposure as the failed
candidate. A reference used after its owning Eshkol region has exited has no safety
guarantee; arbitrary freed memory is not probed. K2 retains no per-value native or
Eshkol registry. The fixed native runtime is measured separately from
caller-promoted shells, and 1,024/8,192 lexical region tests must prove equal
retained root-arena state after each region exits.

`capability-discover` returns a new report shell. Report entries and successful
require results are newly constructed entry snapshots. Every string, list, shape,
and constraint carrier returned by an accessor is a fresh deep copy. Ordinary pairs
and strings are not physically frozen by the pinned runtime, but caller mutation
cannot affect an opaque snapshot, another snapshot, matching, or a later accessor
result.

The supported topology is one completed source-composed registry aggregate per
process. Packaging rejects duplicate aggregates. K2 makes no portable runtime
identity claim for arbitrary independently built aggregates loaded together.

## Process, fork, and concurrency

The K1 runtime is one private process-local singleton retained until process exit.
Candidate discovery and both exact audits complete before publication. Failure
destroys the candidate and leaves the slot absent and retryable. Repeated ensure
reuses the same runtime without another allocation.

Fork handling is limited to serialized, single-threaded fork with no K2 call in
flight. A parent-origin shell is rejected in the child before payload/provider use.
No inherited K1 runtime remains eligible. The child must fully discover and audit a
candidate before successful replacement and must clean up exact failure paths. This
does not make inherited P1/I2/O2 receivers or arbitrary multithreaded-fork state
safe. Concurrent or reentrant K2 calls are unsupported; not every data race is
claimed detectable. There is no signal-handler safety claim.

## Private native protocol

The trusted source extension and its native helper coexist in one localized
artifact. Their fixed v1 seam is:

```c
int64_t et_k2_private_runtime_ensure_v1(void *error_bytes, int64_t error_capacity);
int64_t et_k2_private_runtime_pid_v1(void);
int64_t et_k2_private_runtime_generation_v1(void);
int64_t et_k2_private_runtime_require_v1(
    int64_t expected_generation, int64_t sorted_entry_index,
    const void *operation, int64_t operation_bytes,
    const void *dtype, int64_t dtype_bytes,
    const void *device, int64_t device_bytes,
    const void *shape_u64le, int64_t shape_bytes, int64_t rank,
    int64_t deterministic, void *error_bytes, int64_t error_capacity);
int64_t et_k2_private_report_factory_register_v1(const void *closure);
int64_t et_k2_private_report_factory_authenticate_v1(const void *closure);
int64_t et_k2_private_request_factory_register_v1(const void *closure);
int64_t et_k2_private_request_factory_authenticate_v1(const void *closure);
int64_t et_k2_private_entry_factory_register_v1(const void *closure);
int64_t et_k2_private_entry_factory_authenticate_v1(const void *closure);
```

Each borrowed bytevector pointer names the Eshkol carrier's aligned signed-`i64`
declared-length header; its payload starts eight bytes later, and the supplied byte
count must equal that header. The entire nonwrapping header-plus-payload span is
synchronously checked and is never returned or retained. The public boundary
carries no pointer. Native matching copies bounded symbol payloads and little-endian
`u64` shape payload bytes into aligned stack storage, constructs one K1 request, and
returns no entry pointer.

The two diagnostic functions return these E1-aligned categories: 0 success, 1
invalid argument, 2 shape mismatch, 3 dtype mismatch, 4 device mismatch, 5
noncontiguous, 6 unsupported, 7 invalid state, 8 version mismatch, and 9 internal.
For an admitted diagnostic buffer the return equals its category field. K1 version
category 7 maps to K2 category 8 and K1 internal category 8 maps to K2 category 9;
other K1 categories retain the corresponding number.

The exact private record is the 264-byte payload of an aligned 272-byte Eshkol
bytevector carrier on the supported little-endian x86-64 ABI:

| Offset | Field |
|---:|---|
| 0 | `u32-le` category |
| 4 | `u32-le` reason code |
| 8 | `char operation[64]` |
| 72 | `char message[192]` |

Capacity and the carrier header must both equal 264. The payload must be aligned to
`_Alignof(et_kernel_error)`; the full carrier must be nonwrapping, writable, and
disjoint from every borrowed input carrier and protected singleton/factory metadata.
NULL, short, oversized, misaligned, wrapping, or overlapping diagnostic storage
returns invalid argument and leaves the entire 272-byte carrier byte-identical; the
caller does not parse it. Header overlap is rejected even when an aliased input's
declared or supplied length is itself malformed. After admission, native code
validates other transport, generation/index, singleton discovery/audit, then
performs K1 matching. It commits one fully initialized local record to the payload:
all 264 payload bytes zero on success, or bounded NUL-terminated operation/message
strings with zero tails on error. The length header and bytes beyond the exact
carrier remain untouched.

K1 reason codes `0..16` and messages are preserved with E1
`source-domain=kernel-abi`. K2-private codes use
`source-domain=k2-capability-facade`:

| Code | Meaning |
|---:|---|
| `0x4b320001` | rejected diagnostic buffer; bridge-side because the buffer is unchanged |
| `0x4b320002` | malformed borrowed transport/span/symbol/shape |
| `0x4b320003` | stale PID or generation |
| `0x4b320004` | invalid sorted entry index |
| `0x4b320005` | fixed I2 provider absent |
| `0x4b320006` | fixed I2 descriptor or ABI audit mismatch |
| `0x4b320007` | discovered eleven-row runtime audit mismatch |
| `0x4b320008` | factory registration/collision invariant |
| `0x4b320009` | authentication before factory registration |
| `0x4b32000a` | other native invariant failure |

Provider absence maps to unsupported; ABI/version/required-feature mismatch maps to
version mismatch; altered fixed metadata, row mismatch, factory conflict, malformed
native category/code pairing, and trusted invariants map to internal. Stale
generation maps to invalid state. Invalid index or transport maps to invalid
argument. Only an exact well-formed K1 unsupported/code-15 result is a native normal
no-match.

The scalar-only functions use a separate signed result namespace. PID/generation
are positive on success; absent generation is `-7`, and getter invariant failure is
`-9`. Each register returns 1 for new/idempotent registration, `-1` for malformed
input, or `-9` for collision/different re-registration. Each authenticate returns 1
for the exact address, 0 for another address, or `-9` for an unregistered/invariant
state. The Eshkol bridge maps `-1`, `-7`, and `-9` to invalid argument, invalid state,
and internal respectively. These are K2 transport statuses, not invented K1 errors.

## Errors and public behavior

Wrong public argument types and foreign/wrong-kind opaque values raise
`invalid-argument`. Authentic stale values raise `invalid-state`. A valid absent,
unverified, unsupported, nonrepresentable, or nonmatching query makes
`capability-verified?` return `#f`; the same request makes `capability-require` raise
`unsupported`. Other genuine native failures remain visible through E1.

E1 wraps K1 source code/message with `source-domain=kernel-abi`, and private K2
transport failures with `source-domain=k2-capability-facade`. The public error's
operation is the actual A0 operation. Validation completes before publication, and
failure never returns a partial shell, report, entry list, or request.

## Source composition and artifact

The installed `lib/transformer/capabilities.esk` contains only the safe public
facade and the six E1 accessors. `native/k2_wave2_extension.esk` is the complete
trusted K2 extension and loads no aggregate root. `native/k2_wave2_root.esk` loads
the exact I2 root followed by that extension. The fixed twelve wrapper definitions
are reusable by a later C2 bridge without linking the completed K2 archive.

`build/k2/libeshkol_transformer_wave2.a` contains one localized `k2_wave2.o`.
It has exactly 59 global definitions, 53 package exports, and 59 public name strings:
the accepted I2 47/41/47 boundary plus twelve K2 wrappers. It still reports eleven
capabilities. C2's 81/75 count is conditional on its remaining proposed surface and
separate acceptance after K2 merges.

There is no canonical provider symbol, generic dispatcher, raw/private helper
export, Python/PyTorch runtime dependency, cast, copy, transfer, numerical fallback,
or serialized capability proof.

## Verification

`make test-k2` covers exact provider/report audits, both determinism positives,
logical/native request boundaries, all unverified/nonmatching rows, allocation
failpoints and retry, exact diagnostic admission and write behavior, factory
identity/folding/collision/forgery, mutation isolation, bounded fork behavior,
strict fresh-cache AOT and all arities, 1,024/8,192 region retention, deterministic
rebuilds, exact symbol/string/source/dependency manifests, hostile package inputs,
aggregate collision rejection, sanitizers, and production Python isolation.

All affected aggregate gates and the complete repository suite remain required. A
supported Ubuntu 22.04 x86-64 / LLVM-Clang 21.1.8 blocking run is required before
acceptance. Local LLVM 22 results are compatibility evidence only.
