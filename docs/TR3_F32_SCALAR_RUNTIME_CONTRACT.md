# TR3 true-binary32 scalar runtime prerequisite

Status: **upstream contract accepted for implementation; runtime feature not yet implemented or accepted**.

This document freezes the smallest Eshkol runtime addition needed for the A0
trainer metrics path. It is a requirement on a future runtime successor, not a
claim about the pinned runtime, an implementation patch, a runtime pin update, or
a new transformer public API. The runtime audit used exact commit
`81298b4a9608fb92eb6f351a2eabd8392da7d9ef` in the ephemeral checkout
`/tmp/eshkol-rethrow-source-81298b4a-20260923T200234Z`. That checkout was removed
by the host reboot. The audit evidence survives, and implementation is now
underway in a separate isolated Eshkol worktree.

The moving TR3 metrics design stabilized separately at transformer commit
`dee31931308f86553debe207916e5ecf37506a95`, proposal SHA-256
`c118b7f32169d5e537e503f7539ebd8cd49d0c859b84027d87962e5ece11de8a`.
The metrics author subsequently reported coordinated draft SHA-256
`c341b21b8923da6723a1d05b9a504f096e309b74bd7eb6161eb645968e5cdae9`,
which proposes a private E3 bit accessor and a separate lifetime disposition. This
contract does not import either candidate, accept its metrics owner or lifetime
policy, or modify the active E3/runtime integration work.

## 1. Authority and exact need

A0 requires reported losses and metrics to accumulate and return `f32`, and fixes
the three trainer result schemas in `docs/PUBLIC_API_CONTRACT.md:104-105,509-559`.
Automatic dtype narrowing and scalar fallback are forbidden. The existing ordinary
floating Eshkol value is `ESHKOL_VALUE_DOUBLE`, backed by a C `double`; returning it
under an f32 field name would make the dtype claim false.

The selected TR3 refinement additionally requires `metrics-ref` to return a
detached, allocation-free immediate containing the exact binary32 word copied from
the accepted producer. A0 itself does not say “immediate”, “detached”,
“allocation-free”, or “bit-identical”; those are explicit downstream design
choices. This distinction must remain visible in review.

The required runtime value is therefore:

- a first-class, unboxed, inexact real value;
- physically one IEEE-754 binary32 word;
- constructible and inspectable by raw `uint32_t` bits without conversion;
- copied by value through AOT, VM, embedding, containers, and region barriers;
- distinguishable from DOUBLE, integer, tensor, bytevector, and foreign values;
- free of allocation, ownership, release, and hidden scalar fallback.

A boxed f64, rank-zero I2 tensor, bytevector, precision flag on DOUBLE, or
transformer-private scalar wrapper does not satisfy this contract.

## 2. Audited source facts

The following are source observations, not runtime-support claims:

- `inc/eshkol/eshkol.h:99-148` assigns native tags through 10, then 16-19,
  leaving 11 unused. The only ordinary inexact real is DOUBLE=2.
- `inc/eshkol/eshkol.h:164-204` and `inc/eshkol/eshkol_ffi.h:43-58` expose a
  nominal 16-byte value with one 64-bit payload; neither has an f32 member or ABI.
- `lib/backend/type_system.cpp:57-72` fixes LLVM tagged values at
  `{i8,i8,i16,i32,i64}`. `tagged_value_codegen.cpp:115-149,330-353` recognizes,
  packs, and unpacks LLVM `double`, not LLVM `float`.
- `lib/types/hott_types.cpp:151-157` registers nominal `Float32` with
  `RuntimeRep::Float64`; `:691-713,747-822` promotes reals to Float64 and has no
  native Float32 tag mapping.
- `lib/backend/vm_core.c:123-201` defines one inexact `VAL_FLOAT` backed by
  `double`. `lib/backend/vm_numeric.h:28-53` occupies VM value IDs 8-33;
  VM ID 11 is RATIONAL.
- ESKB version 1 has F64 but no F32 constant in
  `lib/backend/eskb_format.h:22-38`.
- Native display, equality, hashing, introspection, errors, FFI, and persistence
  dispatch explicitly on DOUBLE only.
- `lib/core/runtime_regions.cpp:1188-1217,2023-2041,2096-2119` leaves values it
  classifies as non-pointers unchanged and copies complete tagged values through
  the checked barrier.

The type/flags audit found a real but separable legacy defect:

- exactness is documented in `flags`, and normal integer/double constructors plus
  LLVM `buildTaggedValue` store a pure tag and separate flags;
- `ESHKOL_VALUE_EXACT_INT64`, `ESHKOL_VALUE_INEXACT_DOUBLE`,
  `ESHKOL_MAKE_EXACT`, and `ESHKOL_MAKE_INEXACT` instead fold 0x10/0x20 into
  `type`;
- the combined constants/manipulators have no production call sites in the audited
  `inc`, `lib`, and `tests` trees beyond their declarations and one explanatory
  i128 comment;
- fourteen source files contain unconditional 0x0f type masks, while six use an
  `>=8` split; port code still creates four folded pointer/port encodings.

The prerequisite therefore does not authorize a global port/tag migration. It
selects the narrow pure-tag rule in section 4 and makes violations test failures.

## 3. Mandatory slice versus deferred language expansion

The mandatory slice is exactly what a public metrics value can reach:

1. native tagged representation and raw-bit ABI;
2. LLVM AOT/JIT construction, inspection, and value transport;
3. VM representation, host-native construction/inspection, stack and container
   transport;
4. core numeric classification, type reporting, equality, hashing, and formatting;
5. explicit arithmetic promotion to the existing f64 domain;
6. region escape/write-barrier behavior;
7. stable embedding/FFI access;
8. supported-host and native/AOT/VM/embedding parity evidence.

The following are deferred and must not be smuggled into this prerequisite:

- f32 source literals or a public Scheme raw-bit constructor;
- type-preserving `write`/`read` syntax;
- ESKB F32 constants or an ESKB format bump;
- checkpoint, KB, model, or generic persistence of f32 scalar values;
- Python/NumPy convenience conversion;
- f32-preserving arithmetic or f32-preserving transcendental results, plus complex,
  dual, Taylor, or AD behavior; promoted-f64 elementary functions remain mandatory;
- GPU, f16, bf16, tensor storage, or mixed-precision behavior.

VM support remains mandatory even though ESKB constants are deferred: a registered
host native or compiled `metrics-ref` can push a runtime-produced f32 value without
placing an f32 literal in the bytecode constant pool. Old ESKB readers and writers
must reject attempts to serialize the new value rather than convert it to F64 or
NIL.

## 4. Native tag, flags, and canonical bytes

The runtime owner shall reserve:

```c
ESHKOL_VALUE_FLOAT32 = 11
```

Tag 11 is a native tagged-value assignment only. It is not HoTT TypeId 11, VM
value ID 11, an ESKB constant ID, a heap subtype, or a port encoding.

Every canonical value has:

```text
type       11 exactly
flags      ESHKOL_VALUE_INEXACT_FLAG exactly
reserved   0
padding    all zero
data       raw_val[31:0] = binary32 bits; raw_val[63:32] = 0
```

Construction zero-initializes all 16 bytes before writing fields. Raw bits are
moved with integer assignment or `memcpy`, never union-punning through C `float`.
The bits API preserves positive/negative zero, subnormals, infinities, and NaN
payloads. TR3 producers separately require finite metric values; the runtime bits
carrier does not erase nonfinite IEEE encodings.

The selected type/flags decision is **strict prohibition**, not global repair:

- no f32 producer may OR exactness, port, or other bits into `type`;
- f32 checks compare `type == ESHKOL_VALUE_FLOAT32`; they do not use 0x0f/0x3f
  normalization or the legacy combined-type macros;
- checked inspection additionally requires the exact canonical flags, reserved,
  padding, and zero upper payload;
- generic code that folds flags into the new type must be fixed before acceptance;
- legacy combined integer/double and port encodings remain a separately scoped
  runtime hardening issue.

This is safe for the bounded slice because current production integer/double
constructors already use pure tags and separate flags, and tag 11 is not any
currently generated port encoding. It avoids changing unrelated legacy pointer and
port behavior while making the new representation unambiguous.

The implementation must strengthen ABI checks to require, on supported targets:

```text
sizeof(eshkol_tagged_value_t) == 16
alignof(eshkol_tagged_value_t) == 8
offsetof(type) == 0
offsetof(flags) == 1
offsetof(reserved) == 2
offsetof(data) == 8
```

Equivalent assertions apply to `eshkol_ffi_value_t` and the LLVM tagged layout.

Tag 11 is numerically inside ranges that current generic code sometimes treats as
pointer-bearing. VM tag 34 is likewise adjacent to heap-backed VM values. Adding a
named switch case is insufficient. Before either tag is constructed, the runtime
owner must produce an exhaustive disposition of:

- every 0x0f/0x3f tag mask, `< 8`/`>= 8` range test, switch default, and table
  indexed by a tag;
- every native `ptr_val` cast/dereference, release/free path, GC/region traversal,
  unknown-tag path, and immediate/pointer allowlist;
- every VM `as.ptr` or heap-index access, evacuation/parallel-copy classifier,
  release path, unknown-value default, and immediate/heap allowlist.

Each site must explicitly classify FLOAT32 as an immediate, reject it, or prove it
unreachable. No catch-all native or VM range may interpret arbitrary f32 payload
bits as an address or heap index. The required inventory is acceptance evidence,
not an optional cleanup, and must be repeated against the actual compatible
successor because the separately owned allocator fix may change these sites.

## 5. Required future C ABI

These names are the proposed runtime-owned ABI. They do not exist at the audited
commit and must not be consumed until an accepted runtime exports them.

```c
enum {
  ESHKOL_VALUE_F32_OK = 0,
  ESHKOL_VALUE_F32_INVALID_ARGUMENT = 1,
  ESHKOL_VALUE_F32_INVALID_VALUE = 2
};

int32_t eshkol_value_f32_from_bits_v1(
    eshkol_tagged_value_t *out, uint32_t bits);

int32_t eshkol_value_f32_to_bits_v1(
    const eshkol_tagged_value_t *value, uint32_t *out_bits);

int32_t eshkol_value_is_f32_v1(
    const eshkol_tagged_value_t *value);

uint32_t eshkol_runtime_has_f32_scalar_v1(void);
```

`from_bits` accepts every `uint32_t` bit pattern and writes one canonical value.
`to_bits` accepts only a canonical f32 value. Null arguments are
`INVALID_ARGUMENT`; a wrong tag or noncanonical byte pattern is `INVALID_VALUE`.
Failed calls preserve every caller output byte. `is_f32` returns one only for a
canonical value and returns zero for null, malformed, or other values.

`eshkol_runtime_has_f32_scalar_v1` is an exported detection symbol and returns
exactly one. The installed header defines `ESHKOL_HAS_F32_SCALAR_ABI_V1` to 1.
Dynamic embedders detect support by resolving the symbol and then requiring return
one; symbol absence means unsupported. Static consumers compile the f32 calls only
when the header macro is present and equal to one. No older runtime is assumed to
provide a generic feature query or a reserved feature bit.

Pointer/out-parameter forms are mandatory because the runtime already documents a
platform C by-value/JIT tagged-struct calling-convention mismatch and uses thunks at
such boundaries. Header-only by-value convenience wrappers may exist, but they are
not the cross-module authority.

No numeric `float` constructor is required. Callers that own a C `float` may obtain
its bits with `memcpy`; keeping the trusted boundary bit-oriented prevents hidden
conversion or NaN canonicalization.

## 6. LLVM AOT/JIT contract

`TaggedValueCodegen` must add exact operations equivalent to:

```text
packFloat32Bits(i32)    = zero-extend i32 to i64, tag 11, INEXACT flag
unpackFloat32Bits(tv)   = require tag 11, truncate canonical i64 to i32
promoteFloat32(tv)      = canonical qNaN for NaN; otherwise bitcast i32 to float
                          then fpext float to double
isFloat32(tv)           = exact tag/canonical representation check
```

`ensureTagged` must recognize raw LLVM `float`; it must never map it to NULL or
DOUBLE. HoTT `Float32` must map to `RuntimeRep::Float32` and native tag 11 in both
directions. Existing decimal literals remain Float64; this contract adds no f32
literal syntax.

Generated return values, arguments, closures, cons/vector/hash slots, globals, and
FFI thunks must preserve all canonical bytes. No path may unpack tag 11 with the
existing blind double bitcast.

## 7. VM and host-native contract

The VM shall append, without renumbering existing values:

```c
#define VAL_FLOAT32 34
```

Its `Value` union carries `uint32_t f32_bits`; construction uses a dedicated
`FLOAT32_BITS_VAL(bits)` helper. Native tag 11 and VM tag 34 are translated
explicitly at boundaries and are never copied as equivalent numeric IDs.

The registered host-native ABI shall add:

```c
enum {
  ESHKOL_VM_F32_OK = 0,
  ESHKOL_VM_F32_INVALID_ARGUMENT = 1,
  ESHKOL_VM_F32_STACK_UNDERFLOW = 2,
  ESHKOL_VM_F32_WRONG_TYPE = 3,
  ESHKOL_VM_F32_STACK_OVERFLOW = 4
};

int32_t eshkol_vm_host_pop_float32_bits_v1(VM *vm, uint32_t *out_bits);
int32_t eshkol_vm_host_push_float32_bits_v1(VM *vm, uint32_t bits);
```

Both calls use the frozen status domain `0=OK`, `1=INVALID_ARGUMENT`,
`2=STACK_UNDERFLOW`, `3=WRONG_TYPE`, and `4=STACK_OVERFLOW`. Pop returns
INVALID_ARGUMENT for a null VM or output, STACK_UNDERFLOW for an empty stack, and
WRONG_TYPE when the top value is not `VAL_FLOAT32`; every failure leaves both the
stack and `out_bits` unchanged. Success writes the raw bits and removes exactly one
stack value. Push returns INVALID_ARGUMENT for a null VM and STACK_OVERFLOW when
the existing stack has no capacity; failure leaves the stack unchanged. Success
accepts every bit pattern, appends exactly one value, and allocates nothing.

VM stacks, upvalues, globals, pairs, vectors, hashes, continuations, parallel
transport, and region evacuation must copy the immediate value without treating
its payload as a heap index.

ESKB version 1 remains unchanged in the mandatory slice. Its compiler/writer must
diagnose a requested f32 constant as unsupported; its reader cannot manufacture
`VAL_FLOAT32`. A later readable-literal proposal must assign a distinct ESKB
constant, encode four explicit little-endian bytes, bump the format/version, and
prove old-reader rejection.

## 8. Numeric and type semantics

The minimal slice treats f32 as a narrow transport representation whose arithmetic
join is the existing f64 domain. This avoids pretending that existing double
arithmetic executes binary32 operations.

- `number?`, `complex?`, and `real?` return true.
- `exact?` returns false; `inexact?` returns true.
- `rational?` returns true exactly when the binary32 value is finite.
- `integer?` returns true exactly when it is finite and mathematically integral.
- `finite?`, `infinite?`, `nan?`, `zero?`, `positive?`, and `negative?` inspect the
  binary32 value with IEEE semantics. Both zero signs satisfy `zero?`.
- `type-of` reports the distinct runtime type `float32`; `float32?` returns true
  only for the canonical tag. These are Eshkol runtime names, not transformer A0
  names.

**Public type reflection decision.** `(type-of value)` is a total unary,
first-class procedure on native AOT, native JIT, and the bytecode VM. It returns
a canonical interned Scheme symbol naming the semantic runtime type, never a
string, native/VM numeric tag, heap/callable subtype ID, or ESKB constant ID.
Results with the same spelling compare true under `eq?`, including comparison
with a literal symbol. The public C `eshkol_type_of` returns that same symbol in
tagged-value form. Internal numeric IDs may differ between substrates.

The current semantic names are `null` for the empty value; `integer` for int64
and bignum; `real` for binary64; `float32` for canonical binary32; `boolean`,
`char`, and `symbol`; `complex`, `rational`, `dual-number`, `hyper-dual-number`,
and `i128`; `pair`, `string`, `vector`, `tensor`, `hash-table`, `bytevector`,
`record`, `values`, `exception`, `port`, `promise`, `future`, and `eof-object`;
`closure`, `primitive`, `continuation`, `lambda-sexpr`, and `ad-node` for the
declared callable kinds; `substitution`, `fact`, `knowledge-base`,
`factor-graph`, `workspace`, `logic-variable`, `prng`, `parameter`, `ad-tape`,
`manifold`, `riemannian-adam-state`, `dnc`, `sdnc`, and `taylor` for the declared
domain kinds; `handle`, `buffer`, `stream`, and `event` for resource kinds; and
`void` for an unspecified result. A valid callable without a more specific
classification reports `procedure`. Native exactness/direction flags do not
change the semantic name.

Only a canonical f32-v1 carrier reports `float32`; malformed f32 carriers and
undeclared direct tags report `unknown`. A declared heap or callable value with
an undeclared subtype reports `heap-object` or `procedure`, respectively.
Deprecated pointer tags report the semantic name of their consolidated
replacement. Adding a declared tag or subtype requires a symbolic mapping and
cross-substrate tests in the same change. This decision supersedes the current
native integer-tag and VM string implementations; their observable results are
not accepted type-reflection behavior.

Every ordinary numeric operation that currently accepts DOUBLE must accept f32 by
the promotion rule below and return the same result kind as its DOUBLE path. This
includes f32/f32, f32/integer, f32/DOUBLE, unary negation/absolute value, min/max,
remainder where already defined for inexact values, comparisons, and every
elementary function in the implementation's required DOUBLE-dispatch inventory.
Comparisons return boolean. Min/max return DOUBLE even when the selected operand
was f32; they do not leak the original tag through an operand-selection shortcut.
The explicit exclusions in section 12 remain unsupported rather than inheriting
this rule.

The shared promotion helper preserves every finite binary32 value exactly as a
binary64 value, including the sign of zero, and maps infinities by sign. Any
binary32 NaN maps to the fixed positive quiet binary64 NaN bit pattern
`0x7ff8000000000000`; arithmetic promotion therefore does not preserve a NaN sign,
payload, or signaling state. Raw-bit inspection remains lossless. This is an
explicit promotion, not a fallback or a claim of binary32 arithmetic. A future
f32-preserving arithmetic contract would be a separate numerical workstream with
rounding-mode, contraction, libm, and gradient evidence.

The mandatory equality policy is:

- numeric `=` compares after applying the section 8 promotion rule;
- `eqv?`, atomic `equal?`, and the default hash-key equality require the same
  representation tag, then use IEEE numeric equality;
- positive and negative zero therefore compare equal within f32;
- every NaN compares unequal, including to itself;
- F32 and DOUBLE may be `=` while remaining unequal under `eqv?`/`equal?`;
- the f32 hash includes the representation tag and canonicalizes both zero signs to
  one hash so equal f32 keys always hash equally.

Raw-bit inspection is the sole equality operation that distinguishes zero signs or
NaN payloads.

## 9. Formatting and read/write boundary

`display`, `write`, error rendering, and `number->string` must recognize tag 11 and
must never reinterpret its low bits as a double payload or integer. Native and VM
must call one reviewed f32 formatting helper; the current independent native and
VM formatting branches are not sufficient. The helper emits `0.0`, `-0.0`,
`+inf.0`, `-inf.0`, and `+nan.0` for the corresponding classes. For a finite
nonzero value it applies the deterministic promotion rule in section 8 and the
existing shortest-f64 algorithm, adding a decimal marker when needed by the
runtime's number grammar. The result round-trips through the ordinary reader to
the same widened binary64 bits. It may contain more digits than shortest-f32 text;
it is a shortest round-tripping decimal approximation of the widened value, not a
complete exact decimal expansion. Native and VM output must be byte-identical.

Ordinary decimal `read` and `string->number` continue to produce DOUBLE. Therefore
mandatory `write`/`read` compatibility is numeric, not representation-preserving:
reading the emitted finite text produces the same numeric value as a DOUBLE, not an
f32 tag. Signed negative zero and existing R7RS infinity/NaN spellings retain their
current numeric behavior. NaN payload preservation is available only through the
raw-bit ABI.

A syntax such as `#f32(XXXXXXXX)`, shortest-f32 formatting, and tag-preserving text
round-trip are deferred language features. No review may claim them from this
contract.

## 10. Regions, containers, and allocation

Native region classification and VM evacuation must include explicit FLOAT32 cases
as pointer-free immediates. Region escape and the checked write barrier copy all 16
canonical bytes unchanged and allocate no transaction or payload storage for a lone
f32 value.

Cons cells, vectors, hashes, closures/upvalues, globals, parameters, continuations,
and exception irritants must retain the tag, flags, padding, and payload. Optimized
double-only slot helpers must either add an f32 case or route f32 through the full
tagged-value copy. No default branch may interpret it as a pointer, DOUBLE, INT64,
or unknown NIL.

`metrics-ref` construction and inspection must be allocation-free after its metrics
record has been authenticated. The runtime scalar has no release operation or
lifetime owner.

## 11. FFI and embedding

The stable FFI shall add:

```c
#define ESHKOL_FFI_TYPE_FLOAT32 11

int32_t eshkol_ffi_float32_from_bits_v1(
    uint32_t bits, eshkol_ffi_value_t *out);

int32_t eshkol_ffi_float32_to_bits_v1(
    const eshkol_ffi_value_t *value, uint32_t *out_bits);

int32_t eshkol_ffi_is_float32_v1(const eshkol_ffi_value_t *value);

int32_t eshkol_ffi_float32_to_double_v1(
    const eshkol_ffi_value_t *value, double *out);
```

The FFI representation is byte-compatible with the native canonical value and has
the same statuses and output-preservation rules as section 5; `is_float32_v1`
returns zero for null or malformed input. `float32_to_double_v1` applies the exact
promotion rule from section 8. The existing by-value `eshkol_ffi_to_double` remains
outside the new cross-module authority and must continue to reject unknown tag 11;
it must never read the f32 payload as a C double. Old embedders observe an unknown
tag and must not be told that tag 11 is DOUBLE.

Embedding feature detection uses the exact header macro and exported probe from
section 5. Python conversion and other language bindings remain deferred.

## 12. Explicit unsupported cases

The first accepted slice must report `unsupported` rather than silently widen or
reinterpret f32 when passed to:

- forward/reverse AD entry points, duals, hyper-duals, and Taylor towers;
- complex constructors or operations that require an f32 component policy;
- persistence encoders without a versioned f32 scalar format;
- source/bytecode constant construction without accepted literal/ESKB support;
- any foreign-device or accelerator scalar path.

Ordinary numeric arithmetic is not in this list because section 8 explicitly and
truthfully promotes it to DOUBLE. The unsupported AD rule does not affect the A0
metric fields, which carry no graph and require no gradient.

## 13. Required implementation files

The bounded runtime owner must at least review and update the following surfaces;
an unchanged file requires an explicit audit disposition:

```text
inc/eshkol/eshkol.h
lib/backend/type_system.cpp
inc/eshkol/backend/tagged_value_codegen.h
lib/backend/tagged_value_codegen.cpp
inc/eshkol/backend/arithmetic_codegen.h
lib/backend/arithmetic_codegen.cpp
lib/backend/llvm_codegen.cpp
inc/eshkol/types/hott_types.h
lib/types/hott_types.cpp
lib/types/type_checker.cpp
lib/core/runtime_display_hosted.cpp
lib/core/runtime_errors_hosted.cpp
lib/core/runtime_deep_equal.cpp
lib/core/runtime_hash_table.cpp
lib/core/introspection.cpp
lib/core/runtime_arena_core.cpp
lib/core/runtime_regions.cpp
lib/core/runtime_tagged_cons.cpp
lib/core/runtime_vector_mutation.cpp
lib/core/system_builtins.c
inc/eshkol/core/dtoa_shortest.h
inc/eshkol/eshkol_ffi.h
lib/ffi/eshkol_ffi.cpp
lib/backend/vm_core.c
lib/backend/vm_numeric.h
lib/backend/vm_run.c
lib/backend/vm_native.c
lib/backend/vm_region_evac.c
lib/backend/vm_parallel.c
inc/eshkol/backend/vm.h
lib/backend/eshkol_vm.c
lib/backend/eskb_format.h
lib/core/kb_persistence.cpp
```

The owner must search every `ESHKOL_VALUE_DOUBLE` and `VAL_FLOAT` dispatch and
record why f32 is included, deliberately promoted, or explicitly unsupported. It
must also complete the exhaustive mask/range/default/pointer/heap classifier audit
in section 4; a name-only search does not satisfy that requirement.

Parser/AST, runtime reader, REPL literal lowering, Python bindings, and positive f32
serialization formats remain deferred unless the implementation expands scope and
obtains separate acceptance first. Existing ESKB and KB readers/writers are in the
mandatory audit: writers must reject a runtime f32 explicitly and failure-atomically,
while readers must retain their unknown-code rejection and have no code that maps
input to tag 11/34. This is negative-path hardening only; it does not add an
encoding or version bump.

## 14. Acceptance tests

### Representation and ABI

- Compile-time size, alignment, offset, native/FFI/LLVM layout equality.
- Raw-bit round-trip for `+0`, `-0`, minimum/maximum subnormal, minimum normal,
  maximum finite, both infinities, multiple quiet-NaN payloads, and signaling NaNs.
  The bits APIs preserve every pattern exactly; promotion canonicalizes every NaN
  exactly as section 8 specifies.
- Every constructor byte outside the low payload word is zero.
- Null, DOUBLE, integer, bool, rank-zero tensor, bytevector, malformed flags,
  nonzero reserved/padding/high payload, and foreign tags reject without changing
  outputs.
- Adversarial payload words shaped like aligned addresses, region pointers, object
  headers, and VM heap indices cross every audited classifier without dereference,
  release, tracing, or evacuation as a pointer.
- Static and dynamic embedding tests prove the exact feature-detection behavior,
  including symbol absence for the prior runtime.

### Semantics

- Complete predicate/type matrix in native AOT and VM.
- Arithmetic witnesses prove exact finite widening, canonical NaN promotion, and
  DOUBLE result tags.
- A table enumerates every admitted arithmetic, remainder, comparison, and
  elementary-function entry point; each entry has native/AOT and VM witnesses or
  an explicit unsupported diagnostic. Min/max, unary operations, signed zero,
  infinities, and canonical NaN promotion agree between AOT and VM.
- `=`, `eqv?`, `equal?`, and hash tests pin cross-tag and signed-zero behavior and
  prove the hash/equality invariant.
- Native and VM formatting is byte-identical for all boundary classes.
- Finite formatting, signed zero, `read`, and `string->number` tests prove the
  numeric widened-binary64 round-trip while separately proving that the read value
  is DOUBLE rather than FLOAT32.
- AD/dual/Taylor/complex and unsupported persistence attempts fail explicitly.

### Transport and lifetime

- AOT return/argument, failure-atomic VM host push/pop, FFI, closure capture,
  global, cons, vector, hash, continuation, parallel transport, VM region
  evacuation, and exception-irritant round-trips preserve bits.
- Nested region escape and checked write barriers preserve the full canonical value
  and prove zero allocations on the immediate path.
- Repeated metric-style construction/inspection has flat scalar allocation counts;
  any metrics-registry retention is measured separately by TR3.

### Toolchain evidence

- Supported Ubuntu 22.04/LLVM 21 native, strict AOT, VM, and embedding lanes.
- Two fresh AOT compilations produce deterministic output.
- Sanitizer coverage for C/FFI/VM boundaries.
- x86-64 ABI evidence; any additional architecture is unsupported until its exact
  layout/calling convention is tested.

The current host's missing `libLLVM.so.21.1` or `libopenblas.so.0` is an environment
limitation, not evidence that the runtime feature is unsupported. Acceptance needs
the supported container/toolchain lane.

## 15. Downstream E3 and metrics seam

This runtime contract does not grant TR3 access to E3's private destination tensor
pointers. The latest coordinated minimal downstream seam follows the existing E3
scalar `counter_ref` shape and is proposed as an amendment to the E3-owned
[private contract](E3_PRIVATE_CONTRACT.md):

```c
int64_t et_e3_private_selected_metric_bits_ref_v1(
    void *frame, int64_t selector);
```

Selectors zero and one mean loss and mask-weight; no other metric is exposed. The
call registry-authenticates before dereference, requires the accepted post-finish
idle/published state, copies one private rank-zero I2 word locally, and returns its
zero-extended `uint32_t` bits as an exact nonnegative i64. Failure returns -1 with
the existing E3 diagnostics. It allocates nothing and exposes no destination
pointer, perplexity, or accuracy.

TR3 must retain its trainer/evaluation guard and no-concurrency authority across
both bit reads and the existing two counter reads. It stages and validates all four
returned i64 values before writing any public metrics field. If another caller could
republish the frame between those reads, the scalar seam would be insufficient and
E3 would need one atomic aggregate copy instead. E3 owns the final name, state/error
mapping, and acceptance. This proposed seam is not an API supplied by this commit.

The TR3 metrics owner is separate from the runtime scalar. This prerequisite
selects no registry quota or arbitrary operation limit. An uncapped strong
process-lifetime registry grows without an operation-count bound, and bounded-run
measurements are profile evidence rather than a lifetime bound. TR3/root must accept
the actual lifetime policy separately before public trainer completion.

## 16. Dependency and adoption order

1. Root accepts or revises this prerequisite contract.
2. The active runtime owner finishes its current union proof and the separately
   reviewed unchecked-allocation crash prerequisite reaches a compatible accepted
   union. A new isolated f32 workstream starts from that successor, re-audits every
   source anchor above, and does not modify the busy runtime checkout.
3. Runtime implementation and independent review prove the complete mandatory slice
   and supported-host evidence without touching the user's busy runtime checkout.
4. F0/R0 and A0 adopt the accepted successor commit and rerun declaration/toolchain
   evidence. No transformer pin moves earlier.
5. E3 records and accepts its private selected-metrics copy seam in an E3-owned
   contract and retains destination ownership.
6. TR3-A implements the metrics authority and `metrics-ref` against the accepted
   runtime ABI; TR3-B/O/C and joint restore retain their existing dependencies.
7. Public trainer wrappers run schema, type, transaction, lifetime, and supported
   union CI before any completion claim.

No dependent implementation may guess tag IDs, helper names, or behavior from this
proposal before the runtime owner publishes and the transformer project accepts the
actual successor ABI.
