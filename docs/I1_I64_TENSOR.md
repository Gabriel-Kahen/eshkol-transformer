# I1 exact signed-i64 tensor container

## Evidence boundary

I1 adds a separate native C11 ABI 1.0 for an owned exact signed-`i64` CPU
container. It does not change the K1 ABI, any A0 public Eshkol name or arity, or
the provider-free K1 baseline report. The normal build artifact is
`build/i1/libeshkol_transformer_i64.a`; there is no install, dynamic discovery,
serialization, or persisted-format contract.

The I1 provider verifies only this exact K1 request tuple:

- capability `tensor.i64`;
- operation `storage.copy`;
- dtype `i64` and device `cpu`;
- deterministic execution;
- exactly two shape alternatives: rank 0 and rank 1 with extent
  `0..SIZE_MAX/8` (`0..2305843009213693951` on the frozen ABI); and
- exactly one dense row-major zero-offset input and one disjoint caller-owned
  output of the same shape.

The provider validates every fallible condition before its non-failing full
`memcpy` commit. It allocates, casts, transfers, relabels, materializes, and
falls back to nothing. Container support for ranks through 64 is not a generic
`tensor.i64` capability claim, and ranks 2 through 64 remain outside the verified
K1 provider entry.

## Native ABI 1.0

The public header is `include/eshkol_transformer/i64_tensor.h`. It exposes an
opaque `et_i64_tensor`, an opaque tracked `et_i64_tensor_borrow`, ABI version
accessors, owned construction/destruction, metadata accessors, exact full-buffer
copy operations, borrowed K1 views, and an explicit provider-descriptor accessor.
Every exported I1 ABI entry point uses the `_v1` symbol suffix. I1 has a separate
fixed caller-owned error record because frozen K1 error categories do not include
A0 `invalid-state`.

Construction requires a null output slot, deep-copies shape metadata, and
zero-initializes storage. Rank 0 is a scalar with one element and eight bytes.
For ranks 1 through 64, any zero extent canonically means zero elements, zero
bytes, null data, and zero byte strides in every dimension. Otherwise element
count and byte length use checked multiplication, and the dense row-major byte
stride for dimension `d` is `8 * product(shape[d+1..rank))`. Nonempty data and
caller copy buffers are aligned for `int64_t`.

Values are copied as exact C `int64_t` values across the full
`INT64_MIN..INT64_MAX` range. There is no parsing or numeric conversion. Copy
calls require the exact element count and nonaliasing buffers; an empty tensor
accepts a null buffer. Validation failures leave tensor storage, caller output
buffers, scalar output slots, and handle output slots unchanged.

The creator owns shape, stride, and data storage until successful destruction.
Destroy is null-idempotent and nulls the caller's handle on success. A borrow
lease owns K1 view metadata whose shape and data are borrowed from the tensor until
`borrow_end`; at most one lease may be active. The lease authorizes direct external
use, including passing the view as a K1 output, so I1 cannot prevent writes through
the live view. The active lease blocks owner-side `copy_from` and destruction, while
`copy_to` may read. Caller serialization is required. Fabricated,
copied-after-release, and otherwise stale raw pointers are outside the C contract;
the live-handle checks do not make arbitrary invalid-address dereferences safe.

## K1 conformance and E1 mapping

Every borrowed view has the unchanged K1 v1.0 prefix, dtype `i64`, device `cpu`,
dense-row-major layout, zero offset, exact rank/shape/byte span, null data only
when empty, and aligned data otherwise. K1 has no explicit stride field; I1's
stride accessors are the independent evidence for its canonical dense layout.

The native I1 categories map at an Eshkol boundary as follows:

| I1 category | E1/A0 category |
|---|---|
| `INVALID_ARGUMENT` | `invalid-argument` |
| `SHAPE_MISMATCH` | `shape-mismatch` |
| `NONCONTIGUOUS` | `noncontiguous` |
| `INVALID_STATE` | `invalid-state` |
| `VERSION_MISMATCH` | `version-mismatch` |
| `INTERNAL` or an unknown category | `internal` |

The compiled-Eshkol test fixture uses only fixed-arity `ptr` and `i64` externs,
copies native diagnostics before releasing their context, and wraps failures with
the reviewed E1 `transformer-error-wrap-foreign` boundary using source domain
`i64-tensor`. Canonical source inspection and executable probes show that the
pinned compiler does not range-check an `extern i64` argument: the fixture
therefore proves `integer?`, `exact?`, `real?`, and inclusive signed-i64 range
validation before every such call. The fixture is not a production scalar/list
tensor adapter, a finalizer, or generic Eshkol tensor evidence.

Native allocation failures use I1 `INTERNAL/ALLOCATION_FAILED`, matching K1's
existing ABI-local treatment because A0 has no resource-exhausted category. The
stable source code/message preserves that cause rather than presenting it as a
shape, argument, state, or capability error.

## Verification

Run:

```sh
/usr/bin/bash -c 'make test-i1'
```

The focused gate warning-cleans C11 and C++17 consumers, runs the normal native
contract twice and compares output bytes, exercises deterministic allocation
failpoints in the sanitized build, runs ASan/UBSan, and AOT-compiles/links/runs
the canonical Eshkol interop fixture twice. LeakSanitizer can be enabled with
`I1_ASAN_DETECT_LEAKS=1` where supported; it is disabled under the local traced
executor where LeakSanitizer itself is unavailable.

The final compatibility-lane run passed 661 normal native checks and 795
test-hook ASan/UBSan checks. Both AOT executions passed 59 checks with output
SHA-256
`de78f6cb1b0ea6ffd169a4dfc1d7500f1f3f1db69dd113f8fd737a978c92d217`.
The compiler binary SHA-256 was
`caa295b19a6e9388963aa0def99dade63656d2dcbffccad421bd1daaa1db3750`,
and its clean source checkout was the canonical commit
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`.

## Explicit limitations and downstream obligations

I1 supplies no tokenizer, token range check, dataset, mask, numerical/model
operation, gradient, bool/f32 storage, dtype conversion, device transfer,
accelerator, performance, checkpoint, or file format. It contains no Python or
PyTorch production path. It makes no concurrency or thread-safety claim.

T1 may consume the owned rank-1 CPU substrate only through a separately reviewed
production wrapper that supplies tokenizer behavior, E1 public errors, and safe
native-handle lifetime. D2 may use the rank-2 container only as unverified owned
storage; the K1 provider explicitly does not advertise rank 2, and I1 supplies
neither masks nor loader/cursor/packing/resume behavior. The pinned runtime exposes
no proved native-pointer finalizer, so this
workstream deliberately does not claim an automatic production Eshkol ownership
bridge; T1 and D2 must coordinate explicit lifetime rather than leak or invent one.
