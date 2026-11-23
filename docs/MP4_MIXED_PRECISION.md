# MP4 mixed-precision foundation

MP4 1.0 exposes `et_mp4_kernel_provider_v1()` for explicit K1 1.0 resolver
registration and two standalone loss-scaler functions in
`eshkol_transformer/mp4_mixed_precision_abi.h`. Link
`build/mp4/libeshkol_transformer_mp4.a` and K1 with the platform math library.
The MP4 archive exports these three functions and has no allocator imports.

## K1 requests and tensor schemas

Every verified request uses compute dtype `f32`, device `cpu`, and admits
deterministic execution. All six operations use the same exact shape rows
`[1]`, `[8]`, `[16]`, `[64]`, `[256]`, `[256,4]`, `[1,4,4]`, and
`[2,8,16]`. In the table, `S` is the request shape and `[]` is a
rank-zero scalar. Inputs and outputs appear in descriptor order.

| Capability | Operation | Inputs | Outputs |
| --- | --- | --- | --- |
| `kernel.cast-f16` | `cast.f32-to-f16` | `f32[S]` | `f16[S]` |
| `kernel.cast-f16` | `cast.f16-to-f32` | `f16[S]` | `f32[S]` |
| `kernel.cast-bf16` | `cast.f32-to-bf16` | `f32[S]` | `bf16[S]` |
| `kernel.cast-bf16` | `cast.bf16-to-f32` | `bf16[S]` | `f32[S]` |
| `kernel.grad-unscale` | `grad.unscale-f16-to-f32` | `f16[S]`, `f32[]` inverse scale | `f32[S]` master gradient, `bool[]` overflow |
| `kernel.grad-unscale` | `grad.unscale-bf16-to-f32` | `bf16[S]`, `f32[]` inverse scale | `f32[S]` master gradient, `bool[]` overflow |

Each view is borrowed, dense row-major, and zero-offset. Data addresses are
aligned to their element width. For `N` elements K1 requires exactly `2N`
bytes for the narrow tensor and `4N` bytes for its f32 counterpart. For
example, `[256,4]` needs 2,048 versus 4,096 bytes. The tests dispatch both
forms for each shape and verify the two-to-one storage ratio. This measurement
covers tensor payloads.

Descriptors and inputs remain stable through validation and invocation.
Output spans must be disjoint from inputs, other outputs, and live call
metadata. That metadata includes request and call prefixes and extensions,
both descriptor tables and their full strides, shape arrays, and symbol
strings through their NUL terminators. K1 requires the caller's error record
to be disjoint from the call metadata and all tensor buffers. Validation
rejects malformed requests and numeric cast errors before any tensor write.
Dispatch retains no borrowed storage and performs no heap allocation.

The serial x86-64 lane requires round-to-nearest on x87 and SSE, masked
exceptions on both units, and MXCSR with flush-to-zero and denormals-are-zero
disabled. Validation checks these controls and snapshots and restores the
floating-point environment and `errno`. Compatible pre-existing sticky flags
are permitted and preserved. Gradient multiplication also restores the
caller's MXCSR before committing outputs.

## Conversion and overflow

The casts use integer operations on IEEE-754 object representations and
require only the base x86-64 instruction set. Narrowing rounds to nearest,
ties to even, preserves signed zeros, and handles subnormal rounding and
carry into the next exponent. Widening is exact, including signed zeros
and subnormals.

- Binary16 narrowing compares the discarded bits with half a target ulp and
  rounds an exact tie to an even significand. The caller's original f32
  magnitude must be at most 65,504. Casting 65,504 succeeds, and both the next
  f32 above it and 65,520 reject. All f32 infinities and NaNs reject.
- Bfloat16 narrowing adds `0x7fff + ((bits >> 16) & 1)` before retaining
  the high 16 bits. All f32 infinities and NaNs reject, as do finite values
  whose rounded encoding is bfloat16 infinity.
- Binary16 widening rebiases the exponent and normalizes subnormals.
  Bfloat16 widening shifts the storage word left by 16. All narrow
  infinities and NaNs reject during cast validation.

Unscale requires a finite, strictly positive f32 inverse scale. This includes
positive subnormals. An invalid inverse scale is a K1 error that leaves both
outputs untouched. For each finite gradient the provider widens it exactly
and multiplies once in binary32. Ordinary rounded subnormals and signed zeros
are accepted. If any gradient is infinite or NaN, or if any product rounds
to infinity, dispatch succeeds with the overflow byte set to `1` and the
entire f32 master gradient set to positive zero. A clean call writes every
unscaled gradient and sets the byte to `0`.

## Loss-scaling state and optimizer contract

`et_mp4_loss_scaler_init` accepts finite fields satisfying
`0 < min_scale <= scale <= max_scale`, `growth_factor > 1`,
`0 < backoff_factor < 1`, and `growth_interval > 0`. It initializes
`clean_steps` to zero. The state contains five binary32 fields and two
`uint32_t` counters in a 28-byte, pointer-free x86-64 layout. For persistence,
encode the seven named fields in header order as little-endian 32-bit words
in the caller's versioned, checksummed state format. Persist `clean_steps`
with the scale and policy. A restored state must satisfy the init invariants
and `clean_steps < growth_interval`, which update validates before use.

| Update outcome | State transition | Caller action |
| --- | --- | --- |
| `STEP_ACCEPTED` | Increment clean steps. At the interval, multiply by `growth_factor`, clamp to `max_scale`, and reset the counter. | Apply the optimizer step to the f32 master weights. |
| `OVERFLOW_BACKOFF` | Multiply by `backoff_factor` and reset the counter. | Skip the optimizer step and use the backed-off scale for the next attempt. |
| `UNDERFLOW_FAULT` | The exact backoff product is below `min_scale`. Leave state byte-for-byte unchanged. | Skip the optimizer step and surface the fault. |
| `INVALID_ARGUMENT` or `UNSUPPORTED_ENVIRONMENT` | Leave state byte-for-byte unchanged. | Skip the optimizer step and surface the error. |

The scaler forms an exact binary64 product of its binary32 operands, applies
the bound in binary64, then rounds any committed product to binary32 with
ties to even. A backoff exactly equal to the minimum is accepted. The
underflow fault also covers products below the minimum that would round to
the minimum in binary32. Growth clamps safely even if the unclamped value
exceeds finite binary32. An overflow flag outside `0` and `1` is invalid.
The caller is responsible for keeping its inverse scale representable and
for treating an upstream gradient-production error as a skipped step.

## Validation and scope

Ubuntu 22.04.5 x86-64 and Clang 21.1.8 in
`eshkol-r0-debug:20260928` passed
`CC=clang-21 CXX=clang++-21 MP4_ASAN_DETECT_LEAKS=1 bash scripts/test-mp4.sh`.
The run passed the C and C++17 ABI checks, deterministic archive comparison,
symbol audit, ASan, UBSan, LSan, Q0 Python isolation, and CI topology checks.
The C suite exhausts all 65,536 storage patterns in each format, including
finite round trips and rejection of every infinity and NaN during widening.
An independent Python test used exact rational rounding and standard-library
binary16 conversions to cross-check 55,690 casts through native K1 dispatch.

The momentum-training smoke uses narrow storage for the working weights and
scaled gradients, and f32 for the master weights and optimizer state. Each
format accepts 12 updates and skips one injected overflow with master weights
and optimizer state unchanged. Backoff lowers the scale from 64 to 32, and
the next attempt resumes loss reduction. Replayed runs and the control that
omits the failed attempt agree bit-for-bit on weights, momentum, and loss.
The initial loss is 4.6875 and final losses are approximately 0.0000148133
for binary16 and 0.0000161482 for bfloat16.

This foundation covers bounded serial CPU storage conversions, f32 master
gradients, and loss-scaling state. GPU tensor cores, other devices and shapes,
TR3 trainer wiring, and checkpoint integration are outside this ABI.
