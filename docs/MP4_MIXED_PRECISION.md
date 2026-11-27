# MP4 mixed-precision foundation

MP4 1.0 exposes `et_mp4_kernel_provider_v1()` for explicit K1 1.0 resolver
registration and two standalone loss-scaler functions in
`eshkol_transformer/mp4_mixed_precision_abi.h`. Link
`build/mp4/libeshkol_transformer_mp4.a` and K1 with the platform math library.
The archive also exports two buffer record functions through
`eshkol_transformer/mp4_loss_scaler_record.h`. It has no allocator imports.

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
`uint32_t` counters in a 28-byte, pointer-free x86-64 layout. A restored state
must satisfy the init invariants and `clean_steps < growth_interval`, which
update validates before use.

| Update outcome | State transition | Caller action |
| --- | --- | --- |
| `STEP_ACCEPTED` | Increment clean steps. At the interval, multiply by `growth_factor`, clamp to `max_scale`, and reset the counter. | Apply the optimizer step to the f32 master weights. |
| `OVERFLOW_BACKOFF` | Multiply by `backoff_factor` and reset the counter. | Skip the optimizer step and use the backed-off scale for the next attempt. |
| `UNDERFLOW_FAULT` | The exact backoff product is below `min_scale`, or rounds in binary32 to a value not strictly smaller than `scale`. Leave state byte-for-byte unchanged. | Skip the optimizer step and surface the fault. |
| `INVALID_ARGUMENT` or `UNSUPPORTED_ENVIRONMENT` | Leave state byte-for-byte unchanged. | Skip the optimizer step and surface the error. |

The scaler forms an exact binary64 product of its binary32 operands, applies
the bound in binary64, then rounds any committed product to binary32 with
ties to even. A backoff exactly equal to the minimum is accepted provided the
rounded binary32 scale is strictly smaller than the current scale. The
underflow fault also covers products below the minimum that would round to
the minimum in binary32, as well as products whose rounded binary32 scale
equals the current scale (preventing stalled backoff). Growth clamps safely
even if the unclamped value exceeds finite binary32. An overflow flag outside
`0` and `1` is invalid.
The caller is responsible for keeping its inverse scale representable and
for treating an upstream gradient-production error as a skipped step.

## Loss-scaler record v1

`et_mp4_loss_scaler_encode_v1` and `et_mp4_loss_scaler_decode_v1` take an
exactly 44-byte caller-owned buffer disjoint from the scaler. The record may
be unaligned. Callers synchronize access to both spans. Fields occupy
consecutive bytes and all multibyte fields are little-endian.

| Byte offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | Magic `45 54 4d 50 34 4c 53 00` (`ETMP4LS` plus NUL) |
| 8 | 4 | Record version, unsigned value 1 |
| 12 | 4 | `scale` binary32 bits |
| 16 | 4 | `growth_factor` binary32 bits |
| 20 | 4 | `backoff_factor` binary32 bits |
| 24 | 4 | `growth_interval` uint32 |
| 28 | 4 | `clean_steps` uint32 |
| 32 | 4 | `min_scale` binary32 bits |
| 36 | 4 | `max_scale` binary32 bits |
| 40 | 4 | CRC-32/ISO-HDLC over bytes 0 through 39 |

The CRC uses the reflected polynomial `0xedb88320`, initial value
`0xffffffff`, and final XOR `0xffffffff`. It detects corruption. An intact
v1 record remains reusable at any age. The caller controls which snapshot to
restore and how it is stored.

Both functions return `ET_MP4_LOSS_SCALER_RECORD_OK` on success. Invalid spans
or an invalid source scaler return `INVALID_ARGUMENT`. Decode returns
`INVALID_RECORD` for bad magic, a checksum mismatch, or an invalid decoded
scaler, and `UNSUPPORTED_VERSION` for another version with valid magic and
checksum. Each failure leaves the destination bytes intact. Decode replaces
all seven fields on success. The codec uses integer operations and preserves
the caller's floating-point environment and `errno`.

The focused MP4 gate verifies the wire bytes against an independent fixture,
rejects all 352 single-bit mutations and checksummed malformed records, and
compares the full scaler state and outcome at every step across a save and
restore with a pending growth step, subsequent growth and clamping, overflow
backoff, and an underflow fault. This record covers standalone scaler state.
TR3/O2 trainer checkpoint integration remains open.

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

The explicit opt-in Make entry is
`CC=clang-21 CXX=clang++-21 MP4_ASAN_DETECT_LEAKS=1 make test-mp4`;
it runs the same focused gate and is outside the default `make test`,
shared full-coverage matrix, and required CI evidence inventory.
The optional `MOD5/MP4/PA4-Wider provider diagnostic` also runs this target
on the exact head of a `codex/mod5-*` pull request; hosted qualification of
the combined candidate is pending.
An isolated-worktree rerun of that gate on source `4ad7dbbc` passed in the
same image on 2026-11-25, including the 55,690-cast oracle, both exhaustive
storage sweeps, the overflow-recovery smoke, ASan/UBSan/LSan, Q0, and CI
topology. The Make gate took 5.43 seconds with 170,104 KiB peak RSS. Its
sealed log is
`/home/gabe/.codex/evidence/eshkol-transformer/mp4-standalone-4ad7dbbc-20261125/SHA256SUMS`
(SHA-256 `b73e058d79defe2271a4b17f6f5d63cffa6e5ffdd71c6a21bca64b5e539a5739`).

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
and TR3 trainer wiring remain future work.

## MOD5 actual-gradient composition witness

`make test-mp4-mod5` is a separate development-only gate on the accepted
CPU-f32 MOD5 GQA block fixture. It computes real parameter gradients through
MOD5's forward and backward provider calls, scales and stores each eight-value
tile using MP4 f16 or bf16 K1 casts, and unscales through MP4 into f32 before
updating f32 master weights. The four accepted SGD updates are compared with a
direct f32 baseline at every active parameter and gradient. Tolerances are
`atol=2e-5, rtol=0.005` for f16 and `atol=1e-4, rtol=0.02` for bf16.

Between the second and third accepted updates, a deterministic target offset
is calibrated against the real MOD5 gradient maximum. For f16, the scaled
gradient exceeds 65,504; for bf16, it exceeds the finite-to-infinity rounding
boundary `0x1.ffp127f`. The witness separately checks that every active raw gradient
and every scaled f32 value is finite. MP4 narrowing rejects a finite out-of-range
tile in each format. The optimized and sanitized gates both measured peak
scaled values of 70,400 (f16) and `3.3995006e38` (bf16). The scaler backs off
by half, all master parameters remain byte-identical, and the next normal
attempt succeeds.

For each format, an independent trajectory starts with the same fixture and
half the maximum scale. After its first accepted update, `clean_steps` is one
of the two steps required for growth. The witness encodes the scaler into its
44-byte v1 record and decodes it into a fresh scaler alongside a separate
in-memory copy of the f32 master weights. The first resumed update grows the
scale to its maximum. The following rejected cast skips SGD and backs off.
Two more accepted updates recover and grow the scale again. Each attempt
compares the scaled f32, narrow storage, and unscaled f32 bytes produced by
both trajectories, their gradient and scaler outcomes, all master-weight
bytes, the encoded scaler records, and the loss bits. The final loss after
all four accepted updates also matches bit for bit.

The opt-in gate runs optimized and ASan/UBSan/LSan builds with leak detection
enabled. MOD5 forward and backward compute use f32. This continuation covers
the standalone scaler record and the caller-owned master snapshot in the GQA
fixture. Mixed compute, production trainer checkpoint integration, distributed
synchronization, and AOT remain open.


## Live MOD5 gradient storage through AdamW

The opt-in native witness runs the admitted CPU F32 MOD5 grouped-query
fixture with two batches, three tokens, four query heads and two KV heads.
It applies the existing MP4 F16 or BF16 narrowing and unscaling path to
each actual backward gradient. The accepted native F32 gradient plan and
O2 AdamW update consume the unscaled results for all nine parameter tensors.
Each next backward pass uses the updated native weights.

Two clean attempts grow the scale. The next target produces a finite F32
gradient whose scaled value exceeds the narrow format's limit. The MP4
cast rejection backs off the scaler. The test observes zero gradient-plan
or optimizer-step calls during that attempt and checks the live weights,
both moments, optimizer options, schedule factor and completed-update
counter byte for byte. Retrying the same target at the lower scale computes
the same raw gradient and loss, then unscales and updates successfully.
One further clean update grows the scale again.

```bash
bash scripts/test-mp4-mod5-o2.sh
```

Normal, repeat and ASan/UBSan/LSan processes produce identical per-format
traces across four accepted updates and one rejected attempt. The captured
scaled peaks at rejection were 70,400 for F16 and approximately
`3.3995008e38` for BF16. The model computation, master weights and optimizer
state use CPU F32. This witness covers gradient storage and scaler behavior
during an in-process optimizer trajectory.
