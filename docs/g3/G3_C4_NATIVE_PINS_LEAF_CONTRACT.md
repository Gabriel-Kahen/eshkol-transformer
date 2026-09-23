# G3-C4 native fixed-14 pin leaf contract

**Accepted bounded implementation leaf.** Root accepted and dispatched this
leaf after reviewing the exact pin section of the proposal at `0c67676`. Its
base is the accepted seeded-owner commit
`bd977aa9f968500a1cd0a8729e01a72286e50a95`. This leaf adds only the private C4
fixed-14 pin sibling in the existing shared I2 registry/implementation lineage.
It adds no active-owner seam, P1/Eshkol construction, transport, cache, sampler,
restore, shared restore matcher, runtime pin, package manifest, CI entry, push,
PR, or merge.

The only production amendments are conditional declarations in
`src/eshkol_transformer/m3_call_pins.h` and conditional implementation in
`src/eshkol_transformer/m3_call_f32_integration.c`, both under
`ET_G3C4_NATIVE_PINS_PRIVATE`. With that conditional absent, ordinary objects,
symbols, bytes, diagnostics, and the accepted C2 `et_g3t_model_pins_*` behavior
remain unchanged. There is no second f32 registry or copied independent pin
implementation.

The source-private C4 surface is:

```c
typedef struct et_g3c4_model_pins_internal {
  struct et_g3c4_model_pins_internal *self;
  et_f32_parameter *parameters[14];
  const void *identities[14];
  et_f32_tensor *values[14];
  et_kernel_tensor_view_v1 views[14];
  uint16_t held_mask;
} et_g3c4_model_pins_internal;

int32_t et_g3c4_model_pins_begin_internal(
    et_f32_parameter *const parameters[14],
    const void *const identities[14],
    et_g3c4_model_pins_internal *pins,
    et_f32_tensor_error *error);
int32_t et_g3c4_model_pins_check_internal(
    const et_g3c4_model_pins_internal *pins,
    et_f32_tensor_error *error);
void et_g3c4_model_pins_end_internal(
    et_g3c4_model_pins_internal *pins);
```

The exact owner-order geometry is `[4,4]` four times, `[4,8]`, `[8,4]`, four
`[4]` vectors, `[256,4]`, two `[4]` vectors, and final `[4,4]`: 1,192 f32
values and 4,768 bytes. It differs from the accepted C2 table only at index 13,
which remains `[2,4]` for `et_g3t_model_pins_*`. `FULL` has `self == pins`,
`held_mask == 0x3fff`, exact parameter/identity/value snapshots, and exact
read-only dense f32 views. Canonical `IDLE` has every named field zero.

Begin uses operation `g3c4-model-pins-begin`. It authenticates all fourteen
parameters, values, gradients and exact identities through the one actual I2
registry before acquisition. It validates exact shapes, strides, byte lengths,
pairwise-distinct parameters/identities/value storage, disjoint caller and live
I2 storage, canonical IDLE output, and idle parameter/value/gradient borrow and
plan controls. Only after complete preflight does it acquire indices 0 through
13 by setting each parameter control pin, exact value-borrow sentinel and mask
bit. An injected prefix failure preserves its first diagnostic, drains only the
acquired prefix in reverse, and returns canonical IDLE.

Check uses operation `g3c4-model-pins-check`, is read-only, and accepts only
exact `FULL`. End uses internal preflight operation
`g3c4-model-pins-end`; canonical `IDLE` is idempotent. Exact `FULL` drains indices
13 through 0 and then clears the record. Copied, forged, aliased, stale,
partially held, corrupted or otherwise non-IDLE/non-FULL records fail-stop
before releasing any control. No allocator, mapper, callback or recoverable
branch occurs after the first release write.

Focused tests live in `tests/g3c4/test_pins.c`, with the standalone supported
runner `scripts/test-g3c4-pins.sh`. They use a genuine accepted seeded C4 owner
and cover the complete existing M3CG adversarial classes: malformed and
overlapping spans, foreign/stale controls, duplicate parameter/identity/value,
every geometry/stride/byte defect, error-buffer aliases, parameter/value/
gradient borrows and every plan-pin class, all fourteen prefix cuts and reverse
rollback, copied/corrupt/partial pin objects, IDLE idempotence, exact FULL
checking, fail-stop end subprocesses, and explicit C2-final-row versus
C4-final-row rejection. Ordinary object/symbol/byte preservation, unchanged C2
regression, deterministic repeat output, normal execution and ASan/UBSan/LSan
are required. Supported evidence is reported separately and makes no transport,
cache, sampling, persistence, package or full-G3 claim.
