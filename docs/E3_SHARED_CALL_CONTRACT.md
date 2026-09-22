# E3-CG: exact shared M3 guard and fixed14 pin contract

**Exact shared design accepted; source implementation under review, runtime acceptance open.** [Issue #101](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/101)
consolidates the shared prerequisite of [E3](E3_EVALUATION_PROPOSAL.md) and
[G3-T](g3/G3_T_PRIVATE_CONTRACT.md). Source base is PR #100 merge
`27c99f7c9f27a227f0e237f1571ac89d5cdfa225`. Binding architecture decisions
[5771430755](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5771430755)
and [5771464954](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5771464954)
authorized the exact-design review. The contract merged in PR #102 as
`2ac0b5bc5a385e79700f46bf899bab841f33d1cb`; subsequent
[dispatch 5771713077](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/101#issuecomment-5771713077)
authorizes the bounded [shared source implementation](M3_SHARED_CALL.md).

E3-CG owns the common definitions below. The original G3-T owner reviews affected
G3 compatibility; E3 owns its frame, P1 modes, D2 cursor, metrics and publication.
[Root decision 5771662657](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/102#issuecomment-5771662657)
accepts the behavioral contract at `5c554421ec37ac3a2cc0e7bc960ad126664fa6f5`,
tree `18f725bc19b8eab15342608de83f5edbb79b659e`, following independent
exact-head review. Contract merge and explicit root dispatch precede implementation;
remaining E3 consumer design is not accepted by this decision. G3-T retains its
hold for both independently approved N/S implementation merges. This proposal
includes no sampler review, runtime capability, public operation or numerical change.

## 1. Exact source ownership

The three canonical source paths implement the accepted design in the current
candidate; sections 8 and its original validation record describe the preceding
documentation-only phase:

| Path | Definitions and inclusion |
|---|---|
| `native/m3_call_adapters.esk` | Exactly 38 `g3t-checked-m3t-*` definitions, loaded once immediately after `native/m3_package_root.esk` and before consumer extensions |
| `src/eshkol_transformer/m3_call_pins.h` | The one type and three C-only declarations in section 3; private, never installed |
| `src/eshkol_transformer/m3_call_f32_integration.c` | Includes `m3t_f32_integration.c` once, then `m3_call_pins.h`; defines the three helpers with access to the existing I2 static registries |

Keep `m3-call-state` and `m3-call` in `native/m3_model_extension.esk` unchanged.
Do not move, duplicate or redefine the original `m3t-public-*` functions. There
is no common owner registry, native active-token function, runtime initializer,
provider, mode policy, restore callback, generic selector or additional Eshkol
binding. Internal C utilities, if needed, are file-local and part of the later
measured local-symbol inventory; they are not additional callable seams.

Both consumer packages select the same f32 replacement **instead of** the
`m3t_f32_integration.c` compile entry, never alongside it or `f32_tensor.c`.
The included predecessor remains a depfile input, not another object. Original
I2/M3T/M3 packages retain their existing recipes and behavior. The common layer
is source, not a separately linked registry-owning archive.

G3's proposed `g3t_f32_integration.c` and `g3t_model_pins.h` are superseded by
these common paths, without compatibility shim files or aliases. Actual accepted
`et_g3t_*` and `g3t-checked-m3t-*` spellings remain; a historical private name
does not import G3 runtime. `g3t_transport.c` still includes `m3_model.c` once.
E3's separately owned `e3_frame.c` likewise includes `m3_model.c` once in its
own artifact; `e3_frame_internal.h` exposes only its reviewed private consumer
contract. The shared f32 TU includes neither consumer nor `m3_model.c`.

## 2. All 46 checked-map targets and exact operation identity

The table is the successor map: the eight M3 rows are unchanged; only the 38
M3T left-hand prefixes change. The C ABI targets and order remain byte-identical
to `native/m3_package_private_renames.txt`. Arity is checked against current
Eshkol definitions. Each new adapter has this literal pattern (zero-argument
rows omit arguments), with its exact public operation symbol from the table:

```scheme
(define (g3t-checked-m3t-input-copy! input values)
  (m3-call 'diagnostic-input-copy!
    (m3t-public-input-copy! input values)))
```

No callback, `apply`, variadic adapter, new boundary macro or runtime dispatch is
needed. The adapter enters the existing guard once before calling its trusted
original. A nested public entry rejects before argument validation or mutation
as E1 `invalid-state` for that operation, with the existing fixed message
`model operation is already active`. Other failures retain M3T's bounded mapping
and same-operation rethrow behavior. The eight M3 functions already own their
outer guard. Their schedules continue directly into the original M3T functions;
calling the checked adapters from those schedules would be erroneous reentry.

| Successor Eshkol map name | Arity | Operation symbol | Unchanged C ABI target |
|---|---:|---|---|
| `m3-public-diagnostic-model-logits-bits` | 1 | `diagnostic-model-logits-bits` | `et_e1b_private_m3_diagnostic_model_logits_bits_cabi_v1` |
| `m3-public-diagnostic-model-logits-release!` | 1 | `diagnostic-model-logits-release!` | `et_e1b_private_m3_diagnostic_model_logits_release_cabi_v1` |
| `m3-public-diagnostic-output-vjp!` | 4 | `diagnostic-output-vjp!` | `et_e1b_private_m3_diagnostic_output_vjp_cabi_v1` |
| `m3-public-model-forward` | 3 | `model-forward` | `et_e1b_private_m3_model_forward_cabi_v1` |
| `m3-public-model-output-logits` | 1 | `model-output-logits` | `et_e1b_private_m3_model_output_logits_cabi_v1` |
| `m3-public-model-output-loss` | 1 | `model-output-loss` | `et_e1b_private_m3_model_output_loss_cabi_v1` |
| `m3-public-model-output-release!` | 1 | `model-output-release!` | `et_e1b_private_m3_model_output_release_cabi_v1` |
| `m3-public-model-output-rng` | 1 | `model-output-rng` | `et_e1b_private_m3_model_output_rng_cabi_v1` |
| `g3t-checked-m3t-attention!` | 1 | `diagnostic-attention!` | `et_e1b_private_m3t_attention_cabi_v1` |
| `g3t-checked-m3t-attention-vjp!` | 1 | `diagnostic-attention-vjp!` | `et_e1b_private_m3t_attention_vjp_cabi_v1` |
| `g3t-checked-m3t-embedding!` | 2 | `diagnostic-embedding!` | `et_e1b_private_m3t_embedding_cabi_v1` |
| `g3t-checked-m3t-embedding-vjp!` | 2 | `diagnostic-embedding-vjp!` | `et_e1b_private_m3t_embedding_vjp_cabi_v1` |
| `g3t-checked-m3t-gelu!` | 1 | `diagnostic-gelu!` | `et_e1b_private_m3t_gelu_cabi_v1` |
| `g3t-checked-m3t-gelu-vjp!` | 1 | `diagnostic-gelu-vjp!` | `et_e1b_private_m3t_gelu_vjp_cabi_v1` |
| `g3t-checked-m3t-heads-merge!` | 1 | `diagnostic-heads-merge!` | `et_e1b_private_m3t_heads_merge_cabi_v1` |
| `g3t-checked-m3t-heads-merge-vjp!` | 1 | `diagnostic-heads-merge-vjp!` | `et_e1b_private_m3t_heads_merge_vjp_cabi_v1` |
| `g3t-checked-m3t-heads-split!` | 2 | `diagnostic-heads-split!` | `et_e1b_private_m3t_heads_split_cabi_v1` |
| `g3t-checked-m3t-heads-split-vjp!` | 2 | `diagnostic-heads-split-vjp!` | `et_e1b_private_m3t_heads_split_vjp_cabi_v1` |
| `g3t-checked-m3t-initializer-algorithm` | 1 | `diagnostic-initializer-algorithm` | `et_e1b_private_m3t_initializer_algorithm_cabi_v1` |
| `g3t-checked-m3t-initializer-state` | 1 | `diagnostic-initializer-state` | `et_e1b_private_m3t_initializer_state_cabi_v1` |
| `g3t-checked-m3t-initializer-words` | 1 | `diagnostic-initializer-words` | `et_e1b_private_m3t_initializer_words_cabi_v1` |
| `g3t-checked-m3t-input-copy!` | 2 | `diagnostic-input-copy!` | `et_e1b_private_m3t_input_copy_cabi_v1` |
| `g3t-checked-m3t-input-copy-tokenizer!` | 2 | `diagnostic-input-copy-tokenizer!` | `et_e1b_private_m3t_input_copy_tokenizer_cabi_v1` |
| `g3t-checked-m3t-input-create` | 0 | `diagnostic-input-create` | `et_e1b_private_m3t_input_create_cabi_v1` |
| `g3t-checked-m3t-input-release!` | 1 | `diagnostic-input-release!` | `et_e1b_private_m3t_input_release_cabi_v1` |
| `g3t-checked-m3t-layer-norm!` | 2 | `diagnostic-layer-norm!` | `et_e1b_private_m3t_layer_norm_cabi_v1` |
| `g3t-checked-m3t-layer-norm-vjp!` | 2 | `diagnostic-layer-norm-vjp!` | `et_e1b_private_m3t_layer_norm_vjp_cabi_v1` |
| `g3t-checked-m3t-linear!` | 2 | `diagnostic-linear!` | `et_e1b_private_m3t_linear_cabi_v1` |
| `g3t-checked-m3t-linear-vjp!` | 2 | `diagnostic-linear-vjp!` | `et_e1b_private_m3t_linear_vjp_cabi_v1` |
| `g3t-checked-m3t-logits-bits` | 1 | `diagnostic-logits-bits` | `et_e1b_private_m3t_logits_bits_cabi_v1` |
| `g3t-checked-m3t-logits-copy-bits!` | 2 | `diagnostic-logits-copy-bits!` | `et_e1b_private_m3t_logits_copy_bits_cabi_v1` |
| `g3t-checked-m3t-logits-create` | 0 | `diagnostic-logits-create` | `et_e1b_private_m3t_logits_create_cabi_v1` |
| `g3t-checked-m3t-logits-release!` | 1 | `diagnostic-logits-release!` | `et_e1b_private_m3t_logits_release_cabi_v1` |
| `g3t-checked-m3t-model-create` | 2 | `diagnostic-model-create` | `et_e1b_private_m3t_model_create_cabi_v1` |
| `g3t-checked-m3t-model-initializer` | 1 | `diagnostic-model-initializer` | `et_e1b_private_m3t_model_initializer_cabi_v1` |
| `g3t-checked-m3t-model-profile` | 1 | `diagnostic-model-profile` | `et_e1b_private_m3t_model_profile_cabi_v1` |
| `g3t-checked-m3t-residual!` | 2 | `diagnostic-residual!` | `et_e1b_private_m3t_residual_cabi_v1` |
| `g3t-checked-m3t-residual-vjp!` | 2 | `diagnostic-residual-vjp!` | `et_e1b_private_m3t_residual_vjp_cabi_v1` |
| `g3t-checked-m3t-sum!` | 2 | `diagnostic-sum!` | `et_e1b_private_m3t_sum_cabi_v1` |
| `g3t-checked-m3t-workspace-begin!` | 2 | `diagnostic-workspace-begin!` | `et_e1b_private_m3t_workspace_begin_cabi_v1` |
| `g3t-checked-m3t-workspace-check-primals` | 1 | `diagnostic-workspace-check-primals` | `et_e1b_private_m3t_workspace_check_primals_cabi_v1` |
| `g3t-checked-m3t-workspace-copy-logits!` | 2 | `diagnostic-workspace-copy-logits!` | `et_e1b_private_m3t_workspace_copy_logits_cabi_v1` |
| `g3t-checked-m3t-workspace-create` | 1 | `diagnostic-workspace-create` | `et_e1b_private_m3t_workspace_create_cabi_v1` |
| `g3t-checked-m3t-workspace-release!` | 1 | `diagnostic-workspace-release!` | `et_e1b_private_m3t_workspace_release_cabi_v1` |
| `g3t-checked-m3t-workspace-reset!` | 1 | `diagnostic-workspace-reset!` | `et_e1b_private_m3t_workspace_reset_cabi_v1` |
| `g3t-checked-m3t-workspace-vjp-begin!` | 2 | `diagnostic-workspace-vjp-begin!` | `et_e1b_private_m3t_workspace_vjp_begin_cabi_v1` |

## 3. Exact private pin declaration and geometry

The accepted private header declaration for `m3_call_pins.h` is:

```c
#ifndef ET_M3_CALL_PINS_H
#define ET_M3_CALL_PINS_H
#include "../../native/f32_parameter_internal.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct et_g3t_model_pins_internal {
  struct et_g3t_model_pins_internal *self;
  et_f32_parameter *parameters[14];
  const void *identities[14];
  et_f32_tensor *values[14];
  et_kernel_tensor_view_v1 views[14];
  uint16_t held_mask;
} et_g3t_model_pins_internal;
int32_t et_g3t_model_pins_begin_internal(
    et_f32_parameter *const parameters[14], const void *const identities[14],
    et_g3t_model_pins_internal *pins, et_f32_tensor_error *error);
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *error);
void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins);
#ifdef __cplusplus
}
#endif
#endif
```

The fixed fields, order and signatures preserve G3's accepted three-helper
contract. This is a source-private C/C++ declaration, not a serialized layout,
installed ABI, Eshkol extern or authorization to accept arbitrary pointers.
No size/offset ABI is promised. `views` are read-only by closed-role discipline:
K1's `data` is `void *`, so the C type does not enforce immutability. Only fixed
forward input positions may consume them; no view or data pointer escapes.

| Index | Canonical parameter | Shape | Bytes |
|---:|---|---|---:|
| 0 | attention key | [4,4] | 64 |
| 1 | attention output | [4,4] | 64 |
| 2 | attention query | [4,4] | 64 |
| 3 | attention value | [4,4] | 64 |
| 4 | FFN down | [4,8] | 128 |
| 5 | FFN up | [8,4] | 128 |
| 6 | norm1 beta | [4] | 16 |
| 7 | norm1 gamma | [4] | 16 |
| 8 | norm2 beta | [4] | 16 |
| 9 | norm2 gamma | [4] | 16 |
| 10 | tied token/head | [256,4] | 4096 |
| 11 | final beta | [4] | 16 |
| 12 | final gamma | [4] | 16 |
| 13 | position | [2,4] | 32 |

Total: 14 unique canonical identities, 1,184 f32 values, 4,736 bytes. Native
`owner.p[14]`/`owner.handles[14]` and Eshkol model slot8 supply this exact order;
no caller parameter-index or role-selected owner input is introduced. The tied
head uses index10 without acquiring a fifteenth pin. Shapes, rank, exact count,
byte span, CPU/f32/dense/zero-offset and dense strides must match this table.

## 4. Authentication and token boundary

The mechanics do **not** authenticate a G3 generator or E3 frame. `self` only
checks address stability/copying. The closed consumer must first authenticate
its Eshkol entry and native context by exact registry lookup, before reading any
payload or computing an embedded address. An arbitrary mapped pointer, one-word
tag, copied struct or self-consistent `self` is not admission. The signatures
are not reachable through externs, boxed wrappers, installed headers or observers.
Trusted actual arrays, pin/error storage and enclosing context lifetime are
preconditions; span arithmetic cannot prove arbitrary C memory is accessible.

| Checkpoint | G3-T exact authority | E3 exact consumption requirement |
|---|---|---|
| Eshkol admission | Existing generator/call entries and kinds; generator slot9/model slot10 idle | E3's separately authenticated reusable frame entry; model slot10 idle; no G3 registry access |
| Native admission | Existing generator registry, live stable context and independently admitted M3T OWNER | E3 frame registry before dereference, then independently admitted M3T OWNER in the included M3 lineage |
| Pin source | Authenticated owner's `p`/`handles`, exact embedded `ctx->pins` | Authenticated owner's `p`/`handles`, exact embedded `frame->pins` |
| Token publication after successful begin | Existing native `owner.active=ctx`; generator9/model10 reference the exact rooted call entry | Native `owner.active=frame`; model10 references that exact authenticated E3 frame entry |
| Every later step | Reauthenticate context/model/call, exact generator9/model10/native match | Reauthenticate frame/model, exact model10/native match and closed E3 phase |
| End | Shared helper drains pins only | Same helper drains pins only; model/outer exclusion survives restoration and publication |

The consumer native boundaries derive all three begin operands internally after
admission. They accept no separate pin pointer, replacement model owner, raw token,
owner-kind flag, parameter index or callback. Native shared helper registration
is unnecessary: each consumer's existing transport TU can use the included M3T
`admit(..., OWNER, ...)`, `owner` definition and canonical arrays directly. No
G3/E3 union, common context registry or dependency on an unmerged G3 runtime is
needed. Exact E3 function names and frame phase encoding remain E3-owned.

After the one outer `m3-call` entry, reject occupied consumer/model tokens and
independently check native owner idle. Authenticate/canonicalize all rooted
entries before native acquisition. Begin returns either no pins or all14 pins.
Publish exact tokens only after success through preallocated nonraising stores;
record each acquired flag so interruption cannot clear another call's token.
No arbitrary code/callback/reentry occurs between these steps. Each consumer
checks its own native active field and model identity on every boundary.

Between invocations an existing manual M3T frame blocks only its model; the outer
guard is free for another idle model. During an invocation all eight M3 and 38
checked M3T outer entries, and both consumer outer entries, share aggregate-wide
exclusion. This is serialized execution, not a thread-safe lock or signal/fork API.

## 5. Begin, check and end

Canonical IDLE has every named field zero/null (not a comparison of C padding).
FULL has `self==pins`, `held_mask==0x3fff`, all14 exact parameter/identity/value
snapshots and descriptors. A pin object remains embedded at its admitted address
until released; no copying, moving, reallocating, region exit or parent teardown
while held. Existing `et_f32_tensor_scoped_*_internal` guards remain synchronous
and stack-only; no ordinary I2 borrow shell is allocated for these pins.

**Begin (arity4):** validate trusted control/error/array spans and canonical IDLE,
then preflight all14 in increasing order before changing any pin/control. Admit
each parameter through the existing I2 live registry before dereference; admit
its value **and gradient** independently before reading their fields. Require
exact non-null identities, distinct parameters/identities/value owners, canonical
value/gradient shapes and distinct storage; no value/gradient/control span aliases
any other live operand or pin/error/array storage. Check parameter `plan_pins==0`,
value and gradient `active_borrow==NULL`, and both tensor `plan_pins==0`.
Do not inspect/rewrite gradient contents or change gradient presence/count/weight.
Mode/profile/topology/tokenizer authentication belongs to the consumer, not I2.

After complete preflight, populate `self`, snapshots and each K1 descriptor from
its authentic value owner. In increasing index order set parameter `plan_pins`
0→1, value `active_borrow` to
`(et_f32_tensor_borrow *)(void *)&pins->views[i]`, then the corresponding held bit.
Each index is an indivisible protocol step: no fallible call, injection point or
callback between those three stores. No value/gradient tensor pin count changes.
The success boundary exposes FULL only. Any injected acquisition failure between
indices retains the first error, validates the acquired prefix, reverses just its
held bits without allocation and clears the pin fields back to canonical IDLE.
Normal all14 preflight failures leave pins and controls unchanged. Partial masks
are internal rollback states and never cross the native consumer boundary.

**Check (arity2):** read-only full-frame preflight. Require FULL and re-admit every
parameter/value/gradient before dereference, exact identities and canonical value
relationship, shape/count/span/stride, and all descriptor fields (`struct_size`,
data, byte_length, layout, offset, rank and exact shape pointer; dtype/device
pointers equal the fixed TU-owned `f32`/`cpu` string storage set by begin before
any string read). Require each parameter count exactly1 and each value sentinel exactly
its own `&views[i]`; require both tensor counts0 and no gradient active borrow.
Consumer separately rechecks its canonical arrays/model/token/phase. This may
return a recoverable precommit error without mutation; it cannot repair state.
Detecting an illicit gradient borrow does not make parameter pins a gradient lock.

**End (arity1):** canonical IDLE is an idempotent no-op. Otherwise require FULL,
preflight the entire frame as above before the first release write, and fail-stop
on invalid/copy/partial/null/invariant-defective storage; there is no error output
or recoverable result. Under the closed serialized protocol, drain indices13→0:
clear the exact value sentinel, decrement the exact parameter count1→0, clear
that held bit. Clear all views/parameters/identities/values/self only after every
held control is released. No allocator, provider, callback, destructor, error
mapper or fallible branch occurs inside the drain. Begin's private partial drain
uses the same mechanics after validating its acquired prefix; public end does not
accept caller-forged masks. Never route these sentinels through ordinary I2 borrow
end, whose registry/type is different.

Check is the consumer's explicit prepare-release operation. It is not a reusable
authorization token: end rechecks at the actual drain, and any intervening provider
activity must have ended its views. An earlier prepare cannot license mutation of
pins. No extra prepare flag, callback or fourth pin helper is proposed.

`plan_pins` excludes parameter destruction and gradient reset/contribution-plan
preparation. Existing privileged `et_f32_parameter_gradient_borrow_begin_v1`
does not consult that count. Preservation therefore also requires the closed
no-gradient transcript, serialization and no caller callbacks/reentrancy; this
proposal neither strengthens I2 nor claims universal gradient-read exclusion.

## 6. Errors and cleanup ownership

Pin helpers use existing I2 category/code enums only; status equals category,
zero means success. A valid error record is cleared on success. A fixed trusted
consumer owns the actual error object; it is disjoint from its enclosing context,
arrays, pin storage and all I2 storage/control spans. The core checks the spans
available in its signature; whole-context disjointness remains a fixed-adapter
precondition. Unsafe error storage receives no writes and returns invalid-argument.
The core performs no numeric work or floating-environment admission.

| Failure in begin/check | I2 category/code | Numeric pair |
|---|---|---|
| Null required operand/identity | INVALID_ARGUMENT / NULL_ARGUMENT | 1/1 |
| Unsafe/overlapping control or buffer span | INVALID_ARGUMENT / INVALID_BUFFER | 1/4 |
| Foreign/stale parameter or tensor | INVALID_STATE / INVALID_HANDLE | 4/7 |
| Begin non-null identity mismatch or duplicate parameter/identity/value | INVALID_ARGUMENT / INVALID_HANDLE | 1/7 |
| Begin wrong canonical shape/count/span/stride | SHAPE_MISMATCH / INVALID_SHAPE | 2/3 |
| Begin already active/nonempty pin object or borrowed/pinned owner controls | INVALID_STATE / ACTIVE_BORROW | 4/8 |
| Copied pin object; check wrong self/mask/identity/sentinel/count/descriptor | INVALID_STATE / INVALID_HANDLE | 4/7 |

Precedence: safe error/control spans, required storage, pin lifecycle, then each
canonical index's live-owner admission, identity/uniqueness, geometry and idle
controls. Check's FULL/self gate precedes per-index checks; changed geometry or
identity in a previously FULL frame is invariant invalid-state, not a fresh shape
admission. Fixed operation strings are `m3-call-pins-begin` and
`m3-call-pins-check`; messages are bounded constants selected by these cases.
No raw message pointer escapes. Null end or an impossible tail defect terminates,
never reports a recoverable partial release. Native first-error triples are saved
before cleanup calls can overwrite them.

G3's unchanged mapper transports these failures as domain3 plus the I2 pair;
invalid triples map to its internal/invariant error. No new G3 code is fed through
M3T's domain0 mapper. E3 must consume the same I2 domain/category/code meaning
through its separately reviewed bounded mapper; it adds no common error domain
or shared extern accessor. Consumer identity/phase errors retain consumer codes. Before rethrow, the inner
consumer boundary freezes/normalizes the first structured error to exactly the
outer `m3-call` operation. Existing `m3t-rethrow-raw` otherwise replaces details
when operation identities differ; no common guard change is authorized.

G3 remains eval-only, with unchanged 29 native calls, exactly-once prepared
finish/abort and cache/RNG/result publication. Its cleanup checks pins before
commit and ends them in its admitted tail. E3 alone owns its authenticated17-mode
and cursor restore tokens, forward storage and staged results. It ends views and
owned batch, validates/drains pins, restores cursor/all modes, then publishes all
four existing I2 scalars and two i64 counters atomically while BOTH model and
outer exclusion remain held. Pins do not block P1 mode writes. Finalized staging
survives until E3 success commit; failure preserves the original error and every
caller output. No shared helper restores, publishes or clears any active token.

Consumer cleanup clears only its matching native/model/consumer tokens, after
restoration/publication or admitted G3 finish. The enclosing `m3-call` then owns
clearing `m3-call-state`. Its existing catch must run **after** the consumer's
inner cleanup handler. Shared pin-end never clears the outer guard, including on
failure. No allocating public mode setter, seek, reset or ordinary borrow release
is asserted to be an infallible rollback primitive.

## 7. Package deltas and implementation evidence boundary

Shared extraction adds exactly38 named Eshkol bindings already counted by G3's
95, plus its already counted three C-only pin functions; it adds zero externs,
boxed/public wrappers, facade files or registry definitions. G3 retains29 native
calls,95 named Eshkol bindings,93 globals,87 boxed exports and seven facades.
E3 counts its38 shared bindings once within its own private tuple; it cannot
borrow G3's29/95 or infer a public evaluator count. Actual compiler-local and
undefined symbols must be measured later, not guessed from source counts.

The G3 successor tuple remains
`native/g3t_package_{root.esk,bridge.c,private_renames.txt,public_exports.txt}`,
`scripts/{build-g3t,g3t-package-policy,g3t-native-inputs}.sh` and one final
`g3t_package.o`. Its root now loads M3 root, common adapters, then G3 extension;
its bridge includes the unchanged M3 bridge once. Its map is exactly section2.
G3 source closure gains common adapters; native closure gains common pin header/TU
instead of the two superseded proposed G3 pin paths. Included
`m3t_f32_integration.c`, `f32_tensor.c`, `m3t_f32_scoped.h`, `m3_model.c` and
`m3t_transport.c` stay in the raw transitive closure. No extra provider object is
introduced by the common layer; G3 retains the six objects in its accepted contract.

E3's private tuple likewise loads common adapters after the inherited18-file M3
Eshkol closure, before its extensions; replaces the f32 compile slot with common
TU and model slot with `e3_frame.c`; retains both predecessor include chains.
E3 owns its remaining exact tuple/manifests in its consolidated design. It imports
no G3 extension, decoder, G3-N/S object or A2-cache. Each artifact has one I2 registry,
one M3T registry and one inherited E1/P1/T1 bridge lineage. No linking of completed
M3/D2/G3 aggregates, no whole-archive admission and no duplicate registry owner.

Later `scripts/build-e1b-consumer.sh` must admit exact successor tuples before
predecessor M3/M3T detection and select these substitutions only after admission.
Consumer-owned policy scripts enumerate the complete lexical source tuple, native
compile/object/depfile closure, manifest paths and ordered trusted include roots;
there is no generic common-layer builder switch. Preserve component symlink checks,
C `../` include-spelling checks, unfiltered source/native depfiles, scrubbed compiler
environment, exact public strings and localization. G3's include roots remain
`internal/p1/lib`, `internal/c1/lib`, `internal/t1/lib`, `src` in order. E3's
additional D2/P1 integration remains its own acceptance item. Predecessor archive
and public boundaries do not change merely because a successor exists.

Required implementation proof: all38 checked entries plus8 M3 entries reject reentry,
trusted M3 schedules still run; idle different-model/manual-frame cases; forged,
copied, stale, wrong-kind and cross-owner contexts reject before payload access;
all14 acquisition failure positions, exact partial rollback and reverse drain;
corrupt descriptor/sentinel/count/self/mask fail before writes; error-span alias
negatives and original-error preservation; actual parameter/gradient bytes and
counts unchanged; no gradients/graphs in E3; no allocating or callback cleanup;
consumer restoration/publication order; exact source/ELF/archive isolation and
normal-versus-instrumented separation. The [shared implementation record](M3_SHARED_CALL.md)
tracks common-layer runtime/AOT/sanitizer and live plus cumulative retention
evidence. Consumer authentication, restoration and publication remain separately
owned downstream proof obligations.

## 8. Bounded design evidence and dispositions

This phase reads merged source and compiles only a standalone declaration probe;
it runs no production build, numerical tests, toolchain build or full CI. Host
session metadata verified task `01a0c789-bd3f-7b21-b865-8b9365c7a282`,
`gpt-6-astra` with high reasoning, `danger-full-access`, approval policy `never`;
commands use `/usr/bin/bash`, `login=false`.

Independent source/packaging and lifetime/adversarial Astra/high subagents both
approve documentation submission, with no blocking finding, at full-draft SHA-256
`f10190df0dc8daf70d8b796cc3a7ed0ca4f69c45df1346063f06fdb439725aaa`
(before this evidence-only record). Both read the complete contract and all three
companion-document diffs. The source review independently checked bridge arities,
operation identity and exact targets; lifetime review checked full/prefix release,
span/authentication boundaries and consumer token/error preservation.

Commands executed in local ignored `.tmp/e3-cg/`, with Clang 22.1.6 (host
compatibility declaration evidence only, not supported runtime evidence):

```text
clang -std=c11 -Wall -Wextra -Werror -I . -I include -c .tmp/e3-cg/declaration.c -o .tmp/e3-cg/declaration-c.o
clang++ -std=c++17 -Wall -Wextra -Werror -I . -I include -x c++ -c .tmp/e3-cg/declaration.c -o .tmp/e3-cg/declaration-cxx.o
nm -u .tmp/e3-cg/declaration-c.o .tmp/e3-cg/declaration-cxx.o
python3 .tmp/e3-cg/check_design.py
/usr/bin/bash scripts/check-ci-topology.sh
python3 -m unittest discover -q -s tests/ci -p 'test_*.py'
python3 -m unittest -q tests.q0.test_python_isolation
git diff --check
```

Both compilation commands exit0, warning-clean; function-pointer assignments
check all three signatures, static assertions check14 parameter/view slots, and
both objects have exactly the three unmangled unresolved pin helper names.
The header is extracted from section3 with only its include path relocated for
scratch compilation. It supplies no helper definitions, link or runtime evidence.
Static checks pass:46 ordered rows (8+38), names/arities/operations/C targets,
4,736 bytes, exact declaration equality, inherited87/93/7 manifests,43 local
links, exactly four Markdown changes and byte-preserved G3/N/S/L3S contracts
outside the explicit G3-T change. Topology check passes18 suites/26 commands;
98 CI tests and2 Python-isolation tests pass. Their synthetic evidence output is
a test fixture, not a new full-coverage run. Whitespace passes.

The original G3-T owner approves affected-contract compatibility for root review
and a docs-only PR, with no correction, at the same shared-draft hash and G3-T
file hash `b8797d91ba5bc8d42215af50c31f5a5b44d5bf0b6f59ecb43ccb55eeb333a910`.
That owner independently checked all46 rows and three helper declarations and
confirms29/95/93/87/7 and the N/S hold are preserved.

E3 also approves the complete shared contract for design submission at behavioral
hash `f10190df0dc8daf70d8b796cc3a7ed0ca4f69c45df1346063f06fdb439725aaa`;
its independent packaging reviewer found no shared-map/closure blocker. E3 confirms
exact pins/source/authentication/domain3 and retained exclusion are compatible.
Its own error contract may return a fixed failure sentinel normally through
`m3-call` after cleanup and raise a prepared final-operation error outside the
guard, because pinned Eshkol raise allocates. This requires no common change;
normalization applies before any rethrow through the canonical handler. Exact
E3 frame/restore/error design remains separately unaccepted. Root acceptance
covers only this shared design; runtime implementation and acceptance remain open. Compilation cannot prove acquisition, release, authentication,
source-composed execution, gradients, determinism, memory bounds or rollback.
No production implementation, public API, installed artifact, full numerical CI,
toolchain build, merge or runtime capability is delivered by this proposal.
