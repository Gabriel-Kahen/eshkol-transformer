# G3-T source-private manual logits owner leaf

The accepted native `et_g3t_private_logits_reserve_v1(ptr ctx) -> ptr` stem is
conditional on `ET_G3T_MANUAL_LOGITS_PRIVATE`. It authenticates a live active
kind-0/1 manual call with the exact model/pin authority and no existing pending
result. Its generation budget does not govern manual reservation; an exhausted
categorical RNG is also admissible. It allocates a kind-3 header and owned CPU
f32 `[1,256]` I2 tensor before enrolling a single pending result in the same
G3-T registry and attaching it to the call. Header or I2 allocation failure
leaves registry, pins, cache, RNG, and result linkage unchanged. A generate call
cannot reserve kind-3 logits.

The accepted `tensor_release(ptr owner)` gains exact kind-3 dispatch. A pending
result may be released while its call is active and frame-idle, clearing its
link; a dead result releases idempotently. Borrowed, forged, wrong-kind, busy,
and invalid-state owners reject before destruction. `call_abort` first checks
that the pending I2 can be destroyed, then destroys it exactly once and leaves
an inert tombstone before draining pins. No successful manual frame or logits
publication exists in this leaf. The result's live publication transition is
reserved for a later accepted frame prepare/commit leaf.

The private tests use only testing-compiled I2 allocation and borrow hooks.
They cover both manual call kinds with G0 and exhausted G1 configurations,
duplicate/wrong-kind/stale admission, typed release, borrowed release/abort,
and header plus I2 allocation cuts. No Eshkol
shell, numerical schedule, public facade/package, N>1 route, or CLI is added.
