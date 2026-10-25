# G3-R C4 sealed native owner topology leaf

**Source candidate only.** `ET_G3R_C4_OWNER_TOPOLOGY_PRIVATE` adds
`int64_t et_g3c4_private_sealed_owner_topology_status_v1(const void *exact_owner)`
to the source-private owner ABI. The feature requires
`ET_G3C4_NATIVE_PINS_PRIVATE`; the symbol is absent when disabled. A zero
return and zero last-error triple mean success. Failure returns the category
and leaves the first typed domain/category/code in the existing C4 last-error
state. The caller must serialize owner operations externally, as for all owner
calls.

The helper admits the exact pointer through the native owner registry before
dereferencing it, requires a sealed, initialized fourteen-parameter owner
with successor state and no active native call, and performs the existing
identity/idle preflight. It then scopes the fixed C4 fourteen-pin
begin/check/end over the owner's own parameters and bound identities. The
pin profile checks the fourteen C4 shapes, including position `[4,4]`, CPU
f32 contiguous row-major value storage, disjoint backing and idle gradients.
The helper returns no parameter, identity, tensor, pin, array or workspace.
Failure cleanup preserves the first native error triple; partial pin
acquisition is drained by the pin provider and the helper ends its pin scope.

The intrusive focused native test uses one genuine sealed owner, rejects
open/prepared/foreign/active/not-ready owners, wrong binding, duplicate
identity, and final position shape, and injects each of fourteen pin cuts. It compares
all 4,768 parameter-value bytes and checks no remaining parameter or tensor
pin/borrow after every call. Local Clang normal and ASan/UBSan passed 25
checks each; feature-off compilation omitted the helper symbol and feature-on
compilation defined it. This is local focused source testing, not a pinned
compiled admission gate. Reproduce with
`scripts/test-g3r-c4-native-owner-topology-local.sh`.

This native status alone does not establish a trainable C4 shell, exact
current P1 handle-shell to native bound-identity association, or future
workspace/generation/trainer lease exclusion. Those are required before any
workspace allocation. The proposed association gate must compare each
current P1 handle to the native binding through a reviewed status-only route,
with swapped-handle and stale-shell negatives.
