# G3-C4 private I2 construction-admission bridge leaf contract

**Accepted bounded leaf.** Its exact
base is the accepted owner-and-pins head
`e11e61f4a3da6682c49c5422040baefdd32d8d4e` (tree
`b044c0cc92f52c3dfa1960abf0028c68313e8a2d`). It adds only the two private I2
C entry points already frozen by the proposal at `0c67676`. It does not add P1
state, an Eshkol construction operation, an owner-active seam, pins, transport,
cache, forward execution, sampler, restore, package inventory, or CI entry.

## Dependency proof and larger-unit blockers

The accepted owner already implements the required target:

```c
int32_t et_g3c4_construction_parameter_preflight_internal(
    const void *exact_owner, void *parameter,
    const void *exact_handle, et_f32_tensor_error *error);
```

That function authenticates the actual process-lifetime C4 owner registry,
requires the exact `staged_c4_owner` in `OPEN` or `PREPARED`, proves unique
membership among its fourteen parameters, compares the exact bound handle (and
accepts null only for a genuinely unbound member), then performs the accepted
whole-owner idle teardown preflight. The existing I2 package bridge already owns
one thread-local `et_i2_last_error`, its clear/getter path, and the analogous
conditional M3T construction preflight. No callback, owner probing, registry or
new error storage is required.

The preferred owner-active/call seam is not implementable at this base. The
proposal requires `owner.active` to equal an authenticated address-stable native
generator context inside a five-way native/Eshkol active tuple. No C4 generator
type, registry or context-admission API exists. A standalone setter accepting an
arbitrary pointer or pin `self` would invent authority.

A native full-prefix forward is also blocked. The only accepted C4 numerical
surface is the carrier-neutral `et_g3c4_kernel_provider_v1`, which supplies
primitive rows but no cache owner, role scratch/context, provider-union routing,
input carrier, frame state or 21-role call. Several T1/T2 roles require accepted
predecessor providers. Implementing a composed forward now would invent those
missing APIs. These conclusions do not inspect or depend on the held sampler
review.

## Exact private surface and compile gates

The exact functions are the proposal's existing declarations:

```c
int64_t et_i2_private_g3c4_construction_available_v1(void);
int64_t et_i2_private_g3c4_construction_parameter_preflight_v1(
    void *exact_owner, void *parameter, void *exact_handle);
```

Changes are conditional in `native/i2_wave2_package_bridge.c` under
`ET_G3C4_I2_CONSTRUCTION_PRIVATE`. Without that macro, the accepted ordinary I2
bridge object, bytes and symbols remain unchanged. This leaf does not update an
installed package manifest or Eshkol extern table.

A surface-only build defines `ET_G3C4_I2_CONSTRUCTION_PRIVATE` without
`ET_G3C4_NATIVE_OWNER_PRIVATE`. It defines both private symbols, availability
returns -1, preflight returns -1 without reading its arguments or clearing the
existing I2 error, and the object has no undefined C4-owner reference.

The genuine C4 source tuple defines both macros. Only in that tuple does the
bridge include `g3c4_model_owner_internal.h`; availability returns 0. Preflight
clears `et_i2_last_error` exactly once and directly calls
`et_g3c4_construction_parameter_preflight_internal(exact_owner, parameter,
exact_handle, &et_i2_last_error)`, returning its exact status. It never calls the
M3T adapter. It creates no registry or callback, performs no allocation, and
does not mutate ownership, parameters, or owner lifecycle state. The bridge does
clear the existing I2 diagnostic before the call, and the accepted target resets
the existing C4 diagnostic as part of its established diagnostic behavior.

This two-level gate preserves the proposal's availability rule while keeping all
ordinary and inherited package artifacts byte-identical until a later accepted
P1/Eshkol/package source tuple explicitly enables the private surface.

## Exact files and evidence

After root acceptance, the only production amendment is:

- `native/i2_wave2_package_bridge.c`

Focused additions are:

- `tests/g3c4/test_i2_construction_bridge.c`
- `tests/g3c4/test_i2_construction_unavailable.c`
- `scripts/test-g3c4-i2-construction-bridge.sh`

The test composes the accepted owner and I2 sources directly. It covers ordinary
absence; surface-only availability/preflight -1 and no owner dependency; genuine
C4 availability 0; unbound-member/null-handle and bound-member/exact-handle
success; foreign, stale and wrong owners; nonmember parameters; null-for-bound,
nonnull-for-unbound and wrong handles; every owner lifecycle state; OPEN and
PREPARED success; SEALED and ABORTED rejection; duplicate membership/identity
corruption; and every parameter/value/gradient borrow and plan-pin class through
the whole-owner idle preflight. Every failure must expose the exact I2
category/code through the existing I2 getters without ownership, parameter, or
lifecycle-state mutation or allocation. Tests must also verify the established
I2-clear and C4-reset diagnostic side effects on genuine calls.

The runner requires:

- accepted ordinary I2 bridge object byte identity when the new macro is absent;
- exactly two added globals in each surface-enabled object;
- no C4-owner undefined symbol in the surface-only object and exactly the
  accepted preflight dependency in the genuine source-tuple delta;
- unchanged existing M3T construction bridge tests;
- unchanged owner and pin tests;
- deterministic normal/repeat output and ASan/UBSan/LSan;
- supported Ubuntu 22.04/Clang 21 evidence with network disabled and read-only
  source/root mounts.

Explicit exclusions are P1 prepare/commit/abort additions, I2/P1 Eshkol
construction ledgers, model/generator registries, owner-active mutation, pins or
call lifecycle, workspace/full-prefix execution, cache, transport, sampler,
restore/matcher, public API, package manifests, CI integration, runtime pin,
push, PR and merge.
