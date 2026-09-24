# G3-C4 Step 4a native context and fixed-cache contract

Status: **root accepted for bounded Step 4a implementation**. Root accepted the
independently reviewed draft at commit
`9d4f97af37c54cc84bf6205c46c3090c32959ba3`, tree
`41bf7218f74b1f8ec071972ea9e2644dfd2dce3c`, with all three decisions in
Section 8. That disposition authorizes only the files and behavior below.

This contract narrows Section 4 of
[`G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md`](G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md)
to the native context registry, one fixed A2 cache, and context close. It adds no
Eshkol generator registry, seed or RNG policy, normalized constructor policy,
fixed-14 pins, `owner.active` mutation, active-call operation, forward schedule,
sampler, result, persistence, public surface, package registration, or G3-S
source.

## 1. Source and build ownership

The context implementation is a conditional block in the existing production
translation unit
`src/eshkol_transformer/g3c4_model_owner.c`, enabled only by
`ET_G3C4_CONTEXT_PRIVATE`. This is the recommended composition because that
translation unit already owns the sole C4 owner type, registry, admission,
lifecycle, canonical parameter identities, and `active` field. It can therefore
authenticate the owner without exposing its layout or adding owner getters and
setters.

The block includes `eshkol_transformer/a2_kv_cache.h` only when enabled. A new
source-private header,
`src/eshkol_transformer/g3c4_context_internal.h`, declares the two context
operations below and contains no owner layout or A2 structure definition.
The conditional include and implementation block are appended after the
existing owner implementation so the disabled preprocessed translation unit
and optimized object can remain byte-identical to the accepted owner source.

Existing owner, model-authority, I2, and pin builds do not define
`ET_G3C4_CONTEXT_PRIVATE`. Their compile and link closures remain unchanged and
must retain exact defined/undefined symbol inventories and existing output.
The context-enabled object is the same single owner implementation with two
additional private entry points; it is never linked beside another
`g3c4_model_owner.o`.

The production context closure adds the already accepted canonical
`a2_kv_cache.o`. It does not copy A2 source into the owner file, add a cache
registry, or link the A2 attention provider. The focused failure-injection test
may compile `native/a2_kv_cache.c` with `ET_A2_KV_CACHE_TESTING`; production
links the canonical object.

## 2. Exact private boundary

```c
void *et_g3c4_private_context_create_v1(void *c4_owner);
int64_t et_g3c4_private_context_close_v1(void *context);
```

Both operations use the existing C4 thread-local error state and its three
existing getters. They require external serialization. No alias using the
proposed `generator_seed` or `generator_rng` names is added in Step 4a.

This context-only constructor is intentionally narrower than the later
generator constructors. Step 4b may call it only after its Eshkol pending entry,
normalized policy, and seed/RNG ownership rules are separately frozen. Step 4b
does not change the context registry or create a second native context.

## 3. Exact context authority and record

One file-local process-lifetime registry is the sole native C4 context
authority. It admits a candidate by exact pointer comparison before any
candidate dereference. Registry nodes are never freed or reused.

The Step 4a record contains only these fields in this logical order:

1. registry link;
2. fixed context magic;
3. fixed kind `G3C4_CONTEXT`;
4. lifecycle `LIVE` or `DEAD`;
5. reserved busy field, exactly zero in every Step 4a operation;
6. exact authenticated sealed C4 owner while live, otherwise `NULL`;
7. exact owned A2 cache while live, otherwise `NULL`.

The record contains no pins, call phase, transaction, view, read-borrow,
scratch, policy, RNG, result, or provider runtime. Later work may append such
fields after a separate contract. It may not move or duplicate this registry,
owner link, or cache handle.

## 4. Construction

`et_g3c4_private_context_create_v1` resets the existing C4 error state, then:

1. authenticates `c4_owner` through the file-local owner registry before
   dereference;
2. requires the exact owner lifecycle `SEALED` and `owner.active == NULL`;
3. allocates one zeroed context record;
4. invokes exactly
   `et_a2_kv_cache_create_v1(1, 1, 2, 4, 2, &cache, &error)`;
5. installs the exact owner and cache in the complete record; and
6. enrolls the context as `LIVE` only after every fallible action succeeds.

The A2 tuple is layers 1, batch 1, KV heads 2, capacity 4, and head dimension
2. The cache remains opaque and never enters an Eshkol shell or public API.

An owner in `OPEN`, `PREPARED`, or retained `ABORTED`, or a sealed owner with
nonnull `active`, rejects as C4 `invalid-state/lifecycle`. A foreign owner
rejects as C4 `invalid-argument/identity`. Context allocation failure is C4
`internal/allocation`. An A2 failure preserves its exact K1 category and code in
the shared C4 error state.

Before registry enrollment, every failure destroys any successfully created
cache, frees the unregistered context, and restores the first error. A2 create
already guarantees an unchanged null output and complete internal cleanup on
failure. No failed constructor leaves a native context tombstone, changes the
owner, or changes the context-registry baseline.

Multiple live idle contexts may reference the same sealed owner. Mutual
exclusion begins only in the later active-call contract through
`owner.active`; Step 4a neither reserves nor mutates that field.

## 5. Close

`et_g3c4_private_context_close_v1` authenticates the exact context through the
context registry before dereference. A foreign pointer rejects as C4
`invalid-argument/identity`. After registry admission it validates the fixed
magic and kind for every lifecycle. An exact valid retained `DEAD` context then
returns success without touching A2 or the owner.

A `LIVE` close requires all of the following before mutation:

- fixed magic and kind;
- reserved busy field zero;
- a nonnull owner link that still authenticates as the exact `SEALED` owner;
- `owner.active == NULL`; and
- a nonnull owned cache.

It calls `et_a2_kv_cache_destroy_v1(&context->cache, &error)`. A2 performs the
authoritative cache-registry check and rejects before mutation when a cache
transaction or read borrow is live. Step 4a exposes no operation that can open
such a lease; the focused intrusive test creates each busy state solely to
prove close rejection for later consumers.

After successful cache destruction, the no-fail tail sets the owner link to
`NULL` and lifecycle to `DEAD`. The context record remains in its registry as a
stable tombstone. Close never mutates or destroys the sealed owner, its fourteen
parameters, their identities, values, gradients, or `active` field.

If A2 destroy rejects, close preserves the exact K1 error and leaves the live
context, owner link, cache handle, and owner byte-for-byte unchanged.

## 6. Concrete implementation and test files

An accepted implementation is limited to:

- `src/eshkol_transformer/g3c4_model_owner.c` — conditional context block;
- `src/eshkol_transformer/g3c4_context_internal.h` — opaque private boundary;
- `tests/g3c4/test_context_cache.c` — focused native test;
- `scripts/test-g3c4-context-cache.sh` — supported native gate;
- `scripts/check-g3c4-context-cache.py` — fast source/closure check;
- `native/g3c4_context_source_closure.txt` — exact native/test closure; and
- a final implementation note under `docs/g3/`.

No Eshkol source, package bridge, installed header, Makefile, CI entry, owner
layout header, pin source, A2 source/header, or predecessor test is modified.
The focused development test may include the owner and A2 implementations in
one intrusive test translation unit to inspect file-local registries and create
busy A2 leases. That inclusion is forbidden in every production object; the
production context closure compiles the owner and canonical A2 sources as
separate objects.

The source closure includes the modified owner source and its private header,
the new context header, the accepted M3T/f32 owner dependencies, canonical A2
cache header/object source provenance, K1 header, N3K initializer provider
dependency, focused test, and gate. It excludes A2 attention, G3-C4 pins,
Eshkol transport, sampler, and G3-S.

## 7. Required focused evidence

The new gate must establish:

- exact symbol delta of only the two production context operations when
  `ET_G3C4_CONTEXT_PRIVATE` is enabled, plus an exact undefined-symbol delta of
  only `et_a2_kv_cache_create_v1` and `et_a2_kv_cache_destroy_v1`;
- unchanged existing owner object/symbol inventory and existing owner test when
  the feature is absent;
- exact cache tuple and distinct cache ownership across repeated contexts;
- foreign, stale, wrong-lifecycle, and owner-busy rejection;
- failure at the context allocation and every A2 cache allocation, with exact
  owner/context/A2 registry baselines and first-error preservation;
- live close, repeated dead close, foreign close, and close rejection for an
  active A2 transaction, transaction view, and read borrow;
- unchanged sealed owner lifecycle, `active`, parameter identities, parameter
  bytes, and gradients across create, failed create, failed close, and close;
- deterministic repeat output; strict C/C++ header checks; and
- normal plus ASan/UBSan/LSan execution under the supported toolchain.

The ordinary C4 owner, fixed-14 pins, A2 cache, Step 3 model-authority static
check, and exact source-closure checks remain affected gates. AOT, Docker,
Eshkol generator construction, and package/CI integration are outside Step 4a.

## 8. Root acceptance decisions

Root acceptance fixed all three decisions below before implementation:

1. authorize the context-only two-function private boundary instead of
   partially implementing or ignoring the later seed/RNG arguments;
2. authorize the conditional implementation inside the existing owner
   translation unit, with no production direct-source include or replacement
   object; and
3. allow multiple idle contexts per sealed owner while retaining every closed
   native context as a process-lifetime dead tombstone.

If any decision changes, this contract must be revised and reviewed again. The
earlier broader Section 4 text alone authorizes no additional implementation.
