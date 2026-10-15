# PR214 same-registry C2 partial-claim source union

Status: **private source prerequisite; linked owner runtime gate pending**.
This branch starts at accepted original-owner preclaim `6b49e29a`. The C2/O2
partial-claim source and runtime negatives are ported byte-for-byte from the
independently reviewed `b5fbbb0` child of `ec97f61a`. Its two-bit progress
cell marks the O2 claim *attempt* before the fallible call. On a typed or `#f`
exception after mutation, cleanup reconciles the exact registered O2 state
and exact C2 token before rolling P1 back; any uncertain cleanup exits while
the local child shells remain rooted. An equal-looking foreign token has no
authority.

`native/g3r_original_owner_c2_claim_root.esk` loads the accepted preclaim
root, then the exact reviewed post-compose adopter check from `d50d2bd`.
CLI3/TR3 already loads C2 and O2 into this one registry, so the new root does
not load either extension again. The post-compose check requires the cleared
request slots, registered live C2 shell, exact P1 and O2 children, exact O2
native pointer and owner token, and the original held optimizer. It fails
stop on an uncertain observation and marks only the private ledger claimed.
The preclaim root remains usable alone and does not transfer a C2 child.

The source/host gates check load closure, causal post-compose mutations,
existing G3/O2/TR3 hold negatives, and C2 rollback handler order. The C2
runtime fixture includes ordinary compose/release, O2 mutate-then-throw,
O2 mutate-then-raise-`#f`, and equal-looking foreign-token rejection. A
linked Eshkol aggregate must compile and run those cuts with the exact
feature-gated native objects before any transaction can be admitted. No
coordinator drives the full hold/adopt/C2/release sequence here; no C2 SAVE,
LOAD, file publication, restored generator or replay is claimed.
