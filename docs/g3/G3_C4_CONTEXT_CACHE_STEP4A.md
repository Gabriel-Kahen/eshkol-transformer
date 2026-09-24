# G3-C4 Step 4a native context and fixed cache

This bounded native leaf implements the root-accepted
[`G3_C4_CONTEXT_CACHE_STEP4A_CONTRACT.md`](G3_C4_CONTEXT_CACHE_STEP4A_CONTRACT.md).
It adds a process-lifetime native context registry and the two private operations
`et_g3c4_private_context_create_v1` and
`et_g3c4_private_context_close_v1`.

The implementation is conditional inside the sole C4 owner translation unit.
It directly authenticates the file-local owner registry, requires a sealed idle
owner, allocates a context, creates exactly one A2 cache with tuple
`(1,1,2,4,2)`, and enrolls the context only after complete success. Close uses
A2's authoritative lease-aware destroy and retains a dead context tombstone.
It never changes the sealed owner or its parameter storage.

Seed/RNG policy, Eshkol generator authority, fixed-14 pins, `owner.active`
mutation, active calls, forward execution, sampling, persistence, and public
surface remain downstream.

The focused native regression covers owner/context identity and lifecycle,
context and all five A2 allocation failures, first-error preservation, active
transaction/view/read-borrow close rejection, dead close idempotence, owner and
parameter preservation, exact A2 registry baselines, symbol deltas, and
deterministic normal/sanitizer output. Its retention loop measures the accepted
tombstone cost:

| Completed create-close cycles | Retained registry records | Logical retained bytes |
|---:|---:|---:|
| 1,024 | 1,024 | 49,152 |
| 8,192 | 8,192 | 393,216 |

Each record is 48 bytes in the reviewed x86-64 build. A2 cache allocations are
destroyed each cycle. The native context registry intentionally grows linearly
for process lifetime; this leaf makes no flat-memory claim and does not measure
allocator overhead or resident-set size.

The supported native gate is `scripts/test-g3c4-context-cache.sh`; it also runs
the unchanged native owner regression. During local preparation the fast static
check and host-native normal and sanitizer builds passed. The configured
supported-toolchain gate, Docker, AOT, and package gates remain separately
queued.
