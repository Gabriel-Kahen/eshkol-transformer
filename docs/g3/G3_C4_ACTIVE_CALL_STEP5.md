# G3-C4 Step 5 native active calls

This bounded native leaf implements the accepted
[`G3_C4_ACTIVE_CALL_STEP5_CONTRACT.md`](G3_C4_ACTIVE_CALL_STEP5_CONTRACT.md).
It adds the four private acquire, prepare-end, finish, and abort operations to
the sole C4 owner/context translation unit.

The active-enabled context embeds the accepted fixed-14 pin record and retains
the exact Step 4a owner and A2 cache. Acquire accepts only `(0,0)`, `(1,0)`,
`(2,0)`, and `(2,1)`, proves A2 idle through a transient read borrow, acquires
all pins, records call metadata and busy state, then publishes `owner.active`
last. Test-only publication failures unwind the recorded native mask in reverse.

Prepare-end is read-only in net effect. Abort completes the fallible A2 idle
probe before cleanup. Finish is the allocation-free trusted tail. Both drain
pins before clearing `owner.active` and then zero call metadata, busy state, and
the acquired mask. No Eshkol call entry, seed/RNG policy, frame, forward,
sampler, persistence, public API, package surface, or G3-S source is added.

On the reviewed x86-64 build, the embedded pins increase each retained context
record from 48 to 1,432 bytes. The focused 8,192-cycle measurement is:

| Completed create-close cycles | Retained records | Logical retained bytes |
|---:|---:|---:|
| 1,024 | 1,024 | 1,466,368 |
| 8,192 | 8,192 | 11,730,944 |

This remains intentional linear process-lifetime tombstone retention. The
figures exclude allocator overhead and resident-set size. Every A2 cache is
destroyed on context close, and active-call A2 probes retain no lease.

The supported gate is `scripts/test-g3c4-active-call.sh`. Host compatibility
and static results are recorded separately; the fresh supported Docker/AOT gate
remains held until the shared supported-build slot is released.
