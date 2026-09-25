# G3-T private authenticated T1 input leaf

This leaf adds the accepted source-private `input_from_t1(ptr sealed_t1)`
stem. The Eshkol wrapper first requires the exact same-aggregate T1 registry
member. Native code then uses T1's existing pointer-admitted length, read and
status calls: it requires a sealed shell of length one or two, reads each ID
into local staging and requires every ID in 0..255. It creates a distinct
dense CPU I1[1,P] owner, copies the staged IDs, and enrolls the G3-T kind-2
input last. No T1 shell, borrow, registry entry or tokenizer authority is
retained. The existing private scalar/pair constructors remain test witnesses;
they use inline storage and are not claimed as the accepted T1-owned I1 path.

The T1-created input owns its I1 tensor canonically. Its two inline ID slots
are a validated staging copy for the pre-existing private frame ABI, never a
separate public input shape. Typed release rejects an active I1 borrow before
mutation, destroys the tensor once, clears staging and retains the native
tombstone. Construction cleans any partial I1 owner before returning failure.
The Eshkol wrapper preallocates and roots its shell, entry, ledger and list
before native construction and releases a newly constructed native owner if
later wrapper publication raises.

The focused P2 witness tests foreign, copied, withdrawn, unsealed and stale
T1 identity, P0/P3 and out-of-range IDs, native header and four I1 allocation
cuts, borrowed input release, detached P1/P2 copies, idempotent release, and
P1/G0 and P2/G0 frame consumption of T1-backed inputs. It retains the
accepted P1/G1 and clone regressions. Reusing one sealed P2 shell through
1,024 and 8,192 create/release cycles shows at most one live input/I1 and
returns live I1 count to baseline after each release; native/Eshkol identity
tombstones remain cumulative. The production object excludes test
observers. This is a private input prerequisite only; it adds no G3-G facade,
export or CLI, and P2/G1 still requires a capacity-three contract.

The pinned Ubuntu 22.04/LLVM 21 network-disabled focused gate passes 46,469
checks each in normal, repeat and ASan/UBSan/LSan modes with identical stdout
and empty compile/runtime stderr. Seven source contracts and Q0 4/4 pass.
The sanitizer launch uses `detect_leaks=1`, `halt_on_error=1` and arena
poisoning. This is candidate evidence, not public package acceptance.
