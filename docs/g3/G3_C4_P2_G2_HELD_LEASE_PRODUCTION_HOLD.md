# P2/G2 terminal held-lease authority boundary

This audit starts from test-only head `f31ec2f35752767b83326e5ac19b046c03bcc3c5`
(tree `4fe1d4eef5b7cc801bedd27178f3e41cf5e4e899`). Its successor
proposes a scoped correction to the
[terminal-close contract](G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md).
It does not change the production composer or irreversible zero-return tail.

## Existing authority

The native terminal and extensible protected-scope contracts require a held
genuine pending-output I1 or A2 lease to reject abort without mutation, and
permit its holder to release and retry inside the protected body. The fixed
composer entry takes only `(generator input)` at
`native/g3c4_p2g2_terminal_composer_extension.esk:188`; it creates `call`,
`ledger`, and `terminal` in that lexical invocation (`:195-226`, `:309-330`).
Its prezero caught path calls `g3c4-p2g2-terminal-abort-clean!`, then exits
134 on abort rejection (`:240-248`). The helper calls protected abort before
scrub or unlink (`:155-175`); protected abort invokes native abort before
mutating Eshkol edges (`native/g3c4_p2g2_protected_scope_extension.esk:150-175`).

Native abort preflights pending-output I1 at
`src/eshkol_transformer/g3c4_model_owner.c:8213-8217` and a READY A2
transaction view at `:8229-8247`, before token-frame discard or pending
output destruction. Holder release requires its captured pointer slot:
`et_i64_tensor_borrow_end_v1(et_i64_tensor_borrow **)` in
`include/eshkol_transformer/i64_tensor.h:152-153`, or
`et_a2_kv_cache_transaction_view_end_v1(et_a2_kv_cache_transaction_view **)`
in `include/eshkol_transformer/a2_kv_cache.h:113-114`. The production
composer's extern list (`native/g3c4_p2g2_terminal_composer_extension.esk:5-18`)
has neither release entry nor holder capability. Its fixed coordinator does
not retain an I1 borrow or A2 view across the terminal prezero tail.

The `m3-call` macro rejects entry while busy and clears its busy cell when a
condition escapes (`native/m3_model_extension.esk:18-30`). An external
holder cannot release by reentering a public/private model operation after
abort rejection while preserving the original live region. The diagnostic
overlay in `f31ec2f3` works because a same-aggregate **test shim** retains
the genuine lease pointer and releases it before raising a prezero condition
to the existing outer guard. Its `#f` negative is a mismatched output
argument; it is not a live foreign owner. Its G2 READY/P3 cases remain
source-reviewed and compiled-runtime unproved.

## Supported holder reachability

The pending I1 is newly created by native output reservation inside an active
call (`src/eshkol_transformer/g3c4_model_owner.c:3045-3103`), and the pending
lookup is `static` (`:2133-2155`). The source-private context ABI returns an
opaque output handle, not its I1 or A2 transaction
(`src/eshkol_transformer/g3c4_context_internal.h:137-144`). The Eshkol
reservation enrolls its pending entry in the private global `g3c4-registry`
and retains the local reference
(`native/g3c4_output_envelope_extension.esk:134-162`); the fixed serialized
invocation has no supported callback or yield exposing it to another caller.
The composer returns only the terminal shell after close. Snapshot and logical clone
authenticate a published terminal output, so they cannot expose the pending
handle (`native/g3c4_p2g2_terminal_composer_extension.esk:513-540,623-640`).

An idle token frame has no transaction
(`src/eshkol_transformer/g3c4_model_owner.c:1496-1504`). A committed A2 read
lease present before generation blocks native call acquire's idle-cache
preflight (`:7995-8011`); while the frame transaction exists, A2 refuses a
new committed read lease (`native/a2_kv_cache.c:804-818`). The native
prefix/terminal helper's temporary I1 borrows and A2 views are ended before
return on success and failure (`src/eshkol_transformer/g3c4_model_owner.c:7077-7130,7250-7297,7352-7391,7452-7533,8355-8392`).
The raw V256 T1 decoder has no user callback
(`native/g3c4_t1_output_decode_extension.esk:1-52`). Native C4 operations
require external serialization
(`docs/g3/G3_C4_NATIVE_OWNER_LEAF_CONTRACT.md:43`,
`docs/g3/G3_C4_ACTIVE_CALL_STEP5_CONTRACT.md:157-163`), and `m3-call`
rejects reentry while busy (`native/m3_model_extension.esk:18-30`). Thus no
supported caller obtains and retains a genuine pending I1 borrow or READY A2
view across this fixed entry's prezero abort. Direct private-struct access,
concurrent unsynchronized C calls, and same-aggregate test injection are
outside that supported boundary.

## Contract correction and future decision

No accepted production API identifies a cooperative holder, transfers its
captured release authority into this lexical call, or invokes that holder
after typed native abort rejection while `m3-call` stays live. The revised
terminal-close contract therefore keeps native and protected-body held-lease
requirements but does not demand an unreachable production external-holder
retry from this fixed entry. Its test-only overlay still proves the same-call
mechanism when the test shim owns and releases the captured pointer. On an
unsupported injected persistent lease, protected/terminal records remain
unchanged up to fail-stop; no recoverable return is claimed.

If a future supported caller is permitted to hold a lease across this call,
first review a private call-bound ownership contract. It must define exact
authority transfer and original holder, a prezero release point while the
original call is live, typed busy discrimination, reauthentication of the
call/ledger/root identities/native pending/cache/RNG before one bounded
abort retry, and fail-stop for missing/mismatched release. It must forbid
foreign-pointer release and any callback or native readback after terminal
zero. No implementation API is proposed or admitted by this document.

## Finite completion gate

Review the contract correction and exact source/return-shape proof. If a
future holder interface is accepted, test genuine I1 and A2
leases with a second live foreign owner negative, first native abort's exact
typed busy result, byte/identity preservation of pending output, cache, RNG,
and all Eshkol root cons cells, captured-holder release, and one successful
retry in the original `m3-call`. Test missing release, wrong holder, repeated
busy, P2 and P3 rollback, and postzero fail-stop; normal/repeat/sanitizer,
fault overlays, feature-off and predecessor gates must pass on exact pins.
Preserve failed evidence. The published test overlay establishes a diagnostic
possibility only; it does not establish production external-holder support.
The fixed composer compiled close gate remains pending, and no Docker run was
performed for this authority audit.
