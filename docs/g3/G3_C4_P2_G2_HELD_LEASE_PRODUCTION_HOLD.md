# P2/G2 terminal active-call held-lease production HOLD

This audit starts from test-only head `f31ec2f35752767b83326e5ac19b046c03bcc3c5`
(tree `4fe1d4eef5b7cc801bedd27178f3e41cf5e4e899`). It does not
change the accepted [terminal-close contract](G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md),
production composer, or irreversible zero-return tail.

## Existing authority

The accepted contract requires a held genuine pending-output I1 or A2 lease
to reject abort without mutation, retain the live pending terminal anchor
and protected record, and allow owner release and abort retry inside the
same call region. The composer entry takes only `(generator input)` at
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

## Required upstream decision

No accepted production API identifies a cooperative holder, transfers its
captured release authority into this lexical call, or invokes that holder
after typed native abort rejection while `m3-call` stays live. Adding such
an API here would invent an upstream ownership contract. Choose one of:

1. Specify and review a private, call-bound cooperative holder capability.
   It must identify the exact pending I1/A2 lease and original holder,
   permit only that holder to end its captured pointer, and offer a prezero
   in-region release point. Then a bounded retry can reauthenticate the
   original call, ledger, terminal/protected/prefix root identities, native
   pending output, staged cache and RNG before retrying protected abort.
   Scrub and unlink only after abort succeeds. A missing/mismatched release
   retains fail-stop. No callback or native readback may run after terminal
   zero.
2. Narrow the accepted held-lease clause only after proving that no genuine
   external holder can survive across this fixed coordinator's prezero
   abort point. The current source establishes no intentional retained
   holder; that alone is not a proof against every supported caller/alias.

## Finite completion gate

After the upstream ownership decision, review the exact contract and source
before implementation. If option 1 is accepted, test genuine I1 and A2
leases with a second live foreign owner negative, first native abort's exact
typed busy result, byte/identity preservation of pending output, cache, RNG,
and all Eshkol root cons cells, captured-holder release, and one successful
retry in the original `m3-call`. Test missing release, wrong holder, repeated
busy, P2 and P3 rollback, and postzero fail-stop; normal/repeat/sanitizer,
fault overlays, feature-off and predecessor gates must pass on exact pins.
Preserve failed evidence. Until then, the published test overlay establishes
a diagnostic possibility only, and the production held-lease clause remains
unproved. No compiled gate or Docker run was performed for this audit.
