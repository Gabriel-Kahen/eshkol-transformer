# G3-C4 private Eshkol manual transport witness

This leaf connects the already accepted native C4 manual frame path to the
existing private Eshkol C4 model, generator, call, and T1 registries. It is a
development witness, not the G3-T package root or a public generation API.
The native owner remains the `diagnostic-c4` model with its C4 profile. The
G3-T contract calls for an authentic M3T/C2 model owner and its own native
boundary; neither is supplied by this leaf. The C4 owner must not be relabeled
as M3T to make a root appear complete.

`g3c4_manual_transport_extension.esk` is loaded after the private output
envelope. It uses the shared `g3c4-registry`, model registry, T1 registry, and
`m3-call` guard. Every manual call enters that guard once, acquires the native
context with the exact call kind and zero budget, and records both model and
generator active links. The lexical macro supplies cleanup only. Its body
provides the 21 individual `g3t-role-step` calls. No production numeric loop
or prefill/decode wrapper is added.

The pending logits shell and 12-slot entry enter the Eshkol registry before
native reservation. The call ledger roots the canonical entry before the
allocation. Each transport operation authenticates the exact call entry,
active cells, model, tokenizer, pending result, and call ledger before passing
native pointers. `frame_begin` also admits the sealed T1-derived input owner.
Native `frame_begin` takes its own copy of the token IDs. Reservation failure,
step failure, or a wrong result shell aborts the native call, tombstones the
pending logits and call entries, and clears active links before reraising.

The successful sequence is `logits_reserve`, `frame_begin`, 21 explicit
`role_step` calls, `frame_prepare`, Eshkol tokenizer/eval and identity
rechecks, native `call_prepare_end`, native `frame_commit`, native
`call_finish`, and Eshkol active-link cleanup. `frame_commit` is the sole
manual publication point. Immediately afterward the lexical phase becomes
committed; any failure in the tail is fail-stop. The result entry becomes a
detached live logits owner, which `g3t-logits-release!` authenticates and
releases through the existing typed native tensor release.

The source-backed witness covers P1 and P2 manual prefill, a decode frame
following committed P1 prefill, exact cache lengths and detached result
ownership, out-of-order ordinal rollback, wrong-result rollback, native
reservation allocation failure, tombstones, release, and repeated normal and
sanitized execution. The prior native role-step gate remains the authority for
all 21 provider routes, retry cuts, bitwise comparison, pin checks, and
native publication failure cuts. This leaf adds no installed source and no
public symbol manifest entry.

Pinned gate: `scripts/test-g3c4-manual-transport.sh`; exact dependency union:
`native/g3c4_manual_transport_source_closure.txt`. The gate invokes Q0 Python
isolation, predecessor static checks, native role-step and output-envelope
regressions, and the normal/repeat/ASan+UBSan Eshkol witness.
