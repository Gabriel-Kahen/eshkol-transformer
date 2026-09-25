# G3-T pending G1 ID staging and raw text readiness

The pending numeric output now supports two source-private native calls:
`et_g3t_private_output_copy_decode_ids_v1(context, output, staging)`
and `et_g3t_private_output_accept_text_v1(context, output, raw)`.
They authenticate the active G1 call, unique pending output, committed
parameter binding, and once-only readiness order. The staging carrier is a
preallocated bytevector with an i64 length header and eight payload bytes.
The raw carrier has one payload byte. Both calls reject overlap with enrolled
transport records, model owner, pinned parameters, live I1 storage, and A2
storage. The exact output I1 `[1]` is borrowed synchronously, checked for
a byte ID, and released before readiness changes. Abort scrubs staged ID
bytes and all readiness flags.

`g3t-t1-decode-output!` checks the rooted call/output linkage, stages the
ID, invokes the existing same-package
`t1-private-g3-decode-raw-into!` decoder, then accepts the raw byte.
Staging and raw bytevectors are allocated at output reservation. The T1
decoder authenticates the actual registry/core and raw V256 profile; the
valid decode path enrolls no tokenizer shell or retained borrow.

This leaf supports G1 only. Final frame preparation and commit, cache/RNG
publication, live output transition, public accessors, and G0 remain open.
