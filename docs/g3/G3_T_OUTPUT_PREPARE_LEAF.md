# G3-T private G1 numeric output preparation leaf

This bounded successor to the sampled-token frame gives the pending native
output its accepted numeric owner fields: `parent_ctx`, `P`, `G`, exact I1
`ids[G]`, generated and cache lengths, four RNG words, and the three readiness
flags. Reservation now creates I1 `[1]` before enrolling the output. Failure
leaves no enrolled partial owner. Abort preflights an active I1 borrow before
discarding the token transaction; successful abort destroys I1 and scrubs the
unpublished output while retaining the committed P1 cache and old RNG.

`et_g3t_private_output_prepare_v1(context, output)` authenticates the active
generate call and its unique pending output, requires the committed P1 binding
and the completed 21-role sampled frame, then copies the sampled byte ID into
I1 `[1]`. That I1 copy is the last recoverable operation. The no-failure tail
records generated length 1, cache length 2, the staged successor RNG and
`numeric_ready=1`. A second preparation and wrong owner, early frame, stale
binding or active I1 borrow reject before output mutation. The committed
cache, generator RNG and speculative append remain unchanged.

The source-private Eshkol `g3t-output-prepare` wrapper checks the rooted call
and output link before the native call. No public G3 generation name or
installed symbol is added. G0 preparation remains blocked by the existing
G3-T generate admission, which accepts only `max-new-tokens=1`, and by the
absence of a G0 prefill/output path. ID byte staging, T1 raw decode, text
readiness, final prepare/commit and live output publication remain downstream.

The focused witness injects all four I1 reservation allocation cuts, retries
on the same active call, rejects preparation before prefill and before the
token frame is ready, and proves a live I1 borrow blocks both preparation and
abort before frame mutation. It checks the staged ID, lengths and successor
RNG, once-only readiness and complete abort scrubbing under normal, repeated
and ASan/UBSan/LSan execution.
