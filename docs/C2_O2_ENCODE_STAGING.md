# Private C2 O2 encode staging

`c2-o2-encode-staging-internal` is a private, source-composed staging slice. Its
exact request is `[live-O2-state, live-P1-state, operation-symbol, #f]`. Success
sets the final slot to `[O2-metadata, moment-payload]`; failure leaves that slot
unchanged. It adds no public O2 operation, package export, provider row, or K2
dependency.

The adapter reads O2's deep-owned validated lexical configuration, canonical
paths, aliases, and completed-update count. It obtains the full ordered model
entry projection through the real P1 internal state APIs, derives canonical
unique parameter indices after excluding secondary aliases, and checks exact
path, alias, rank, extent, shape, count, dtype, device, and layout agreement.
The native O2 state/count/completed and per-entry shape seams are rechecked
before any moment copy. Moment bytes come only from
`o2-state-copy-moment-bits-internal`, which copies exact binary32 bits directly
into bytevectors and allocates or retires no O2/I2 borrow shell.

The result uses the accepted O2 1.0 metadata layout and interleaved
exp-average/exp-average-square payload order. Both 32-byte digest slots in every
parameter record are deliberately all-zero placeholders. Therefore staging is
not a valid independently parseable checkpoint and must not be published. The
private `et_c2_checkpoint_encode_v1` contract ignores and recomputes every
moment digest from the final record and payload, then recomputes the outer
digest; C2 must run the complete parser before publication.

Limits are the accepted wire limits: 4,096 full P1 model entries, 1,365 unique
O2 parameters and groups, rank/path depth 64, 65,536 UTF-8 bytes per path
segment, 256 GiB per moment tensor, 6,826 aggregate tensors, 256 MiB O2
metadata, and 1 TiB staging metadata plus moment payload. Actual allocation is
still limited by the host and Eshkol bytevector allocator; allocation failure is
reported and cannot publish a partial output.
