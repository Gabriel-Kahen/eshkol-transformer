# Private P2/G2 first-frame Eshkol carrier bridge

`ET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE` exposes the source-private
`et_g3c4_private_p2g2_first_frame_carrier_v1(context, input,
staging_header, carrier_bytes)` only with the accepted P2/G2 prefix-commit
and private f32 storage-inspection features. It is a prerequisite for the
[proposed same-aggregate T1 coordinator](G3_C4_P2_G2_T1_PREFIX_OWNERSHIP_CONTRACT.md),
not that coordinator. The call requires an acquired `(kind,budget)=(2,2)`
generation call, an authentic owned P2 I1 input, one linked pending P2/G2
output, an idle first-token frame, and an Eshkol bytevector with exactly eight
payload bytes. The complete header-plus-payload extent is 16 bytes. The
caller must check `bytevector?` and `bytevector-length==8`, then authenticate
the original owner identity, backing extent, and no callback replacement
within the same Eshkol call region before passing its pointer;
neither its self-declared length header nor a caller-supplied C extent can
prove backing allocation on its own.

The bridge checks the reported extent, alignment, exact signed-i64 length
header, active owners and output, and full-range overlap against transport
records, model owners, pinned parameter views, I1/f32 tensor storage and A2
storage. Native stack `float[256]` buffers and a scalar `i64` carry the
accepted P2 prefill, last-logit sample and selected-token forward in that
order. The selected ID comes only from the native sampler. A selected EOS
fails this nonterminal leaf before forward. After a ready frame and binding
reauthentication, the bridge writes the ID as eight little-endian bytes to
the staging **payload**. It never writes the length header, publishes an
output, or commits the prefix cache/RNG; the accepted prefix-commit leaf
remains a separate call after genuine same-registry T1 raw decode.

Preflight, prefill, sampler and forward failures leave the carrier untouched
by this bridge. A failed prefill preserves the entry cache; a later failure
leaves the committed P2 `2/1100` cache and original generator RNG. The active
call owns any candidate frame until authenticated abort. The local bridge
test checks greedy/categorical IDs and Philox successors against the
independent oracle, bitwise next logits and staged K/V against the direct
native route, full-carrier canaries, malformed lengths/extents, aliases,
stale/repeated calls, provider cuts and feature closure. Genuine Eshkol T1
decode provenance, output ownership and rollback are separate next gates.

## Supported gate status

The independently source-and-test-reviewed `810c08b985bd979796878625db8d41c01591dbd3`
(tree `48f7dbd89d648f948fe85153322c8926e3a03047`) passed its pinned
`fe9dfd52` compiler / f31 LLVM 21 image gate. The single rootless,
network-disabled launch exited zero in both Docker and launcher records.
Normal, repeat and ASan+UBSan+LSan runs each passed 972 checks with
byte-identical stdout and empty stderr. The independent two-mode G3-S/Philox
oracle, predecessor gates, exact private-symbol delta and isolated-macro
rejection passed. The verified 44-file evidence seal is
`/home/gabe/g3c4-p2g2-carrier-810c08b-fe9-f31-20260930-prepared-b/SHA256SUMS-RUN`
(SHA-256 `d68020558419dc1e3886f7f4df3dca5e8fe3729d5076669b992382a6ef21bc0b`).
Independent evidence review accepted the sealed gate; hosted CI and merge
remain pending. The next Eshkol leaf
still must authenticate the caller-owned bytevector allocation and retained
identity, perform genuine same-aggregate T1 decode and own the call/output
rollback before invoking the accepted native prefix commit. This gate makes
no second-token, EOS publication, terminal result, public ABI or CLI claim.
