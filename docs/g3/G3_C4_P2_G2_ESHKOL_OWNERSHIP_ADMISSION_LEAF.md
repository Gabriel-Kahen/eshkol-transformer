# Private P2/G2 Eshkol ownership admission leaf

This leaf adds only source-private capacity admission for the accepted
[`(P,max G,N)=(2,2,1)` ownership contract](G3_C4_P2_G2_T1_PREFIX_OWNERSHIP_CONTRACT.md).
The Eshkol gate starts off. Loading
`native/g3c4_p2g2_ownership_extension.esk` after the existing call/output
envelope enables `max-new-tokens=2` in the private C4 policy and exact
`(call-kind,budget)=(2,2)` in the private call-tuple predicate. The pending
output reservation then admits only `(prompt,requested G)=(2,2)` and allocates
the existing raw/staging auxiliary bytevectors at lengths two and sixteen.
The future private call must retain requested budget two in slot 7; this
leaf's reservation and output validators require that marker, exact
auxiliary lengths and the active call-ledger child edge. Native reservation
owns the pending zero/unready
`I1[2]`; requested capacity is not an emitted-token count.

The old `g3c4-with-call-internal` terminal scope explicitly rejects budget
two. The P2/G1 publication scope and public C4 adapter retain their one-token
guards. No P2/G2 caller can reach prefill, T1 decode, prefix commit, text,
copy-out, terminal finish or a returned output through this leaf. The
compiled Eshkol witness enrolls a budget-two call **only in its test fixture**
to check native-backed pending ownership, rejected wrong tuples, malformed
auxiliary lengths and child ledger, unready output and abort cleanup. The
runner compiles separate feature-off and feature-on fixtures: the former never
loads the enabling extension and checks default rejection; the latter loads it
before exercising pending ownership. This separation matches Eshkol AOT source
loading, which enables a loaded top-level gate before the original fixture's
runtime assertions. The public C4 constructor's one-token guard and budget-two
rejection fixture have source checks in this leaf; compiled public rejection
remains outside its evidence. The accepted P2/G1 and C2 regressions remain
predecessors of the supported gate.

This is an admission prerequisite, not the protected one-region coordinator.
The generic pending validator does not authenticate same-length replacement
of the raw/staging bytevector objects. The next leaf must capture their
original identities and backing allocation/extent in a protected call
record, allocate the private prefix carriers, handle held native I1/A2
leases, and provide exact pre/post-prefix rollback. Only then may it call the
merged native carrier bridge, genuine same-registry T1 decode and native
prefix commit. First-token EOS publication, second-token forward, terminal
output, public ABI and CLI remain separate.

Source and static/Q0/CI topology checks are local at this stage. The
compiled Eshkol normal/repeat/sanitizer gate and independent source-and-test
review are pending on an immutable commit; no supported execution claim is
made here.
