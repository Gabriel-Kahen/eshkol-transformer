# E3 N1 distinct seal authority contract

Status: **proposed successor contract; HOLD**. This specifies the authority
missing from the [source-only v2 synthetic durable receipt candidate](E3_N1_BROKER_CANDIDATE.md)
on exact `ed2856b8`. It does not change that candidate's source, tests, or
accepted synthetic-consistency result. It supplies no production implementation,
real release pin, export, or N1/M1 admission.

## Authority and trust boundaries

The output broker owns source validation, the protected export parent and
stages, the protected HOLD intents, and receipt/commit publication. A separate
seal authority owns its seal signing capability and protected seal store. The
broker must have no seal private key, symmetric seal key, signing RPC credential,
writable seal-store fd, or ability to ask the sealer to sign unchecked bytes.
The seal authority must have no broker receipt signing key, writable broker
parent/receipt-store fd, or ability to issue an acquisition receipt. Their UIDs,
credentials, writable ancestors, ACLs, mount access, `/proc` access, and
inherited fds must be reviewed as distinct boundaries. A shared login with two
files marked `0700` is not separation.

The read-only consumer/reconciler receives only frozen release metadata,
non-minting receipt and seal verification capabilities, and read access to
protected evidence. It must not hold either signing key or accept a
caller-selected verifier key, verifier service, release descriptor, or seal
store as authority. The issuance token remains a separate capability returned
by the receipt issuer to the acquisition caller after its checked durable
commit; the sealer neither creates nor delivers it. A token hash in a readable
commit does not replace the returned token.

The v2 `Config` currently gives the broker a symmetric
`seal_verification_key`; that is also a seal-minting key. The sealer cannot be
given the broker's symmetric `signing_key` merely to validate a receipt,
because that would let it mint broker receipts. A successor therefore needs
independently verifiable broker receipts and seals, or a separately reviewed
trusted receipt-verification boundary that does not expose receipt-minting
material to the sealer or consumer. The receipt and seal authorities, their
non-minting verification interfaces, and failure behavior must be specified
and tested before either can release admission. No asymmetric verifier or
trusted signing/verification RPC is pinned in this tree's N1 dependencies:
the N1 Python modules use the standard library, while the available Python
lockfiles cover development reference/oracle tests. Algorithm, library/service,
exact API, and key provisioning remain HOLD; this document does not select or
pretend to provide them.

## Frozen release and custody pins

Before a real layer read, review one immutable release descriptor that binds
an authentic image index and selected manifest/config/layer digests, ordered
source paths and pre-read identities, source and output ancestor device/inode/
mode chains, byte/time/RSS caps, target basename, receipt-store path and
ancestor chain, compiler/source heads and trees, and the planned export policy.
It also binds distinct broker-receipt and seal verification key identities,
verification algorithms or service identities, and the protected seal-store
path and ancestor chain. A changed key, identity, path, chain, source, image,
or release digest is a new release and cannot inherit an earlier receipt or
seal. The current `synthetic:` descriptor is not an authentic f31 pin.

The broker's export parent, intent and receipt stores, and the sealer's store
must deny untrusted rename, unlink, write, and fd access throughout issuance
and reconciliation. Pin the exact seal file identity and digest after checked
publication through a protected channel; a caller-created `SealRef` or
path/hash/inode tuple is only an assertion. Independently check the live
export target and files through nofollow fd walks against the receipt and
release; a signed statement about an old target is insufficient.

## Issuance and crash rules

1. The broker keeps the existing protected, synced pre-stage HOLD intent and
   quarantines all receiptless or uncertain export outcomes. A visible target,
   `VERIFIED.json`, or readable receipt/commit has no admission authority.
2. After target verification and sync, the broker publishes and reads back the
   attempt-keyed out-of-export receipt and commit, then syncs their directory.
   Only a checked completion returns the issuance token and receipt through
   the separate delivery channel. Unknown write/sync/return outcomes remain
   HOLD; recovery cannot derive the token from commit bytes.
3. The seal authority accepts the exact frozen release and attempt only after
   independently validating the broker receipt authority, durable receipt and
   returned-token evidence through a non-minting boundary. It checks the
   protected intent, target identity, ordered file hashes, receipt/commit
   identities, and export custody before signing a canonical seal. The seal
   binds the release/image digest, both authority key identities, attempt,
   target device/inode, exact file list, and receipt/commit digests. It
   publishes the seal with checked no-replace write, readback, and parent sync
   in its protected store, then returns a protected seal reference. It cannot
   retroactively seal a failed or receiptless attempt.
4. A crash before checked receipt or seal issuance, or before external
   delivery of the receipt, token, and seal reference, remains HOLD, even if
   the corresponding files are readable. A crash after all three have been delivered can reconcile only
   while both protected stores, target, files, identities, key pins, and seal
   verify. An incomplete or ambiguous sync is HOLD. Reconciliation is read-only
   and never manufactures a receipt, token, seal, or missing reference.

A consumer can report `MATCHED` only for the same frozen release and attempt
when the externally held receipt and token match the durable receipt/commit,
the protected seal reference resolves to the exact verified seal, both
non-minting authority checks pass, and current export bytes and custody pins
match. Any missing, duplicate, unreadable, stale, replaced, or ambiguous
record, token, target, seal, key identity, or store is HOLD. `MATCHED` in the
existing synthetic tests does not establish this admission rule.

## Required successor evidence

Generated tiny OCI fixtures must reject broker-crafted seals, sealer-crafted
receipts, shared or substituted keys, caller-selected verification material,
foreign release/image pins, copied stores, forged seal references, target or
file drift, duplicate attempts, and all receipt/commit/seal write, rename,
readback, sync, and delivery fault boundaries. Hard exits after readable
`VERIFIED`, receipt, commit, and seal bytes but before their checked return
must remain HOLD; a fully delivered valid attempt may reconcile after restart.
The tests must distinguish receipt issuance from seal issuance and must not
use real f31 layers or claim production custody.

A separate live cross-UID authority gate must prove, with distinct real
UIDs, that the caller cannot mutate any broker-owned ancestor, stage, intent, receipt store,
or target; the broker cannot read seal signing material or write the seal
store; the sealer cannot read broker signing material or mutate broker output;
and the consumer cannot mint either authority. Check ACLs, mount/namespace
routes, `/proc` and inherited fds, including pre-open rename/replacement and
post-publication mutation attempts. Review the exact deployed key/provider
pins and service behavior as well as source/tests. Until these gates and the
authentic release descriptor pass independent review, real-layer acquisition
and N1/M1 admission remain HOLD.
