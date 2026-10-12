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
and tested before either can release admission. The
[docs-only provider decision](E3_N1_PROVIDER_DECISION.md) selects OpenSSL
3.5.9 PureEd25519 EVP as a v3 candidate and pins its upstream source
artifact, role-separated messages, and verification API. Built provider and
verifier bytes, deployment, keys and live authority proof remain HOLD; the v2
Python modules still use standard-library HMAC and cannot implement that
decision. The frozen release must carry the actual public
verification key bytes and their digest (or an immutable verifier trust
anchor), not a key ID alone. If a trusted service is selected, pin its exact
endpoint, transport trust root, deployed service binary/config identity, and
credential-to-role policy; a display name or caller-supplied address cannot
serve as a trust anchor. Both receipt and seal verification authorities need
these pins, even when they use different mechanisms.

## Frozen release and custody pins

Before a real layer read, review one immutable release descriptor that binds
ordered source paths and pre-read identities, source/output ancestor
device/inode/mode chains, byte/time/RSS caps, target basename,
receipt-store path and ancestor chain, compiler/source heads and trees, and
the planned export policy. Bind the exact raw OCI descriptor chain from the
[accepted input plan](E3_N1_EXACT_INPUT_PLAN.md):

| Ordered object | SHA-256 | Size (bytes) | Current state |
| --- | --- | ---: | --- |
| `index.json`, image ID | `f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6` | 855 | small bytes sealed |
| sole `linux/amd64` manifest | `3649921cfd39ef9b2c4a54d0c6ca01cd6fc49768806297fe61e05a70648804fb` | 676 | small bytes sealed |
| selected manifest config | `584fc09fcf7a84e381d8ef1f2903143b7b1debbbd5661468ed31d56863087ee2` | 2,326 | small bytes sealed |
| ordered gzip layer 0 | `20c3783cc497b5b0df1fc5f92bd64c3d2fbb24057c88692c8fda205b6ea8a2f2` | 29,750,251 | descriptor only |
| ordered gzip layer 1 | `399066089356a53b7c7046be9a8a2e938647e6a4076724bf42034912ea6ace98` | 449,869,772 | descriptor only |

Freeze the exact media-type strings, platform, sizes, and digest fields from
the sealed raw index, manifest, and config bytes: index to the unique
`linux/amd64` manifest (never the `unknown/unknown` attestation), manifest
to its config and two ordered gzip layers, and config to its ordered expanded
diff IDs `sha256:ea16cace89338c84eb6bcb91a7efdfcae6838fff359efe951858227436486c34`
and `sha256:5ef3a0f7c37c1e91eaf25f50ed9c318abd80e417aef95d9352435bb165eeb458`.
The gzip digest/size and diff IDs remain unverified claims until the
separately reviewed bounded read and decompression gates pass. Preserve raw
bytes and descriptor order; reserialized JSON, a config digest substituted
for the image ID, or a matching local size cannot prove this chain.

Bind distinct broker-receipt and seal verification trust anchors, their
actual public key bytes/hashes or exact trusted-service identities, the
protected seal-store path/ancestor chain, and the target's planned immutable
handoff. A changed key, endpoint, path, chain, source, image, or release
digest is a new release and cannot inherit an earlier receipt or seal. The
current `synthetic:` descriptor is not an authentic f31 release pin.

The broker's export parent, intent and receipt stores, and the sealer's store
must deny untrusted rename, unlink, write, and fd access throughout issuance
and reconciliation. Pin the exact seal file identity and digest after checked
publication through a protected channel; a caller-created `SealRef` or
path/hash/inode tuple is only an assertion. Independently check the live
export target and files through nofollow fd walks against the receipt and
release; a signed statement about an old target is insufficient.
An inode/hash check at seal time does not prevent later same-inode mutation.
Before signing, a reviewed handoff must put the verified target into an
immutable snapshot or equivalent protected ownership state that neither the
broker nor acquisition caller can write, rename, replace, or regain through
consumer use. Pin that snapshot's path, device/inode, custody chain, and file
hashes in the seal and release/evidence record; the consumer reads that
protected object. A review must establish the exact handoff and lifetime,
including what happens if read-only state or ownership transfer fails. The
current v2 target has no such proven handoff and remains HOLD.

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
   protected intent, immutable target snapshot/ownership handoff, ordered
   file hashes, receipt/commit identities, and export custody before signing
   a canonical seal. The seal binds the release/image digest, both authority
   key identities, attempt, immutable target device/inode and custody
   identity, exact file list, and
   receipt/commit digests. It publishes the seal with checked no-replace
   write, readback, and parent sync in its protected store, then returns a
   protected seal reference. It cannot
   retroactively seal a failed or receiptless attempt.
4. A crash before checked receipt or seal issuance, or before external
   delivery of the receipt, token, and seal reference, remains HOLD, even if
   the corresponding files are readable. A crash after all three have been
   delivered can reconcile only while both protected stores, target, files,
   identities, key pins, and seal verify. An incomplete or ambiguous sync is HOLD. Reconciliation is read-only
   and never manufactures a receipt, token, seal, or missing reference.

A consumer can report `MATCHED` only for the same frozen release and attempt
when the externally held receipt and token match the durable receipt/commit,
the protected seal reference resolves to the exact verified seal, both
non-minting authority checks pass, and the protected target snapshot,
current bytes, and custody pins match throughout consumer use. Any missing,
duplicate, unreadable, stale, replaced, or ambiguous record, token, target,
seal, key identity, or store is HOLD. `MATCHED` in the existing synthetic
tests does not establish this admission rule.

## Image-selected N1 closure

Export custody alone cannot admit the failed F tuple. A later N1 admission
descriptor, rooted in the immutable export-release digest, and its independent
evidence seal must bind a reviewed finite image-selected guest-path
inventory: exact compiler, linker, resource-directory and
`readelf`/`objdump`/`addr2line` tool paths, hashes, modes, layer origins and
symlink chains; interpreter, loader path, provider candidates and complete
ELF dependency/search configuration; direct and transitive system headers;
exact per-object compiler/linker argv and environment; and the final
optimized `puts` producer and per-call ABI at all 39 sites. Record the
resolved loader/provider observation separately from path candidates. Bind
F's host mount routes and host files separately; host candidates cannot stand
in for selected guest bytes. Each selected file or symlink must have the
accepted plan's role, absolute guest path, final layer, kind, mode, and exact
hash/size or symlink target. Seal the raw selection, discovery procedure,
loader observation, and producer/ABI evidence before accepting a capture
recipe. The small OCI seal and current synthetic v2 supply none of this guest
closure. Until it is obtained and independently reviewed, N1 import, M1, and
any claim of F parity remain HOLD even if export custody later passes.

## Required successor evidence

Generated tiny OCI fixtures must reject broker-crafted seals, sealer-crafted
receipts, shared or substituted keys, caller-selected verification material,
foreign release/image pins, copied stores, forged seal references, target or
file drift, duplicate attempts, and all receipt/commit/seal write, rename,
readback, sync, and delivery fault boundaries. Include post-seal same-inode
write and replacement races before and during consumer reads; each must be
denied by the reviewed custody boundary or return HOLD. Hard exits after
readable `VERIFIED`, receipt, commit, and seal bytes but before checked return
must remain HOLD; a fully delivered valid attempt may reconcile after restart.
The tests must distinguish receipt issuance from seal issuance and must not
use real f31 layers or claim production custody.

A separate live cross-UID authority gate must prove, with distinct real
UIDs, that the caller cannot mutate any broker-owned ancestor, stage,
intent, receipt store, or target; the broker cannot read seal signing
material or write the seal store; the sealer cannot read broker signing
material or mutate broker output; and the consumer cannot mint either authority. Check ACLs, mount/namespace
routes, `/proc` and inherited fds, including pre-open rename/replacement and
post-publication mutation attempts. Review the exact deployed key/provider
pins and service behavior as well as source/tests. Until these gates and the
authentic release descriptor pass independent review, real-layer acquisition
and N1/M1 admission remain HOLD.
