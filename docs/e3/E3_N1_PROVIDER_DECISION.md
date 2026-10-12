# E3 N1 receipt and seal verifier provider decision

Status: **docs-only proposed v3 contract; HOLD** on exact `d833c583`. This
selects a route for the [distinct seal authorities](E3_N1_SEAL_AUTHORITY_CONTRACT.md).
It changes no v2 Python source or synthetic receipt and supplies no service,
deployed key, real release pin, export, or N1/M1 admission.

## Source and binary pins

Select PureEd25519 through native OpenSSL 3.5.9 LTS `libcrypto` EVP and its
`default` provider. The exact upstream source artifact is
`openssl-3.5.9.tar.gz`, SHA-256
`603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a`.
This is the official
[3.5.9 checksum](https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz.sha256)
linked from the [OpenSSL download page](https://openssl-library.org/source/).
`toolchain/eshkol.lock` fixes Ubuntu 22.04/x86-64 and the Eshkol/LLVM
toolchain; it does not pin an N1 OpenSSL build, verifier, or deployment.

Before implementation or release, review and freeze the OpenSSL build recipe,
compiler, generated and system headers, flags and link mode; the built
`libcrypto` bytes, build ID, SONAME and complete ELF/loader closure; the actual
`default` provider bytes or proof it is built into that exact library; provider
selection and configuration (`OPENSSL_CONF`, `OPENSSL_MODULES`, property query,
search paths and relevant environment); and the verifier executable/library
bytes, ABI, symbol resolution and deployment image identity. The source pin
does not establish these binary pins. A missing, changed or ambiguous provider
returns HOLD. No host OpenSSL, other version, provider or HMAC fallback is
implicit.

## Verification ABI

The verifier receives only two distinct raw 32-byte Ed25519 **public** keys
from the frozen release. Reject equal keys, malformed lengths, changed key
hashes and caller/envelope/environment-selected keys. In a dedicated
`OSSL_LIB_CTX`, explicitly retain `OSSL_PROVIDER_load(ctx, "default")`.
Construct a verification-only key with
`EVP_PKEY_new_raw_public_key_ex(ctx, "ED25519", "provider=default", key, 32)`.
Allocate `EVP_MD_CTX` with `EVP_MD_CTX_new()`, then call
`EVP_DigestVerifyInit_ex(mdctx, NULL, NULL, ctx, "provider=default", pkey,
NULL)` and one-shot `EVP_DigestVerify(mdctx, signature, 64, message, length)`.
Only return `1` verifies; `0`, errors and unsupported fetches are HOLD. Free
contexts and keys on every path. PureEdDSA requires a NULL digest and one-shot
message; do not prehash or use Update/Final. The official
[Ed25519](https://docs.openssl.org/3.5/man7/EVP_SIGNATURE-ED25519/),
[raw-key](https://docs.openssl.org/3.5/man3/EVP_PKEY_new/),
[provider](https://docs.openssl.org/3.5/man3/OSSL_PROVIDER/) and
[verification](https://docs.openssl.org/3.5/man3/EVP_DigestVerifyInit/)
manuals define these symbols and return rules. Any signer uses a matching
one-shot API only inside its own authority; the verifier has no private key.

## v3 bytes and authority

V3 explicitly supersedes the synthetic v2 HMAC envelope. Sign the exact
canonical **body bytes**. The v3 body is UTF-8 JSON with exact schema-allowed
fields and types; object keys sorted by ASCII byte order; arrays kept in
specified order; no whitespace; schema-restricted string bytes `0x20` through
`0x7e` except quote and backslash, without escapes; and unsigned decimal
integers from zero through `2^64-1` without leading zeros. Reject
floats, NaN, duplicate/unknown keys, alternate escapes and noncanonical
re-encoding. Raw evidence
and the separately reviewed release descriptor retain their own bytes and
SHA-256; signed bodies bind those digests rather than reserializing evidence.

```
receipt_message = ASCII("eshkol:e3:n1:broker-receipt:v3\0")
                  || uint64_be(len(receipt_body)) || receipt_body
seal_message    = ASCII("eshkol:e3:n1:export-seal:v3\0")
                  || uint64_be(len(seal_body)) || seal_body
```

The `\0` denotes one zero byte. Each envelope is canonical JSON under the
same rules, with exactly `schema`, `algorithm`, `key_sha256`, `body`, and
`signature`. `algorithm` is `Ed25519`; `key_sha256` is the role-specific raw
public-key SHA-256; `signature` is 64 bytes as lowercase hex. `body` is the
canonical JSON object whose exact byte span is signed. The receipt body's
exact fields are
`schema`, `release_sha256`, `image_id`, `attempt`, `target`, `device`, `inode`,
`files`, and `verified_record_sha256`. The seal body's exact fields are
`schema`, `release_sha256`, `image_id`, `attempt`, `broker_key_sha256`,
`seal_key_sha256`, `receipt_envelope_sha256`, `commit_record_sha256`,
`commit_device`, `commit_inode`, `issuance_token_sha256`,
`snapshot_path`, `snapshot_device`,
`snapshot_inode`, `snapshot_custody_sha256`, and `files`. Each `files` entry
has exactly `path`, `size`, and `sha256`, in the frozen source order. Body
schema strings are respectively `eshkol-e3-n1-receipt-v3` and
`eshkol-e3-n1-export-seal-v3`; digests and image ID use lowercase SHA-256
hex, attempt is exactly 32 lowercase hex characters, and path/target values
must match the frozen release. `device`, `inode`, and `size` are unsigned
integers. Envelope schemas are respectively
`eshkol-e3-n1-signed-receipt-v3` and `eshkol-e3-n1-signed-seal-v3`.
`snapshot_custody_sha256` hashes a separately reviewed exact-byte handoff
record naming the protected path, owner, device/inode, ancestor chain and
ordered files. The token itself stays in the separate delivery channel.
An envelope never chooses a trusted key; its
algorithm and key hash must match the frozen role-specific release values.

## Durable v3 file identity and size limits

This is a proposed **v3 source change**; the v2 receipt wrapper and commit
files remain untouched. The broker publishes `receipt-<attempt>.json` as the
exact canonical signed receipt **envelope bytes** above, with no extra outer
wrapper. The externally delivered receipt must equal that file byte for byte.
`signed_receipt_sha256` means SHA-256 of those exact file bytes, including
the signature; it never means a reserialized body or a v2 wrapper.

The broker next publishes `commit-<attempt>.json` as one exact canonical JSON
record with only `schema` = `eshkol-e3-n1-receipt-commit-v3`,
`release_sha256`, `attempt`, `signed_receipt_sha256`, `receipt_device`,
`receipt_inode`, and `issuance_token_sha256`. The receipt device/inode come
from the checked opened receipt file. The token is a fresh 32-byte out-of-band
value; the commit stores only its SHA-256. The broker uses protected
attempt-keyed no-replace publication, checked write/readback and parent sync
for both files, and returns the receipt and token only after checked commit
completion. A readable file after an uncertain step does not establish
issuance. The release freezes the protected receipt-store path and ancestor
chain; the commit does not select a new store.

The sealer verifies the exact receipt envelope bytes and checks the commit
record, release, token hash, receipt file device/inode and protected store.
Only then may it publish its seal by checked no-replace write, readback and
parent sync in the separately pinned seal store. `receipt_envelope_sha256`
in the seal is SHA-256 of the exact receipt file bytes;
`commit_record_sha256` is SHA-256 of the exact checked commit file bytes.
`commit_device` and `commit_inode` are the opened protected commit file's
identity after publication. The consumer requires these values, the external
receipt bytes and token, the checked commit bytes and both protected store
path/ancestor pins to match. The reference chain is acyclic: release pins
stores and keys; signed receipt binds release; commit binds signed receipt;
seal binds receipt and commit. No object in this chain hashes or contains a
later object.

Reject sizes above these **inclusive byte caps before allocation**, with no
truncation or partial parse: exact raw v3 release descriptor 65,536; each
receipt or seal body 16,384; each signed envelope, receipt file or seal file
32,768; and commit file 4,096. Check encoded size before publication and
`fstat` plus bounded read before parsing; a cap hit, short read, extra byte,
overflow or changed size is HOLD. Any later source implementation must retain
these caps or undergo a new reviewed contract revision.

The v3 verifier receives the exact frozen release descriptor digest and both
actual raw public keys (lowercase hex), with their SHA-256 digests, through an
independently reviewed protected release channel. It checks the descriptor's
raw bytes against that pinned digest before using either key. The production
bootstrap digest and channel are not provisioned, so verification remains
HOLD. A key ID alone is insufficient; no caller, envelope or environment may
inject or replace keys or release authority. Broker and seal private keys
remain in different authorities with reviewed UIDs, ACLs, mounts, `/proc` and
fd isolation. Replacing either key, verifier/provider pin, trust anchor or
custody pin requires a newly reviewed descriptor digest and a new attempt
namespace, with new receipts and seals. An old release can be checked only
under its own exact pinned digest; there is no implicit overlap or
alternate-key retry.

## Remaining gates

The sealer verifies the broker receipt with the broker public key, then checks
protected receipt/commit, returned-token evidence, immutable handoff and live
bytes before signing. The consumer verifies both roles with frozen public
keys and checks protected stores and snapshot throughout use. Uncertain
write/sync/return or undelivered receipt, token or seal remains HOLD;
reconciliation is read-only and cannot mint missing authority.

Generated tiny OCI negatives must reject role/domain/key swaps, a shared key,
malformed or noncanonical signatures/bodies, unknown algorithm, foreign
release/image, provider substitution/failure, forged seal references, drift
and duplicate attempts. Also reject oversized/truncated release, body,
envelope, receipt, commit and seal bytes; mismatched exact external receipt,
token hash, receipt/commit digest or inode. Inject receipt/commit/seal write,
rename, readback, sync, delivery and crash faults, including readable bytes
before checked return. Check post-seal same-inode mutation and replacement
before and during consumer use. The live cross-UID and immutable-custody gates
in the seal contract remain required; no real f31 layer, Docker or deployed key is used
by this docs-only decision.
