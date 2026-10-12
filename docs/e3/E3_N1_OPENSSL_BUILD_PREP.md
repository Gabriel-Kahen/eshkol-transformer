# E3 N1 OpenSSL 3.5.9 source and build preparation

Status: **SOURCE/PREP only; HOLD** on the accepted provider decision and seal
authority contract at `c8c49e4a`/tree `d8075e70`. This document creates no
OpenSSL build, verifier, keys, prepared execution marker, release, export, or
N1/M1 admission. Every binary hash and live observation below is **unmeasured:
HOLD**. The [provider decision](E3_N1_PROVIDER_DECISION.md) fixes PureEd25519,
v3 bytes, verification ABI, and the source archive; it does not pin executable
bytes. The [seal authority contract](E3_N1_SEAL_AUTHORITY_CONTRACT.md) still
requires separate receipt and seal authorities and immutable export custody.

## Source and later build recipe

The sole source candidate is the official `openssl-3.5.9.tar.gz`, SHA-256
`603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a`.
Its upstream checksum is linked in the provider decision. Also require the
detached release signature, verified against a **separately trusted** OpenSSL
signing-certificate primary fingerprint
`B146 647E 45A7 B339 47AB 226B 2A2C 87D1 6169 2D40`, as documented on the
[official download page](https://www.openssl-library.org/source/). A valid
signature using a key fetched alongside the tarball without an independently
accepted fingerprint does not establish authenticity. Record the exact
signature and certificate bytes, signing subkey, verification tool/argv and
result; if trust or signature cannot be established, HOLD. A later isolated,
network-disabled build must compare the **raw archive bytes** with the digest,
verify that detached signature, and reject any mismatch, unsafe member path,
link escape, or unexpected archive top-level name before extraction. A Git
checkout, distro `libssl-dev`, or another OpenSSL version is not a substitute.
Record the archive and extracted source inventory hashes before configuration.

The build lane is the `toolchain/eshkol.lock` Ubuntu 22.04 x86-64 lane with
LLVM/Clang 21.1.8. It needs the OpenSSL 3.5.9
[INSTALL.md](https://github.com/openssl/openssl/blob/openssl-3.5.9/INSTALL.md)
prerequisites, including Perl and `Text::Template`, and the
[Unix notes](https://github.com/openssl/openssl/blob/openssl-3.5.9/NOTES-UNIX.md).
Freeze the exact image digest, package versions, tool paths and hashes, build
UID, protected absolute source/build/install/config paths, `--prefix`,
`--openssldir`, `--libdir`, `CC`, `CXX`, compiler/linker flags, feature options,
environment, job count, and link mode **before** executing Configure. The
proposed candidate is `linux-x86_64`, shared `libcrypto.so.3`, an isolated
prefix, and `--libdir=lib`; these are review inputs, not verified build pins.
The install path must never replace system OpenSSL. Do not silently add a
feature-disabling flag or inherit `CFLAGS`, `CPPFLAGS`, `LDFLAGS`, `PERL5LIB`,
or other host build settings. Freeze the complete environment, including
`PATH`, locale and reproducibility variables, as exact data.

After ROOT freezes those inputs, the **later** build gate executes this fixed
order in a fresh protected stage; each step records exact argv, environment,
exit status, stdout/stderr hashes, and input/output identities:

1. Verify archive SHA-256 and detached signature against the separately trusted
   fingerprint, check safe extraction, then record source inventory.
2. Invoke `./Configure linux-x86_64 shared --prefix=<frozen-prefix>
   --openssldir=<frozen-openssldir> --libdir=lib` with the frozen compiler,
   flags, feature options, and environment. Preserve `configdata.pm`, generated
   `Makefile`, generated headers, and their hashes. Any added option is a new
   review input. The placeholder paths are intentionally not executable pins.
3. Run frozen `make -j<jobs>`, `make test`, then `make install_sw` into only the
   isolated prefix. Capture the full test result. Failure or skipped tests is
   HOLD, not an alternate recipe. No network fetch is permitted.
4. Compile and link the later verifier against only that installed OpenSSL
   header/library set with frozen argv, link search and runtime search policy.
   Verify symbol versions and loaded library identity before any signature
   result is accepted. The verifier source/ABI and executable are separate
   review items; this document does not implement them.

The OpenSSL default provider is built into `libcrypto`; the
[provider manual](https://docs.openssl.org/3.5/man7/OSSL_PROVIDER-default/)
defines its `provider=default` property. A later runtime must still prove it
explicitly loaded `default` into its dedicated `OSSL_LIB_CTX` and fetched
Ed25519 with that property. `OPENSSL_MODULES` is recorded and constrained as
part of the environment, but its value alone does not establish a different
default-provider implementation. No host OpenSSL, alternate provider, HMAC,
or implicit fallback is allowed.

## Immutable evidence manifest required before a prepared gate

Produce a versioned, canonical, checksummed, non-executable manifest rooted in
one reviewed build-input digest. Each path entry records its role, absolute
path, file kind, mode, owner, symlink chain, byte size and SHA-256. Record
`UNMEASURED/HOLD` for an unavailable value; do not copy a candidate hash into
an observation field. The reviewed manifest and sealed raw evidence must bind:

| Closure | Required exact evidence |
| --- | --- |
| Source/build tools | Archive URL, raw digest, detached signature and independently trusted signing fingerprint; extracted tree inventory; image/OS/architecture digest; exact `toolchain/eshkol.lock`; `Configure`, Perl and `Text::Template`, Make, Clang/C++ driver, linker, assembler, archiver and `pkg-config` paths, versions and hashes; all build argv/env; `configdata.pm`, generated `Makefile`, flags and complete command log. |
| Headers | Actual compiler include search order and every direct/transitive header read for each OpenSSL and verifier translation unit, including generated OpenSSL, installed OpenSSL, compiler resource, libc and other development/system headers. Cross-check depfiles against a full include/file-open trace; hash bytes and symlink chains. A package/version list or host candidate is insufficient. |
| Installed binaries | `libcrypto.so.3` bytes, SHA-256, ELF build ID, SONAME, architecture, exported and consumed symbol versions; `libssl` and `openssl` if produced; installed generated headers and any module bytes. Record the exact verifier source/tree, compile/link argv, executable bytes/build ID, ABI (public key and signature lengths, signed message construction, return semantics), undefined symbols, relocations, and resolved symbol providers. |
| Provider/config | Source-backed built-in default-provider proof plus runtime provider name, explicit load/fetch results and actual `provider=default` selection. Pin `OPENSSL_CONF`, `OPENSSL_MODULES`, config file bytes/ancestry, module search policy and any attempted module loads, property query, `OPENSSL_CONF_INCLUDE`, `OPENSSL_ENGINES`, and all other relevant environment. Missing or ambiguous provider resolution is HOLD. |
| ELF/loader | Interpreter and loader bytes/build IDs, all `DT_NEEDED`, `RPATH`/`RUNPATH`, `ld.so` configuration/cache and search order, direct/transitive dependency paths, hashes, modes, symlinks and symbol resolution. Preserve a controlled actual loader observation and process maps, separately from `readelf` candidates. Pin `LD_LIBRARY_PATH`, `LD_PRELOAD`, `LD_AUDIT` and namespace/mount routes; reject unreviewed preload or dependency substitution. |

The later evidence collector must compare discovery to the frozen input plan
and fail on missing, duplicate, added or changed paths, unknown provider loads,
wrong SONAME/version, host-library resolution, different include closure, or
unresolved symbols. `readelf`/depfiles alone cannot prove runtime resolution;
a successful Ed25519 test alone cannot prove library identity. The exact
reviewed loader/provider and verifier pins must enter a new frozen release
descriptor through the protected release channel before any v3 use. All such
pins are currently **HOLD**.

## One-use prepared execution gate (design only)

ROOT reviews the complete source/build-input manifest and supplies a fresh
unpredictable nonce through a protected channel. The later preparer binds
`(schema, accepted base commit/tree, source SHA-256, build-input manifest
SHA-256, nonce, authorized next step)` in a canonical prepared record in a
protected directory. A separate exact-attempt `O_EXCL`/no-replace consumption
record is durably written, read back, and parent-synced **before** the later
build starts. The gate rejects missing review, reuse, mismatch, preexisting
marker, changed ancestors, uncertain write/readback/sync, or a second caller.
After a crash or uncertain consumption it remains spent/HOLD; a new nonce and
review are needed. No marker or nonce is created or consumed by this document.
Prepared means only that one exact build attempt may start; it conveys no
receipt issuance, seal authority, release admission, or N1/M1 status.

## Subsequent synthetic and negative gates

Use generated tiny OCI fixtures and **synthetic** Ed25519 key pairs with
separate broker/seal private keys held only by their respective test
authorities. The verifier receives only two distinct frozen raw 32-byte public
keys. Prove a positive PureEd25519 result against an independent
[RFC 8032](https://www.rfc-editor.org/rfc/rfc8032) vector or separate oracle,
then test valid role-separated v3 receipt and seal messages;
swapped/shared/substituted keys, key hashes and domains; malformed public-key
length, signature length, JSON, duplicate fields, noncanonical body and
oversized/truncated records; wrong release/image, receipt/commit/token digests
and inode; and `0`/error/unsupported EVP return outcomes. Fault inject
provider load/fetch failure, wrong `libcrypto`/symbol/loader path, config or
module-policy drift, and unreviewed ELF/header bytes. Those return HOLD,
without falling back. A test must demonstrate explicit default-provider
selection from the pinned library; an environment-only `OPENSSL_MODULES`
change is not treated as proof of provider substitution.

Fault the one-use marker at create, write, readback, rename/no-replace, sync,
delivery and crash boundaries. Test two concurrent consumers of one nonce,
replay after success, changed manifest/base/source, ancestor replacement,
and readable marker bytes before checked completion. All uncertainty is HOLD.
The separate v3 successor must also test the receipt/commit/seal and immutable
snapshot fault matrix in the seal authority contract. Synthetic success cannot
stand in for cross-UID isolation, real keys, authentic f31 inputs, or N1/M1
admission.
