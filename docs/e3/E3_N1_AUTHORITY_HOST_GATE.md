# E3 N1 protected authority host gate

Status: **deployment HOLD; no real source read authorized**. This 2026-10-12
review is a bounded
host and source review of the [attempt boundary](E3_N1_F32_ATTEMPT_BOUNDARY.md),
[broker candidate](E3_N1_BROKER_CANDIDATE.md),
[seal contract](E3_N1_SEAL_AUTHORITY_CONTRACT.md), and
[v3 provider decision](E3_N1_PROVIDER_DECISION.md). It supplies no installed
launcher, protected store, signing key, receipt, seal, or resource proof.
`CAPTURE_RUN_ENABLED = False` and `REVIEWED_CAPTURE_SHA256 = None` remain closed.

## Observed host authority

The login is UID 1000 with `CapEff=0`. `sudo -n true` reports `a password is
required`. Neither `eshkol-n1-broker` nor `eshkol-n1-sealer` exists in the
host user database. A generated-only probe,
`timeout 4s systemd-run --user --wait --collect --pipe -p User=nobody /usr/bin/id -u`,
exited `216/GROUP` without running `id`. The installed systemd is 262.
The installed OpenSSL is 3.6.4, whereas the reviewed v3 candidate pins a
source artifact for OpenSSL 3.5.9 and has no reviewed local build/provider or
verifier bytes. No real OCI or Docker data was accessed for these checks.

The [recorded scope witness](E3_N1_F32_SCOPE_CUSTODY_PROBE.md) demonstrates
direct placement, reaps, peaks, pipe EOF and empty child cgroup for two fixed
generated marker actors. Its delegated user scope stays under UID 1000; that
UID can rearrange tasks within the delegated subtree. The witness omits a
protected broker/sealer, setup manager peaks, durable records and final
delivery, so it cannot establish the complete attempt graph or custody.

The [installed systemd `User=` contract](https://man.archlinux.org/man/systemd.exec.5.en#User/Group_Identity)
does not allow a nonroot per-user manager to switch to another real UID. A
caller-owned user namespace also cannot make its mapped identities
independent of the caller. A system manager can launch other UIDs, but the
current attempt contract counts a manager that dispatches attempt work; a
persistent manager has no per-attempt final `wait4` peak. Treating it as an
unmeasured service would change that contract. Linux requires `CAP_SETUID`
and `CAP_SETGID` or equivalent privileged authority to assume unrelated
host identities; this login has no effective capability for that step.

## Minimum candidate for a later generated-byte deployment

An administrator would provision two static, nonlogin **host** UIDs with no
shared write groups, protected key material and root-owned ancestor paths.
The broker may write only its stage, HOLD-intent and receipt/commit stores;
the sealer may read checked broker records but may write only its distinct
seal store. The caller and consumer get no signing key, writable store fd,
mutable executable or authority to rename an ancestor. Review ACLs, mounts,
`/proc`, inherited fds and actual UID transitions, rather than relying on
`0700` names alone. The target must pass to a protected snapshot or equivalent
state after broker fds close, with no later broker/caller write path. This
requires an implemented and reviewed handoff, not only an inode/hash check.

A root-owned, immutable, finite per-attempt launcher would have to create a
private cgroup, directly launch the fixed broker, reader and sealer graph
under their reviewed identities, deny unapproved spawn and cgroup migration,
and retain authenticated per-process closure evidence. Any privilege entry
point and its configuration must be pinned and audited; an arbitrary `sudo`
command, setuid wrapper or persistent systemd service is not an approved
launcher. A generated-byte prototype must precede any real input.

Before any source open, the broker must create a unique protected `HOLD`
intent with no-replace publication, checked write/fsync/readback and parent
directory sync, then attest that checked completion over a protected channel.
Only then can the pinned reader send complete raw stdout, stderr and failure
bytes to the broker's bounded stage. Unknown byte count, stream break, fault
or sync leaves the protected prefix in HOLD. After target verification and
sync, the broker publishes a role-signed receipt and checked commit in its
protected store, reads both back, syncs the parent and returns the exact
receipt and fresh issuance token through a protected channel. The separate
sealer verifies that receipt using a **public** broker key, checked commit,
returned-token evidence and immutable handoff. It publishes a role-signed
seal with no-replace write, readback and directory sync, then delivers a
protected seal reference. A crash or uncertain delivery never promotes
readable records. Reconciliation stays read-only. The v2 HMAC implementation
cannot implement this split because its broker holds seal-minting symmetric
verification material; the v3 Ed25519 design is still docs-only.

For the frozen 30-second and 268,435,456-byte **current RSS** limits, an
independent witness must start one `CLOCK_BOOTTIME` interval before launcher
setup. Its endpoint follows final seal readback/sync/delivery, stream EOF,
every declared worker's cessation and final `wait4` usage, and a verified
empty process scope. Include the finite privileged setup/launcher, broker,
reader, sealer and any active supervisor, witness or helper. An independent
passive witness could be excluded only under a separately reviewed boundary.
Authenticate nested
`wait4` reports, obtain the setup worker's own final peak from its parent,
then conservatively sum **all** individual peaks in bytes. Only a closed,
nonmigrating finite graph with total peaks at or below 268,435,456 and a
completed interval at or below 30,000,000,000 ns could pass that post-run
test. A timeout request, `memory.max`, `memory.peak`, sampled `/proc` RSS or
two marker peaks cannot substitute. An uninterruptible task after the deadline
remains HOLD, even if killed later.

## Exact next gate

There is no safe deployable artifact yet: this login has no reviewed privilege
entry point available to create and launch protected real-UID
roles, and source lacks the finite
launcher, protected channel/snapshot, v3 signer/verifier and pinned build/key
deployment. First implement and independently review the generated-only
launcher, protocol, provider pins, fault tests and whole-attempt accounting.
The candidate host gate after that review is one administrator-backed
provisioning session for the two static UIDs, protected stores and narrowly
scoped launcher. Provisioning alone does not authorize a real read. No
current HOLD gate changes.

References: [systemd user identity](https://man.archlinux.org/man/systemd.exec.5.en#User/Group_Identity),
[Linux capabilities](https://man7.org/linux/man-pages/man7/capabilities.7.html),
[user namespace ownership](https://man7.org/linux/man-pages/man7/user_namespaces.7.html),
and [cgroup v2 delegation/migration](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).
