# E3 N1 f32 generated scope and custody HOLD probe

Status: **fixed generated-byte diagnostic; protected custody and §9 HOLD**.
`tests/e3_private/e3_n1_scope_custody_probe.py` accepts no request or source
path. Its only actor output is generated role/cgroup text. It does not open
f32, OCI or Docker data, issue a broker receipt or seal, or change the public
`CAPTURE_RUN_ENABLED = False` and `REVIEWED_CAPTURE_SHA256 = None` gates.

## What the supported host actually exercised

On this host, a transient user `systemd-run --scope -p Delegate=yes` unit
provided a writable cgroup v2 subtree. A finite setup worker created a child
cgroup and used Linux `clone3(CLONE_INTO_CGROUP | CLONE_PIDFD)` to place two
fixed marker actors directly into it. The actors are named `broker` and
`sealer` solely to test the planned finite graph; neither performs a custody
or signing action. Each reported its actual cgroup, exited zero, yielded an
individual `wait4` peak RSS, and closed its pipe. After both were reaped,
`cgroup.events` reported `populated=0`; the worker removed the child cgroup.
A network syscall filter was installed before either actor launched. The
probe always returns `status=HOLD` and `resource_proof=HOLD`.

The scope and setup worker remain under the same caller UID. The delegated
user can rearrange processes within its delegated subtree; an empty child
cgroup plus two reaps proves closure of **these fixed actors**, not that an
unlisted process could not move elsewhere. The setup worker, `systemd-run`
and user manager are not all captured by per-process `wait4` peaks. The
result therefore does not establish a whole-procedure current-RSS bound or a
30-second completed-attempt interval. The preserved caps remain 30 seconds
and 268,435,456 bytes of **current RSS**; `memory.max` or `memory.peak` is not
used as a substitute. Missing delegation, `clone3`, pidfd, pipe EOF, actor
usage or empty-scope evidence yields HOLD before any source read.

## Why custody cannot pass here

The current login is UID 1000 with no passwordless elevation or deployed
independent broker/sealer service. A caller-created user namespace can map
subordinate IDs and demonstrate different file owners, but its creator owns
that namespace and retains capabilities there; the caller can assume the
mapped broker identity. That is not a protected distinct-UID authority
boundary. The probe therefore opens no generated source and reports
`protected_pre_read_custody=HOLD`, `source_opened=false`, and no receipt or
seal. No same-UID `0700` directory, namespace-only UID split, or checksummed
file is promoted to custody.

The next engineering gate is a reviewed authority outside the acquisition
caller's control: finite broker and sealer workers with distinct protected
credentials and stores, a durable HOLD intent before the first generated
source open, complete raw-stream preservation, protected receipt and seal
delivery, and a pinned process scope from setup through all workers' verified
cessation. It must include setup and authority workers in the time/RSS
accounting and test failed launch, migration, writes, syncs, readbacks,
delivery, and stuck actors. Until then the private-scope diagnostic and
custody HOLD cannot be combined into §9 proof or real-index authority.

Linux contracts: [cgroup v2 delegation and `populated`](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html),
[`clone3` placement/pidfd](https://man7.org/linux/man-pages/man2/clone.2.html),
and [user namespace ownership/capabilities](https://man7.org/linux/man-pages/man7/user_namespaces.7.html).
