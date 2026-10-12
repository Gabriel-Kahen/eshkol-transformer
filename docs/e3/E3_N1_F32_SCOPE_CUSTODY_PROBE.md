# E3 N1 f32 generated scope and custody HOLD probe

Status: **fixed generated-byte diagnostic; protected custody and §9 HOLD**.
`tests/e3_private/e3_n1_scope_custody_probe.py` accepts no request or source
path. Its only actor output is generated role/cgroup text. It does not open
f32, OCI or Docker data, issue a broker receipt or seal, or change the public
`CAPTURE_RUN_ENABLED = False` and `REVIEWED_CAPTURE_SHA256 = None` gates.

## Recorded local witness

The generated-only result in
[`E3_N1_F32_SCOPE_CUSTODY_WITNESS.json`](E3_N1_F32_SCOPE_CUSTODY_WITNESS.json)
was captured from this worktree on 2026-10-12 with:

```sh
python3 -I -B tests/e3_private/e3_n1_scope_custody_probe.py probe > docs/e3/E3_N1_F32_SCOPE_CUSTODY_WITNESS.json
sha256sum docs/e3/E3_N1_F32_SCOPE_CUSTODY_WITNESS.json
```

The file SHA-256 is
`e92b1f2ef87dbea41598867548069d829b2f3931814628d0b6311479690a72ed`.
The local invocation used UID 1000, x86_64, Python 3.14.7, systemd 262,
`XDG_RUNTIME_DIR=/run/user/1000`, and cgroup v2
membership `0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-org.chromium.Chromium-122839.scope`.
Its broker and sealer markers each exited zero with pipe EOF and reported
direct placement in the same `codex-e3-n1-scope-1c30914ec31d48c7.scope/n1-generated-actors`
cgroup. Their individual `wait4` peak RSS values were 10,854,400 and
10,883,072 bytes; the worker reported `actor_scope_empty=true` after
reading `cgroup.events` and removing the child cgroup. The outer elapsed
time was 51,784,935 ns. These are observations from that invocation, not
guarantees for another host or run. A skipped live test or
`scope-worker-unverified` result gives no direct-placement witness for that
invocation.

In the recorded invocation, a transient user
`systemd-run --scope -p Delegate=yes` unit provided a writable cgroup v2
subtree. A finite setup worker created a child
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
