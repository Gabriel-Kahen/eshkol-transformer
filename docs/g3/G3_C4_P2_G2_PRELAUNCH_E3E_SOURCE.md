# Exact e3e private terminal prelaunch source

This successor supplies a host-only, one-use launcher for the detached
`e3ebca2d0e7122a8335029a0d94c8c04c8a20d4b` source tree
`38963dbcca449c59fcace054292309fd8aa8da44`. The external prepared
bundle contains its own immutable source, compiler, image, and resource pins.
Source-only inspection and fake/process controls do **not** authorize a Docker
dispatch, consume the real claim, validate the pinned image live, or establish
compiled terminal acceptance.
The current bundle pins `launch_enabled=false`, so `--launch` rejects before
claim. Any future runnable bundle needs a distinct seal and independent
lifecycle review after the hosted run is complete.

The launcher takes the fixed heavy-slot flock before reading the prepared plan
or writing results. It authenticates the prepared manifest, creates and fsyncs
one exclusive `CLAIMED`, then records `UNSEALED_QUARANTINE` immediately in
`RESULT`. A consumed claim is never retried. A finite whole-gate alarm starts
at claim. Source Git is checked through a sanitized absolute Git executable;
Docker CLI commands use a pinned rootless Unix socket and an isolated empty
config with inherited Docker/Git/Python override variables removed. The source
mount is read-only and contains a real, empty ignored `.deps` mountpoint;
the nested compiler tree is read-only. Container root is read-only, networking
is disabled, and memory, CPU, PID and temporary-storage limits are explicit.

A private cgroup bootstrap joins before the CLI executes. Every CLI invocation
retains exact argv/cwd/environment, a signed direct-child identity and numeric
wait, and separate raw stdout/stderr through EOF. The daemon's unique claim
label listing supplies the CID; create stdout is only a consistency check.
Before start, attach, cleanup and recovery, inspection must match the pinned
image, command, entrypoint, user, working directory, environment, labels,
mounts and material host settings. No command uses `--rm` or name-only cleanup.
The guardian inherits the heavy-slot descriptor. It holds that slot through
the result seal and quarantines it after parent loss; recovery requires signed
parent absence and, for an unsealed run, the guardian's quarantine state.
Ambiguous CID, incomplete raw wait/EOF, changed cgroup identity, an observed
same-UID result writer or uncertain cleanup stays physically unsealed.
Recovery is itself serialized and one-use. The writer check is a current
`/proc` snapshot; live descendant and daemon closure remain unrun.

The source-only controls exercise a real guardian child, lock/claim races,
preflight retention, a real bootstrap command/timeout, inspected-config
mutations, environment poisoning, daemon-issued CID mismatch, and seal drift.
They do not run the pinned Docker daemon or prove live cgroup delegation and
guardian recovery on this host. Exact daemon `inspect` default-field
normalization must be calibrated in a separately reviewed, non-dispatch live
check before any launch-enabled seal. The terminal runner's predecessor subtree and
deeper checker calls still lack complete per-command receipts. Retained IR and
symbol inventories remain insufficient for successful-zero no-allocation and
transitive linked-import proof. These are explicit admission HOLDs. A sealed
guest exit records execution and preserves evidence; it does not remove them.
