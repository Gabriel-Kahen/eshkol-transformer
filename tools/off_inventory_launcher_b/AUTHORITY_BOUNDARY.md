# OFF inventory authority boundary: blocked design

This is a design finding, not an accepted launcher or runner API. Runner J's
FD3 and child lineage checks remain source-only. B2 pins Runner I. No external
binding, protected install, ROOT RELEASE, or F/G product parity is authorized
by this note.

## Current-host evidence

The launcher drops all three UIDs to the invoker before ordinary Python
`execve` with FD3 open (`LAUNCHER.c`); `run_child` also execs ordinary Python
with FD3 (`RUNNER.py`). A synthetic test confirms that ordinary exec resets
`PR_SET_DUMPABLE(0)` to dumpable 1. On this host, Yama `ptrace_scope=1`, the
invoker has no effective capabilities, and a synthetic same-UID ancestor
successfully attached and duplicated a child's open descriptor with
`pidfd_getfd`. This proves the current host supplies no FD3 exclusion at
either Python exec boundary. It does not claim that a protected launcher was
installed or run.

The host-only probe and its JSON result are preserved at
`/home/gabe/.codex/evidence/eshkol-transformer/g3r-cold-s-dual64-tagged-native-off-authority-boundary-probe-20261011-k/`.
`SHA256SUMS-PROBE` has SHA-256
`893b9bb1947376e008551488a1adc0740d9dfe6ee26ee2d24e329c0840247a89`.

The launcher currently chowns its fresh output child to the invoker before
work. The runner expects that ownership, writes the result there, then uses
`chmod(0444/0555)`. A synthetic check shows the owner can restore write
permission and replace result bytes. Device/inode checks establish which
directory was opened; they do not authenticate its contents against its
owner. A caller-owned child beneath a root-owned parent can still be changed
by the caller. A caller-owned parent also lets the caller rename its child.

The current source/evidence paths pass through `/home/gabe` (UID 1000, mode
0700), and the F/G run directories are UID 1000, mode 0700 with mode 0600
manifests. They fail B2's root-owned-ancestor check and cannot be read by a
distinct worker UID. This process is UID 1000 with no effective capabilities;
unattended `sudo -n` requires a password. A protected runtime cannot be
installed or exercised from this checkout.

## Smallest defensible next contract to review

1. A root-owned, nonwritable installation contains a versioned launcher,
   runner, parser, Python closure, reviewed binding and input snapshot.
   Stage the sealed F/G inputs behind access granted to a dedicated nonlogin
   worker UID; verify staged bytes and paths before parser admission or parity
   comparison. Staging itself is outside this source task.
2. A privileged launcher records the **invoking UID/GID** separately from the
   **worker UID/GID**, claims and fsyncs the one-use root ledger before forking,
   and binds the exact external digest, invoker identity, worker identity,
   output device/inode and process identity into its sealed handoff. Runner
   and parser children both run only as the dedicated UID, without ambient
   capabilities or supplementary groups. They never share the invoker UID.
3. Keep the working output directory owned by the worker, mode 0700, under
   ancestors that are all root-owned and nonwritable to the invoker. The
   invoker must have no write path to it during execution. A privileged
   supervisor waits for runner and parser descendants to exit, validates the
   pinned directory and regular,
   single-link files through directory FDs, and publishes the canonical
   result into root-owned protected storage. Any caller-owned export is a
   mutable copy whose bytes must be checked against that protected result at
   consumption.
4. Version and review the new launcher/runner pins and UID/output checks
   together. A protected runtime test must show the exact invoker cannot
   `PTRACE_ATTACH`, `pidfd_getfd`, open FD3 through `/proc`, or modify working
   output through any ancestor for both parent and parser child during startup
   and execution. It must prove no worker writer remains before publication.
   Preserve direct/missing/stale-FD negatives and consumed-ledger behavior.

A root-verified Yama `ptrace_scope>=2` gate could protect a same-UID runner
and child from unprivileged attach, but this host is at mode 1 and that gate
alone does not protect caller-owned output. A `PR_SET_DUMPABLE` call before
ordinary Python exec is reset; a call inside Python leaves startup exposed.
An unprivileged user namespace mapping only the invoker's host UID does not
establish an inaccessible worker identity against that invoker. The next
prerequisite is a root-provisioned protected installation and input/output
staging contract; only then can runtime
authority be tested. Do not read or stage F/G products for this source task.

Kernel contracts: [execve(2)](https://man7.org/linux/man-pages/man2/execve.2.html),
[pidfd_getfd(2)](https://man7.org/linux/man-pages/man2/pidfd_getfd.2.html),
and [Yama](https://docs.kernel.org/admin-guide/LSM/Yama.html).
