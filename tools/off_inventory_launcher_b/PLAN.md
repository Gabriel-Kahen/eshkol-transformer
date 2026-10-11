# OFF inventory launcher B2 and Runner J source candidate

Source and synthetic tests only. Launcher A and Runner H seals and reviews stay immutable. This version repairs A's protected-code/output-permission contradiction by keeping the runner and evidence BASE behind root-owned, nonwritable ancestors while the privileged launcher creates one fresh output child under BASE, changes that child to the invoking UID/GID with mode 0700, and binds its device/inode to a one-use handoff. The output parent remains protected; the caller owns only the output child.

The launcher claims a fixed ledger name in a root-only 0700 directory with `openat(O_CREAT|O_EXCL|O_NOFOLLOW)`, writes the authorized binding digest, and fsyncs the file and directory before output creation. A failure after this point consumes the attempt. It then creates a root-owned sealed memfd containing the binding digest, caller UID, PID and process start time, output device/inode, and random nonce. The descriptor is inherited as FD 3 across the privilege drop and `execve`. Runner I accepts only that sealed root-created descriptor and an empty 0700 caller-owned directory with the pinned inode. The root-private ledger blocks a second launcher; the immutable handoff cannot be rewritten for a direct Python invocation. A direct invocation without the handoff fails before release work.

Protected installation, the root-private authorization file, external binding, Runner I independent/ROOT source reviews, runtime ROOT RELEASE authority, and one-use execution are absent. The current user-owned evidence tree deliberately fails `protected_path`. Root/mount actors, ptrace into the same running process, and the final protected-install identity are separate trust gates. No F/G product bytes or parity execution are part of this candidate.

Runner J adds an internal child handoff to close I's direct `--child` route. The
admitted `--run` process retains launcher FD3 after validating its sealed
root-owned record, passes only FD3 to each child, and closes its copy when the
run ends. Every `--child` entry requires FD3 before release review, metadata
admission or product parse. The child checks the record's binding digest,
caller UID, live parent PID/start time, and output directory device/inode;
it then closes its copy. The argv binding digest alone grants no authority.
Launcher B2 remains pinned to I and is not an installed J launcher. J's
release, binding and review paths are versioned `-j` and intentionally absent.

This is a source-only handoff improvement, **not a one-use runtime release**.
`/proc/<pid>/fd/3` reopening of the root-owned `0600` memfd should be denied
by inode permissions, but `pidfd_getfd` can duplicate an open fd if the actor
has ptrace attach rights; the invoking same-UID caller is an ancestor of the
runner on the current Yama mode 1 host. Unrestricted ptrace into the run
process defeats in-process handoff enforcement. The protected installation
and anti-ptrace or separate-UID authority gate must be designed and reviewed
before J could claim replay exclusion. No source-only test here proves that
runtime boundary.

## Same-UID authority boundary: current-host HOLD

The bounded J handoff cannot be made replay-safe by changing only Python or
launcher-local process flags on this host. B2 drops all real/effective/saved
IDs to the invoking UID, then execs ordinary `/usr/bin/python3` with FD3 open.
Linux resets dumpability to 1 for that exec (confirmed by the synthetic
`test_plain_python_exec_resets_nondumpable_state`), so a `PR_SET_DUMPABLE(0)`
before exec is lost. Calling it in Python leaves the interpreter startup
window. The invoking process is an ancestor under this host's Yama
`ptrace_scope=1`; a synthetic ancestor `PTRACE_ATTACH` succeeded here. The
`pidfd_getfd` permission check also uses ptrace attach rights, though other
host controls may deny that syscall. Ptrace control of the runner can alter
its flow or spawn an admitted child, defeating the in-process lineage check;
a copied FD3 alone does not pass that check. The same gap recurs when
`run_child` execs ordinary Python with `pass_fds=(3,)`.

The one-use root ledger still limits *launcher admissions*, but it cannot
make the exposed descriptor exclusive to the admitted Python processes.
No source-only test establishes a protected runtime. A defensible next
candidate must cover both runner and parser children: for example, a
root-verified Yama `ptrace_scope>=2` installation gate (with privileged
ptrace actors in the trusted base), or a separately identified worker UID
with reviewed input access and output handoff. Neither condition exists in
the current installed evidence tree; B2 also still pins Runner I. Do not
issue external binding or ROOT RELEASE on J from this source candidate.

The output child is owned by the invoker during execution. That invoker can
modify or remove files there even if ptrace is excluded, so a future release
must either protect work products until completion or explicitly exclude
caller-controlled output bytes from its authority claim.
