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
