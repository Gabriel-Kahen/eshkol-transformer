# Private P2/G2 terminal pending allocation anchor

Status: bounded private candidate with a focused compiled gate; independent
result review and integration remain pending. This leaf does not call native
terminal commit, snapshot, or clone and cannot publish a terminal output.

The accepted native P2/G2 terminal API requires the Eshkol caller to retain
the original pending output and exact raw-header backing allocations before
sampling. The existing protected scope reserves a genuine capacity-two pending
output, but its abort-only record has no distinct second raw/staging pair or
logical text[1]/text[2] allocations. The private extension
`native/g3c4_p2g2_terminal_pending_anchor_extension.esk` adds those allocations
after authenticated reservation and before any frame. It roots a working
record and separate allocation-time anchor, binds the original protected
record/anchor, pending output shell/native owner, first and second carrier
identities, and expected header-plus-payload sizes 10/24, 9/16, 9/16, 9/10.
The four newly allocated second/text objects and exact list nodes/successors
are also captured in a lexical root inaccessible to the diagnostic record.
Paired replacement of mutable records cannot change those originals.

`g3c4-p2g2-terminal-pending-check` is read-only. Its status is valid only
while the authentic protected call and pending output are live. Equal-length
working-record substitutions, changed expected sizes, output drift, foreign
calls, replaced list nodes or successors, and duplicate reservation reject
before mutation. The pending-only abort
checks the complete anchor before invoking accepted native abort through the
protected scope. A held genuine I1 lease leaves the anchor and text unchanged;
successful abort scrubs the second carriers/text and removes both anchor
nodes. The private diagnostic wrapper also cleans up a pre-sampling exception.

The existing protected and prefix wrappers remain unchanged and abort-only.
This leaf deliberately has no terminal-success path. A later separate
same-call coordinator must prove first/second authentic sampling and T1/EOS,
the native-zero fail-stop tail, a surviving live terminal anchor/output, and
detached snapshot/I1 clone. Actual bytevector backing extents and native
pointer conversion are not proven by this leaf. The new second/text allocation
failure cuts are source-reasoned through the inherited protected wrapper, not
compiled failpoint-tested here. Native terminal and clone source acceptance
alone does not establish any of those Eshkol claims or public G2 admission.

On candidate source based at `c71d39a984`, the pinned `fe9dfd5`
Eshkol runtime in Ubuntu 22.04/Clang 21 compiled the private and feature-off
fixtures in normal and ASan/UBSan/LSan modes. Private normal, repeat and
sanitizer runs each reported 23 checks; feature-off runs each reported four.
Each mode's stdout matched byte for byte and all compile/run stderr files were
empty. The static checker passed seven source mutations. The runner exited
128 only at its final `git diff --check`, because the container mounted the
worktree files without its external Git worktree directory. The same diff
check and the output/stderr comparisons passed in host postflight. Exact
source, runtime and logs are sealed in
`build/g3c4-p2g2-terminal-pending-anchor-20261114/SHA256SUMS`; the result is
a focused candidate gate, not public generation or terminal admission.
