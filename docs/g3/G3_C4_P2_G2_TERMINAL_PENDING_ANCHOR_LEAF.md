# Private P2/G2 terminal pending allocation anchor

Status: source candidate. Compiled Eshkol normal/repeat/sanitizer admission is
not yet established. This leaf does not call native terminal commit, snapshot,
or clone and cannot publish a terminal output.

The accepted native P2/G2 terminal API requires the Eshkol caller to retain
the original pending output and exact raw-header backing allocations before
sampling. The existing protected scope reserves a genuine capacity-two pending
output, but its abort-only record has no distinct second raw/staging pair or
logical text[1]/text[2] allocations. The private extension
`native/g3c4_p2g2_terminal_pending_anchor_extension.esk` adds those allocations
after authenticated reservation and before any frame. It roots a working
record and separate allocation-time anchor, binds the original protected
record/anchor, pending output shell/native owner, first and second carrier
identities, and exact header-plus-payload extents 10/24, 9/16, 9/16, 9/10.

`g3c4-p2g2-terminal-pending-check` is read-only. Its status is valid only
while the authentic protected call and pending output are live. Equal-length
working-record substitutions, changed extents, output drift, foreign calls,
and duplicate reservation reject before mutation. The pending-only abort
checks the complete anchor before invoking accepted native abort through the
protected scope. A held genuine I1 lease leaves the anchor and text unchanged;
successful abort scrubs the second carriers/text and removes both anchor
nodes. The private diagnostic wrapper also cleans up a pre-sampling exception.

The existing protected and prefix wrappers remain unchanged and abort-only.
This leaf deliberately has no terminal-success path. A later separate
same-call coordinator must prove first/second authentic sampling and T1/EOS,
the native-zero fail-stop tail, a surviving live terminal anchor/output, and
detached snapshot/I1 clone. Native terminal and clone source acceptance alone
does not establish any of those Eshkol claims or public G2 admission.
