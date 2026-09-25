# TR3-C private lease unenrollment

`tr3-lease-unenroll-internal!` accepts only the exact shell of an enrolled,
idle trainer. It returns `#t` after unlinking one record from the private
`tr3-trainers` registry. Subsequent use of that shell, including repeat
unenrollment, raises `invalid-argument` with operation
`tr3-lease-unenroll-internal!` (or the operation of another private consumer).
Malformed or copied shells also raise `invalid-argument`. An active trainer
phase or parent, active acquisition or another unenrollment, or a busy M3T,
O2 or D2 receiver raises `invalid-state` before the unlink.

The operation authenticates the exact record and retained receiver identities,
requires the existing M3T/O2/D2 idle admission, then reauthenticates and
checks registry head, list predecessor, exact shell identity and trainer phase immediately before
one list-link write. The private guard rejects same-thread unenroll/create
reentry and is cleared on every recoverable failure. Head, interior and tail
unlink preserve all other enrolled records; the former five-operand tuple can
be leased again as a new trainer shell.

Unenrollment ends only TR3's exclusive lease and registry retention. The
configuration, tokenizer, dataset, model and optimizer remain caller-owned and
live. It does not call D2 close, destroy O2/model storage, free arena memory,
or add a public `trainer-release!` facade. The pinned runtime's process-wide
root arena may retain bytes after the registry stops retaining a record; no
physical memory reclamation is claimed. The linked package localizes this
operation and rejects external C links to its raw symbol. It does not yet
make the package C-constructible: a same-package producer for
the five operands and a C handle/status contract remain separate gates.
