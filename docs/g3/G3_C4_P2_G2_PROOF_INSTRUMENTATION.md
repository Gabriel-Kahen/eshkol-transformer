# P2/G2 terminal-close proof instrumentation

This c7e-derived branch adds an opt-in capture path. Native and Eshkol
production source files are unchanged. An external one-use launcher must create
an empty private receipt directory and export `G3C4_PROOF_RECEIPTS` to enable
per-command capture in the terminal-close runner and its eight nested runners.
The default path calls each original command directly. In capture mode each
wrapped external command keeps its argv and cwd, records numeric exit and
monotonic wall interval, hashes argv/environment/raw streams, and relays both
streams. A raw stream over 16 MiB terminates the command and fails with exit
86. Shell builtins, command substitutions in `common.sh`, and subprocesses
inside Python/Eshkol programs are not individually wrapped. Generated C++
link wrappers are wrapped.
`check-g3c4-proof-source.py` rejects known unwrapped external forms in the
nine runner scripts. Each receipt carries its Bash file:line site, and
`check-g3c4-proof-receipts.py` requires every listed wrapper site and every
nested predecessor invocation after a full run. This is a conservative shell
audit, not an exhaustive Bash parser; `common.sh` toolchain probes and Python
subprocess internals remain outside it and keep total per-process completeness
on HOLD.

Set `G3C4_PROOF_EMIT=1` for emitted artifacts. Every normal and sanitizer
fixture, including feature-off and overlays, receives a separate
`--emit-object --dump-ir --emit-depfile` compile, raw object/bitcode/IR/depfile,
linked map, and relinked executable. The runner compares each relinked
executable bytewise with its direct-compile baseline. A mismatch fails closed.
This extra path changes the compiler invocation and cannot by itself prove the
direct build's emitted tail. Independent review must establish that the IR,
depfile, link map, symbol closure, and address trace support the compiled claim.

With receipt capture enabled, successful `build.*` trees are retained. The
external launcher must cap total evidence, own the private cgroup and daemon
container identity, establish both stream EOFs and a numeric direct wait, seal
the run, and quarantine any failure. No Docker run is part of this source change.
