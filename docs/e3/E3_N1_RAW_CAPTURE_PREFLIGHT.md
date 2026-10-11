# E3 N1 image-specific capture preflight (source candidate)

Status: **inert, noninstalled, HOLD**. This source is a preflight and raw
receipt structure checker, not a Docker launcher or a semantic ABI review.
`scripts/e3-n1-raw-capture.py` never starts a subprocess, executes a guest
file, calls a provider, or changes import admission. Its source-reviewed plan
SHA-256 is deliberately `None`; every production invocation writes a bounded
`HOLD` receipt and exits nonzero. N1/M1 remain HOLD.

The preflight binds the exact failed F source head/tree, fe9 compiler commit,
f31 image config digest, failed seal hash and F private/native object hashes.
It requires a separately reviewed JSON plan whose SHA-256 must later be
compiled into the source. The plan must contain OCI layout path, index SHA-256,
indexed manifest descriptor, config descriptor, ordered layer descriptors and
uncompressed diff IDs. It hashes each blob while streaming and checks the
selected guest tool, interpreter, loader, provider and header file entries
against final layer entries, including modes, symlink targets and selected
file bytes. Uncompressed OCI tar layers alone are supported in this candidate;
gzip/zstd layers fail closed pending a reviewed streaming implementation.
Selected guest symlink targets must also appear in the pinned selection.
Whiteouts affecting a selected path fail closed. This scanner does not claim
a complete overlay filesystem reconstruction or actual dynamic-loader
resolution.

The plan also requires all seven F host/guest mount routes, including a fresh
`/out` route, mount-root `lstat`/resolved path records, selected host tool,
interpreter and loader bytes, and explicit positive time, blob, selected-byte,
raw-record, output and RSS caps. Preflight reserves twice the maximum output
size in free disk space. It streams retained F objects and raw records;
resource and partial-failure conditions emit `HOLD`, never a truncated pass.
The reviewed plan must spell out direct argv arrays, environment, per-command
time/output caps and a pinned guest tool for call relocations, final producer,
ABI/declarations, provider ELF and symlinks, loader observation, system
headers and argv/env. These are declarative records only; this source cannot
run them. A future runner must separately enforce those caps and prove the
tool and loader selected from the pinned image were actually used.

A later capture receipt must carry 39 unique `(object, section, offset)` call
rows distributed 25 `private.o` puts, 2 native-5 bcmp, 12 native-6 bcmp. Each
row must reference raw producer, ABI and declaration evidence. The receipt
must also contain provider graph, actual loader observation, complete
direct/transitive system-header inventory, NUL-delimited per-object argv/env,
declarations, F object parity and immutable raw path/hash/size records. The
checker enforces presence, unique row identity, F object parity and raw byte
hashes. **It does not prove those claims true or complete.** Independent
semantic review must inspect raw records and pin a reviewed claims hash in a
separate source change; actual loader observation and any replay require
separate source review and a free heavy slot. A replay attributed to F must
match all three retained F object hashes and relevant depfile hashes.

No OCI export, indexed manifest/layers, resolved guest tool/loader/provider
bytes, host capture-tool pin, complete header trace, per-object argv/env or
actual loader map was present in the accepted F/plan evidence. Therefore no
capture command is enabled. The next gate is independent source review of
this preflight and a proposed exact f31 OCI export plus host/guest input pin
file; only then may a separate bounded capture runner and its heavy execution
be considered. Import-policy review and a fresh full Numerical F gate remain
later gates.
