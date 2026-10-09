# Public G2 installed caller and F artifact attestation — source only

This leaf adds an API-only installed G2 caller using the existing public
facades. It derives a first EOS byte from observed public raw text under a
legal initial EOS policy, then tests one- and two-byte terminal results.
The public API has no reviewed I1 word-copy accessor; the separate white-box
fixture checks exact ID words. The caller is staged, not compiled or run.

`tests/g3c4/f_artifact_attest.py` is a test-only producer for a **new**
attestation over a fresh copy of selected F artifacts. The historical F gate
and its original result remain failed. The attester pins F's original full
inventory, four ordinary copied compiler/cache/runtime artifacts, real
configure/build argv and exit receipts, clean compiler commit/tree, one exact
external fe9-to-6803 lock, a full copied project source inventory, the three
exact `common.sh` substitutions, and actual image library resolution. It
authenticates all ten original generated LLVM symlinks across five build
directories and excludes the two selected build-off links from the copy. Missing
image pins or loader receipts fail closed. Its compatibility TSV is explicitly
new and test-only; F and `bootstrap-eshkol.sh` did not produce it.

The source checker and owned negative tests exercise wrong/missing identities,
dirty source, changed cache/receipts, partial or altered copies, loader and
library drift, wrong external lock, copied-source extras, and fixture closure.
No Docker, native build, installed AOT, sanitizer, emitted symbol inventory,
mathematical comparison, or heavy gate has run. ROOT must separately accept
the offline F observation, exact image/input graph, source/test review,
physical lifecycle owner and one heavy-slot admission before any compiled
claim. Production `scripts/common.sh` and `toolchain/eshkol.lock` remain exact.
