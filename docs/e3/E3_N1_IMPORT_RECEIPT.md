# E3 N1 numerical import receipt verifier

This is a noninstalled, candidate-specific **pre-admission** checker. It does
not change the E1B import policy, the numerical package, or the evaluator.
Numerical F at 94b8fecd/tree d28cffc1 failed before AOT/runtime when the
sanitized object added bcmp and puts. N1 and M1 remain open.

The script scripts/verify-e3-n1-import-receipt.py pins F's source tree,
image, compiler, builder, six retained object/IR hashes, and exact 39-call
relocation roster (14 bcmp, 25 puts). It requires per-call final producer,
ABI, declaration and provider fields; immutable raw argv/environment,
relocation, disassembly, optimized IR, source, header, provider-symbol and
loader/link evidence; all 17 F depfiles; a sorted complete header trace with
matching SHA-256 bytes; and pinned provider files and closure. It rejects
duplicate JSON keys, missing or changed bytes, changed pins/calls, unresolved
file symlinks and incomplete provider records. A structurally complete receipt
still needs independent semantic review of the raw evidence before any
tuple-scoped source admission.

The separate immutable candidate record at
/home/gabe/.codex/evidence/eshkol-transformer/e3-section9-n1-receipt-candidate-20261011-a/candidate.json
holds the exact F pins, object hashes, 39 observed relocations and 17 depfile
hashes. It intentionally leaves unavailable final producer/ABI, declarations,
complete system-header trace, actual provider bytes and resolved loader
closure absent. The verifier must reject it. The F private.ll declares
printf but not puts; this does not prove the final 25-call lowering.
libprivate.so's GLIBC_2.2.5 requests do not identify the supplying library.

Synthetic negative tests run with:

    python3 -m unittest tests.e3_private.test_n1_import_receipt -v

These tests exercise the checker's structure and drift rejection. They do not
execute a compiler, linker, loader, Docker, provider or numerical gate, and
their complete synthetic fixture is not an N1 admission receipt. The next
gate is actual image/candidate producer, ABI, provider and complete dependency
bytes plus independent review. Only then may a separate scoped import-policy
change and fresh full Numerical F gate be considered.
