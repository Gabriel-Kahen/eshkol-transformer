# E3 N1 numerical import receipt verifier

This is a noninstalled, candidate-specific **pre-admission** checker. It does
not change the E1B import policy, the numerical package, or the evaluator.
Numerical F at 94b8fecd/tree d28cffc1 failed before AOT/runtime when the
sanitized object added bcmp and puts. N1 and M1 remain open.

The script tests/e3_private/verify-e3-n1-import-receipt.py pins F's source tree,
image, compiler, builder, six retained object/IR hashes, and exact 39-call
relocation roster (14 bcmp, 25 puts). It requires per-call final producer,
ABI, declaration and provider fields; immutable raw argv/environment,
relocation, disassembly, optimized IR, source, header, provider-symbol and
loader/link evidence; all 17 F depfiles; a sorted complete header trace with
matching SHA-256 bytes; and pinned provider files and closure. It rejects
duplicate JSON keys, missing or changed bytes, changed pins/calls, unresolved
file symlinks and incomplete provider records. The 17 depfile paths and hashes are now
compared against the independently sealed F forensic receipt
(SHA-256 0b47f72d89d770975d5583a26964e8a36424f59ebeb674d4d9acef5abae6ef16);
changing a depfile and updating its candidate hash is rejected.

Candidate A's independent/root review held its broader verifier claim: nonempty
ABI/provider/version changes and self-rehashed depfiles could pass. The B
successor compares all per-call producer, ABI, declaration, provider, header
and loader claims against a separate reviewed-claims record. Its production
anchor is deliberately absent because no image-specific raw provider/ABI
receipt has been accepted. Thus even a structurally complete production
candidate exits HOLD. The synthetic tests supply an isolated anchor only to
check drift rejection. This checker certifies the anchored F object, call,
import and depfile inventory; it cannot certify ABI/provider safety until
independent raw-record review and a later explicit source pin.
The [image-specific acquisition plan](E3_N1_RAW_RECEIPT_ACQUISITION.md)
enumerates the missing bytes and capture order. Reviewed claims now bind raw
record paths as well as hashes.

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

These tests exercise the checker's structure and drift rejection, including
nonempty calling-convention, declaration, provider version/binding, loader
path and self-rehashed or substituted depfile mutations. They do not
execute a compiler, linker, loader, Docker, provider or numerical gate, and
their complete synthetic fixture is not an N1 admission receipt. The next
gate is actual image/candidate producer, ABI, provider and complete dependency
bytes plus independent review. Only then may a separate scoped import-policy
change and fresh full Numerical F gate be considered.
