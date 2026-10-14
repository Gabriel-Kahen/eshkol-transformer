# E3 N1 image-specific raw receipt acquisition plan

Status: **inert plan, no source admission**. The target remains Numerical F
head 94b8fecd/tree d28cffc1, f31 image ID, fe9 compiler, N1/M1 HOLD.
Candidate B's production reviewed-claims anchor remains absent. This plan
names the bytes required before that anchor can be proposed; it authorizes no
Docker, compiler, loader, provider or numerical target execution.

## Retained bytes and their limit

The failed F seal (SHA-256 8269be7b...) contains private.ll (917,990,074
bytes), private.o, combined.o, libprivate.so, combined.map, native objects,
normal/sanitized import inventories, source closures and all 17 MMD depfiles.
The 39-call forensic receipt (14 native bcmp, 25 private.o puts) and its
bounded disassembly are separately sealed; its root review is offline-only.
F's plan.tsv and launch.sh pin the image ID, fe9 source/tree, runtime/runner
hashes, test script, mounts and container command. They do **not** retain an
OCI filesystem export, actual provider file, resolved loader map, exact
per-object argv/environment, final optimized producer trace or complete
system-header inventory. The F private.ll declares printf but no puts. Its
MMD depfiles omit system headers; four Eshkol external include paths are
identified, and host copies exist, but their current presence is not
image-specific captured header proof.

## Capture order and immutable inputs

1. **Preflight without mutation.** Verify F's failed seal, B's accepted source
   and review seals, source head/tree, image ID, compiler tree and runtime
   hashes. Inventory file sizes, free disk/RSS and tool versions. Use a fresh
   evidence directory and never edit F or the accepted B checkout. Freeze
   absolute path, symlink chain, mode, size and SHA-256 for every input.
2. **Image filesystem acquisition.** Obtain a read-only OCI image
   manifest/layer export for exactly f31, with image config, layer digests,
   architecture, environment, loader configuration and every resolved file
   byte sealed. If no independently pinned image filesystem is available,
   HOLD provider/header work. A host libc or same-name header is not a
   substitute. Do not execute any extracted binary while collecting bytes.
3. **Producer and call ABI.** From F's private.o, package-native-5.o and
   package-native-6.o, retain complete readelf relocation/symbol/version,
   objdump disassembly and debug-location outputs for all 39 unique
   (object, section, offset) calls, with command/tool hashes. Correlate
   native sites to exact memcmp expressions and generated puts sites to
   final optimized lowering, not just private.ll. Capture each call's
   argument/return types, widths, pointer/address-space semantics, calling
   convention, visibility/binding, version and declaration bytes. If F's
   retained object/debug/IR cannot prove the final lowering, a separate
   pinned compiler capture must reproduce the exact F object SHA-256 before
   its trace can be attributed to F; otherwise HOLD.
4. **Provider and loader closure.** Statically inspect the pinned image's
   candidate provider files and symlink chain, exported bcmp/puts symbol
   type/binding/visibility/version, SONAME, DT_NEEDED and RUNPATH. Hash all
   direct/transitive libraries and loader/config bytes. Preserve a
   machine-readable resolution graph under the recorded image search path.
   Static ELF metadata is not an actual resolved loader map; that last
   observation needs a separately reviewed bounded loader capture. HOLD
   source admission until actual resolution is corroborated.
5. **Headers, argv and environment.** Reconstruct every compiler command
   from the fixed builder only as a hypothesis. F did not retain exact
   per-object argv/env. A later pinned, constrained replay must capture
   NUL-delimited argv/env, include roots, compiler resource directory and
   full (not MMD-only) direct/transitive system-header trace for private,
   bridge, all native and provider objects. Resolve and hash every path,
   symlink, mode and byte. Require the replayed object/depfile bytes to match
   F or treat it as a new candidate requiring separate review.

Seal raw outputs and a versioned claim table with one row per call, plus
provider, loader, header and argv records. The independent reviewer must
compare the claims to raw bytes and pin the exact reviewed-claims file SHA-256
in a later verifier source change. The verifier now projects both path and
SHA-256 for raw records; a same-byte path substitute is a negative. Also
reject changed call producer/signature/version/binding, provider or symlink
bytes, loader search/closure, header/depfile path or bytes, argv/env and any
extra noninstrumentation import. Each deliberate mutation must fail before
any source admission.

## Resource and completion boundaries

F's retained scratch and gate trees already occupy about 1.35 and 2.24 GB.
Stream/hash existing large files; do not duplicate private.ll merely to
inspect it. Before any later capture, reserve measured free space for the
expected output plus margin, set per-command time/RSS/output limits and
record peak usage. A cap hit is HOLD with the raw partial output sealed, never
a truncated pass. No target execution or provider call is part of this
acquisition plan. An independently accepted complete receipt is only the
precondition for a separate tuple-scoped import-policy review; N1/M1 still
require the fresh full Numerical F gate and evidence verdict.
