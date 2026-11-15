# Standalone P2/G2 constructor-owned carrier prerequisite

Status: private candidate with focused compiled normal/repeat/sanitizer evidence;
integration remains pending.
This implements only the original-allocation prerequisite of the accepted
[terminal-close contract](G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md).
It does not compose a native pending output or a generation call.

`native/g3c4_p2g2_terminal_carriers_extension.esk` exposes three private
operations: `g3c4-p2g2-terminal-carriers-create` creates a fresh bundle;
`g3c4-p2g2-terminal-carriers-check bundle operation` authenticates it read-only;
`g3c4-p2g2-terminal-carriers-retire! bundle` authenticates, scrubs, and removes
its private root. A copied/foreign bundle, equal-length replacement, paired
swap, alias, changed expected extent, changed length header or diagnostic tag
rejects without changing the originals. Repair permits retry. Retired bundles
and repeated retirement reject; changing a tag cannot recreate the removed
root. This standalone lifetime differs from terminal output's separately
required idempotent release. Root-list rebuilding finishes before scrubbing;
no allocation follows revocation. Dropping a live bundle without retirement
retains its root; callers must retire it. No flat-memory claim is made.

Bundle slots 0..7 are first raw[1]/staging[8], pending raw[2]/staging[16],
second raw[1]/staging[8], text[1]/text[2]. Slot8 is the diagnostic requested
extent vector `9,16,10,24,9,16,9,10`; slot9 is a diagnostic lifetime tag.
An inaccessible lexical closure captures all eight objects directly from
their literal `make-bytevector` construction sites, the original bundle and
extent-vector identities, and a separate lifetime cell. Mutable slots never
establish allocation provenance. No enrollment operation accepts external
buffers, no native pointer conversion occurs, and no call/output/terminal
authority is conferred.

The provenance argument is bounded to pinned Eshkol `fe9dfd5`: its
`lib/backend/llvm_codegen.cpp:16286-16318` requests `8+length` bytes and its
`lib/core/runtime_object_alloc.cpp:20-61` allocates the extra object header
plus eight-byte rounding. Requested header-plus-payload extents above are
distinct from actual rounded backing allocations `24,24,24,32,24,24,24,24`.
The checker combines those fixed construction facts with original identities
and exact current lengths; it does not infer backing from a mutable header,
read memory before a supplied pointer, or certify arbitrary bytevectors.

The development-only allocator ledger adapted from `83667506` independently
records exact construction requests and aligned backing calls. It is evidence
for this compiled construction path, not production authentication. Its old
pending-record fixture is not reused: that fixture predates the independently
rooted pending-anchor replacement rejection. The dedicated fixture covers all
eight constructor returns, foreign objects/bundles, paired swaps, aliasing,
expected/header corruption, repair, scrub and independent retirement.

The initial lowering stores the length through the allocator result, but the
pinned hosted AOT final pass `hardenGeneratedArenaAllocations`
(`llvm_codegen.cpp:263-340,3458-3473`) inserts a null-result emergency raise
before that store. The second gate's normal run passed 170 checks, then its
classifier failed because it incorrectly expected SIGSEGV: the measured first
cut exited 1 with `Unhandled exception: object or exception-handler allocation
failed`. That failure is retained, not recast as a passing gate. The corrected
fixture tests eight guarded synthetic null-return cuts for caught exceptions,
no published answer, earlier-bundle preservation and successful retry after
disarm; separate unguarded subprocesses require the exact emergency exit 1
diagnostic in normal and sanitizer modes. Neither proves genuine heap
exhaustion, vector/closure/root allocation failure cleanup, or process-wide
rollback. Expected unhandled diagnostics are separate from successful
sanitizer stderr.

The corrected pinned Ubuntu 22.04/Clang 21.1.8 gate passed 530 checks each in
normal, repeat and ASan/UBSan/LSan runs, with identical stdout and empty
compile/successful-run stderr. Each run includes eight guarded null-return
cuts, preserved earlier-bundle identities and all 39 nonzero payload bytes,
and successful constructor retry/retirement. Sixteen separate unguarded cuts
(eight normal, eight sanitizer) matched the exact emergency exit 1 diagnostic
without sanitizer errors. The dedicated gate took 3:02.58 and peaked at
3,804,144 KiB. The unchanged pending/feature-off predecessor fixtures passed
23/4 checks per normal/repeat/sanitizer mode in 5:55.49 at 4,058,852 KiB.
Source mutation checks pass 22 new and seven predecessor mutations; Q0's
four tests pass with exact development-only Python helper admission and
near-miss rejection. The inherited missing pending-checker admission is
repaired alongside the two new development helpers.

Evidence is under `build/g3c4-p2g2-terminal-carriers-20261115-attempt3/`.
The first parse failure and second classifier failure remain in their own
attempt directories. The older 170-check binary has an independently
reviewed eight-site null-guard disassembly; it is supporting pipeline evidence
only. The passing 530-check gate cleans its temporary binaries, so no binary
hash or disassembly for that gate is retained. Its source/compiler/runtime/
image/commands/logs are sealed separately. No additional AOT was run solely
to replace that missing artifact.

After compilation, only this documentation, the roadmap, and the new source
manifest/checker were completed. The final 244-path manifest preserves the
inherited pending closure and adds the trusted M3/I2 package manifests,
23 transitive fixture loads, 57 native source/include dependencies and exact
runner/toolchain inputs. It closes inherited manifest omissions in this new
leaf only. The original compile-time manifest and source digests are retained;
the final receipt proves compiled code and fixture inputs unchanged. Runtime
sources, compiler and archives are pinned/sealed outside the repository-file
manifest. These metadata corrections do not establish any wider admission.

Run `scripts/test-g3c4-p2g2-terminal-carriers.sh` under the shared heavy lock.
The existing protected/prefix wrappers, phase admission, abort behavior and
native ABI remain unchanged. A later separately reviewed terminal constructor
must use these same original carriers as its sole pending output's carriers;
allocating another bundle or importing mutable upstream slots does not close
that ownership gap. Authentic T1/EOS/second-token composition, native-zero
tail, snapshots/clones, public G2, trained-G2 sanitizer acceptance and
save/reload remain open.
