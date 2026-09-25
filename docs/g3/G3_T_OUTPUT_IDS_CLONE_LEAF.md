# G3-T private generated-ID clone leaf

This source-private leaf implements only the accepted
`output_ids_clone(ptr out)` stem and kind-5 typed release from the
[G3-T contract](G3_T_PRIVATE_CONTRACT.md). It is compiled only under
`ET_G3T_OUTPUT_IDS_CLONE_PRIVATE` with final publication. It adds no Eshkol
accessor, public facade, package export, other output clone, or CLI.

An authentic independently live output with completed numeric/ID/text readiness
may be cloned after call finish. The native path rejects wrong-kind, pending,
dead or borrowed output before allocation. It allocates a new rank-one I1[G]
owner, copies the published IDs (G=0 or 1), and enrolls the kind-5 record only
after the copy succeeds. Header and every I1 allocation failure leave output,
cache, RNG and live-clone membership unchanged. The exact clone survives output
release and generator close. `tensor_release` accepts the kind-5 owner, rejects
an active I1 borrow before mutation, destroys its I1 payload, clears its roots,
and retains an idempotently releasable tombstone. Wrong-kind release remains
`invalid-argument`; authentic dead non-release use remains `invalid-state`.

The focused P2-enabled source witness covers live P2/G0, P1/G0 and P1/G1,
empty and one-ID clones, header and I1 allocation cuts, pending/borrowed source,
wrong-kind and active-borrow release, parent-release survival and idempotence.
The normal production object excludes every development observer. It does not
prove public list wrapping, clone shell enrollment, other detached result kinds,
manual logits, or generation package acceptance; those require G3-G review.

On the pinned Ubuntu 22.04/LLVM 21 image, the P2-enabled focused gate passed
242 checks in normal, repeat and ASan/UBSan/LSan modes with identical stdout
and empty runtime stderr. Six inherited source contracts and Q0 4/4 passed.
The supported image ran with Docker network disabled, `ESHKOL_ARENA_POISON=1`,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`. This is candidate evidence, not G3-G
acceptance or a long-running retention measurement.
