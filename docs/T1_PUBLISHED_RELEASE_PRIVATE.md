# Private published T1 shell release source contract

This is a source-only, feature-gated prerequisite for the accepted public
P2/G2 history continuation. It adds no public tokenizer operation or installed
ABI. A private aggregate may load `native/t1_published_release_private.esk`
after `native/t1_wave1_root.esk` and link `native/t1_i64_shell.c` with
`ET_T1_PUBLISHED_RELEASE_PRIVATE`.

`t1-private-published-release!` accepts only the exact identity in the T1
encoded-tensor registry. A live entry first passes native preflight. A failed
preflight leaves its admission and owner unchanged. The owner then marks the
entry `releasing` before native teardown, so decoder and generation admission
reject it. Native release ends the shell's sole I1 borrow, destroys the I1
tensor, and marks the shell dead. Checked failures after withdrawal keep the
entry nonadmitted and the native shell retryable; a second private release
resumes teardown. Success marks the Eshkol entry `dead`. Repeat release of
that exact dead identity succeeds without touching another owner. Foreign and
unpublished values fail. A native dead shell remains registered as a small
tombstone, so its pointer can never name a new shell; native length/read and
the private C4 pair read reject it. The owned I1 payload, borrow, and tensor
are reclaimed, while the identity tombstone remains for process lifetime.

The seam assumes serialized T1 use, as the existing T1 contract requires.
It must not release a source during an in-flight decode or before a dependent
consumer has copied its IDs. F must compile and exercise this exact private
owner path after C4 copy and issuer enrollment, then prove continuation from
the independent native IDs. The focused source/native gate here does not
prove that aggregate runtime behavior or change the public process-lifetime
contract.
