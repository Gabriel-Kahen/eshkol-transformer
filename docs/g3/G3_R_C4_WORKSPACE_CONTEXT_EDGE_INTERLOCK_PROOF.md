# G3-R C4 native context-entry edge proof

**Local source/test evidence only.** The existing native base-context,
seeded-generator, and typed-RNG-generator entry paths all reject a non-null
sealed-owner `active` edge before context allocation. The extended native
fixture uses a distinct sentinel as a proposed workspace edge and verifies
all three paths return C4 `invalid-state/lifecycle` without changing the owner,
transport registry, or successful context-allocation counter. Normal, repeat,
and ASan+UBSan runs pass 221 checks with byte-identical output.

This is an interlock proof for an already published edge, not a workspace
implementation. It does not authenticate the sentinel as a workspace,
serialize the preflight/commit gap, publish either workspace edge, or prove
that a racing context cannot enter before publication. The workspace gate
must recheck under one serialized aggregate before its no-failure edge tail.
No pinned compiled gate has run for this extended fixture.
