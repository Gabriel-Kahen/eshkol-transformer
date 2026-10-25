# G3-R C4 native context exclusion leaf

**Source candidate with local native tests only.** Under
`ET_G3R_C4_CONTEXT_EXCLUSION_PRIVATE`, the source-private
`et_g3c4_private_no_live_owner_context_status_v1(owner)` returns a closed
status: zero only for an exact sealed owner with no live same-owner native
context and no active owner edge. It traverses the existing mixed generation
transport registry, validates each record before interpreting its kind, and
rejects both idle cached generators and active calls. Dead context tombstones
and live contexts for another owner do not exclude this owner. It returns no
context, pointer array, cache or lease and changes no owner/context record.
The symbol is absent without the private flag; that flag requires the
generation transport registry.

The local fixture checks foreign owner, mixed RNG record validation, malformed
record, foreign-owner live generator, same-owner idle generator with
`owner->active == NULL`, active call, closed/dead generator, live base context,
and a non-null active edge. It compares owner records and registry heads
across each status call. Normal, repeat and ASan+UBSan output must match;
the runner also verifies the feature-off symbol is absent.

This is a **create preflight only**. It cannot prevent a new context after the
status returns, authorize a workspace, authenticate P1 handles, or prove
lease exclusion. The [workspace ownership contract](G3_R_C4_WORKSPACE_OWNERSHIP_CONTRACT.md)
requires a recheck under the serialized aggregate and a native context-create
interlock while a workspace edge lives. No pinned compiled gate has run.
