# TR3 I2 restore integration

This candidate selectively replays the accepted I2 restore leaf at
`154d9c3aeabd74027ca9899952ae6534d63ff7b3`, tree
`2421a7569834fbe1c303d8a27d014293e9d1f169`, onto reviewed union
`d5fa9a71b000ab1139a25e9b734b230ab5bda759`, tree
`185869ef7f79b9086860c9fdc6ea9aaaa8d7cd94`. It adds the feature-gated owned-clone
matcher and authority-bound 42-destination restore builder, tests, and standalone
Make target. It does not register a CI suite or install a public trainer API.

Only the ten I2 paths changed between leaf predecessor
`ce996a99148a6b202407174d933b943dab48c68c` and the accepted source are replayed.
The unaccepted O2 participant in that ancestry is excluded. The import preserves
the union's SHARED-R2 rank-two copy shapes, overflow guards, provider 1.1/v2
identity, and seven shape ranges. Replacing the complete older `f32_tensor.c`
would lose those accepted changes; this candidate applies the feature delta.

The [mode/blob inventory](TR3_I2_RESTORE_INTEGRATION_BLOBS.tsv) identifies exact
imports and the two intentional compositions. `native/f32_tensor.c` retains the
union provider and adds the reviewed feature, testing hooks, and three constructor
initializers. The standalone gate retains every leaf assertion and uses `d5fa9a7`
as its frozen ordinary-source predecessor. It applies exactly the three approved
initializer changes to that predecessor before comparing GCC/Clang O0 and O2
strict-aliasing object bytes and symbols. This proves the testing hooks and
disabled feature add no further production-object change.

The leaf's original supported gate passed on Ubuntu 22.04.5, Clang 21.1.8, and
GCC 11.4, including strict-aliasing builds, normal repetitions, exact manifests,
negative admission, failure atomicity, terminal behavior, and ASan/UBSan/LSan.
That leaf result alone does not accept the composed source. Exact-candidate
supported evidence and independent integration review are required separately.

Each restore retains a measured 64-byte tombstone; total retained control grows
linearly with restore count. This is not constant-total retention. No native
trainer, O2 restore participant, joint live restore, public evaluator, overfit,
full-state resume, or end-to-end CLI acceptance is claimed here. The runtime
union's inherited package gates and full CI remain separate acceptance gates.
