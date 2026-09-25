# G3-T private generator RNG input admission

This leaf extends the accepted source-private `g3t-generator-create/3`
constructor to accept the exact eight-pair `:rng` alternative to `:seed`.
Policy normalization still checks the same C2 scalar bounds and greedy
constraints. The RNG route looks up the exact same-aggregate shell in
`g3t-registry`, requires kind `rng` and a live detached entry, then passes
only its native kind-8 pointer to the accepted `generator_rng` stem. Native
admission independently rejects forged, dead, wrong-kind or busy pointers and
copies all four words into a new idle generator before enrollment. The Eshkol
generator retains its model, tokenizer and normalized policy, not the source
RNG shell. The seeded route still calls `generator_seed` with its previous
validation and failure cleanup.

The focused witness covers forged/wrong-kind/dead/busy shells, invalid
policy, native header and A2 allocation cuts, retry, source release
independence, exact P1/G0, P1/G1 and P2/G0 words, and no draw or cache
publication at construction. The historical seeded constructor gate remains
valid with the accepted native feature dependencies linked. Its current build
closure is `native/g3t_generator_constructor_rng_source_closure.txt`;
`native/g3t_generator_constructor_source_closure.txt` records the historical
seed-only predecessor and is not the complete RNG-enabled link closure. No public facade,
package export, CLI, or general generation claim is added.
