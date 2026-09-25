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

The supported pinned image `eshkol-checked-promotion-llvm21:20260922`
(`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`)
strictly compiled both touched test units, then ran the focused constructor
and P2 aggregate gates against source commit `750f7c7aedb852e033d9b9526ecbfe291c9aa8b0`
(tree `08073dd11dcd97e768701a850d365c681bbfb6f3`). The constructor
reported 53 checks, and P2 reported 46,540; normal, repeat, and
ASan+UBSan+LSan outputs matched with empty compiler and runtime stderr.
Nine P2 source contracts and Q0 4/4 passed. The sealed evidence is
`g3t-generator-rng-input-750f7c7`.
