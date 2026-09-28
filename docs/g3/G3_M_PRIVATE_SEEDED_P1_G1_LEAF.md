# G3-M private seeded P1/G1 implementation leaf

The private `(g3m-generate-p1-g1! generator input)` implements the independently
reviewed [P1/G1 schedule](G3_M_PRIVATE_SEEDED_P1_G1_PROPOSAL.md). It binds the
existing native full-request preflight before pins, reserves the pending
output, runs the two explicit 21-role T1 schedules with one G3-S sample
between them, accepts owned ID/raw T1 text, and delegates atomic final
publication to the existing G3-T `g3t-final-commit!` tail. Generated EOS is
included after its one append; zero budget remains on the separate P1/P2 G0
routes. A `ET_G3T_TESTING`-only observer snapshots the prepared generated
append logits for an independent M3T second-row bit comparison; it is absent
from the production object. This leaf adds no production native symbol or
public G3-G operation.

The exact source/test commit, tree, supported normal/repeat/sanitizer and Q0
results, feature-off inventory and durable seal are recorded in the handoff
after the focused gate. Hosted integration CI and public facade/package
acceptance remain separate.
