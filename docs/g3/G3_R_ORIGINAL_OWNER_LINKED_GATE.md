# G3-R original-owner linked runtime gate

Status: **prepared source and host gate; linked Eshkol runtime unrun**.

`tests/g3r/original_owner_linked_runtime.esk` loads the accepted one-registry
`g3r_original_owner_c2_claim_root.esk`. It builds a genuine M3T/G3 generator
with a committed P1/G1 history, changes that exact model to train mode, and
enrolls it with the same raw tokenizer, dataset, and original O2 optimizer in
TR3. The normal cut begins the G3/O2 originals under the trainer lease,
enrolls the held O2 state, clones P1, composes C2, checks the exact adopted
child and token, drains O2 then G3/lease, and releases the detached C2 owner.

Separate processes cover O2 claim mutation followed by a typed error or
raised `#f`. Both require unchanged C2/request publication and live P1/O2
children before release. Two post-adoption cuts raise after the C2 claim and
prove that original holds and the detached C2 child can be cleaned up while
the first typed or `#f` exception is preserved. A final process corrupts a
request slot after compose and requires the post-compose check to fail stop.

`scripts/test-g3r-original-owner-linked-host.sh` checks source closure,
causal mutations, prior native feature-off and failure cuts, and runner
syntax. On a supported pinned local toolchain,
`scripts/test-g3r-original-owner-linked.sh FRESH_EVIDENCE_DIRECTORY` builds
the feature-gated native archive and linked Eshkol fixture, runs all cuts,
and records provenance. The compiled entrypoint has not yet been run. This
fixture installs no public API and admits no SAVE, LOAD, file publication,
restored generation, or replay.
