# TR3 fixed one-batch overfit acceptance

This development-only witness trains the installed public `trainer-train!`
and evaluates through installed public `trainer-evaluate!` and `metrics-ref`.
It uses one deterministic D1 shard with tokens `(3, 197)`, the accepted
N1/T2/V256/D2 CPU-f32 diagnostic model with seed 1729, a byte tokenizer,
one D2 batch with one masked target, AdamW constant learning rate 0.1,
and accumulation of one microbatch per update. The evaluation dataset is
separate but has identical bytes. The same model is evaluated twice before
training to establish a stable baseline; then the public trainer performs
exactly 32 updates and the model is evaluated again.

The predeclared acceptance criterion is **at least 25% lower final loss**
than baseline, using positive finite binary32 loss values decoded from the
public metrics bit boundary. Both evaluations must report exactly one token
and one batch with mask weight 1.0. The training summary must report 32
updates and 32 tokens. A no-op training path fails the quantitative loss
criterion even if it falsely reports update counts. An independent public
deterministic model forward on the same input is byte-stable before training
and must change after the 32 updates. Public `module-state-dict` snapshots
then restore the initial and trained parameter states in turn: the same
forward must recover its initial and trained bytes respectively. This ties
the functional change to actual model parameter state rather than metrics
publication. The public installed API does not expose raw parameter bytes,
so this does not identify which parameter tensor changed.

The installed trainer package includes `transformer/model.esk` and the eight
accepted M3 public forward, output, and diagnostic boxed entries in its
existing same-aggregate bridge. Their trusted entry names are renamed and
localized to the lease-aware `tr3-public-*` gates with the rest of the trainer
package; the witness links no second M3 object or registry. A public forward
on the enrolled model must raise `invalid-state` at `model-forward`, while
forwards before enrollment and after release still succeed.

Forward observations and state snapshots occur before trainer enrollment or
after `trainer-release!`: the public facade rejects direct model calls on an
enrolled trainer component. `:deterministic? #t` validates a deterministic
forward request without changing module mode. The model stays in the required
train mode for `trainer-train!`; public state-dict loads copy parameter values
and leave that mode untouched.

This checks fit to the training batch only. It does not establish held-out
improvement, generalization, exact resume, GPU behavior, mixed precision, or
long-run memory bounds. The linked fe9 build and its ordinary installed API
and package gates remain prerequisites for this runtime witness.
