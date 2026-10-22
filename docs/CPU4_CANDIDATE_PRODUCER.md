# Private CPU4 candidate producer

This feature creates the canonical C4 context-four model through a capability
issued before its first P1 token. It preserves the ordinary constructor, its
fourteen parameters, tied embedding and entry slot7=false. It requires
`ET_CPU4_CANDIDATE_PRIVATE`, `ET_CPU4_TRAINING_PRIVATE`, and
`ET_G3R_CANDIDATE_RETIRE_PRIVATE`; it adds no installed API.

The native nine-function surface is declared in `g3c4_model_owner_internal.h`.
Its capability retains original owner, fourteen parameter/handle identities and
exact native P1 scope. States are CONSTRUCTING, UNPUBLISHED, ADOPTED,
RETIRE_PREPARED, GRAPH_REVOKED and DEAD. A bound open abort requires proof that
P1 already aborted; direct owner abort cannot bypass this rule. An adopted
model cannot be failure-retired.

Retirement checks storage provenance, all parameter/value/gradient pins and
borrows, exact P1 handle membership, training/context leases and O2 consumers
before mutation. It then closes an explicitly supplied sole idle generator,
retires the exact optimizer, revokes P1, invalidates captured Scheme P1/I2
nodes, and destroys C4 payloads. Unexpected failures after mutation exit134.
Control tombstones remain rooted; CPU4 F32 parameter controls retain80 bytes
rather than the ordinary64 to authenticate original value/gradient children.
Ordinary feature-off controls remain64 bytes.

The initial Scheme surface is four closed functions in
`native/cpu4_candidate_extension.esk`: create, model, adopt and retire.
It retains its own construction graph only. Retirement passes null optimizer
and generator, rejecting live children; adoption accepts generator=false only.
It does not yet own checkpoint, transport, optimizer or generator children.
A checkpoint consumer needs an explicit typed child-enrollment extension before
claiming compound restore failure cleanup. Active O2 builders or states block
retirement conservatively, including independent ones without origin rosters.

Validation commands:

```
bash scripts/test-cpu4-candidate-source.sh
bash scripts/test-cpu4-candidate-native.sh /absolute/fresh/native-evidence
bash scripts/test-cpu4-candidate-producer.sh /absolute/fresh/producer normal
bash scripts/test-cpu4-candidate-producer.sh /absolute/fresh/producer-san sanitize
```

Run compiled/sanitizer commands under the shared heavy lock and supported
pinned toolchain. The native test uses genuine native P1 tokens; it does not
prove Eshkol shell cleanup. The separate producer test exercises genuine
Eshkol graphs, both independent retirement orders, adoption and ordinary-model
denial. Source/planner checks alone do not admit compiled behavior, restore,
resume equivalence or the complete CPU4 train/SAVE/reload/eval/generation smoke.
