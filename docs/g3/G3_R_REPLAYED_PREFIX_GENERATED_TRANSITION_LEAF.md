# Private G3-R replayed-P1 generated-call transition

This bounded source leaf adds a feature-gated native entry,
`et_g3t_private_replayed_prefix_call_acquire_v1(context, expected_model)`,
and the same-registry Eshkol composer `g3r-generate-replayed-p1!(generator)`.
It enters the existing G3-T generated kind-2 transaction from an idle
committed manual P1 prefix. It is private to the CPU-f32, one-row,
diagnostic-C2, one-new-token profile. It does not load a C2 pair or admit
full G3-R continuation.

Native admission authenticates the live expected M3 owner against the
generator's model, no active model/call/frame/result/save hold, valid exact
policy and four-word RNG, committed P1 history with no generated/manual
append or pending G1, pinned 14-parameter identity/value binding, and the
owned A2 cache's length-one mask and geometry. The P2 full-context case and
categorical exhausted counter reject before a draw. The Eshkol wrapper
rechecks its canonical generator/model/T1 entries and the exact registered
raw V256 fingerprint. The existing seeded P1/G1 entry and full-request
preflight stay unchanged.

Admission pins the model, sets generated kind 2 and a private replay flag,
and enrolls the existing P1 history as pending G1. Only this flag permits
the normal pending output reservation on a committed prefix. The composer
then calls the existing sampler on the committed last logits and runs the
normal 21-role generated decode, output copy/text readiness, and final
commit. A precommit abort destroys the pending output and frame transaction,
clears pending G1 and the replay flag, and retains the committed cache,
logits, history and RNG. The generated frame commit publishes the sampled
byte (including EOS), successor RNG and length-two cache; its existing
postcommit finish is fail-stop.

`scripts/test-g3r-replayed-prefix-native.sh` compiles feature-off and
feature-on objects, checks symbol closure, and runs a synthetic in-memory
owner fixture under host normal and ASan+UBSan. The fixture verifies
cross-owner/forged model, P2, exhausted-counter and bad-cache rejection;
pending-output enrollment; sampler failure/abort/retry; and production native
sample/final-commit/finish calls. Its fake sampler asserts the exact saved
logits and RNG and returns a deterministic byte. The fixture stages the
frame and output that the 21-role Eshkol composer would produce; it does
not execute those numerical roles. LeakSanitizer cannot start in the current
ptrace sandbox. These host results are source-development evidence only.

Acceptance remains **HOLD** pending an independent review and a pinned
fresh-process normal/repeat/ASan+UBSan+LSan gate that compiles and runs the
Eshkol composer with real manual P1 prefill, generated numerical roles and
G3-S provider, including exact tokenizer/model/logit/cache parity and
negative fault cuts. The C2 SAVE receipt, persisted pair loader, larger
profile and repeated decode remain separate downstream gates.
