# TR3-B private objective bridge contract

Status: issue #116 implementation candidate. Nothing in this contract is a
public facade or an accepted downstream dependency until independent review.

The source aggregate `native/tr3b_objective_root.esk` composes one T2-owned T1,
E1, P1, I1, I2, M3T, M3 and D2 identity universe. It loads the accepted M3
forward/reverse schedule unchanged and dispatches the accepted L2 and L3S
providers through genuine K1 runtimes. It does not link an M3 or D2 package
archive.

`tr3-stage-d2-input!/2` has the exact logical signature `(batch, input-owner)`.
It requires a live authentic D2 `N=1,T=2` batch and a live genuine M3T input
owner. It opens one synchronous D2 input view, validates exact CPU dense i64
`[1,2]` metadata and byte-vocabulary IDs, copies the two IDs through M3T's
existing native ingress, closes the view on every path, and returns `#t`.

`tr3-objective-vjp!/6` has the exact logical signature `(output, graph-logits,
batch, seed, expected-ordinal, observation-bits)`. The output and graph logits
must be live exact leases of the same accepted M3 graph. The batch must be live
and authentic. Seed must be a live genuine M3T logits owner. The ordinal is a
nonnegative exact i64. Observation is an actual Eshkol bytevector with exactly
12 bytes, allocated by the caller before forward and disjoint from the supplied
owners and protected tensor storage. Its three little-endian f32 words are batch
numerator, positive mask weight and batch mean.

Before the gradient mutation, the call allocates its 16-byte target and 2-byte
mask scratch, copies them under two nonoverlapping D2 view borrows, closes both
borrows, copies graph logits into native stack storage, and performs genuine K1
dispatch in this order:

1. L2 `indexed-cross-entropy.forward`;
2. L3S `l3s.masked-objective.reduce.bool`;
3. L3S `l3s.masked-objective.numerator-seed.bool`;
4. L2 `indexed-cross-entropy.backward`.

The native prepare validates every descriptor, span and alias before writing the
genuine seed or the observation destination. A preparation failure preserves the
observation bytes and all fourteen gradients. Successful preparation may change
the reusable seed and observation scratch. They are not metrics and must not be
consumed as success unless the call returns `#t`.

The call then invokes the accepted `diagnostic-output-vjp!` exactly once with the
numerator dlogits seed, the exact positive weight bits as metadata and the supplied
ordinal. This is the sole gradient mutation boundary. Failure preserves all
fourteen gradient bytes/counts/weights, although prepared seed and observation
scratch may remain changed. Success contributes all fourteen unique gradients
once and immediately returns `#t`; there is no result allocation or publication
after the contribution.

The caller continues to own output, logits, batch, seed and observation. It must
release output/logits/batch after the call. These accepted public releases use
fallible outer guard frames, so TR3-B does not classify them as a no-fail tail.
The later trainer transaction must clear all transient gradients if cleanup fails
before optimizer commit. Issue #116 proves successful leased cleanup and live-owner
baselines without adding a private M3 release authority.

Only normal D2 batches establish reachable masks. For `T=2`, the focused corpus
proves normal `11` and final padded `10`. The `01` case is a clearly named
test-only injected mask that exercises objective ordering; it is never reported as
a D2-produced batch.

The witness constructs the canonical exact-`V=256` raw byte tokenizer from the
same aggregate's resolved X1 model configuration. Its D1 corpus is bound to that
tokenizer fingerprint, and every staged input and target is below 256. M3 staging
and L2's exact `V=256` target check retain the fixed model profile; no tokenizer
special-ID range is admitted or silently truncated.

M3 graph, output, logits and I2 plan tombstones remain the accepted cumulative
controls after release. Evidence reports them separately from live owners and
makes no flat-total-memory claim.
