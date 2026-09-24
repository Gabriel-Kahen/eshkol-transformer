# G3-C4 Step 15A: authenticated T1 prompt copy

Step 15A adds one bounded private prompt owner. It consumes an already sealed,
same-aggregate T1 encoded-ID shell and synchronously copies one or two byte-token
IDs into a new native I1 owner. It adds no public facade, generation call,
numerical schedule, result or text owner, EOS behavior, persistence format, or
package export.

## Exact private boundary

The native feature macro is `ET_G3C4_PROMPT_T1_BORROW_PRIVATE`. It requires the
accepted generator transport and adds exactly these source-private calls:

```c
void *et_g3c4_private_input_from_t1_v1(void *sealed_t1);
int64_t et_g3c4_private_tensor_release_v1(void *owner);
```

The corresponding Eshkol operations are
`(generation-input-create-internal tokens)` and
`(generation-tensor-release-internal! owner)`. The constructor returns a new
kind-`input` identity in the existing private G3-C4 registry. Its native payload
is an owned dense CPU i64 tensor with exact shape `[1,P]`, where `P` is one or
two. Release is idempotent for the exact issued identity, rejects foreign or
wrong-kind identities, destroys the native tensor, and scrubs every retained
Eshkol payload root.

The operation has no generator argument. It does not alter generator or call
entry layout: their slot 6 continues to retain the exact Step 14A tokenizer
shell. The new input retains no generator, model, tokenizer, T1 encoded shell,
or T1 registry entry. It therefore makes no tokenizer-provenance claim beyond
the authenticated synchronous copy.

## Authentication and copy order

Eshkol first requires `t1-wave1-tensor-admitted?` for the exact encoded shell.
Only after that same-aggregate check may it pass the opaque sealed-shell value to
the native constructor. Native code uses the existing T1 ABI in this order:

1. call `et_t1_i64_shell_length_v1` and inspect
   `et_t1_i64_shell_last_status_v1`;
2. require length one or two;
3. call `et_t1_i64_shell_read_v1` for every element, checking status after each
   read and requiring every ID in `[0,255]`;
4. create a new I1 tensor with shape `[1,P]`, copy the staged values, and enroll
   the complete input owner.

The T1 reads finish before I1 publication. No T1 pointer, payload address,
borrow lease, view, registry link, or ownership claim is retained. This leaf
does not add to `t1_i64_shell.h`, does not expose a raw T1 span, and does not
infer a new T1 borrow API. A foreign native pointer, unsealed shell, wrong
length, or out-of-range value fails before input enrollment.

All Eshkol shell, entry, ledger, and registry-list allocation needed for
publication precedes the native constructor. Native failure leaves that staged
entry dead and scrubbed. Failure after native construction releases the exact
native owner before reraising. Native partial construction destroys any I1
tensor it created and preserves the first error. No failure publishes a live
input or changes an existing generator, call, model, tokenizer, RNG, cache, or
T1 shell.

## Failure categories and lifetime

Foreign Eshkol identities are `invalid-argument`. Native T1 identity failure is
`invalid-argument`; an admitted but unsealed T1 shell is `invalid-state`.
Lengths other than one or two, including zero and three, are
`shape-mismatch`. IDs outside `[0,255]` are `shape-mismatch`. Allocation and
invariant failures retain their accepted native error provenance.

Once constructed, the input is independent of the source shell. Test-only T1
registry withdrawal, tokenizer fingerprint drift, model mode changes, generator
close, and source reachability changes do not prevent input release and do not
change its copied values. Exact release is idempotent; non-release use of a dead
input is `invalid-state`.

The accepted T1 registry intentionally retains sealed encoded shells and has no
sealed-shell release. Retention evidence therefore reuses one P2 T1 shell
across 1,024 and 8,192 input create/release cycles. It must show at most
one live native input payload at a time and no live I1 payload after release;
fresh `tokenizer-encode` calls inside that loop are not a bounded-memory proof.

## Evidence and limits

The focused witness covers P1 and P2 copies including IDs 0 and 255, P0/P3
rejection, foreign and copied identities, direct-native unsealed/range/wrong-kind
rejections, independent copies, exact and repeated release, allocation and
publication cuts, source-registry withdrawal, and 1,024/8,192 retention. It runs
normally twice with byte-identical output and under ASan, UBSan, and leak
detection. The complete Step 14A gate remains the tokenizer/eval and call-entry
regression gate; existing T1 and I1 tests remain the upstream ABI and ownership
gates.

This leaf deliberately cannot invoke the current fixed-P3 prefill boundary:
P3 is rejected, and no prompt-to-prefill adapter or raw pointer accessor is
introduced. Public generation remains blocked on a compatible numerical prompt
route and on result/text ownership.
