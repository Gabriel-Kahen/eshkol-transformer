# C4 native stable-prefix close: first gate

Status: private native candidate. This is not a SAVE hold, a completed
Eshkol `m3-call`, or an idle-P3 generation entry.

`et_g3c4_private_p2g2_stable_prefix_close_v1(context, pending_output)`
accepts only the authentic live `(call_kind,budget)=(2,2)` non-EOS prefix
created by `et_g3c4_private_p2g2_prefix_commit_v1`. The output must be the
sole pending capacity-two output owned by that exact context. The token frame
is idle, the pinned fourteen parameters still match the prefill binding, the
I1 IDs tensor is clean and unborrowed, and the committed A2 cache has exact
P3 shape, length three and mask `1110`. No other pending logits may exist.

The native record stays inside that same generator context. It retains the
P2 prompt IDs, selected first ID/raw byte, bit-exact 256 successor logits,
four successor RNG words, and P3 A2 K/V bytes. Its read-only status check
reauthenticates these against the still-live context, provider cache and
model owner; it exports no pointers or carrier arrays. A successful close
destroys the clean pending I1 output, clears the active-prefix marker and
drains native call pins/owner active edge, leaving the generator idle with
its committed P3 cache. Ordinary call acquire rejects a retained stable
record; generator close may retire it. Repeated close and foreign/forged
edges fail.

All fallible validation precedes the closed tail. Validation failure leaves
the active prefix retryable or abortable. Any failure during pending I1
destruction or later tail mutation fails stop. Feature-off object behavior
and predecessor private APIs remain unchanged.

The Eshkol coordinator currently requires the first prefix to stay inside
one original `m3-call` region. This native close does not transition its
same-region ledger or rooted object authority. That ledger transition is a
required next gate before this state can become a SAVE source. A separate
read-only SAVE hold and genuine idle-P3 second-token entry also remain open.
