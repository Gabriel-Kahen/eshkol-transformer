# Private C4 head-layout VJP leaf

Status: bounded private native leaf accepted by independent source/test and
compiled result review. Exact receipts are recorded in `docs/ROADMAP.md`.

The genuine G3-C4 provider admits split and merge rows with request shape
`[1,4,2,2]`. Split maps `X[1,4,4]` to `[1,2,4,2]`; merge maps
`X[1,2,4,2]` to `[1,4,4]`. Both are materialized permutations. For token
`t`, head `h`, channel `d`, the token index is `(t*2+h)*2+d` and the head
index is `(h*4+t)*2+d`. The feature-private
`et_m3t_c4_head_layout_vjp_private_v1` accepts an exact split/merge selector,
input, upstream, and input-gradient CPU f32 dense K1 views. Split VJP copies
the upstream head index to the token index; merge VJP copies the upstream
token index to the head index. No arithmetic or attention backward is added.

The function returns status only and retains no borrow. It validates the
exact ranks, shapes, lengths, dtype, device, layout, aligned storage, finite
input/upstream values, and disjoint output. It stages the 16 output values
before copying them, preserving all destination bytes on rejection.

The focused test performs all 16 central differences per operation through
the genuine forward provider and checks every backward permutation bit,
including negative zero. It also checks malformed selector, rank, shape,
length, dtype, device, layout, overlap, nonfinite values, null views, and
unchanged output on rejection. The script runs normal/repeat/native
ASan+UBSan and feature-off isolation. This leaf is not integrated with a
trainable C4 M3T; attention VJP, training, checkpoint, and generation remain
separate dependencies.
