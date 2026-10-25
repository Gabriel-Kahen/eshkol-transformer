# Private C4 empty-cache A2 stage/view VJP leaf

Status: bounded source and local numerical candidate. No composed C4 backward
graph, workspace, model owner association, or checkpoint is admitted.

The accepted four-token C4 forward stages K/V `[1,2,4,2]` into a one-layer
A2 transaction with capacity four, old length zero and append count four. In
`et_a2_kv_cache_transaction_stage_layer_v1`, source index
`((head * 4 + token) * 2 + channel)` copies to the same view index because
`old_length=0` and `capacity=append_width=4`. The transaction view returns
immutable K/V with that same shape. The reverse mapping is therefore
`dStagedK=dCacheK` and `dStagedV=dCacheV`; the causal keep mask and effective
length have no f32 derivative.

Under `ET_M3T_C4_A2_EMPTY_STAGE_VJP_PRIVATE`, the direct status-only function
accepts two disjoint upstream and two disjoint destination CPU f32 dense K1
views `[1,2,4,2]`. It rejects malformed shapes, dtype, device, layout,
overlap and nonfinite upstream values before writing either destination. It
also rejects destination overlap with every supplied view descriptor's declared
extent, four-element shape array, and validated dtype/device strings before
either copy; otherwise the first copy could corrupt the second output pointer.
It retains no pointer and does not touch the A2 cache. Feature-off compilation
rejects the leaf; ordinary A2 exports no private symbol.

The focused fixture computes an objective through genuine A2 create, begin,
stage, view, end, abort and destroy operations. All 32 source K/V central
differences match the leaf's gradients. Mutation controls check unchanged
destinations and control bytes after typed rejection, including destination
aliases into each descriptor, shape array, dtype string and device string.
The fixture's cache is freshly empty for
every objective evaluation, and verifies four visible positions and K/V byte
equality with the staged source.

This primitive does **not** authenticate that a caller's gradient came from
the same live A2 view or original C4 owner. The forward A2 API does not retain
its borrowed source pointers, and the full-prefix function returns logits
without a backward tape. A later reviewed composition must authenticate those
edges, capture or recompute intermediates, and prove end-to-end gradients
through the actual 21-dispatch provider before training admission.

The original staged K/V buffers are borrowed only during `stage_layer` and
are never retained by A2. The returned cache-view descriptors and storage
exist only from `view_begin` through `view_end`; abort then scrubs the staged
cache slots. A future backward coordinator must either keep a valid view
lease through its attention VJP and map detached dCacheK/V into separately
owned dStagedK/V before ending the lease, or reproduce and authenticate the
same view and intermediates in a fresh transaction. It cannot retain a view
pointer or use the forward function's stack scratch after return.
