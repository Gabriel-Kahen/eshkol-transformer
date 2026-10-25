# Private C4 trainable forward/tape/VJP composition schedule

**Source proposal, not runtime admission.** This schedule is pinned to the
`0c441e565467cffd4636e0af5bdb76f1e4fc3cd7` workspace source. Its
Eshkol fixture and the tagged constructor/P1/association predecessors have
not passed compiled admission. The separately reviewed C4 VJP leaves are
individual kernels, not one linked training graph. No trained checkpoint,
optimizer step, or multi-token generation follows from this document.

## Authority and inputs

The only model is the live `trainable-c4` entry in the sole C4 registry. Its
entry slot 3 is the sealed native owner, slot 8 holds the exact fourteen P1
handles, and slot 10 must identify the sole live workspace shell. The native
owner's exclusive active edge, workspace row, fourteen original handle/carrier
identities and scalar associations must all match. Compute reauthenticates
these edges and full train-mode topology at entry and before publication;
`module-eval!` can currently drift mode while a workspace lives. A foreign,
stale, released, seeded, context-two, or generation-owned identity rejects.
No generation context may coexist, even idle with retained cache. In
particular, `et_g3c4_private_full_prefix_forward_v1` cannot serve as the
workspace forward entry: it admits an active generator call and borrows its
pins/A2 cache. Its accepted 21-role operation/shape schedule is the numerical
reference for a distinct workspace-owned dispatch path.

Input IDs are owned i64 `[1,4]`, each in `[0,255]`; positions are fixed
`[0,1,2,3]` i64 `[1,4]`; causal keep is u8 `[1,4,4]` with key `k <= query q`.
The attention row is `[1,2,2,4,4,2]`; the head layout is
`[1,4,2,2]`; layer-norm epsilon has f32 bits `0x3727c5ac`.
An independent caller-owned f32 upstream `dz` has shape `[1,4,256]`.
The composition does not choose a loss or normalize `dz`.

## Canonical parameter snapshot

Before computing a frame, copy all fourteen live f32 parameter values from
their exact P1/I2 bindings into workspace-owned immutable-for-frame storage:

| Index | Parameter | Shape |
| ---: | --- | --- |
| 0, 1, 2, 3 | key, attention output, query, value | each `[4,4]` |
| 4, 5 | FFN down, FFN up | `[4,8]`, `[8,4]` |
| 6, 7, 8, 9 | norm1 bias/weight, norm2 bias/weight | each `[4]` |
| 10 | tied token embedding/head weight | `[256,4]` |
| 11, 12 | final norm bias/weight | each `[4]` |
| 13 | learned position embedding | `[4,4]` |

This is 1,192 f32 values or 4,768 bytes. Index 10 has one snapshot and one
final gradient; its embedding and head contributions are summed, never
treated as separate trainable parameters. The three norm calls pass
`(gamma,beta)` as indices `(7,6)`, `(9,8)`, `(12,11)`. Snapshot acquisition
must be transactional under the exact workspace and P1/I2 lease authority:
on any failure release every acquired lease, publish no frame or output, and
leave model/optimizer/RNG bytes unchanged. A borrowed live parameter pointer
must not survive acquisition; forward and VJP read saved values only.

## Forward tape and reverse order

Each named f32 forward role is an independent workspace-owned tape slot. Its
shape and reverse edge follow the accepted C4 full-prefix provider and the
reviewed VJP row; no `[1,2,*]` M3T carrier is reused.

| Forward order / tape role | Shape | Reverse edge, in reverse order |
| --- | --- | --- |
| token `et`; position `ep`; embedding residual `x` | each `[1,4,4]` | `dx -> det,dep`; scatter `det -> g10_token`, `dep -> g13` |
| norm1 `n1` | `[1,4,4]` | `dn1 -> dx_norm,g7,g6` |
| query/key/value `qt,kt,vt` | each `[1,4,4]` | `dqt,dkt,dvt -> dn1_q,dn1_k,dn1_v,g2,g0,g3` |
| split heads `qh,kh,vh` | each `[1,2,4,2]` | `dqh,dkh,dvh -> dqt,dkt,dvt` |
| causal attention `ah` | `[1,2,4,2]` | `dah -> dqh,dkh,dvh` using saved Q/K/V and fixed keep |
| merged heads `at`; attention output `ao` | each `[1,4,4]` | `dao -> dat,g1`; `dat -> dah` |
| attention residual `r`; norm2 `n2` | each `[1,4,4]` | `dn2 -> dr_norm,g9,g8`; `dr -> dx_skip,dao` |
| FFN up `fu`; GELU `fg` | each `[1,4,8]` | `dfg -> dfu`; `dfu -> dn2,g5` |
| FFN down `fd`; FFN residual `y` | each `[1,4,4]` | `dy -> dr_skip,dfd`; `dfd -> dfg,g4` |
| final norm `nf` | `[1,4,4]` | `dnf -> dy,g12,g11` |
| tied head logits `z` | `[1,4,256]` | `dz -> dnf,g10_head` |

Reverse dispatch runs from `z` back to `et/ep`, retaining each saved operand
until its last VJP consumer has finished. Fan-in uses owned f32 sums:
`dn1 = dn1_q + dn1_k + dn1_v`, `dr = dr_skip + dr_norm`,
`dx = dx_skip + dx_norm`, and `g10 = g10_head + g10_token`.
All fourteen final parameter gradients have the exact parameter shapes above.
The causal attention VJP consumes Q/K/V and the fixed mask directly; the
generation A2 transaction/view is neither a tape owner nor a backward input.
The separate A2 empty-stage VJP leaf does not license using a generation
cache in training.

## Frame lifetime and next leaf

The proposed compute lifetime is idle -> captured forward -> consumed reverse
-> idle, with release terminal. A forward frame owns IDs, parameter snapshots,
all tape slots, mask and logits. Reverse snapshots `dz` before dispatch,
produces private gradients, and publishes complete gradients only after every
VJP and fan-in succeeds. Reset/discard destroys an active frame in reverse
allocation order; failure never exposes partial logits or gradients. Release
must reject a held lease/active dispatch, then free frame and workspace once.
Training updates cannot write parameters while a frame is active; stale-primal
checks before reverse/commit compare all fourteen current values and topology
against the captured frame. Exact byte restoration may pass that value check;
an epoch claim is not implied.

**Smallest numerical composition leaf after ownership admission:** authenticate
the live workspace; acquire/release all fourteen genuine parameter bindings;
snapshot the 4,768 bytes; run token embedding, position embedding and their
residual into owned `[1,4,4]` slots; prove deterministic values, repeat-ID
scatter and position gradients with numerical checks, plus every acquisition
and dispatch failure cut. This leaf must expose no partial frame and cannot
claim a full forward or gradient. It depends on compiled normal/repeat and
ASan+UBSan acceptance of the genuine constructor, P1, association and
workspace ownership path, including same-owner generation exclusion. The
remaining 18 roles, VJP union, loss, optimizer, persistence and EOS follow
later gates in that order. No numerical implementation should preempt the
ownership gate by fabricating a native owner, using a seeded generator, or
inventing a parameter-borrow API.
