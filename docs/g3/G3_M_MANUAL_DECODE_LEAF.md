# G3-M source-private one-token manual decode leaf

Status: isolated implementation candidate under the accepted
[composer contract](G3_M_MANUAL_DECODE_COMPOSER_PROPOSAL.md). The private
`(g3m-decode-one! generator input)` uses one guarded kind-1 G3-T call, one
authentic owned i64 `[1,1]` input, 21 ordered T1 roles, and the existing
prepare/preflight/commit/finish tail. Native validates the length-one prefix
and owns the A2 append and detached f32 `[1,256]` result. No sampler, draw,
new native ABI or public facade is installed.

The focused aggregate checks P1 and P1/G0 predecessors, independent M3T
logit/KV parity, cache length/mask, rollback and failure status, detached
ownership and production feature-off behavior. Independent source review and
hosted integration CI remain pending.
