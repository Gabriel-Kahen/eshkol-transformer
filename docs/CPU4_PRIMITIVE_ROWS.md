# CPU4 primitive shape admission

The A2/N2/N3K native algorithms retain their existing ABI, names and strict
floating-point recipes. This bounded extension adds exact CPU-f32 alternatives:

| Provider capability | New request rows |
| --- | --- |
| N3K embedding forward/backward | `[1,4,256,4]`, `[1,4,4,4]` |
| N3K linear forward/backward without bias | `[1,4,4,4]`, `[1,4,4,8]`, `[1,4,8,4]`, `[1,4,4,256]` |
| N3K GELU | `[1,4,8]` |
| N3K residual and activation sums | `[1,4,4]` |
| N3K head split/merge | `[1,4,2,2]` |
| N2 LayerNorm | `[1,4,4]` |
| A2 causal attention | `[1,2,2,4,4,2]` |

Head-layout requests use `(N,T,H,Dh)`, while head storage uses `(N,H,T,Dh)`
`[1,2,4,2]`. All predecessor rows remain admitted. No rectangular range or
neighboring T3/T5 shape is added. A2's raw provider validator supports generic
geometry; exact evidence admission and adjacent-shape rejection happen through
K1 discovery/dispatch. The change does not restrict that raw callback.

The current-source N3K predecessor manifest updates only the two affected A2/N2
source entries. The original baselines at commit
`574b98e42ee0d81491458fc5439c7efb64082014` remain:
A2 `fdb4ca172ca7aa282e787ff42b2998592c8b77a30f67d35680f7e38b2f550b02`,
N2 `81a213545f2e9256eedf57455b0ea33f56b05d908f1165813da9a2f5077137c1`.
Their historical numerical evidence is unchanged; new rows need this gate.

Run `bash scripts/test-cpu4-primitives.sh OUTPUT normal` and repeat with a
separate output directory and `sanitize`, using the pinned supported toolchain.
The gate uses existing provider build recipes and does not acquire a global
resource lock itself; coordinated runs must hold the team's heavy-build lock.

The independent Python standard-library oracle creates temporary fixtures for
28 forward/VJP calls. It evaluates binary64 mathematics and central differences
of that independent forward with step `1e-5 * max(1, abs(x))`, including every
differentiable input element. Native binary32 outputs/gradients must differ by
at most `3e-5 + 3e-4 * abs(reference)`, fixed before execution. Generated tensor
data are not checked into Git. Cases include repeated embedding IDs, every
position row, four linear shapes, nonuniform upstream gradients, asymmetric
T4/H2 layouts, and causal attention with a mask hole.

Each call also checks unchanged output bytes on adjacent request shapes,
truncated output, wrong dtype/device, nonfinite input, and applicable invalid
index/mask errors. Layout calls reject transposed tensor metadata and head
storage metadata used as a request. These primitive checks do not establish
model training, C4 ownership, checkpoint compatibility, evaluator admission or
generation correctness; those require the upstream composition gates.

Device negatives distinguish two K1 stages: capability lookup for a `cuda`
request returns `UNSUPPORTED / CAPABILITY_NOT_VERIFIED`; dispatch of that
request with CPU operands returns `DEVICE_MISMATCH / INVALID_BUFFER` during
generic operand validation, before capability lookup or provider invocation.
