# E3 exp-before-global mutant source

This noninstalled §9 M2 source candidate starts from exact T2 `44c0418`.
The authentic packed three-batch D2 corpus has masks `11,11,10` and frozen
compiled CE numerators `4131ddf7`, `41314df0`, `40b294f5`. They produce
unequal batch losses. An independent rational and Decimal-exp oracle gives
normal `exp(global N/W)` word `43815d49` and the mutant's token-weighted
mean of three batch perplexities `438160e2`. The gap is much larger than the
allowed libm tolerance.

Only `ET_E3_NUMERICAL_EXP_BEFORE_GLOBAL` adds frame-local binary32 state.
After each successful real K1 accumulation, it reads actual L3S batch N/W,
computes `expf(N_b/W_b) * W_b`, and adds the contribution in batch order.
After normal K1 finalization, it substitutes only the frame's perplexity
result with this weighted mean before publication. The real D2 rows, M3
forward roles, CE, reductions, serial N/W/C, loss, accuracy, counters and
cursor remain unchanged. No expected value is injected into the runtime.
Canonical, diagnostic, installed and public packages cannot select the flag.

The fresh private gate demands duplicate normal and mutant builds, package
closure checks, complete normal/repeat/poison/sanitizer traversal, a frozen
normal transcript, then four mutant failures first at the final perplexity
comparator. The mutant verifier independently recomputes both perplexities
from actual observed batch reductions, allowing at most four binary32 ULP
for three libm calls and their aggregation. No rejected mutant transcript may
be published.

Status: 16 source/host tests pass. Supported compiled-private execution and
independent review are **NOT RUN**. The test-only mutation does not prove
production E3, public/installed behavior, hardware, or the full §9 matrix.
