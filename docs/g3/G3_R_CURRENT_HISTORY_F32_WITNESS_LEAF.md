# G3-R current-history F32 bit-witness prerequisite

This bounded feature-gated port brings the source-private F32 SAVE witness
from divergent `c9bad736` onto merged main `86f4eb36`, under the reviewed
[held SAVE proposal](G3_R_CURRENT_HISTORY_HELD_SAVE_CONTRACT.md). It changes
no public F32 caller ABI. `et_f32_parameter_save_witness_begin_v1` requires
an already admitted genuine M3/G3 fixed14 value pin, the expected identity,
and canonical rank/shape. It makes detached bit baselines for the value and
the absent-gradient storage. `check_v1` compares exact owner and child
identity, backing/shape/stride identity, all value and gradient bits,
positive-zero absent-gradient state, and metadata. Numeric equality is
insufficient: signed zero and NaN payload changes must be visible. A witness
does **not** authenticate the M3T model owner on its own.

`end_v1` destroys both baselines, invalidates the witness, and clears the
caller slot even after a failed check; it is also the abort/drain path. A
partial clone/allocation failure destroys any first baseline before returning
the original typed error and leaves the output slot empty. Released witness
shells remain inert so stale pointer addresses cannot become new authority.
Feature-off compilation exports none of the three witness symbols. The
bounded token budget is 896 lifetime captures; exhaustion is an explicit
invalid-state result, not a silent fallback.

The focused native fixture constructs fourteen actual F32 parameter owners
at merged-main M3 call-pin shapes, including position embedding `[2,4]`, and
pins them through the real fixed14 pin implementation. It changes value and
absent-gradient bits separately, checks mismatch and recovery, injects a
second-clone allocation cut, compares live tensor/borrow counts, and drains
after a failed check. Normal, repeat, and ASan/UBSan builds run the same
fixture; feature-off symbols are absent. These checks do not establish a
sealed M3T roster or same-owner G3/A2/O2 hold. Those are separate prerequisites
before C2 composition, paired publication, or a SAVE/continuation claim.
