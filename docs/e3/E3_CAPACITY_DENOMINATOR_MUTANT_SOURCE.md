# E3 physical-capacity denominator mutant source

This is a noninstalled, test-only candidate for the E3 §9 M2 denominator row.
It reuses the frozen sum-order packed-single corpus and its authentic D2 masks
`11,11,10`. The source variant `capacity-denominator` keeps the third-row CE
values, mask, targets, L3S numerator and E3 correct count. The L3S provider
instead returns physical capacity `W_b=2` and `loss_b=N_b/2` on the final `10`
row. A numerical-tuple-only E3 admission exception lets that exact third-row
weight/active mismatch reach serial aggregation and final publication; all
ordinary production checks retain their original code.

`verify_capacity_denominator_mutant.py` checks the normal AOT observation
against the frozen independent M3/D2/CE witness first. For a mutant, it requires
the same 63 model roles, all prior batches, final row, CE and correct count.
Its first allowed difference is third-row `reduction_f32[1]`, from `1` to `2`.
The rational binary32 oracle then checks `loss_b`, serial `W`, unchanged active
token and batch counters `(5,3)`, and all four final words. The ordinary
transcript validator must reject at the third reduction weight; the gate must
leave no mutant transcript.

`test-e3-capacity-denominator-mutant.sh` prepares fresh normal and mutant
private AOT packages twice, compares objects, archives, depfiles and executables,
then runs normal, poison and sanitizer variants through the genuine three-batch
evaluator. Package inventory and canonical observer-isolation checks remain
required. The script is a proposed compiled gate; source and host checks alone
do not prove compiler admission or a compiled kill. The active E3 gate owns the
sole heavy slot. The full E3 §9 matrix remains open.
