# G3-R C4 sealed-owner parameter lease status

**Source candidate, not compiled Eshkol admission.** This leaf adds the private
native `et_g3c4_private_sealed_owner_parameter_lease_status_v1(const void *)`
behind `ET_G3R_C4_OWNER_PARAMETER_LEASE_PRIVATE`. It returns an integer status
and the existing C4 last-error triple; it returns no owner, parameter, borrow,
view, or workspace. The caller must serialize owner operations and must have
already authenticated the exact C4 P1/native association.

The helper requires an exact registered, sealed owner with fourteen initialized
parameters and no active native call. It reuses the existing read-only owner
idle preflight, which validates the fourteen parameter identities and rejects
live value or gradient borrows and parameter, value, or gradient plan pins.
It acquires no pin and changes no owner, parameter, or registry state. An
unknown owner gives C4 `invalid-argument/identity`; an unsealed or active owner
gives C4 `invalid-state/lifecycle`; a borrowed or pinned parameter retains the
first I2 `invalid-state/invalid-handle` error.

The local native test uses genuine I2 value and gradient borrow begin/end on
all fourteen parameters, a gradient reset plan, staged/aborted and foreign
owners, and an active-call sentinel. It checks status/error triples, owner
bytes, parameter bits, lease release, and retry. The feature-off object must
not define the symbol. Normal, repeated, and native ASan+UBSan output must
match.

This status covers only **currently represented** C4 active calls and I2
parameter leases. No C4 workspace or trainer lease ledger exists in this
source. A later workspace-create gate must add and check those ledgers, then
compose this leaf with the accepted topology and exact P1/native association
checks before any workspace allocation. TR3's context-two trainer predicate
does not establish a C4 lease exclusion.
