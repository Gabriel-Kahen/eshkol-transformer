# G3-R private M3 owner-roster prerequisite

The feature-gated `et_m3t_private_owner_save_roster_v1` returns the fourteen
parameter owners and their original bound identities from one sealed M3T
registry owner. It validates the owner, completed initialization, every
parameter-to-identity binding and uniqueness of owners and handles before
copying either output array. A failed call leaves both arrays untouched.
The output spans must be disjoint from each other and from the owner and its
initializer controls; overlap is rejected before either array is written.
The API is private to the same-owner SAVE work; feature-off builds do not
export it.

This ports the bounded roster proof from divergent `c9bad736` onto merged
main `86f4eb36`, without importing the old G3-T ledger. It supplies the
original M3T identity roster needed by the proposed current-history held
capture. It does not itself retain a graph lease, prove value or absent-
gradient bits, hold O2 or G3/A2, clone P1, publish C2, or authorize SAVE.
Those operations require the separately reviewed F32/O2/G3 held-owner
prerequisites and compiled failure cuts.

The native M3T transport fixture checks an unsealed owner, a valid sealed
fourteen-slot roster, null and foreign owners, null outputs, a corrupted
identity binding, and unchanged output arrays after rejection. It also rejects
identical and shifted overlapping outputs and
owner-backed output storage without changing the sealed roster. Final
integration still requires supported compiled union validation before this
private roster can be used by the joint capture coordinator. Independent
source/tests review is accepted at `57a8e80a`/tree `8c5fcf94` (receipt
SHA-256 `c30e5c26ee68e7b51fb0779fc1e8e86d002e94d486fae818271f9fe79f9a438c`).
The direct native fixture passed normal and ASan+UBSan with leak checking
disabled; the supported M3 script stopped at its Ubuntu 22.04 host gate on
CachyOS. The private F32/M3/G3 source union is accepted with integration HOLD
(`g3r-current-main-57a8-union-independent-review-20261013-a/SHA256SUMS-REVIEW`,
manifest SHA-256 `cf7dce7a0a0f1059576904bc693b1506cd9e4bb865d385dbe9a0725f534bcb81`).
