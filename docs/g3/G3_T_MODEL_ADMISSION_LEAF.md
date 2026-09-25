# G3-T M3T/C2 model admission leaf

This source-private precursor follows the accepted [G3-T contract](G3_T_PRIVATE_CONTRACT.md).
It adds no public generation name, native generator, or transport package. Load
`g3t_model_admission_extension.esk` after `m3_package_root.esk` and
`m3_call_adapters.esk` in the single trusted source aggregate. The root cell is
`g3t-registry = (vector '())`; this leaf never enrolls a G3 owner.

`g3t-model-entry-live(model, operation, active-call)` returns the authentic M3T
model entry only after exact registry lookup. It requires the live 12-slot
model entry, native owner, rooted live original and successor initializers,
resolved two-string config record, 14 distinct canonical P1 handles with the
fixed M3T paths/shapes and f32/cpu metadata, M3T profile, eval mode, and an
active-frame slot exactly equal to `active-call`. Call it inside the existing
`m3-call` guard; the tested idle case passes `#f` as `active-call`. Model
construction, handle/native binding, topology, tying, and sealing remain owned
by M3T/I2/P1. Later G3-T code must independently authenticate native pointers
and acquire the accepted fixed-14 pins before publishing a busy call.

The isolated witness creates a real M3T C2 model through the checked adapter,
admits it, rejects copied/foreign/wrong-kind shells and changed entry metadata,
rejects train mode and a real active M3T workspace, then confirms restoration.
The pinned runner tests normal, repeat, and address/undefined sanitizer modes.
The direct source/AOT witness checks that rejected calls throw and clear the
shared guard; it does not claim a public error envelope for this private leaf.
The next dependency is the native G3-T generator context/registry root: it
must bind this authentic model owner and the accepted A2 cache and policy,
authenticate pointer identities, then implement transactional begin/close and
the fixed pin lifecycle. This leaf alone cannot produce G3-T C2 output.
