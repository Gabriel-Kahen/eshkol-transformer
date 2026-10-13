# G3-R/C2/O2 same-owner SAVE source boundary

Status: **source prerequisite union only**. This branch combines the current
main `bd81379c` C2 detached SAVE image/SHA receipt through `de95fc45` with the
current-main O2 hold through `f9173c1e`. It does not create a C2/G3-R pair,
serialize a live generator, or authorize replay. The G3/A2 held owner is
already on main, but the three leaves have no common transaction coordinator.

## Exact available interfaces

| Owner | Source interface | Authority and limit |
| --- | --- | --- |
| G3/A2/M3 | `et_g3r_private_g3t_save_begin/check/copy/end_v1` in `g3t_transport.h` | Pins the original generator/model/cache and yields 17 current-history words while held. C-only; no Eshkol caller currently invokes it. |
| O2 | `et_g3r_private_o2_save_begin/state/check/issued_match/end_v1` in `o2_optimizer_internal.h` | Pins the original 14-entry optimizer and can issue one checked native state. Its roster must first be authenticated by M3T. C-only; no Eshkol caller currently invokes it. |
| C2 | `c2-save-image-receipt-create/digest-copy/publish/release-internal!` in `c2_save_image_receipt_extension.esk` | Encodes an already detached C2 owner into one retained exact image. The digest authenticates those image bytes, not an original G3/O2 owner. |

The existing `g3r-committed-save-snapshot!` releases its native leases before
returning detached data. Feeding that data and an unrelated detached C2 owner
to an exact-image pair publisher would establish byte agreement only. It
cannot prove both were captured from the same original model, optimizer,
tokenizer, dataset cursor, and trainer-control epoch.

## Next implementation boundary

The next source leaf must be a feature-private coordinator built against these
exact current-history interfaces. It must root and reauthenticate the
original generator, sealed M3T graph, O2 optimizer, X1/T1 tokenizer, D2
dataset, and trainer controls; derive the native 14-handle/identity roster
from the registered model; acquire G3/A2/M3 and O2 holds in one documented
order; and keep both holds live through P1 cloning, O2 state issue and C2
claim. Check both original owners before and after every transfer. Compare the
issued O2 state through `issued_match` before it can enter the C2 owner. Copy
the G3 17 words under hold, preserving their current offsets and
generated-EOS provenance. Bind the original tokenizer fingerprint and
trainer-control epoch to the resulting C2 controls. Release holds before
receipt I/O; then create the image receipt from that exact detached C2 owner.

The coordinator also needs reviewed Eshkol ownership and rollback hooks:
preallocated/rooted child cells, a single-claim rule, first-error retention,
reverse-order cleanup before transfer, and fail-stop behavior after uncertain
release or partial claim. The old `c9bad736` coordinator is a useful ownership
reference but targets the retired 18-word ledger and cannot be copied into
the current 17-word committed-history owner. No current source supplies this
coordinator or a compiled aggregate link for all three leaves.

Admission needs independent source review followed by native feature-on/off,
normal/repeat/sanitizer, compiled post-adoption and partial-claim failure cuts,
then cold C2/model restore and fresh-process no-draw replay. Until those gates
pass, report the union as source-compatible prerequisites only.
