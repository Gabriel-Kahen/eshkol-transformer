# G3-C4 native owner and fixed-14 pin integration

Status: **integration-preparation candidate; supported native closure passed;
independent integration review and root acceptance pending**.

This candidate starts from the independently approved runtime/reference commit
`d5fa9a71b000ab1139a25e9b734b230ab5bda759`, tree
`185869ef7f79b9086860c9fdc6ea9aaaa8d7cd94`. It composes two accepted native
leaves:

- seeded owner `bd977aa9f968500a1cd0a8729e01a72286e50a95`, tree
  `a33476064fc563dbc34fdc167e5dc552bab583c2`;
- fixed-14 pins `e11e61f4a3da6682c49c5422040baefdd32d8d4e`, tree
  `b044c0cc92f52c3dfa1960abf0028c68313e8a2d`.

The complete per-path source and integrated blob comparison is recorded in
[`G3_C4_NATIVE_OWNER_PIN_INTEGRATION_BLOBS.tsv`](G3_C4_NATIVE_OWNER_PIN_INTEGRATION_BLOBS.tsv).
All production blobs are exact to their accepted source. The owner test, pin
test, and both leaf contracts are also exact. The pin runner is the sole
integration successor: its ordinary-object comparison now uses `d5fa9a7` as
the baseline instead of the earlier owner commit, so the comparison includes
the accepted rank-two I2 provider already present in this candidate.

## Shared-source composition

There was no production conflict. Before admission, the four shared source
paths in `d5fa9a7` were byte-identical to the corresponding accepted leaf
bases:

| Path | Integration-base blob | Accepted-base relationship | Integrated blob |
|---|---|---|---|
| `src/eshkol_transformer/m3t_f32_integration.c` | `6ff04fa7648de4f2afcbccf920977f14212971f6` | exact to owner base `e899215c` | `a23de6dba41bd24be689bc6f25d190f3f7cd3a8b` |
| `src/eshkol_transformer/m3t_f32_scoped.h` | `289f216be4230248e206aeec9a601db8bec40e1d` | exact to owner base `e899215c` | `7bf0df09141991424f8cf0af187dcf8c21b6760f` |
| `src/eshkol_transformer/m3_call_f32_integration.c` | `3e3aec697682f94e84ce1d2937dbe17368a05fbb` | exact to pin base `bd977aa` | `fcc503d0ec15a22e70b20a5f13204a75e35d5b27` |
| `src/eshkol_transformer/m3_call_pins.h` | `7a9e28ac6c68f32432a202c7d1ecfd05b4716a6b` | exact to pin base `bd977aa` | `9c403b19f594a6a10fec2d8584a3887164a3a44b` |

`m3_call_f32_integration.c` includes the existing M3T/I2 implementation
translation unit. Both the C2 and C4 pin records therefore authenticate and
hold parameters through the same `native/f32_tensor.c` live and retired I2
registries. This candidate adds no second I2 registry or copied I2
implementation. The seeded owner retains only its accepted C4 owner-lifecycle
registry, which authenticates owner handles and does not replace I2 parameter
authority.

## Scope

The owner helper exists only under `ET_G3C4_NATIVE_OWNER_PRIVATE`; the pin
sibling exists only under `ET_G3C4_NATIVE_PINS_PRIVATE`. With those feature
macros absent, no new symbol is emitted. No Makefile or CI topology entry is
added, and no existing aggregate activates either feature.

The integrated source remains native-only. It adds no Eshkol alias, P1
construction, active model attachment, transport, KV cache, sampler, restore,
checkpoint surface, public API, package manifest, or runtime dependency.
Existing ordinary C2 pin behavior remains in the same implementation and is
exercised with and without the private C4 macro by the focused pin runner.

## Evidence boundary

The frozen supported evidence directories are
`g3c4-native-owner-bd977aa-evidence` and
`g3c4-native-pins-e11e61f-evidence`. Their approved leaf source blobs are
preserved as described above. They are not exact-closure evidence for this
integration tree: both compiled leaves include `native/f32_tensor.c`, whose
accepted `d5fa9a7` blob `1854cfa27e2a5d1fa896010da83ba3f042ec782a`
differs from the older evidence input because SHARED-R2 added the bounded
rank-two storage-copy shapes. The I2 registry and parameter layout are
unchanged. Those older results were not reused; the required exact-closure
rerun is recorded below.

The composed closure was rerun in the immutable supported Ubuntu 22.04/Clang
21.1.8 image
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`,
with networking disabled, a read-only container root, and a read-only source
mount. The evidence directory is
`g3c4-native-owner-pin-union-9650a674-evidence`. The run established:

- byte-identical ordinary M3T and M3-call objects against `d5fa9a7`;
- only `et_f32_parameter_idle_preflight_internal` added by the owner macro and
  only the three `et_g3c4_model_pins_*` symbols added by the pin macro;
- identical normal, repeated, and ASan/UBSan/LSan owner output: 6,835 checks,
  including all 182 K1 and 182 I2 allocation failpoints;
- identical normal, repeated, and sanitizer pin output across the fixed 1,024
  and 8,192-cycle retention checks, 98 busy controls, 172 geometry defects, all
  fourteen injected prefix failures, and 25 fail-stop cases;
- identical C2 normal/sanitizer output with the private C4 pin macro absent and
  present, including 1,024/8,192-cycle retention, fourteen failure prefixes,
  and 25 fail-stop cases; and
- empty stderr for every execution plus a manifest of exact source hashes and
  commands.

The repository `verify_toolchain` wrapper was not executed because this was a
direct native closure rather than a configured Eshkol build. Image, compiler,
OS, network, root-filesystem, and mount provenance were recorded directly.
No Eshkol compiler, package, AOT, or full-CI result is claimed.

This evidence does not establish package integration, retention under a future
active model seam, or any end-to-end generation behavior.
