# Wave 3 G3-C4 owner/pins and I2 restore native integration

Status: **bounded native integration accepted by the orchestrator; supported
combined native gates pass; package/AOT/full CI remain pending**.

This candidate merges two native-only parents over their common reviewed
runtime/reference base `d5fa9a71b000ab1139a25e9b734b230ab5bda759`,
tree `185869ef7f79b9086860c9fdc6ea9aaaa8d7cd94`:

- the root-accepted G3-C4 seeded owner and fixed-14 pin integration
  `25a700be1a763b6c992a4322fd3f2b8401f54492`, tree
  `f0f6c150e49fb51361b5beb2bc4c8ec59315f08e`;
- the independently approved and supported I2 restore integration
  `e20677f2cecd99b5e4aef7b32fa053d5c1ebf5e4`, tree
  `38015f3c2953b7f5ca18e7331b68ef2c6bda8692`. This is the documentation
  successor of native source commit `b3e846417cae69f653645c693ae3a7aa92cc6961`,
  tree `9f6b3c41d731cec3a080aebd04a741af6bfe9455`, which root accepted as a
  bounded native-only integration.

The complete per-path comparison is recorded in
[`WAVE3_G3_I2_NATIVE_INTEGRATION_BLOBS.tsv`](WAVE3_G3_I2_NATIVE_INTEGRATION_BLOBS.tsv).

## Composition

The parent deltas overlap only in `docs/ROADMAP.md`. Its resolution retains the
accepted G3-C4 native row and the accepted I2 restore status in the TR3 row. No
production, test, manifest, or build file required hand resolution.

Every G3 production blob is exact to `25a700b`. Every I2 production, test,
symbol-manifest, build-target, and gate blob is exact to `e20677f`. The only
non-documentation successor is `scripts/test-g3c4-pins.sh`: its frozen
ordinary-object baseline advances from `d5fa9a7` to `e20677f`, so that comparison
includes the three accepted I2 constructor initializers as well as the preserved
SHARED-R2 provider.

The merged `native/f32_tensor.c` is the exact I2 parent blob. It retains provider
1.1/v2, all five bounded rank-two storage-copy shapes, overflow checks, and seven
shape ranges, then adds only the reviewed owned-clone matcher, testing hooks, and
three initializer fixes described by the I2 parent. It is not replaced by the
older pre-SHARED-R2 source.

`m3_call_f32_integration.c` still includes the single M3T/I2 implementation
translation unit. C2 and C4 pins and the I2 restore bridge therefore authenticate
through the same live and retired I2 registries. The merge adds no second I2
registry, copied parameter authority, active G3 owner seam, or new public entry.

## Supported combined evidence

Both gates were rerun from this combined source in the immutable Ubuntu
22.04/LLVM-Clang 21.1.8 image
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`,
with networking disabled and source mounted read-only.

The I2 restore standalone gate passes with Clang 21 and GCC 11 at O0 and O2,
including the ordinary-object and symbol comparisons against the exact
`d5fa9a7` plus three-initializer baseline, strict aliasing, exact defined and
undefined manifests, checked restore bridge, negative and failure-atomic cases,
terminal behavior, normal repetition, and ASan/UBSan/LSan.

The G3 native closure passes after comparing ordinary M3T and M3-call objects
against `e20677f`. Owner normal, repeat, and ASan/UBSan/LSan output is identical
at 6,835 checks including all 182 K1 and 182 I2 allocation failpoints. Pin output
is identical across normal, repeat, and sanitizer runs, including fixed
1,024/8,192-cycle retention, 98 busy controls, 172 geometry defects, fourteen
prefix failures, and 25 fail-stop cases. C2 output is identical in normal and
sanitizer modes with the private C4 macro absent and present. Conditional symbol
diffs remain exactly one owner helper and three pin helpers, and every execution
writes empty stderr.

The evidence bundle is `wave3-g3-i2-native-union-evidence` under the root
visualization directory. It records the exact source tree, parent commits,
container provenance, executed commands, logs, source hashes, and evidence
checksums.

## Boundary

This candidate adds no G3 construction bridge, P1/Eshkol construction, O2
restore, trainer, active owner attachment, transport, KV cache, sampler,
checkpoint format, public API, package manifest, CI registration, or runtime
dependency. It does not establish package/AOT/full-CI behavior, active-model
retention, joint restore, resume equivalence, or generation behavior.


## Root acceptance

The orchestrator accepted source commit
`3928fc01f06a3b2634a34ef917b55428b047c8d8`, tree
`66ff3019c4807f34211ea4281bcd436d15fab81c`, after independently checking
all 29 source mode/blob rows, 30 source hashes, all 64 evidence checksums,
ordinary-object byte identity, normal/repeat/sanitizer transcript identity,
empty execution stderr, and the exact one-line pin baseline adaptation.
The evidence manifest SHA-256 is
`7830227ceb7f91013d477f439414985ab2cd8eeb79733ac84784986845d8ab5c`.
The first I2 attempt failed only because its temporary executable directory was
mounted `noexec`; that log is preserved. The successful retry changed only the
scratch mount policy. This acceptance applies to the bounded native union and
does not extend the boundary described above.
