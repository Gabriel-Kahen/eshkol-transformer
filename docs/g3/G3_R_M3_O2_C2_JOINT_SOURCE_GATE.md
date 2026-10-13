# G3-R M3 owner to O2 and C2 joint source gate

**Noninstalled candidate; integration HOLD.** The source candidate at
`20e86b192c23e59149f47c1bb0814918b19b41d7` started at the accepted
O2-issued C2 fixture `098368be` and ported the feature-gated M3T roster from
`513f50af`. The current-main union starts at `6f6ef9ada4c586573d495c40283ad4b9410d0957`.
Its M3T source, header, native test, and prerequisite document already match
the candidate blobs exactly; this union retains the main F32 witness, G3/A2
17-word held owner, C2 operational changes, and predecessor pins. It adds no
public or installed ABI.

`tests/g3r/m3_o2_c2_joint_runtime.esk` constructs two sealed M3T models with
the same deterministic configuration and distinct native owners. Each O2
optimizer is built from its own model's P1 parameter tree. The test bridge
calls `et_m3t_private_owner_save_roster_v1` on the original sealed M3T owner.
O2's existing `et_g3r_private_o2_save_begin_v1` then authenticates all fourteen
issued parameter and bound-identity pairs against the registered optimizer
before creating a hold. A foreign M3 owner and wrong-kind pointer cannot begin
the hold. The test separately observes exact same-owner and foreign roster
matches. No caller-supplied parameter list can authorize the hold.

After the same-owner hold issues one native O2 state, the existing private
adoption wrapper registers that exact pointer. The existing C2 compose claims
it together with the M3 model's P1 state dictionary. The inherited test covers
foreign/equal-bit O2 states, duplicate adoption, counter and P1 mismatches,
second-claim rollback, same-owner retry, and C2-only retirement. The joint
extension calls the existing private C2 SAVE writer after the O2 hold ends;
it checks an invalid path and deterministic file bytes from two AOT builds.
The SAVE claim is conditional on a passing supported compiled/runtime gate.

Run `scripts/test-g3r-m3-o2-c2-joint.sh /fresh/absolute/output` on the pinned
supported Eshkol/Clang host. The script builds one M3T/O2/C2 native archive,
compiles the Eshkol fixture twice, checks identical executables, output and
checkpoint bytes, and leaves artifacts under the supplied fresh directory.
The earlier source-only candidate passed host Clang syntax checks, a combined
relocatable native object link, the M3T native roster/transport test, and
feature-off symbol absence. This current-main union needs independent review
and a fresh supported Eshkol AOT/runtime gate. Joint allocation-failure cuts
remain required before integration or any G3-R pair/restore claim.

The first one-use supported-host B attempt reached the joint AOT compiler and
failed before runtime at `m3t_transport_extension.esk:265`: its T1-named sealed
token predicate was absent from the D2/T2 aggregate. The B host/guest/home,
claim, and quarantine receipts remain preserved; no SAVE bytes were produced.
This source-only successor binds that existing T1 name to the T2 root's exact
same-aggregate admission predicate before loading M3T, as the accepted TR3
private aggregates already do. A focused closure test checks the upstream
predicate and rejects a missing, late, or second-registry binding. The
separately released C executor has no runtime result in this source gate.
The combined M3/F32/G3/A2/O2/P1/C2 state, pre/post transfer bit checks,
error cleanup, and file-pair publication remain unproved.
