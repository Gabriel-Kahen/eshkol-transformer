# G3-R M3 owner to O2 and C2 joint source gate

**Noninstalled candidate; integration HOLD.** This branch starts at the accepted
O2-issued C2 fixture `098368be` and ports the exact feature-gated M3T roster
implementation, header, and native tests from `513f50af`. The port changes no
public or installed ABI. The earlier O2-only fixture remains untouched.

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
This source-only preparation passed host Clang syntax checks, a combined
relocatable native object link, the existing M3T native roster/transport test,
and feature-off symbol absence. It has not run the supported Eshkol AOT/runtime
gate. Independent source/test review, joint allocation-failure cuts, and
supported compiled execution remain required
before integration or any G3-R pair/restore claim.
