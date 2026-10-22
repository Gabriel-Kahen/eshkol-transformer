# Minimal CPU4 no-grad E3 frame — accepted private source target

Basis: provider source d5970d77cf5ea1c76aa303b42c7b699751bfff01, lease producer063bc4c93ee2c1c62d108d0d4818182613c93a9f, accepted r2 lease SHA f97e2177ae06bbe0e5de6031e1af3b0bf91b95c1281c6e3f1502be4cabdfe8ab. Independently accepted source target; frame implementation/admission not claimed.

```c
void *et_e3_private_cpu4_frame_create_v1(
    void *existing_lease,
    et_f32_tensor *loss, et_f32_tensor *weight,
    et_f32_tensor *perplexity, et_f32_tensor *accuracy);
```

The constructor borrows exactly the existing trainer/evaluator lease and four distinct canonical scalar destinations. No raw C4 owner, caller roster, C2OWNER cast, lease creation/release, graph or backward transport. Bind authenticates lease/getters and destination/parameter aliasing; numerical acquisition alone enters READ. Busy acquisition rejects before scratch/mode/publication changes. Existing opaque E3 operation signatures can be retained in a CPU4-private tuple selecting its lease-bound native implementation; default legacy tuple remains separate.

Persistent E3 control contains registry/lifecycle, lease, destinations, phase and published scalar/counter state. Every call creates a fresh registered workspace identity; no numerical scratch survives leave. The composite kind2 workspace owns unexposed IDs/targets/mask/positions/keep,21forward slots, CE/reduction/metric buffers and provider contexts. Root must explicitly accept inputs/logits as internal subobjects, otherwise kinds1/4 need distinct controls. Register actual C4 read-pin control separately as kind5; use existing et_g3c4_model_pins_begin/check/end_internal under ET_G3C4_NATIVE_PINS_PRIVATE, with14parameter/identity getters from the approved lease. Child-control addresses are single-use and retained as cleared tombstones; scratch storage is released, controls/retention measured separately.

Acquire authenticates control+lease, enters READ, registers workspace, builds scratch and pins, then publishes acquired state. Any partial failure snapshots the first native triple, self-unwinds all pins/storage/children and leaves before returning; Eshkol sets acquired only after native success. Same-bound-frame retry remains valid. READ spans21forward roles, cursor/mode restoration and atomic four-scalar/two-counter publication. Drain ends pins; finish destroys all scratch/provider contexts, unregisters children, then leaves. Post-preflight cleanup invariant failure is fail-stop. No caller-controlled child-count/leave wrapper.

Shapes: IDs/targets/mask[1,4], positions0–3,keep16; actualheads[1,2,4,2] distinct from split/merge request[1,4,2,2]; full logits[1,4,256]. Dispatch existing reviewed native providers only. Scalar bounds are8192batches/32768tokens. No graph/gradient/RNG/parameter mutation.

First-error projection: underlying K1/I1/I2 triples retain existing E3 dependency mapping. C4-native invalidargument→E3identity, invalidstate→E3phase, shape→E3shape, unsupported/determinism→E3profile, allocation→E3allocation, otherinternal→E3invariant; unknown triples reject invariant. Preserve raw C4 snapshot across cleanup before the mapped E3 triple/stage is reported.

Joint Eshkol freeze still required: exact canonical C4 entry/config/seed/14handles ↔ existing lease proof, genuine P1token with17-mode restoration, D2T4 binding/cursor rollback. CLI3 owns that producer. No full training graph FFI prerequisite or invented producer accessor.

## Independent acceptance conditions

The composite kind2 workspace owns unexposed input/logits/scratch subobjects;
no separate kind1/4 controls are required while no subobject handle escapes.
Every stage authenticates the workspace registry, lease and current ordinal.
The actual C4 pin control is separately registered kind5. Both controls are fresh,
single-use retained tombstones on each acquisition. Acquire failure completely
self-unwinds before returning. Unregister only after successful pin/storage
release preflight; READ spans restoration/publication and leaves after all children.

The persistent frame only borrows its lease. Its caller drains/destroys it before
lease release, or explicit stale-frame destruction must remain safe. It never
creates/releases the lease. First raw C4/K1/I1/I2 triples are captured before
cleanup. Map only known category/code pairs, including C4 INTERNAL plus ALLOCATION
code to E3 allocation; unknown triples map to invariant. Retain the raw triple.

The canonical C4/P1/D2 Eshkol producer remains required for integration, but the
native no-grad frame can be implemented against the actual approved lease now.
