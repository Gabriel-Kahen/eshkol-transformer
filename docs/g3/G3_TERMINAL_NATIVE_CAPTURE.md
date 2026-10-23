# Terminal ID native-handle capture on fe9

The trained G2 sanitizer run found a one-byte read seven bytes before the
48-byte native clone control. The control was still allocated. The original
and sanitizer-derived IR both show the pinned compiler probing an Eshkol
borrow header when retiring the captured `original-native` variable.

This is an upstream compiler boundary defect: fe9's `llvm_codegen.cpp` packs
an external `ptr` return with `ESHKOL_VALUE_HEAP_PTR`, while
`emitBorrowMutationCheck` assumes a non-null value with that tag has an Eshkol
header. Native clone controls are opaque C allocations with no such header.
The compiler pin and all borrow/sanitizer checks remain unchanged.

The private adapter now preallocates a managed singleton vector before the
native clone call. Its capture is immutable. Bind and retirement write its
fixed slot, and identity checks and release read that slot. Retirement still
clears the handle; the dead-owner proof also requires the cleared slot.
Vector mutation inspects the actual managed vector's header, without reading
an overwritten native handle's header. On fe9 the region barrier recognizes
external addresses by arena ranges and copies their tagged values unchanged;
clearing to null needs no promotion allocation.

This changes private Scheme allocation from a mutable capture box to a
singleton vector. Actual allocation/retention deltas remain unmeasured and
must be included in the CPU-v1 resource gate. Native layouts, clone payload,
native tombstone retention, public API and numerical behavior are unchanged.

The host regression executes the actual Scheme adapter with native fault
stubs for both logical lengths, including handle substitution, held-release
retry, cleared authority, repeated release and zeroed readback. It is control
flow evidence only. Required compiled proof remains the genuine terminal
publication/copy/release path under the pinned compiler with generated-IR ASan,
followed by the full trained G2 sanitizer gate. Neither a host pass nor source
checks establish that compiled acceptance.
