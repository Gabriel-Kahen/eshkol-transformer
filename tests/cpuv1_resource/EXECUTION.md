# Stage A prepared execution gates

The preserved first native-unit attempt compiled two owner objects and failed
on the third, C4 owner, before the unit ran. Its census referenced optional types
absent under the frozen flags. The source correction guards those references
and explicitly reports kinds8/12 N/A; it does not widen the build profile.
Corrected native compilation/unit and all tiny gates remain NOT RUN.

Only root dispatches after independent source/test/recipe review, under the
existing exclusive heavy lock, 8GiB/3CPU/one compiler job. No sanitizers, alternate
allocator, malloc trimming, tracker reset or full caller are part of this gate.
No execution is authorized merely by preparation. Authenticate exact source tree,
pinned fe9 compiler/runtime/provenance and normal glibc profile before dispatch.

Prepare in a new directory using the authenticated absolute tool paths:

```
python -m tests.cpuv1_resource.prepare_recipes FRESH_EVIDENCE \
  --cc AUTHENTICATED_CC --cxx AUTHENTICATED_CXX \
  --eshkol-source AUTHENTICATED_FE9_SOURCE --eshkol-build AUTHENTICATED_FE9_BUILD
```

Preparation writes one explicit 111-unit generated source graph, source hashes,
exact sequential argv JSON and three separate scripts. It never executes a
compiler, preprocessor, native program or Scheme program. The recipe uses the
original35-TU roster plus the two accepted fixture TUs and exact per-fixture
O2/F32 test flags (`native_overlay.json`), then the test producer and normal-glibc
gates. The shared archive includes the original F32 metadata helper overlay for
both native unit and tiny; the tiny body never invokes corruption helpers.

1. Run `native-unit.sh` under bounded custody. This compiles each owner once and
   links `producer_unit.cpp` against the real runtime/archive. Require exit0,
   the native12-domain coverage, Scheme unavailable rejecting COMPLETE, genuine
   allocation/free/usable conservation, partial K1 cleanup, actual F32 metadata
   corruption and restore, retired payload/control distinction, eight genuine
   P1 callback identities (one provider shell versus nine native records),
   omitted/wrong-slot/wrong-context callbacks, authentic DEAD construction shell
   plus restored nonce/PID mutation negatives, declared runtime static extents,
   actual adopted arena blocks, layout/overflow/duplicate/scratch negatives.
   Require exact C4 compiled/N/A masks0x0eff/0x1100, real layout facts for every
   available kind, and rejection of any N/A membership or event history, even
   balanced allocation/free history. N/A zero fields never establish sizeof0.
   Preserve all logs,
   native depfiles, actual compiler argv, archive/object/ELF hashes and normal
   glibc runtime evidence. A source projection is not an object parity proof.
2. Separately run `tiny-compile.sh`. It builds only generated `tiny.esk`, with one
   real original owning graph/P1 initialization and one original candidate
   creation, train/eval mutation and UNADOPTED retirement. No optimizer/trainer,
   update, dataset, adoption, public generation or 375-step caller runs.
3. BEFORE `tiny-run-after-ir-review.sh`, independently review actual emitted IR.
   Seal a transitive call-path ledger rooted at `cpuv1-tiny-observe` covering open,
   generated owning getters, P1 and candidate envelopes, every visitor/helper,
   native census/runtime/hash bridge, and close. For EACH direct and indirect
   closure callsite record exact IR file SHA/line, resolved target set and source
   definition/capture provenance. Follow every target recursively. The success
   path may contain read-only loads, scalar/stack arithmetic, bounded traversal,
   preallocated packet/BSS writes, proven allocation-free native reads and error
   status writes. Reject reachable arena/scope creation/allocation/adoption,
   heap boxing, list/vector/closure/exception construction, hash resize/keys/
   values materialization, lazy initialization, generic malloc/realloc/free,
   unknown/foreign callees or unresolved indirect dispatch. Native libc metadata
   reads (`malloc_usable_size`, `mallinfo2`, `getrusage`) require the authenticated
   normal-glibc read-only implementation/profile; unresolved library allocation
   effects are a failure. Endpoint equality alone cannot exclude transient
   allocation below an old high-water mark. A source allowlist alone cannot
   establish this gate. No IR => proof NOT RUN; unknown target => proof FAIL.
4. Verify a single actual P1 code/provider/context initialization in emitted IR,
   dependency graph and final link map/symbol census. Every P1 dependency edge
   must target the same generated literal path; reject any original or second
   P1 module. Reverse AST projection of every unit, original75 P1 slots and
   original6 candidate slots must pass. Confirm lexical closure captures refer
   to the same canonical holders used by original APIs, not copied list heads.
5. Only after that proof, execute tiny twice in fresh actual processes. Require
   exit0, empty stderr, the exact tiny PASS line, all source-state/native/arena
   conservation and unchanged consecutive warmed observations at baseline,
   live-model and retired-model phases. Verify same P1 context,31 genuine new
   shell/index/native links, one candidate unpublished then dead, one outer
   owned custody live then dead, zero adopted model, and rejected native
   candidate capability/owner/P1-scope/phase mismatches. Source host predicates
   also reject same-object and copied-row duplicate capability references, even
   when a second same-phase native candidate is omitted to balance counts. Preserve actual IR,
   deps/link map, source/provenance/native/ELF hashes and process exits. Tiny
   cannot establish full frozen375-caller budget admission or flag-OFF object
   parity; both remain separate future gates.

C++ runtime bridge and native-unit commands explicitly retain
`-fvisibility=hidden`, matching the C producer profile. Actual symbol visibility
still requires the executed object/link-symbol gate; this flag is source intent.
The static-byte scalar sums producer declared object extents (including both
static packets and runtime root/page/warmed fields). One-packet extent is a
reported subset; linker padding/code/rodata and unit fixture globals/stack are
excluded from that scalar and must not be silently added or usable-queried.
