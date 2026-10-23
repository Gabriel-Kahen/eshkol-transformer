# CPU4 binding guard publication repair

SOURCE ONLY; actual fe9 constructor handler-OOM/native execution is **NOT RUN**.
The producer baseline is79ea814d994417d646835f5d09f26459869bf4a7. The only
production change moves the constructor's existing guard before publication of
its global pending binding row. Envelope/cons construction can still allocate
before the guard because it owns no native lease and publishes no authority.
The handler is active before the first `vector-set! cpu4-model-bindings`.

At pinned fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9,
`lib/core/runtime_exceptions_hosted.cpp`1014–1032 takes a recycled handler or
calls malloc(sizeof(handler)); failure raises static emergency condition5 to
the previous active frame BEFORE linking the new handler. The guard codegen
in `lib/backend/llvm_codegen.cpp`21765 onward uses this actual push. These
are source observations, not a new runtime experiment. Existing
`tests/m3cg/handler_failure*` and `scripts/test-m3cg-repaired-handler.sh` show how
a real witness must occupy recycled frames until genuine allocation, calibrate
exact static condition5 identity, and fail only handler allocation persistently.
C2's existing release mechanism similarly reserves its own known handler depth
before moving release authority. This binding fix changes neither mechanism,
compiler, runtime, reservation count, pin nor native ABI.

If installation itself fails, the constructor body cannot have published the
pending row, so retry has no durable blocker. If a guarded allocation/P1/roster/
native-create failure occurs after publication, cleanup rereads the actual global
registry. It marks only a row whose immutable native owner equals this attempt's
prechecked owner AND whose state is still `pending`; prior overlap validation
ensures no other live/pending binding for that owner exists. It stops at that
row and preserves every unrelated binding, including unrelated pending rows.
It captures no mutable publication flag or prepromotion canonical row. Failure
before the global store has no matching row to retire. The original caught
value is re-raised unchanged.

The checked P1 capture, strict TRAIN admission, all14 canonical handles/carriers,
roster checks and final native lease allocation are identical to the baseline.
The successful rooted pointer/state/token store tail is identical. All other
production forms, including the f064 private D2 error-normalization adapter,
are AST-identical to79ea. No mode relaxation, source-capability change, public
API, numerical behavior or shared root/roster/manifest change is included.

Cheap commands:

```
bash scripts/test-cpu4-binding-source.sh
GUILE_AUTO_COMPILE=0 guile --no-auto-compile -s tests/cpu4_binding/guard_host_test.scm
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.cpu4_binding.test_guard_source
```

The existing host fixture retains its40 binding /66 total actual D2-semantic
checks. Its default no-op host guard-install/P1-capture hooks permit focused
fault injection only in development tests. `CPU4_BINDING_SOURCE` optionally
selects a temporary source input for baseline sensitivity; it is not a delivered
training/constructor parameter.

The new host fixture executes the actual constructor through those explicit
Guile guard/P1/native doubles. It injects failed handler installation before the
host catch becomes active, models publication promotion by replacing the rooted
row with a different vector before a P1 error, and exercises strict TRAIN and
native NULL failures. It checks exact original caught-object identity, current
canonical row cleanup, untouched stale prepromotion row, unrelated live/pending/
dead rows, and successful retry on the EXACT same model/native owner after
every case. Retry then releases its actual host-double lease. These23 checks
exercise source control flow, not fe9 allocation or native ownership.

Baseline sensitivity is run using unchanged79ea source in a temporary input,
without checkout edits: `CPU4_BINDING_GUARD_CASE=install` fails at `handler install
happens before any pending publication`, and `promoted` fails at `cleanup rereads
promoted canonical pending row`. Both return1; repaired source passes all23.
The Python3 source tests additionally reject the old publication order, owner/
pending/stale-row/rethrow mutations, and prove the original checked native tail
and all other production forms unchanged.

Next executable gate belongs to root: independently accepted source composition,
pin refresh and a focused real fe9 handler-allocation failure/retry witness,
with genuine condition5 identity/free-list exhaustion and preserved source/IR/
artifact provenance. Source order and host promotion/fault doubles cannot replace
that compiled witness. No heavy build, native OOM injection, constructor retry
admission or final candidate-ON trainer claim is made by this leaf.
