# Private sealed-M3 fixture producer (source candidate)

`scripts/generate-e3-sealed-fixture-source.py` creates build-only P1, I2 and
M3T source overlays from pinned canonical bytes. The ordinary constructors,
installed library, public API and default-off native objects are untouched.
This tuple requires `ET_G3R_CANDIDATE_RETIRE_PRIVATE` on its private native
objects; it is not currently registered in the fixed E3 package builder.
There is no runnable aggregate root yet: the canonical E3 root loads the
ordinary P1/I2/M3T sources, so loading this driver through that root would
not select the generated overlays. A later reviewed package composition must
bind all three generated members and their exact native objects together.

The candidate P1 construction begins with the existing native candidate
entrypoint and retains its sealed token roster. I2 retains exact construction
memberships and carriers. Before native begin, M3T roots an empty ledger and
then records the canonical root, original, successor, P1 construction, I2
record, and each of fourteen indexed `(native parameter, P1 handle, I2
carrier)` triples before the next fallible step. The head and token-embedding
paths share handle 10; they do not create another owner.

Retirement admits only the exact sealed fixture, checks live workspaces and
frames, the retained path/handle schedule, indexed M3T bindings, I2 carriers,
and the native P1 roster. The successor preflight precedes the final owner
ticket. Unexpected failure of that final ticket or any postboundary commit is
terminal (`abort`, 134); no P1 validation is attempted after revocation.
The serialized tail revokes P1 tokens, tombstones their exact Eshkol shells,
unlinks I2 memberships and carriers, commits native M3T owner and successor,
then tombstones the model and successor entries. The caller-owned original
remains live. No O2 witness, public setter or production retirement path is
claimed.

`sealed_fixture_driver.esk` is an intended same-process source fixture: it
uses the accepted one-shot wrong-tie producer to check the first typed
prepublication schedule error and abort, then checks an authentic
live-workspace rejection, one sealed retirement, stale model and head-handle
typed errors, and canonical reconstruction and retirement from the unchanged
original. The source checker exercises exact
predecessor pins, generation, stale source rejection and fixture routing.
Actual compiler composition, generated source and native object inventories,
normal/repeat/sanitizer execution, feature-off object/symbol parity, and
first-failure construction cuts are **unrun**. This source candidate must not
be described as a runnable Section 9 gate or accepted stale-M3 proof.
