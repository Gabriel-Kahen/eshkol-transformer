# G3-R C4 live P1 to native owner association

**Source candidate only.** This composes P1 slot 74 (`9772674e`) with the
sealed native topology leaf (`dd0cb754`), without publishing a workspace.
The private Eshkol operation
`g3r-c4-trainable-owner-association-status-internal(model, handles, owner)`
returns one closed integer. It accepts only the exact live trainable C4
registry tuple, its ordered fourteen P1 shells, and its registered native
owner. It rejects pending cleanup, I2 construction and an active C4 entry.

It first asks P1 slot 74 to authenticate the live train-mode seventeen-module
graph, exact fourteen handles and tied embedding/head, CPU f32 shapes and
position `[4,4]`. The sealed native owner status then scopes and releases
all fourteen C4 pins. For each exact P1 handle, it reads the already retained
I2 carrier through the P1 trusted identity accessor, checks the carrier's
exact handle and I2 native live registration, and passes **one** native
parameter pointer plus its handle to
`et_g3c4_private_sealed_owner_binding_status_v1(owner,index,handle,parameter)`.
That private C ABI compares both against the sealed owner's indexed binding
and validates the native I2 identity. It returns a typed status only. P1 and
native topology are checked again before success. Neither API returns a raw
pointer, array, tensor view, pin or workspace.

Closed status values are `0` exact; `1` supplied identity mismatch; `2`
inactive or busy registry tuple; `4` malformed or stale I2 carrier;
`11..14` P1 slot-74 status; `21..26` native topology category; and `31..36`
native scalar binding category. The existing native last-error triple gives
the domain/category/code for the latter two ranges. Calls require external
serialization, as do the underlying owner and P1 APIs.

Local focused Clang normal/repeat/ASan+UBSan passed 48 native checks per mode,
including cross-owner, aborted owner, active owner, wrong indexed parameter,
wrong shell, forged pointer, position-shape corruption and fourteen partial
pin acquisitions. Feature-off compilation omits both private native status
symbols. Twenty-three source assertions pass for the constructor, P1 topology
and association chain. The Eshkol fixture records positive, cross-owner,
forged, swapped-handle, eval-mode and `[2,4]` C2 controls. A test-only
changed-live-carrier control temporarily substitutes another live owner's
native parameter in the original P1 handle's retained I2 carrier. It requires
status `4`, restores the original field in both normal and exceptional paths,
checks exact registry/owner/handle identities, compares all fourteen value
byte payloads for both models, and requires association success again. This
fixture has **not** been compiled or run; the native wrong-parameter scalar
has passed local compilation.

Constructor compiled admission, the combined private package runtime gate,
and future generation/workspace/trainer lease exclusion remain open. No
trainable C4 workspace, checkpoint, or multi-token continuation is claimed.
