# Short-MQA MOD5 C2 continuation witness

`make test-mod5-c2` exercises an opt-in CPU-f32 MQA block with one layer, one
row, two tokens, width four, two query heads, and one KV head. The merged
[AC4 implementation](AC4_CHECKPOINTING.md) provides the block's forward and
backward passes through MOD5 RMSNorm, causal attention, and SwiGLU. The fixture
deterministically maps the two D2 i64 token planes to f32 features and uses
AC4's half mean squared error. Dropout is disabled.

The P1 module holds nine uniquely owned parameters in AC4 order. The two RMS
weights have shape `[4]`, the key and value matrices have shape `[2,4]`, and
the five other matrices have shape `[4,4]`. An I2 gradient plan submits the
nine native gradients in one update. O2 applies AdamW with a linear successful
update schedule and clears gradients before the C2 snapshot. Canonical X1
configuration declares the same geometry and uses a byte tokenizer with the
real D2 cursor pair. C2 stores nine model tensors and eighteen moment tensors
under the already admitted K2 storage-copy shapes.

After two updates, the fixture saves an exact detached owner through C2 SAVE
and reloads it through C2 LOAD with a genuine K2 report. A scoped C2 borrow
copies P1 values, both O2 moments per parameter, O2 configuration and progress,
and the D2 cursor. The fixture builds a separate P1 receiver, restores
AdamW through the existing O2 loader with a separately reconstructed detached
O2 state, and seeks its own D2 reader from the loaded cursor. It releases both
detached O2 and C2 owners before the next update.

The in-process gate compares complete checkpoint bytes at the save boundary and after
each of four further updates, along with each corresponding block loss. A
receiver starts with distinct AdamW options and a constant schedule to exercise
the loaded settings. The witness checks the split cursor moved, verifies
nonzero moment data for every parameter, and covers a forged K2 report and use
of a released C2 owner. It uses the canonical private C2 source composition
and the same public C2 call bodies as the installed facade.

The same executable also runs in separate OS processes. The uninterrupted
process takes six updates. A fresh producer takes two updates and exits. The
receiver starts with the producer's second C2 file, reconstructs the live
components, releases the loaded owner, saves the restored boundary, and takes
updates three through six. The runner compares canonical C2 files and raw
little-endian binary64 AC4 losses at every shared update boundary, plus the
receiver's C2 file immediately after restore. The fixture checks the D2 row
ordinal and successful O2 update count after each update and the native O2
linear schedule factor before each update.

Two more receiver processes use the same resume entry point. One rejects a
checksum-corrupted checkpoint in C2 LOAD. The other loads a valid checkpoint
from an independent producer with a different X1 run seed, then fails the
fixture's identity check before rebuilding the live components. Both leave
their output directories empty.

Set `MOD5_C2_EVIDENCE_DIR` to an empty absolute directory to retain a receipt.
The manifest records a commit and tree when both Git status observations are
clean and identify the same HEAD. Dirty or changing checkouts produce a
working-tree receipt with the observed HEADs and status files. The checksum
inventory covers the manifest, status files, native archive, ELF, compiler and
runtime logs, corpus, checkpoint files, and the fresh-process loss records.

The scope is a detached component continuation fixture. The live receiver is
assembled in fixture code, with no generalized live restore transaction. The
witness covers one shape row and one CPU-f32 trajectory, with RNG control bytes
preserved and no RNG-consuming operations. Larger MOD5 geometries and full TR3
trainer resume are separate gates.
