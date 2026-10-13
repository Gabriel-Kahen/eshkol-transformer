# E3 N frame-bind IR inspector diagnosis

The sealed, one-use N scratch compiler run remains failed evidence. Its run
manifest is `332aba650614d18b26dd3ec938f3b6f681ae80ebd2df7c431abd2721e79a8268`.
The launcher exited 74 and Docker exited 1. The package build itself exited 0
and produced the sanitized archive; the subsequent inspector exited 1 before
writing a function extract or receipt. No N first-evaluation run occurred.

The inspector required exactly one `store ... ptr %model1` in the generated
`e3-frame-bind!` body. The pinned compiler instead kept the immutable `model`
argument in SSA. In the 4,946-line generated function, its only two body uses
are an `m3t-entry` call and the staged entry construction, before canonical
readback. There is no `%model1` slot, reassignment, or
`%model.borrow_check`. The failure was therefore an inspector storage-shape
false rejection, not evidence of the stale borrow seen in M.

The raw sanitized IR SHA-256 is
`74e9bae7442051da42c3989f7358f8ffd1f45eb717f51eb4408338f1a4fb25d4`;
its extracted function SHA-256 is
`c4507d89df4e7c63f9729c9ca700677e4cc779d0750d42316cc02eecc50d13a9`.
Independent line inspection connects `%canonical.load154` to vector slot 2,
`%vref_result174` to `%root-model`, and `%root-model.load` plus
`%root-setup-box.load` to `p1-e3-mode-bind!`. The corrected offline inspector
accepts that function and still rejects M's exact sealed IR for its old-model
borrow check. It accepts either an SSA argument or a single initial stack
store, while rejecting a post-readback old-model use or reassignment.

This is a pattern and dataflow diagnosis of generated IR, not runtime or
sanitizer admission. The corrected inspector is a new source candidate in
`tests/e3_private/inspect_frame_bind_ir.py`; it does not change the sealed
bundle or source commit `f7ebcd6d`. A fresh separately authorized one-use
first-evaluation gate and independent raw review remain required for E3 §9.
