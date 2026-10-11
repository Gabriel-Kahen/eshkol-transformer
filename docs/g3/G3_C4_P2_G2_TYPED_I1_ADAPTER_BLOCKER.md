# G3-C4 P2/G2 typed I1 public adapter gate

Status at `8d44e3c5fd95e6a8072badc0c87d828aa8a3a15d`: blocked before public adapter implementation. This is a source and package composition finding, not a claim about runtime failure.

The fixed public contract at `e55df41474cf9cb4be3ea66988e100f58639ff61` requires the private terminal close, detached logical I1 clone, and typed native result owner to be composed and proved in one registry-owning archive before public dispatch is enabled. The terminal and detached Eshkol source at `84b46f1dd1cd1d26206751560bd8fe47c82d09ee` has source guards, but its same-aggregate compiled, fault, and integration gates remain open. The typed native owner at `8d44e3c5fd95e6a8072badc0c87d828aa8a3a15d` is compiled only under `ET_G3C4_P2_G2_TYPED_RESULT_PRIVATE` in its standalone test script; it has no installed package route.

The current `g3g_manual_package_root.esk` does not load the P2/G2 terminal or detached Eshkol extensions. Its C4 public dispatch accepts only `diagnostic-c4-p2g1`, `max_new=1`, and `eos=#f`, then calls the P2/G1 preflight and producer. Its output adapter validates a one-ID copy with length `1` and cache `3`, and calls the scalar result constructor. Thus an adapter wired to that root cannot receive an authentic terminal output or a typed `[1]`/`[2]` owner. Relabeling the private kind-9 clone or manufacturing IDs from a bytevector would violate the contract.

Next gate: compile the terminal coordinator and detached clone with the typed owner in the same feature-gated, registry-owning archive; run the contract's genuine P2/G2, fault, ownership, and feature-off tests; obtain independent source/tests review. Then implement the public profile/producer dispatch and a separate terminal-output adapter using the typed constructors for exact IDs `[G]`, lengths `[1]` with `G`, and cache lengths `[1]` with `2+G`, plus native length/element/release routing. Do not enable this public profile until that gate passes.

Recheck commands on this tree:

```sh
python3 tests/g3c4/test_p2_g2_terminal_source_prerequisites.py
python3 tests/g3c4/test_p2_g2_terminal_close_source.py
python3 tests/g3c4/test_p2_g2_terminal_detached_source.py
rg -n 'g3c4_p2g2|ET_G3C4_P2_G2_TYPED_RESULT_PRIVATE' native/g3g_manual_package_root.esk native/g3g_manual_package_source_closure.txt scripts/build-g3c4.sh
```

The three Python source suites pass 2, 4, and 5 tests respectively. The final search has no matches; it is a package composition check, not compiled proof.
