# E3 §9 R1 source cut inventory

Status: **source inventory only** on `582c6c7792e7d31ab9bfc66ae140a3a66299574b`.
No row is compiled §9 proof. The E3 call is `e3-evaluate-into!` on a bound
frame, authentic model and D2 dataset. A cut is eligible only if the real call
can reach it before a fallible provider invocation or publication. A hook in a
predecessor or a direct provider test does not establish that reachability.

For every recoverable cut, the next gate snapshots pre-call D2 cursor bytes,
17 modes, 14 parameter/gradient images and metadata, tie identity, initializer
successor and RNG bits, four destination words, two counters, native/source
owner and FD counts. It checks the first mapped E1 category and stage, exact
post-failure restoration and same-authority retry against a fault-free twin.
Bind failures instead retry after a fresh bind. `gap` means no selectable
one-shot cut has been demonstrated through this genuine E3 path.

| Cut | Stage | Genuine E3 reachability | Owner | Cited source seam | Disposition | Required cut assertion or gap |
|---|---|---|---|---|---|---|
| admission | 0 | Call admission precedes P1/D2 mutation. | E3 | `native/e3_private_extension.esk::e3-evaluate-closed!` | existing | Reject stale, foreign or busy authority with six outputs and modes unchanged. |
| p1-prepare | 1 | Call reaches fixed P1 mode prepare. | P1 | `native/e3_p1_modes_extension.esk::p1-e3-mode-prepare!` | gap | No cut hook; inject one-shot rejection before mode writes, then retry. |
| d2-prepare | 2 | Call reaches D2 snapshot before acquire. | D2 | `internal/e3/lib/e3_d2_restore.esk::e3-d2-prepare!` | gap | No selectable prepare cut; assert saved cursor and idle record on rejection. |
| frame-allocation | bind | Real frame constructor allocates before an evaluable frame exists. | E3 | `src/eshkol_transformer/e3_frame.c::et_e3_test_frame_fail_alloc_after_v1` | partial | One candidate `calloc` cut exists under ET_E3_TESTING; enumerate setup allocation indices and retry with a freshly bound frame. |
| tensor-allocation | bind | Real constructor creates 37 I2 scratch tensors. | I2 | `native/f32_tensor.c::et_f32_tensor_test_fail_alloc_after_v1` | partial | Broad F32 allocator countdown exists; prove each reachable constructor index and first diagnostic preservation on unwind. |
| provider-discovery | bind | Six K1 runtimes are discovered inside frame creation. | K1 | `src/eshkol_transformer/e3_frame.c::e3_discover` | gap | No E3-selectable provider admission cut; each index needs failed bind cleanup and fresh bind retry. |
| epsilon-copy | bind | Frame creation copies epsilon bits after discovery. | I2 | `native/f32_tensor.c::et_f32_tensor_copy_bits_from_v1` | gap | No copy cut at the later epsilon copy; assert complete setup unwind and first error. |
| acquisition-pin | 3 | Native acquire reaches fixed14 pin begin before model.active. | shared M3/I2 | `src/eshkol_transformer/m3_call_f32_integration.c::et_g3t_model_pins_begin_internal` | gap | No callable E3 setter reaches the file-static countdown; prove each 0..13 pre-pin cut drains its prefix and permits same-frame retry. |
| p1-enter | 4 | Call enters all 17 modes after successful native acquire. | P1 | `native/e3_p1_modes_extension.esk::p1-e3-mode-enter!` | gap | No one-shot enter/preflight cut; assert exact mixed modes and pin restoration. |
| batch-allocation | 5 | Real next-batch crosses D2 native batch allocation. | D2 | `native/d2_native.c::et_d2_batch_test_fail_stage_v1` | partial | Native stage hook exists; prove each stage is reached from E3 after an earlier completed batch and retry. |
| late-shard-read | 5 | Later D2 next-batch can read a real shard after completed batches. | D2 | `native/d2_native.c::et_d2_exact_read_test_fail_stage_v1` | partial | Retention driver reaches this hook; require a specifically late read, first IO stage, exact rollback and same-authority retry. |
| batch-copy | 6 | Three D2 views feed fixed input/target/mask copies. | E3/D2 | `internal/e3/lib/e3_d2_restore.esk::e3-stage-batch!` | gap | No independently selectable copy cut; target each copy before its write and prove borrow release. |
| provider-preflight | 7–12 | Fixed role, CE, reduction, correct, accumulation and final dispatch preflight actual K1 calls. | E3/K1/provider | `src/eshkol_transformer/e3_frame.c::e3_dispatch` | gap | No E3-selectable preinvoke hook; inject only before validated invoke and verify first stage plus rollback. |
| post-provider | 7–11 | Source schedule resumes after an earlier completed provider call. | E3 | `native/e3_private_extension.esk::e3-forward-role!` | gap | No one-shot orchestration cut after completed invoke; retain named role/stage and whole-call rollback. |
| finalize | 12 | Genuine traversal reaches metric finalization before publication. | E3/provider | `src/eshkol_transformer/e3_frame.c::et_e3_private_finalize_v1` | gap | No private final-metric overflow/preflight cut demonstrated; assert no destination write. |
| final-publication | 12 | After D2/mode restore, native publish preflights eight scopes then writes four words and two counters. | E3/I2 | `src/eshkol_transformer/e3_frame.c::et_e3_private_publish_v1` | gap | No destination-scope/late publication cut; fail before first write, assert six-output byte atomicity and retry. |
| public-seek | outside | Public D2 seek is not called by E3 restoration. | D2 | `internal/d2/lib/d2_dataset.esk::d2-token-dataset-seek!` | excluded | Use seek failure only as a negative scope control; E3 restore writes its saved core ordinal directly. |
| k1-invoke | nonfailing | Validated K1 invoke is reached by real dispatch. | K1 | `src/eshkol_transformer/e3_frame.c::e3_dispatch` | excluded | Invoke remains nonfailing by contract; cut preinvoke or after completed call, never inside invoke. |

The existing late-read failure/retry driver is partial evidence only; it does
not close the allocation, copy, pin, provider, or final publication rows.
Source planning must keep controls noninstalled and one-shot, and must not
fabricate logits, CE values, or an alternate owner. The next gate is independent
review of these reachability/gap dispositions before implementing any hook or
claiming compiled R1 coverage.
