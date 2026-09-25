# G3-T source-private C2 full-request preflight leaf

`et_g3t_private_full_request_preflight_v1(ptr generator, ptr input) -> i64`
is a conditional, source-private two-argument native seam under
`ET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE`. It authenticates the exact live
kind-1 generator and kind-2 input in the G3-T registry before dereferencing
input data. It accepts only one or two byte IDs, a normalized nonnegative
budget with `P+G<=2`, an authentic live model and idle generator/input/cache,
and an uncommitted prefill. Required categorical G1 draws reject exhausted
128-bit RNG counter with G3T category 2/code 12. G0 and greedy G1 accept an
exhausted counter. The check is read-only: failed admission does not acquire
pins, open an A2 transaction, publish output, or change cache/RNG/call state.
It makes no promise about later resource failure during call acquisition.

The older `et_g3t_private_prompt_preflight_v1` remains P2/G0-only and retains
its pre-pin P1 rejection. `call_acquire` maps categorical G1 exhaustion to
2/12 as a defensive second check. The counter setter exists only under both
`ET_G3T_TESTING` and the private preflight flag; it authenticates an idle
kind-1 generator and model before mutation, and is absent from the production
object. This leaf adds no public G3-G facade, package export, manual path,
P2/G1 route, N>1 route, or CLI. Its aggregate test covers the three admitted
P/G pairs, P2/G1 and overlength rejection before pins, identity/lifecycle/
busy cuts, exact exhaustion mapping, and unchanged RNG/cache/pins on rejection.

The pinned f31/LLVM21 network-none aggregate for source commit `cfab282`
passed normal, repeat, and ASan+UBSan+LSan with byte-identical 46,733 checks
and empty stderr, fourteen source contracts, Q0 4/4, and a production-object
symbol audit excluding the testing-only counter setter. One sanitizer launch
used `detect_leaks=1`. External evidence is
`/tmp/g3t-full-request-preflight-cfab282-sealed/SEAL.sha256` (SHA256
`dd89fad37d0c7b5621d7742bd6e9c6444c51cbd785707116818dfe53979461fe`).
