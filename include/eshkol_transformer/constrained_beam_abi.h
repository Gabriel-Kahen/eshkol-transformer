#ifndef ESHKOL_TRANSFORMER_CONSTRAINED_BEAM_ABI_H
#define ESHKOL_TRANSFORMER_CONSTRAINED_BEAM_ABI_H

#include "eshkol_transformer/beam_search_abi.h"
#include "eshkol_transformer/constrained_decode_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Advance one initialized BEAM5 search using a CD5 vocabulary index and
 * contiguous finite CPU binary32 logits [logits_rows, logits_vocab].
 * Rows follow the current active-beam order. The beam, index and logits must
 * be valid, disjoint objects. Row count, vocabulary size and EOS ID must
 * match. Use the same index throughout a search.
 *
 * Replay the generated suffix of each current hypothesis from the DFA start
 * state. The prompt is excluded from the DFA and remains part of the BEAM5
 * processor history. Only an accepting DFA state admits EOS. CD5 masks the
 * logits before BEAM5 processors and binary64 log-softmax. If any row has
 * no admissible candidate after masking/processing, the whole step fails.
 *
 * A forbidden generated history, including an invalid retained completion,
 * returns ET_BEAM_ERR_STATE. Inspect the result with the BEAM5 queries and
 * finalizer. Budget-limited unfinished hypotheses are possible.
 *
 * On every error the entire beam object is unchanged. This operation uses
 * bounded automatic staging storage and performs no heap allocation.
 */
et_beam_status et_constrained_beam_step_v1(et_beam_state_v1 *beam,
                                           const et_cd_vocab_index *index,
                                           const float *logits,
                                           size_t logits_rows,
                                           size_t logits_vocab);

#ifdef __cplusplus
}
#endif

#endif
