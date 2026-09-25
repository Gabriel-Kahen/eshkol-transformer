/* Genuine two-token prefill. One provider dispatch per role, with two real
 * token rows and no padded K/V position. P1 and append keep their T1 path. */
static int64_t g3t_prefill2_attention(g3t_context *c) {
  et_a2_kv_cache_transaction *transaction = NULL;
  et_a2_kv_cache_transaction_view *lease = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_tensor_view_v1 inputs[6], count_view;
  et_kernel_error error;
  uint64_t one_shape[1] = {1}, heads[4] = {1,2,2,2};
  uint64_t query_shape[2] = {1,2}, mask_shape[3] = {1,2,2};
  uint64_t request[6] = {1,2,2,2,2,2};
  int64_t count = 2, positions[2] = {0,1};
  uint8_t mask[4] = {0,0,0,0};
  float result[8] = {0};
  int64_t domain, category, code;

  count_view = g3t_view(&count, sizeof(count), "i64", 1u, one_shape);
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_begin_v1(
          c->frame.candidate_cache, 2u, &count_view, &transaction, &error),
          &error)) goto fail;
  inputs[0] = g3t_view(c->frame.p2.kh, sizeof(c->frame.p2.kh),
                       "f32", 4u, heads);
  inputs[1] = g3t_view(c->frame.p2.vh, sizeof(c->frame.p2.vh),
                       "f32", 4u, heads);
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_stage_layer_v1(
          transaction, 0u, &inputs[0], &inputs[1], &error), &error)) goto fail;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_begin_v1(
          transaction, 0u, &lease, &error), &error)) goto fail;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_tensors_v1(
          lease, &keys, &values, &lengths, &keep, &error), &error)) goto fail;
  if (!keys || !values || !lengths || !keep || !lengths->data ||
      !keep->data || ((const int64_t *)lengths->data)[0] != 2 ||
      ((const uint8_t *)keep->data)[0] != 1 ||
      ((const uint8_t *)keep->data)[1] != 1) {
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
    goto fail;
  }
  for (size_t row = 0; row < 2; ++row)
    for (size_t key = 0; key < 2; ++key)
      mask[row * 2 + key] =
          (uint8_t)(((const uint8_t *)keep->data)[key] && key <= row);
  inputs[0] = g3t_view(c->frame.p2.qh, sizeof(c->frame.p2.qh),
                       "f32", 4u, heads);
  inputs[1] = *keys;
  inputs[2] = *values;
  inputs[3] = g3t_view(positions, sizeof(positions), "i64", 2u, query_shape);
  inputs[4] = g3t_view(positions, sizeof(positions), "i64", 2u, query_shape);
  inputs[5] = g3t_view(mask, sizeof(mask), "bool", 3u, mask_shape);
  if (g3t_dispatch(et_a2_kernel_provider_v1(), "kernel.causal-attention",
          "causal-attention.forward", 6u, request, inputs, 6u,
          g3t_view(result, sizeof(result), "f32", 4u, heads))) goto fail;
  if (et_a2_kv_cache_transaction_view_end_v1(&lease, &error)) abort();
  memcpy(c->frame.p2.ah, result, sizeof(result));
  c->frame.transaction = transaction;
  c->frame.next_ordinal = 11;
  return 0;

fail:
  domain = g3t_error_domain;
  category = g3t_error_category;
  code = g3t_error_code;
  if (lease && et_a2_kv_cache_transaction_view_end_v1(&lease, &error)) abort();
  if (transaction && et_a2_kv_cache_transaction_abort_v1(
          &transaction, &error)) abort();
  return g3t_fail(domain, category, code);
}

static int64_t g3t_prefill2_plain(g3t_context *c, int64_t ordinal) {
  et_kernel_tensor_view_v1 inputs[4];
  const et_kernel_provider_v1 *provider = et_n3k_kernel_provider_v1();
  const char *capability = NULL, *operation = NULL;
  uint64_t ids[2] = {1,2}, table[2] = {2,4};
  uint64_t embedding[4] = {1,2,256,4}, position_embedding[4] = {1,2,2,4};
  uint64_t d4[3] = {1,2,4}, d8[3] = {1,2,8}, z_shape[3] = {1,2,256};
  uint64_t linear[4] = {1,2,4,4}, heads[4] = {1,2,2,2};
  const uint64_t *request = d4, *output_shape = d4;
  size_t request_rank = 3u, output_rank = 3u, input_count = 0u, bytes = 32u;
  int64_t positions[2] = {0,1};
  uint32_t epsilon_bits = UINT32_C(0x3727c5ac);
  float epsilon, result[512] = {0}, *destination = NULL;
  memcpy(&epsilon, &epsilon_bits, sizeof(epsilon));

  switch (ordinal) {
    case 0: case 1:
      inputs[0] = g3t_view(ordinal ? (void *)positions :
                          (void *)c->frame.prompt_ids,
                          sizeof(positions), "i64", 2u, ids);
      inputs[1] = ordinal ?
          g3t_view(c->pins.views[13].data, 8u * sizeof(float),
                   "f32", 2u, table) : c->pins.views[10];
      input_count = 2u;
      request = ordinal ? position_embedding : embedding;
      request_rank = 4u;
      capability = "n3k.embedding-forward";
      operation = "n3k.embedding.forward";
      destination = ordinal ? c->frame.p2.ep : c->frame.p2.et;
      break;
    case 2: case 13: case 18:
      inputs[0] = g3t_view(ordinal == 2 ? c->frame.p2.et :
                           (ordinal == 13 ? c->frame.p2.x : c->frame.p2.r),
                           32u, "f32", 3u, d4);
      inputs[1] = g3t_view(ordinal == 2 ? c->frame.p2.ep :
                           (ordinal == 13 ? c->frame.p2.ao : c->frame.p2.fd),
                           32u, "f32", 3u, d4);
      input_count = 2u;
      capability = "n3k.residual";
      operation = "n3k.residual.forward";
      destination = ordinal == 2 ? c->frame.p2.x :
                    (ordinal == 13 ? c->frame.p2.r : c->frame.p2.y);
      break;
    case 3: case 14: case 19:
      inputs[0] = g3t_view(ordinal == 3 ? c->frame.p2.x :
                           (ordinal == 14 ? c->frame.p2.r : c->frame.p2.y),
                           32u, "f32", 3u, d4);
      inputs[1] = c->pins.views[ordinal == 3 ? 7u :
                                (ordinal == 14 ? 9u : 12u)];
      inputs[2] = c->pins.views[ordinal == 3 ? 6u :
                                (ordinal == 14 ? 8u : 11u)];
      inputs[3] = g3t_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
      input_count = 4u;
      provider = et_n2_kernel_provider_v1();
      capability = "kernel.norm";
      operation = "layer-norm.forward";
      destination = ordinal == 3 ? c->frame.p2.n1 :
                    (ordinal == 14 ? c->frame.p2.n2 : c->frame.p2.nf);
      break;
    case 4: case 5: case 6: case 12: case 15: case 17: case 20:
      inputs[0] = g3t_view(ordinal <= 6 ? c->frame.p2.n1 :
                           (ordinal == 12 ? c->frame.p2.at :
                            (ordinal == 15 ? c->frame.p2.n2 :
                             (ordinal == 17 ? c->frame.p2.fg : c->frame.p2.nf))),
                           ordinal == 17 ? 64u : 32u, "f32", 3u,
                           ordinal == 17 ? d8 : d4);
      inputs[1] = c->pins.views[ordinal == 4 ? 2u :
                                (ordinal == 5 ? 0u :
                                 (ordinal == 6 ? 3u :
                                  (ordinal == 12 ? 1u :
                                   (ordinal == 15 ? 5u :
                                    (ordinal == 17 ? 4u : 10u)))))];
      input_count = 2u;
      linear[2] = ordinal == 17 ? 8u : 4u;
      linear[3] = ordinal == 15 ? 8u : (ordinal == 20 ? 256u : 4u);
      request = linear;
      request_rank = 4u;
      output_shape = ordinal == 15 ? d8 : (ordinal == 20 ? z_shape : d4);
      bytes = (size_t)2u * (size_t)linear[3] * sizeof(float);
      capability = "n3k.linear";
      operation = "n3k.linear.forward-no-bias";
      destination = ordinal == 4 ? c->frame.p2.qt :
                    (ordinal == 5 ? c->frame.p2.kt :
                     (ordinal == 6 ? c->frame.p2.vt :
                      (ordinal == 12 ? c->frame.p2.ao :
                       (ordinal == 15 ? c->frame.p2.fu :
                        (ordinal == 17 ? c->frame.p2.fd : c->frame.p2.z)))));
      break;
    case 7: case 8: case 9:
      inputs[0] = g3t_view(ordinal == 7 ? c->frame.p2.qt :
                           (ordinal == 8 ? c->frame.p2.kt : c->frame.p2.vt),
                           32u, "f32", 3u, d4);
      input_count = 1u;
      request = heads;
      request_rank = 4u;
      output_shape = heads;
      output_rank = 4u;
      capability = "n3k.head-layout";
      operation = "n3k.heads.split.forward";
      destination = ordinal == 7 ? c->frame.p2.qh :
                    (ordinal == 8 ? c->frame.p2.kh : c->frame.p2.vh);
      break;
    case 11:
      inputs[0] = g3t_view(c->frame.p2.ah, 32u, "f32", 4u, heads);
      input_count = 1u;
      request = heads;
      request_rank = 4u;
      capability = "n3k.head-layout";
      operation = "n3k.heads.merge.forward";
      destination = c->frame.p2.at;
      break;
    case 16:
      inputs[0] = g3t_view(c->frame.p2.fu, 64u, "f32", 3u, d8);
      input_count = 1u;
      request = output_shape = d8;
      bytes = 64u;
      capability = "n3k.gelu";
      operation = "n3k.gelu.forward";
      destination = c->frame.p2.fg;
      break;
    default:
      return g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
  }
  if (g3t_dispatch(provider, capability, operation, request_rank, request,
          inputs, input_count,
          g3t_view(result, bytes, "f32", output_rank, output_shape)))
    return g3t_error_category;
  memcpy(destination, result, bytes);
  c->frame.next_ordinal = ordinal + 1;
  return 0;
}
