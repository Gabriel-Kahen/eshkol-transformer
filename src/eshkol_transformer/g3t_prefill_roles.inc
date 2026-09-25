/* Fixed P1 prefill and one-token append roles. */
static int64_t g3t_prefill_attention(g3t_context *c) {
  et_a2_kv_cache_transaction *transaction = NULL;
  et_a2_kv_cache_transaction_view *lease = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_tensor_view_v1 inputs[6], count_view;
  et_kernel_error error;
  uint64_t one_shape[1] = {1}, stage_shape[4] = {1,2,1,2};
  uint64_t head_shape[4] = {1,2,1,2}, query_shape[2] = {1,1};
  uint64_t key_shape[2] = {1,2}, mask_shape[3] = {1,1,2};
  uint64_t request[6] = {1,2,2,1,2,2};
  int64_t count = 1, query_position[1] = {c->frame.kind == 2 ? 1 : 0};
  int64_t key_positions[2] = {0,1};
  uint8_t mask[2] = {0,0};
  float result[4] = {0};
  int64_t first_domain, first_category, first_code;

  et_a2_kv_cache *cache = c->frame.kind == 1 ?
                           c->frame.candidate_cache : c->cache;
  if (!cache || c->frame.transaction)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  count_view = g3t_view(&count, sizeof(count), "i64", 1u, one_shape);
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_begin_v1(
          cache, 1u, &count_view, &transaction, &error),
          &error)) goto fail;
  inputs[0] = g3t_view(c->frame.kh, sizeof(c->frame.kh), "f32", 4u,
                       stage_shape);
  inputs[1] = g3t_view(c->frame.vh, sizeof(c->frame.vh), "f32", 4u,
                       stage_shape);
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_stage_layer_v1(
          transaction, 0u, &inputs[0], &inputs[1], &error), &error)) goto fail;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_begin_v1(
          transaction, 0u, &lease, &error), &error)) goto fail;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_tensors_v1(
          lease, &keys, &values, &lengths, &keep, &error), &error)) goto fail;
  if (((const int64_t *)lengths->data)[0] != query_position[0] + 1 ||
      ((const uint8_t *)keep->data)[0] != 1u ||
      ((const uint8_t *)keep->data)[1] != (uint8_t)(c->frame.kind == 2)) {
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
    goto fail;
  }
  mask[0] = (uint8_t)(((const uint8_t *)keep->data)[0] &&
                      key_positions[0] <= query_position[0]);
  mask[1] = (uint8_t)(((const uint8_t *)keep->data)[1] &&
                      key_positions[1] <= query_position[0]);
  inputs[0] = g3t_view(c->frame.qh, sizeof(c->frame.qh), "f32", 4u,
                       head_shape);
  inputs[1] = *keys;
  inputs[2] = *values;
  inputs[3] = g3t_view(query_position, sizeof(query_position), "i64", 2u,
                       query_shape);
  inputs[4] = g3t_view(key_positions, sizeof(key_positions), "i64", 2u,
                       key_shape);
  inputs[5] = g3t_view(mask, sizeof(mask), "bool", 3u, mask_shape);
  if (g3t_dispatch(et_a2_kernel_provider_v1(), "kernel.causal-attention",
          "causal-attention.forward", 6u, request, inputs, 6u,
          g3t_view(result, sizeof(result), "f32", 4u, head_shape))) goto fail;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_end_v1(
          &lease, &error), &error)) goto fail;
  memcpy(c->frame.ah, result, sizeof(result));
  c->frame.transaction = transaction;
  c->frame.next_ordinal = 11;
  return 0;

fail:
  first_domain = g3t_error_domain;
  first_category = g3t_error_category;
  first_code = g3t_error_code;
  if (lease && et_a2_kv_cache_transaction_view_end_v1(&lease, &error)) abort();
  if (transaction && et_a2_kv_cache_transaction_abort_v1(
          &transaction, &error)) abort();
  g3t_error_domain = first_domain;
  g3t_error_category = first_category;
  g3t_error_code = first_code;
  return first_category;
}

static int64_t g3t_prefill_plain(g3t_context *c, int64_t ordinal) {
  et_kernel_tensor_view_v1 inputs[4];
  const et_kernel_provider_v1 *provider = et_g3n_kernel_provider_v1();
  const char *capability = NULL, *operation = NULL;
  uint64_t ids[2] = {1,1}, table[2] = {2,4};
  uint64_t embedding[4] = {1,1,256,4}, position_embedding[4] = {1,1,2,4};
  uint64_t d4[3] = {1,1,4}, d8[3] = {1,1,8}, z_shape[3] = {1,1,256};
  uint64_t linear[4] = {1,1,4,4}, split[4] = {1,1,2,2};
  uint64_t heads[4] = {1,2,1,2};
  const uint64_t *request = d4, *output_shape = d4;
  size_t request_rank = 3u, output_rank = 3u, input_count = 0u, bytes = 16u;
  int64_t token[1] = {c->frame.token};
  int64_t position[1] = {c->frame.kind == 2 ? 1 : 0};
  uint32_t epsilon_bits = UINT32_C(0x3727c5ac);
  float epsilon, result[256] = {0};
  float *destination = NULL;
  memcpy(&epsilon, &epsilon_bits, sizeof(epsilon));

  switch (ordinal) {
    case 0: case 1:
      inputs[0] = g3t_view(ordinal ? (void *)position : (void *)token,
                           sizeof(token), "i64", 2u, ids);
      inputs[1] = ordinal ?
          g3t_view(c->pins.views[13].data, 8u * sizeof(float), "f32", 2u,
                   table) : c->pins.views[10];
      input_count = 2u;
      request = ordinal ? position_embedding : embedding;
      request_rank = 4u;
      capability = "g3n.embedding-forward";
      operation = "g3n.embedding.forward";
      destination = ordinal ? c->frame.ep : c->frame.et;
      break;
    case 2: case 13: case 18:
      inputs[0] = g3t_view(ordinal == 2 ? c->frame.et :
                           (ordinal == 13 ? c->frame.x : c->frame.r),
                           16u, "f32", 3u, d4);
      inputs[1] = g3t_view(ordinal == 2 ? c->frame.ep :
                           (ordinal == 13 ? c->frame.ao : c->frame.fd),
                           16u, "f32", 3u, d4);
      input_count = 2u;
      capability = "g3n.residual-forward";
      operation = "g3n.residual.forward";
      destination = ordinal == 2 ? c->frame.x :
                    (ordinal == 13 ? c->frame.r : c->frame.y);
      break;
    case 3: case 14: case 19:
      inputs[0] = g3t_view(ordinal == 3 ? c->frame.x :
                           (ordinal == 14 ? c->frame.r : c->frame.y),
                           16u, "f32", 3u, d4);
      inputs[1] = c->pins.views[ordinal == 3 ? 7u :
                                (ordinal == 14 ? 9u : 12u)];
      inputs[2] = c->pins.views[ordinal == 3 ? 6u :
                                (ordinal == 14 ? 8u : 11u)];
      inputs[3] = g3t_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
      input_count = 4u;
      capability = "g3n.layer-norm-forward";
      operation = "g3n.layer-norm.forward";
      destination = ordinal == 3 ? c->frame.n1 :
                    (ordinal == 14 ? c->frame.n2 : c->frame.nf);
      break;
    case 4: case 5: case 6: case 12: case 15: case 17: case 20:
      inputs[0] = g3t_view(ordinal <= 6 ? c->frame.n1 :
                           (ordinal == 12 ? c->frame.at :
                            (ordinal == 15 ? c->frame.n2 :
                             (ordinal == 17 ? c->frame.fg : c->frame.nf))),
                           ordinal == 17 ? 32u : 16u, "f32", 3u,
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
      bytes = (size_t)linear[3] * sizeof(float);
      capability = "g3n.linear-forward";
      operation = "g3n.linear.forward-no-bias";
      destination = ordinal == 4 ? c->frame.qt :
                    (ordinal == 5 ? c->frame.kt :
                     (ordinal == 6 ? c->frame.vt :
                      (ordinal == 12 ? c->frame.ao :
                       (ordinal == 15 ? c->frame.fu :
                        (ordinal == 17 ? c->frame.fd : c->frame.z)))));
      break;
    case 7: case 8: case 9:
      inputs[0] = g3t_view(ordinal == 7 ? c->frame.qt :
                           (ordinal == 8 ? c->frame.kt : c->frame.vt),
                           16u, "f32", 3u, d4);
      input_count = 1u;
      request = split;
      request_rank = 4u;
      output_shape = heads;
      output_rank = 4u;
      capability = "g3n.head-layout-forward";
      operation = "g3n.heads.split.forward";
      destination = ordinal == 7 ? c->frame.qh :
                    (ordinal == 8 ? c->frame.kh : c->frame.vh);
      break;
    case 11:
      inputs[0] = g3t_view(c->frame.ah, 16u, "f32", 4u, heads);
      input_count = 1u;
      request = split;
      request_rank = 4u;
      capability = "g3n.head-layout-forward";
      operation = "g3n.heads.merge.forward";
      destination = c->frame.at;
      break;
    case 16:
      inputs[0] = g3t_view(c->frame.fu, 32u, "f32", 3u, d8);
      input_count = 1u;
      provider = et_n2_kernel_provider_v1();
      request = output_shape = d8;
      bytes = 32u;
      capability = "kernel.activation";
      operation = "gelu.forward";
      destination = c->frame.fg;
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

int64_t et_g3t_private_role_step_v1(void *candidate, int64_t ordinal) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (ordinal < 0 || ordinal > 20)
    return g3t_bad(G3T_ARGUMENT, G3T_CONFIG);
  if (!c->h.busy || c->call_kind != 2 || !c->pending_output ||
      !c->frame.active || c->frame.next_ordinal != ordinal ||
      (c->frame.kind == 1 ? c->sampled :
       (c->frame.kind != 2 || !c->sampled || !c->prefill_committed)))
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  return ordinal == 10 ? g3t_prefill_attention(c) :
                         g3t_prefill_plain(c, ordinal);
}
