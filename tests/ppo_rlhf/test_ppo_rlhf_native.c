#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/ppo_rlhf_abi.h"
#include "ppo_rlhf_eshkol_bridge.h"

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define REQUIRE(cond, msg)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int all_bytes_eq(const void *buf, size_t n, uint8_t byte) {
  const uint8_t *p = (const uint8_t *)buf;
  for (size_t i = 0u; i < n; ++i) {
    if (p[i] != byte)
      return 0;
  }
  return 1;
}

static uint32_t f32_bits(float v) {
  uint32_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static void print_f32_vec(const float *v, size_t n) {
  for (size_t i = 0u; i < n; ++i)
    printf("%s%a", i == 0u ? "" : " ", (double)v[i]);
  printf("\n");
}

static void print_i32_vec(const int32_t *v, size_t n) {
  for (size_t i = 0u; i < n; ++i)
    printf("%s%d", i == 0u ? "" : " ", v[i]);
  printf("\n");
}

static void print_u8_vec(const uint8_t *v, size_t n) {
  for (size_t i = 0u; i < n; ++i)
    printf("%s%u", i == 0u ? "" : " ", (unsigned)v[i]);
  printf("\n");
}

static float det_val(uint32_t seed, size_t idx, float scale) {
  uint64_t x = ((uint64_t)seed * UINT64_C(0x9e3779b97f4a7c15)) ^
               ((uint64_t)idx * UINT64_C(0xbf58476d1ce4e5b9));
  x = (x ^ (x >> 30u)) * UINT64_C(0x94d049bb133111eb);
  x ^= x >> 31u;
  const int32_t centered = (int32_t)(x % 2001u) - 1000;
  return scale * ((float)centered / 1000.0f);
}

/* ---- 1. GAE + Exact All-Equal Advantage Whitening (N = 2..64) ---- */

static void test_gae_and_whitening(void) {
  /* Hand-verifiable 2-sequence GAE with prompt mask=0, mid-sequence mask=0,
   * and trailing padding mask=0. */
  const float r_env[10] = {9.0f, 1.0f, 0.5f, 2.0f, 9.0f,
                           1.5f, 9.0f, -0.5f, 1.0f, 0.25f};
  const float old_lp[10] = {0.0f, -1.0f, -0.8f, -0.5f, 0.0f,
                            -1.2f, 0.0f, -0.9f, -0.7f, -0.4f};
  const float ref_lp[10] = {0.0f, -1.2f, -0.6f, -0.5f, 0.0f,
                            -1.0f, 0.0f, -0.9f, -1.1f, -0.3f};
  const uint8_t mask[10] = {0u, 1u, 1u, 1u, 0u, 1u, 0u, 1u, 1u, 1u};
  float r_shaped[10];
  REQUIRE(et_ppo_kl_shaped_rewards_v1(2u, 5u, 0.1f, r_env, old_lp, ref_lp, mask,
                                      r_shaped) == ET_PPO_OK,
          "kl_shaped_rewards");
  REQUIRE(r_shaped[0] == 0.0f && r_shaped[4] == 0.0f && r_shaped[6] == 0.0f,
          "masked r_shaped must be exact 0");
  REQUIRE(fabsf(r_shaped[1] - (1.0f - 0.1f * (-1.0f - (-1.2f)))) < 1e-6f,
          "r_shaped[1] formula");

  const float values[10] = {99.0f, 0.4f, 0.6f, 0.8f, 99.0f,
                            0.3f,  99.0f, -0.2f, 0.5f, 0.1f};
  float adv[10], ret[10];
  REQUIRE(et_ppo_gae_v1(2u, 5u, 0.9f, 0.8f, r_shaped, values, mask, adv, ret) ==
              ET_PPO_OK,
          "gae");
  /* Sequence 0, t=4 is masked -> t=3 is terminal: delta_3 = r_3 - v_3. */
  const double d3 = (double)r_shaped[3] - (double)values[3];
  const double d2 = (double)r_shaped[2] + 0.9 * (double)values[3] - (double)values[2];
  const double a2 = d2 + 0.9 * 0.8 * d3;
  const double d1 = (double)r_shaped[1] + 0.9 * (double)values[2] - (double)values[1];
  const double a1 = d1 + 0.9 * 0.8 * a2;
  REQUIRE(adv[0] == 0.0f && ret[0] == 0.0f && adv[4] == 0.0f && ret[4] == 0.0f,
          "seq0 masked outputs exact 0");
  REQUIRE(fabs((double)adv[3] - d3) < 1e-6, "gae seq0 t=3");
  REQUIRE(fabs((double)adv[2] - a2) < 1e-6, "gae seq0 t=2");
  REQUIRE(fabs((double)adv[1] - a1) < 1e-6, "gae seq0 t=1");
  REQUIRE(fabs((double)ret[1] - (a1 + (double)values[1])) < 1e-6,
          "gae ret = adv + v");
  /* Sequence 1, t=1 is masked -> t=0 must NOT bootstrap from t=2! */
  const double d10 = (double)r_shaped[5] - (double)values[5];
  REQUIRE(fabs((double)adv[5] - d10) < 1e-6,
          "gae mid-sequence mask resets bootstrap");

  /* All-equal advantages exact +0.0f whitening across N = 2..64 for tricky
   * binary32 values where reciprocal multiplication sum*(1.0/N) would fail! */
  static const float tricky_vals[6] = {0.3f,  -0.7f, 1.1f,
                                       0.001f, 3.1415927f, -17.333334f};
  for (size_t vi = 0u; vi < 6u; ++vi) {
    const float c = tricky_vals[vi];
    for (uint32_t n = 2u; n <= 64u; ++n) {
      float in_a[64];
      uint8_t m[64];
      float out_w[64];
      for (uint32_t i = 0u; i < n; ++i) {
        in_a[i] = c;
        m[i] = 1u;
      }
      REQUIRE(et_ppo_whiten_advantages_v1(1u, n, 1e-8f, in_a, m, out_w) ==
                  ET_PPO_OK,
              "whiten all-equal");
      for (uint32_t i = 0u; i < n; ++i) {
        REQUIRE(f32_bits(out_w[i]) == 0u,
                "all-equal whitened advantage must be exact +0.0f");
      }
    }
  }
}

/* ---- 2. PPO-Clip + Clipped/Unclipped Value + Entropy Central FD ---- */

static void test_ppo_loss_fd_and_ties(void) {
  for (uint32_t use_vclip = 0u; use_vclip <= 1u; ++use_vclip) {
    const et_ppo_loss_config cfg = {2u,   4u,    5u,    use_vclip, 0.2f,
                                    0.25f, 0.2f, 0.5f,  0.03f,     0u};
    float logits[40];
    float values[8] = {0.1f, 0.35f, -0.2f, 0.75f, 0.0f, -0.55f, 0.4f, 0.25f};
    float old_values[8] = {0.1f, 0.30f, -0.15f, 0.20f,
                           0.0f, -0.10f, 0.38f, 0.22f};
    int32_t actions[8] = {0, 1, 2, 3, 4, 2, 0, 1};
    uint8_t mask[8] = {0u, 1u, 1u, 1u, 0u, 1u, 1u, 1u};
    float adv[8] = {0.0f, 0.7f, -0.6f, 0.9f, 0.0f, -0.8f, 0.4f, -0.3f};
    float ret[8] = {0.0f, 0.8f, -0.5f, 1.3f, 0.0f, -0.9f, 0.5f, 0.1f};
    float old_lp[8];
    for (size_t i = 0u; i < 40u; ++i)
      logits[i] = det_val(17u + use_vclip, i, 0.9f);
    REQUIRE(et_ppo_token_logprobs_v1(2u, 4u, 5u, logits, actions, mask,
                                     old_lp) == ET_PPO_OK,
            "fd old_lp");
    /* Craft mix of unclipped, upper-clipped, and lower-clipped tokens, all
     * cleanly away from non-differentiable kinks. */
    old_lp[1] -= 0.05f;
    old_lp[2] += 0.06f;
    old_lp[3] -= 0.45f; /* ratio ~ 1.57 > 1.25 with A > 0 -> clipped */
    old_lp[5] += 0.45f; /* ratio ~ 0.64 < 0.80 with A < 0 -> clipped */
    old_lp[6] += 0.04f;
    old_lp[7] -= 0.03f;

    float d_logits[40], d_values[8];
    et_ppo_loss_metrics m_bwd, m_fwd;
    REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp,
                                   old_values, adv, ret, mask, &m_fwd) ==
                ET_PPO_OK,
            "ppo forward");
    REQUIRE(et_ppo_loss_backward_v1(&cfg, logits, values, actions, old_lp,
                                    old_values, adv, ret, mask, d_logits,
                                    d_values, &m_bwd) == ET_PPO_OK,
            "ppo backward");
    REQUIRE(memcmp(&m_fwd, &m_bwd, sizeof(m_fwd)) == 0,
            "forward and backward metrics must match bit-for-bit");
    REQUIRE(m_bwd.active_tokens == 6u && m_bwd.clipped_tokens == 2u,
            "expected 2 policy-clipped tokens");
    if (use_vclip == 1u) {
      REQUIRE(m_bwd.value_clipped_tokens >= 1u,
              "expected >= 1 value-clipped token");
    } else {
      REQUIRE(m_bwd.value_clipped_tokens == 0u,
              "unclipped value mode must report 0 value_clipped_tokens");
    }

    /* Central finite difference check on d_logits and d_values. */
    const float h_l = 1.0f / 256.0f;
    double max_l_err = 0.0;
    for (size_t i = 0u; i < 40u; ++i) {
      const float orig = logits[i];
      et_ppo_loss_metrics mp, mm;
      logits[i] = orig + h_l;
      REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp,
                                     old_values, adv, ret, mask, &mp) ==
                  ET_PPO_OK,
              "fd l+");
      logits[i] = orig - h_l;
      REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp,
                                     old_values, adv, ret, mask, &mm) ==
                  ET_PPO_OK,
              "fd l-");
      logits[i] = orig;
      const double fd = ((double)mp.total_loss - (double)mm.total_loss) /
                        (2.0 * (double)h_l);
      const double err = fabs((double)d_logits[i] - fd);
      if (err > max_l_err)
        max_l_err = err;
      REQUIRE(err < 1e-5, "d_policy_logits central FD err < 1e-5");
    }

    const float h_v = 1.0f / 64.0f;
    double max_v_err = 0.0;
    for (size_t i = 0u; i < 8u; ++i) {
      const float orig = values[i];
      et_ppo_loss_metrics mp, mm;
      values[i] = orig + h_v;
      REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp,
                                     old_values, adv, ret, mask, &mp) ==
                  ET_PPO_OK,
              "fd v+");
      values[i] = orig - h_v;
      REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp,
                                     old_values, adv, ret, mask, &mm) ==
                  ET_PPO_OK,
              "fd v-");
      values[i] = orig;
      const double fd = ((double)mp.total_loss - (double)mm.total_loss) /
                        (2.0 * (double)h_v);
      const double err = fabs((double)d_values[i] - fd);
      if (err > max_v_err)
        max_v_err = err;
      REQUIRE(err < 1e-5, "d_values central FD err < 1e-5");
    }
    /* Masked positions must have exact 0 gradients. */
    REQUIRE(d_values[0] == 0.0f && d_values[4] == 0.0f,
            "masked d_values exact 0");
    for (size_t v = 0u; v < 5u; ++v)
      REQUIRE(d_logits[v] == 0.0f && d_logits[20u + v] == 0.0f,
              "masked d_logits exact 0");
    printf("PPO native FD (use_vclip=%u): max_d_logits_err=%.3e "
           "max_d_values_err=%.3e PASS\n",
           use_vclip, max_l_err, max_v_err);
  }

  /* Tie behavior at exact clip boundary: strict inequality -> not clipped. */
  const et_ppo_loss_config tcfg = {1u, 1u, 2u, 1u, 0.25f, 0.25f, 0.25f, 0.5f,
                                   0.0f, 0u};
  const float tlogits[2] = {0.0f, 0.0f}; /* logpi = -ln(2) */
  const int32_t tact[1] = {0};
  const uint8_t tmask[1] = {1u};
  float tlp[1] = {0.0f};
  REQUIRE(et_ppo_token_logprobs_v1(1u, 1u, 2u, tlogits, tact, tmask, tlp) ==
              ET_PPO_OK,
          "tie lp");
  /* Value tie: v = 0.5, old_v = 0.25, eps_v = 0.25 -> v_clip = 0.5 == v ->
   * sq_c == sq_u -> NOT value clipped, unclipped gradient active. */
  const float tv[1] = {0.5f}, tov[1] = {0.25f}, ta[1] = {1.0f}, tr[1] = {1.0f};
  float tdl[2], tdv[1];
  et_ppo_loss_metrics tm;
  REQUIRE(et_ppo_loss_backward_v1(&tcfg, tlogits, tv, tact, tlp, tov, ta, tr,
                                  tmask, tdl, tdv, &tm) == ET_PPO_OK,
          "tie backward");
  REQUIRE(tm.clipped_tokens == 0u && tm.value_clipped_tokens == 0u,
          "exact boundary ties must not trigger clip");
  REQUIRE(fabsf(tdv[0] - (0.5f * (0.5f - 1.0f))) < 1e-6f,
          "value boundary tie selects unclipped derivative");
}

/* ---- 3. Multi-Epoch Trainer, KL Early Stopping, Frozen Ref & Fresh-Process
 *         ETPP1000 Checkpoint Resume ---- */

static void run_child_resume_check(const char *ckpt_path,
                                   const uint8_t expected_param_sha[32],
                                   const uint8_t expected_ref_sha[32]) {
  et_ppo_aot_ctx *ctx = NULL;
  REQUIRE(et_ppo_aot_create(16, 4, 8, 6, &ctx) == ET_PPO_OK && ctx != NULL,
          "child create");
  REQUIRE(et_ppo_aot_configure(ctx, 4, 1, 1, 0.2, 0.2, 0.2, 0.5, 0.01, 0.95,
                               0.90, 0.02, 1e-6, 0.25, 0.06, 0.01, 1.0) ==
              ET_PPO_OK,
          "child configure");
  REQUIRE(et_ppo_aot_init_task(ctx) == ET_PPO_OK, "child init");
  REQUIRE(setenv("PPO_AOT_CKPT", ckpt_path, 1) == 0, "child setenv");
  REQUIRE(et_ppo_aot_load_env_ckpt(ctx) == ET_PPO_OK, "child load ckpt");
  for (int64_t iter = 8; iter < 16; ++iter) {
    REQUIRE(et_ppo_aot_sample_rollout(ctx, iter) == ET_PPO_OK,
            "child rollout");
    REQUIRE(et_ppo_aot_train_rollout(ctx) == ET_PPO_OK, "child train");
  }
  REQUIRE(et_ppo_aot_refresh_digests(ctx) == ET_PPO_OK, "child digests");

  for (int64_t w = 0; w < 4; ++w) {
    uint64_t exp_p = 0u, exp_r = 0u;
    for (size_t i = 0u; i < 8u; ++i) {
      exp_p = (exp_p << 8u) | expected_param_sha[(size_t)w * 8u + i];
      exp_r = (exp_r << 8u) | expected_ref_sha[(size_t)w * 8u + i];
    }
    int64_t sp = 0, sr = 0;
    memcpy(&sp, &exp_p, sizeof(sp));
    memcpy(&sr, &exp_r, sizeof(sr));
    REQUIRE(et_ppo_aot_param_sha256_word(ctx, w) == sp,
            "child param sha256 mismatch");
    REQUIRE(et_ppo_aot_ref_sha256_word(ctx, w) == sr,
            "child ref sha256 mismatch");
  }
  et_ppo_aot_destroy(ctx);
}

static void test_trainer_early_stop_and_checkpoint(const char *self_exe) {
  char tmpdir[] = "/tmp/ppo_rlhf_test.XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != NULL, "mkdtemp");

  char ckpt_path[512], tamper_path[512], trunc_path[512], sym_path[512];
  snprintf(ckpt_path, sizeof(ckpt_path), "%s/model.etpp", tmpdir);
  snprintf(tamper_path, sizeof(tamper_path), "%s/tamper.etpp", tmpdir);
  snprintf(trunc_path, sizeof(trunc_path), "%s/trunc.etpp", tmpdir);
  snprintf(sym_path, sizeof(sym_path), "%s/link.etpp", tmpdir);

  REQUIRE(setenv("PPO_AOT_CKPT", ckpt_path, 1) == 0, "setenv ckpt");
  REQUIRE(setenv("PPO_AOT_TAMPER", tamper_path, 1) == 0, "setenv tamper");
  REQUIRE(setenv("PPO_AOT_TRUNC", trunc_path, 1) == 0, "setenv trunc");
  REQUIRE(setenv("PPO_AOT_SYMLINK", sym_path, 1) == 0, "setenv sym");

  et_ppo_aot_ctx *ctx = NULL;
  REQUIRE(et_ppo_aot_create(16, 4, 8, 6, &ctx) == ET_PPO_OK && ctx != NULL,
          "trainer create");
  REQUIRE(et_ppo_aot_configure(ctx, 4, 1, 1, 0.2, 0.2, 0.2, 0.5, 0.01, 0.95,
                               0.90, 0.02, 1e-6, 0.25, 0.06, 0.01, 1.0) ==
              ET_PPO_OK,
          "trainer configure");
  REQUIRE(et_ppo_aot_init_task(ctx) == ET_PPO_OK, "trainer init");

  int64_t ref_init[4];
  for (int64_t w = 0; w < 4; ++w)
    ref_init[w] = et_ppo_aot_ref_sha256_word(ctx, w);

  const double init_r = et_ppo_aot_expected_env_reward(ctx);
  double prev_r = init_r;
  for (int64_t iter = 0; iter < 16; ++iter) {
    REQUIRE(et_ppo_aot_sample_rollout(ctx, iter) == ET_PPO_OK, "rollout");
    REQUIRE(et_ppo_aot_train_rollout(ctx) == ET_PPO_OK, "train");
    const double cur_r = et_ppo_aot_expected_env_reward(ctx);
    REQUIRE(cur_r > prev_r, "expected env reward must strictly increase");
    prev_r = cur_r;
    if (iter == 7) {
      REQUIRE(et_ppo_aot_save_env_ckpt(ctx) == ET_PPO_OK, "save ckpt");
      struct stat st;
      REQUIRE(stat(ckpt_path, &st) == 0, "stat ckpt");
      REQUIRE((st.st_mode & 0777) == 0600, "ckpt permissions must be 0600");
    }
  }
  REQUIRE(prev_r > init_r + 0.5, "substantial reward improvement");
  for (int64_t w = 0; w < 4; ++w) {
    REQUIRE(et_ppo_aot_ref_sha256_word(ctx, w) == ref_init[w],
            "frozen ref sha256 invariant");
  }

  /* Extract expected digests for fresh-process child verification. */
  uint8_t param_sha[32], ref_sha[32];
  for (int64_t w = 0; w < 4; ++w) {
    const int64_t pw = et_ppo_aot_param_sha256_word(ctx, w);
    const int64_t rw = et_ppo_aot_ref_sha256_word(ctx, w);
    uint64_t pu = 0u, ru = 0u;
    memcpy(&pu, &pw, sizeof(pu));
    memcpy(&ru, &rw, sizeof(ru));
    for (size_t i = 0u; i < 8u; ++i) {
      param_sha[(size_t)w * 8u + i] = (uint8_t)(pu >> (56u - 8u * i));
      ref_sha[(size_t)w * 8u + i] = (uint8_t)(ru >> (56u - 8u * i));
    }
  }

  /* Verify tamper, truncation, symlink, and ref-mismatch rejections. */
  REQUIRE(et_ppo_aot_check_ckpt_negative(ctx, 0) == ET_PPO_ERR_CHECKSUM,
          "tamper rejected");
  REQUIRE(et_ppo_aot_check_ckpt_negative(ctx, 1) == ET_PPO_ERR_TRUNCATED,
          "truncation rejected");
  REQUIRE(et_ppo_aot_check_ckpt_negative(ctx, 2) == ET_PPO_ERR_SYMLINK,
          "symlink rejected");
  REQUIRE(et_ppo_aot_check_ckpt_negative(ctx, 3) == ET_PPO_ERR_REF_MISMATCH,
          "ref mismatch rejected");
  et_ppo_aot_destroy(ctx);

  /* Fresh OS process (fork + exec) resumes from iter 8 checkpoint. */
  char phex[65], rhex[65];
  for (size_t i = 0u; i < 32u; ++i) {
    snprintf(phex + i * 2u, 3u, "%02x", (unsigned)param_sha[i]);
    snprintf(rhex + i * 2u, 3u, "%02x", (unsigned)ref_sha[i]);
  }
  const pid_t pid = fork();
  REQUIRE(pid >= 0, "fork");
  if (pid == 0) {
    execl(self_exe, self_exe, "--resume-child", ckpt_path, phex, rhex,
          (char *)NULL);
    _exit(127);
  }
  int wstatus = 0;
  REQUIRE(waitpid(pid, &wstatus, 0) == pid && WIFEXITED(wstatus) &&
              WEXITSTATUS(wstatus) == 0,
          "fresh-process ETPP1000 resume child must succeed with bitwise parity");

  /* Tight target_kl early stopping check. */
  et_ppo_aot_ctx *es = NULL;
  REQUIRE(et_ppo_aot_create(16, 4, 8, 6, &es) == ET_PPO_OK, "es create");
  REQUIRE(et_ppo_aot_configure(es, 4, 1, 1, 0.2, 0.2, 0.2, 0.5, 0.01, 0.95,
                               0.90, 0.02, 1e-6, 1e-4, 0.06, 0.01, 1.0) ==
              ET_PPO_OK,
          "es configure");
  REQUIRE(et_ppo_aot_init_task(es) == ET_PPO_OK, "es init");
  REQUIRE(et_ppo_aot_sample_rollout(es, 0) == ET_PPO_OK, "es rollout");
  REQUIRE(et_ppo_aot_train_rollout(es) == ET_PPO_OK, "es train");
  REQUIRE(et_ppo_aot_report_early_stopped(es) == 1 &&
              et_ppo_aot_report_epochs_completed(es) < 4 &&
              et_ppo_aot_report_approx_kl(es) > 1e-4,
          "target_kl early stopping must trigger");
  printf("PPO native trainer (16 iters): reward %.6f -> %.6f, early_stop "
         "epochs=%lld approx_kl=%.6f, fresh-process ETPP1000 resume: PASS\n",
         init_r, prev_r, (long long)et_ppo_aot_report_epochs_completed(es),
         et_ppo_aot_report_approx_kl(es));
  et_ppo_aot_destroy(es);

  (void)unlink(ckpt_path);
  (void)unlink(tamper_path);
  (void)unlink(trunc_path);
  (void)unlink(sym_path);
  (void)rmdir(tmpdir);
}

/* ---- 4. Fail-Closed Negative & 0xA5 Sentinel Preservation Tests ---- */

static void test_negatives_and_sentinels(void) {
  et_ppo_loss_config cfg = {1u, 2u, 3u, 1u, 0.2f, 0.2f, 0.2f, 0.5f, 0.01f, 0u};
  float logits[6] = {0.1f, -0.2f, 0.3f, 0.2f, 0.1f, -0.3f};
  float values[2] = {0.2f, -0.1f};
  int32_t actions[2] = {0, 2};
  float old_lp[2] = {-1.0f, -1.1f};
  float old_v[2] = {0.1f, -0.1f};
  float adv[2] = {0.5f, -0.5f};
  float ret[2] = {0.6f, -0.4f};
  uint8_t mask[2] = {1u, 1u};

  float d_logits[6], d_values[2];
  et_ppo_loss_metrics metrics;
  memset(d_logits, 0xA5, sizeof(d_logits));
  memset(d_values, 0xA5, sizeof(d_values));
  memset(&metrics, 0xA5, sizeof(metrics));

  /* OOB action -> failure-atomic 0xA5 sentinel preserved. */
  actions[1] = 3;
  REQUIRE(et_ppo_loss_backward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, d_logits, d_values,
                                  &metrics) == ET_PPO_ERR_INDEX,
          "oob action");
  REQUIRE(all_bytes_eq(d_logits, sizeof(d_logits), 0xA5u) &&
              all_bytes_eq(d_values, sizeof(d_values), 0xA5u) &&
              all_bytes_eq(&metrics, sizeof(metrics), 0xA5u),
          "sentinel preserved on oob action");
  actions[1] = 2;

  /* Non-finite logit -> sentinel preserved. */
  logits[4] = NAN;
  REQUIRE(et_ppo_loss_backward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, d_logits, d_values,
                                  &metrics) == ET_PPO_ERR_NONFINITE,
          "nan logit");
  REQUIRE(all_bytes_eq(d_logits, sizeof(d_logits), 0xA5u),
          "sentinel preserved on nan logit");
  logits[4] = 0.1f;

  /* Log-ratio overflow -> sentinel preserved. */
  old_lp[0] = -100.0f;
  REQUIRE(et_ppo_loss_backward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, d_logits, d_values,
                                  &metrics) == ET_PPO_ERR_OVERFLOW,
          "ratio overflow");
  REQUIRE(all_bytes_eq(d_logits, sizeof(d_logits), 0xA5u),
          "sentinel preserved on overflow");
  old_lp[0] = -1.0f;

  /* All-zero mask -> ET_PPO_ERR_EMPTY_MASK. */
  uint8_t zmask[2] = {0u, 0u};
  REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                 adv, ret, zmask, &metrics) ==
              ET_PPO_ERR_EMPTY_MASK,
          "empty mask loss");
  REQUIRE(et_ppo_gae_v1(1u, 2u, 0.99f, 0.95f, ret, values, zmask, d_values,
                        d_logits) == ET_PPO_ERR_EMPTY_MASK,
          "empty mask gae");
  REQUIRE(et_ppo_whiten_advantages_v1(1u, 2u, 1e-6f, adv, zmask, d_values) ==
              ET_PPO_ERR_EMPTY_MASK,
          "empty mask whiten");

  /* Aliasing rejected. */
  REQUIRE(et_ppo_loss_backward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, logits, d_values,
                                  &metrics) == ET_PPO_ERR_ALIAS,
          "alias rejected");

  /* Invalid mask entry (> 1). */
  uint8_t badmask[2] = {1u, 2u};
  REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                 adv, ret, badmask, &metrics) ==
              ET_PPO_ERR_ARGUMENT,
          "bad mask entry");

  /* FP-env traps: FE_TOWARDZERO and x86_64 MXCSR FTZ. */
  const int saved_rm = fegetround();
  REQUIRE(fesetround(FE_TOWARDZERO) == 0, "set FE_TOWARDZERO");
  REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                 adv, ret, mask, &metrics) == ET_PPO_ERR_FP_ENV,
          "FE_TOWARDZERO rejected");
  REQUIRE(fesetround(saved_rm) == 0, "restore FE_TONEAREST");

#if defined(__x86_64__)
  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* Enable FTZ */
  REQUIRE(et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                 adv, ret, mask, &metrics) == ET_PPO_ERR_FP_ENV,
          "MXCSR FTZ rejected");
  _mm_setcsr(saved_mxcsr);
#endif

  /* Bridge negative probe suite (all 13 cases). */
  et_ppo_aot_ctx *ctx = NULL;
  REQUIRE(et_ppo_aot_create(4, 4, 8, 6, &ctx) == ET_PPO_OK, "neg ctx");
  static const int32_t expected_codes[13] = {
      ET_PPO_ERR_NONFINITE, ET_PPO_ERR_NONFINITE, ET_PPO_ERR_ARGUMENT,
      ET_PPO_ERR_ARGUMENT,  ET_PPO_ERR_ARGUMENT,  ET_PPO_ERR_ARGUMENT,
      ET_PPO_ERR_EMPTY_MASK, ET_PPO_ERR_EMPTY_MASK, ET_PPO_ERR_INDEX,
      ET_PPO_ERR_INDEX,     ET_PPO_ERR_FP_ENV,    ET_PPO_ERR_OVERFLOW,
      ET_PPO_ERR_ALIAS};
  for (int64_t c = 0; c < 13; ++c) {
    REQUIRE(et_ppo_aot_check_negative_case(ctx, c) == expected_codes[c],
            "bridge negative case");
  }
  et_ppo_aot_destroy(ctx);
  printf("PPO native negatives & 0xA5 sentinels: PASS\n");
}

/* ---- 5. Oracle Mode for test_ppo_rlhf_oracle.py ---- */

static void run_oracle_dump(void) {
  /* Emit 6 core cases covering GAE, whitening, clipped & unclipped value head,
   * clipped & unclipped policy ratios, entropy, and mask boundaries. */
  static const uint32_t dims[6][4] = {
      {2u, 4u, 5u, 1u}, {2u, 3u, 4u, 0u}, {3u, 3u, 4u, 1u},
      {1u, 5u, 6u, 1u}, {2u, 4u, 3u, 0u}, {2u, 3u, 5u, 1u}};
  static const float hparams[6][9] = {
      /* gamma, lam, beta_kl, adv_eps, clip_lo, clip_hi, v_eps, vf_c, ent_c */
      {0.99f, 0.95f, 0.05f, 1e-6f, 0.2f, 0.2f, 0.2f, 0.5f, 0.01f},
      {0.95f, 0.90f, 0.10f, 1e-5f, 0.15f, 0.25f, 0.0f, 0.5f, 0.02f},
      {0.90f, 0.85f, 0.00f, 1e-6f, 0.2f, 0.3f, 0.15f, 0.4f, 0.05f},
      {1.00f, 1.00f, 0.02f, 1e-6f, 0.25f, 0.25f, 0.3f, 0.25f, 0.00f},
      {0.00f, 0.00f, 0.08f, 1e-6f, 0.1f, 0.2f, 0.0f, 1.0f, 0.03f},
      {0.98f, 0.92f, 0.04f, 1e-6f, 0.2f, 0.2f, 0.25f, 0.5f, 0.04f}};

  for (size_t c = 0u; c < 6u; ++c) {
    const uint32_t B = dims[c][0], T = dims[c][1], V = dims[c][2],
                   use_vc = dims[c][3];
    const size_t bt = (size_t)B * (size_t)T;
    const size_t btv = bt * (size_t)V;
    const float gamma = hparams[c][0], lam = hparams[c][1],
                beta_kl = hparams[c][2], adv_eps = hparams[c][3];
    const et_ppo_loss_config cfg = {
        B,           T,           V,           use_vc,
        hparams[c][4], hparams[c][5], hparams[c][6], hparams[c][7],
        hparams[c][8], 0u};

    float logits[64], values[16], old_values[16], r_env[16], old_lp[16],
        ref_lp[16];
    int32_t actions[16];
    uint8_t mask[16];
    for (size_t i = 0u; i < btv; ++i)
      logits[i] = det_val((uint32_t)(100u + c), i, 0.85f);
    for (size_t i = 0u; i < bt; ++i) {
      const uint32_t t = (uint32_t)(i % T);
      mask[i] = (t == 0u && T >= 3u && (c % 2u == 0u || i == 0u)) ? 0u : 1u;
      if (c == 3u && i == 2u)
        mask[i] = 0u; /* mid-sequence boundary */
      actions[i] = (int32_t)((c * 3u + i * 2u) % V);
      r_env[i] = det_val((uint32_t)(200u + c), i, 1.2f);
      old_values[i] = det_val((uint32_t)(300u + c), i, 0.6f);
      values[i] = old_values[i] + det_val((uint32_t)(400u + c), i, 0.45f);
      ref_lp[i] = -1.1f + det_val((uint32_t)(500u + c), i, 0.4f);
    }
    REQUIRE(et_ppo_token_logprobs_v1(B, T, V, logits, actions, mask, old_lp) ==
                ET_PPO_OK,
            "oracle old_lp");
    for (size_t i = 0u; i < bt; ++i) {
      if (mask[i] != 0u) {
        /* Shift cleanly to produce both clipped and unclipped ratios away from
         * kink boundaries. */
        const float shift = (i % 3u == 0u) ? 0.42f
                            : (i % 3u == 1u) ? -0.38f
                                             : 0.07f;
        old_lp[i] += shift;
      }
    }

    float r_shaped[16], adv_raw[16], ret[16], adv_w[16], d_logits[64],
        d_values[16];
    et_ppo_loss_metrics metrics;
    REQUIRE(et_ppo_kl_shaped_rewards_v1(B, T, beta_kl, r_env, old_lp, ref_lp,
                                        mask, r_shaped) == ET_PPO_OK,
            "oracle r_shaped");
    REQUIRE(et_ppo_gae_v1(B, T, gamma, lam, r_shaped, old_values, mask, adv_raw,
                          ret) == ET_PPO_OK,
            "oracle gae");
    REQUIRE(et_ppo_whiten_advantages_v1(B, T, adv_eps, adv_raw, mask, adv_w) ==
                ET_PPO_OK,
            "oracle whiten");
    REQUIRE(et_ppo_loss_backward_v1(&cfg, logits, values, actions, old_lp,
                                    old_values, adv_w, ret, mask, d_logits,
                                    d_values, &metrics) == ET_PPO_OK,
            "oracle bwd");

    printf("CASE %u %u %u %u %a %a %a %a %a %a %a %a %a\n", B, T, V, use_vc,
           (double)gamma, (double)lam, (double)beta_kl, (double)adv_eps,
           (double)cfg.clip_eps_low, (double)cfg.clip_eps_high,
           (double)cfg.value_clip_eps, (double)cfg.vf_coef,
           (double)cfg.ent_coef);
    print_f32_vec(logits, btv);
    print_f32_vec(values, bt);
    print_f32_vec(old_values, bt);
    print_i32_vec(actions, bt);
    print_u8_vec(mask, bt);
    print_f32_vec(r_env, bt);
    print_f32_vec(old_lp, bt);
    print_f32_vec(ref_lp, bt);
    print_f32_vec(r_shaped, bt);
    print_f32_vec(adv_raw, bt);
    print_f32_vec(ret, bt);
    print_f32_vec(adv_w, bt);
    printf("%a %a %a %a %a %a %a %a %u %u %u\n", (double)metrics.total_loss,
           (double)metrics.policy_loss, (double)metrics.value_loss,
           (double)metrics.mean_entropy, (double)metrics.approx_kl,
           (double)metrics.clip_fraction, (double)metrics.value_clip_fraction,
           (double)metrics.explained_variance, metrics.active_tokens,
           metrics.clipped_tokens, metrics.value_clipped_tokens);
    print_f32_vec(d_logits, btv);
    print_f32_vec(d_values, bt);
  }

  /* Emit 1 multi-iteration Actor-Critic PPO rollout trajectory (3 iterations x
   * 3 epochs, B=2, T=3, D=4, V=4) for exact Python trajectory verification. */
  const uint32_t B = 2u, T = 3u, D = 4u, V = 4u;
  const size_t bt = 6u;
  size_t pcount = 0u, rcount = 0u;
  (void)et_ppo_model_param_count(D, V, &pcount);
  (void)et_ppo_ref_param_count(D, V, &rcount);
  et_ppo_trainer_config tcfg;
  memset(&tcfg, 0, sizeof(tcfg));
  tcfg.loss_cfg.batch_size = B;
  tcfg.loss_cfg.seq_len = T;
  tcfg.loss_cfg.vocab_size = V;
  tcfg.loss_cfg.use_value_clip = 1u;
  tcfg.loss_cfg.clip_eps_low = 0.2f;
  tcfg.loss_cfg.clip_eps_high = 0.2f;
  tcfg.loss_cfg.value_clip_eps = 0.2f;
  tcfg.loss_cfg.vf_coef = 0.5f;
  tcfg.loss_cfg.ent_coef = 0.02f;
  tcfg.optim_cfg.lr = 0.05f;
  tcfg.optim_cfg.beta1 = 0.9f;
  tcfg.optim_cfg.beta2 = 0.999f;
  tcfg.optim_cfg.eps = 1e-8f;
  tcfg.optim_cfg.weight_decay = 0.01f;
  tcfg.optim_cfg.max_grad_norm = 1.0f;
  tcfg.feat_dim = D;
  tcfg.ppo_epochs = 3u;
  tcfg.whiten_advantages = 1u;
  tcfg.gamma = 0.95f;
  tcfg.gae_lambda = 0.90f;
  tcfg.beta_kl = 0.05f;
  tcfg.adv_eps = 1e-6f;
  tcfg.target_kl = 0.5f;

  float feats[24], ref_p[20], params[25], m[25], v[25], r_env[6];
  int32_t acts[6] = {0, 1, 2, 1, 3, 0};
  uint8_t mask[6] = {0u, 1u, 1u, 0u, 1u, 1u};
  for (size_t i = 0u; i < 24u; ++i)
    feats[i] = det_val(700u, i, 0.8f);
  for (size_t i = 0u; i < pcount; ++i) {
    params[i] = det_val(800u, i, 0.2f);
    m[i] = 0.0f;
    v[i] = 0.0f;
  }
  memcpy(ref_p, params, rcount * sizeof(float));
  for (size_t i = 0u; i < bt; ++i)
    r_env[i] = mask[i] ? (0.5f + det_val(900u, i, 0.8f)) : 0.0f;

  printf("TRAJ %u %u %u %u %u\n", B, T, D, V, 3u);
  print_f32_vec(feats, 24u);
  print_f32_vec(ref_p, rcount);
  print_f32_vec(params, pcount);
  print_i32_vec(acts, bt);
  print_u8_vec(mask, bt);
  print_f32_vec(r_env, bt);

  uint64_t step = 0u;
  for (uint32_t it = 0u; it < 3u; ++it) {
    et_ppo_rollout_report rep;
    REQUIRE(et_ppo_train_rollout_v1(&tcfg, feats, acts, r_env, mask, ref_p,
                                    params, m, v, &step, &rep) == ET_PPO_OK,
            "oracle traj rollout");
    printf("STEP %llu %u %u %a %a %a\n", (unsigned long long)step,
           rep.epochs_completed, rep.early_stopped,
           (double)rep.final_metrics.total_loss,
           (double)rep.final_metrics.approx_kl, (double)rep.mean_shaped_reward);
    print_f32_vec(params, pcount);
  }
}

static int parse_hex32(const char *hex, uint8_t out[32]) {
  if (strlen(hex) != 64u)
    return 0;
  for (size_t i = 0u; i < 32u; ++i) {
    unsigned int byte = 0u;
    if (sscanf(hex + i * 2u, "%2x", &byte) != 1)
      return 0;
    out[i] = (uint8_t)byte;
  }
  return 1;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_dump();
    return 0;
  }
  if (argc == 5 && strcmp(argv[1], "--resume-child") == 0) {
    uint8_t psha[32], rsha[32];
    REQUIRE(parse_hex32(argv[3], psha) && parse_hex32(argv[4], rsha),
            "parse child sha");
    run_child_resume_check(argv[2], psha, rsha);
    return 0;
  }

  test_gae_and_whitening();
  test_ppo_loss_fd_and_ties();
  test_trainer_early_stop_and_checkpoint(argv[0]);
  test_negatives_and_sentinels();
  printf("PPO native C11 test suite: ALL PASS\n");
  return 0;
}
