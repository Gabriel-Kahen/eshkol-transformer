#define _GNU_SOURCE
#include "eshkol_transformer/hybrid_mesh_abi.h"
#include "hybrid_mesh_eshkol_bridge.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHILD_CHECK(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "CHILD FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));  \
      _exit(1);                                                                \
    }                                                                          \
  } while (0)

static const uint32_t kDModel = 8u;
static const uint32_t kDFf = 16u;
static const uint32_t kNumClasses = 6u;
static const uint32_t kGlobalBatch = 8u;
static const float kRmsEps = 1e-5f;
static const uint64_t kModelSeed = UINT64_C(20261124);
static const int kTimeoutMs = 15000;
static const int kTrainSteps = 10;

static et_mesh_model_config_v1 make_mcfg(void) {
  et_mesh_model_config_v1 c;
  memset(&c, 0, sizeof(c));
  c.d_model = kDModel;
  c.d_ff = kDFf;
  c.num_classes = kNumClasses;
  c.global_batch = kGlobalBatch;
  c.rms_eps = kRmsEps;
  c.reserved = 0u;
  return c;
}

static et_mesh_optim_config_v1 make_ocfg(void) {
  et_mesh_optim_config_v1 c;
  c.lr = 0.05;
  c.beta1 = 0.9;
  c.beta2 = 0.999;
  c.eps = 1e-8;
  c.weight_decay = 0.01;
  c.max_grad_norm = 0.5;
  return c;
}

static int count_open_fds(void) {
  DIR *d = opendir("/proc/self/fd");
  if (d == NULL) {
    return -1;
  }
  int count = 0;
  struct dirent *ent = NULL;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.') {
      continue;
    }
    ++count;
  }
  closedir(d);
  return count;
}

static int write_exact_file(const char *path, const void *buf, size_t len) {
  const int fd =
      open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) {
    return -1;
  }
  const uint8_t *p = (const uint8_t *)buf;
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, p + off, len - off);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) {
        continue;
      }
      close(fd);
      return -1;
    }
    off += (size_t)n;
  }
  close(fd);
  return 0;
}

static int read_exact_file(const char *path, void *buf, size_t len) {
  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    return -1;
  }
  uint8_t *p = (uint8_t *)buf;
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = read(fd, p + off, len - off);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) {
        continue;
      }
      close(fd);
      return -1;
    }
    off += (size_t)n;
  }
  close(fd);
  return 0;
}

typedef struct rank_train_result {
  et_mesh_coords_v1 coords;
  et_mesh_memory_report_v1 mem;
  et_mesh_comm_stats_v1 stats_after_fwd;
  et_mesh_comm_stats_v1 stats_after_bwd;
  et_mesh_comm_stats_v1 stats_final;
  double step_losses[10];
  double step_grad_norms[10];
  uint32_t any_clipped;
  uint8_t full_sha[32];
  uint8_t local_sha[32];
  float final_full_params[454];
} rank_train_result_t;

/* 1. Central finite-difference gradient check on ALL 454 params + ALL 64 x elements. */
static int test_central_fd_gradients(const char *dump_dir) {
  const et_mesh_model_config_v1 mcfg = make_mcfg();
  const et_mesh_optim_config_v1 ocfg = make_ocfg();
  et_mesh_t *mesh = NULL;
  et_mesh_engine_t *engine = NULL;
  CHECK(et_mesh_open_unix(ET_MESH_ABI_MAJOR, ET_MESH_ABI_MINOR, "", 0u, 1u, 1u,
                          kTimeoutMs, &mesh) == ET_MESH_OK,
        "fd open 1x1");
  CHECK(et_mesh_engine_create(mesh, &mcfg, &ocfg, &engine) == ET_MESH_OK,
        "fd engine create");

  uint64_t fp = 0u;
  CHECK(et_mesh_param_counts(&mcfg, 1u, &fp, NULL, NULL, NULL) == ET_MESH_OK &&
            fp == 454u,
        "fd param count == 454");

  float orig_p[454];
  float work_p[454];
  float analytic_g[454];
  float x[64];
  float work_x[64];
  float analytic_dx[64];
  int32_t targets[8];

  CHECK(et_mesh_init_full_params(&mcfg, kModelSeed, orig_p, fp) == ET_MESH_OK,
        "fd init params");
  CHECK(et_mesh_make_microbatch(&mcfg, 0u, 1u, 0u, x, targets) == ET_MESH_OK,
        "fd make batch");
  CHECK(et_mesh_engine_load_full_params(engine, orig_p, fp) == ET_MESH_OK,
        "fd load params");

  double base_loss = 0.0;
  CHECK(et_mesh_engine_forward(engine, x, targets, 100u, kTimeoutMs,
                               &base_loss) == ET_MESH_OK,
        "fd forward");
  CHECK(et_mesh_engine_backward(engine, 101u, kTimeoutMs, analytic_dx) ==
            ET_MESH_OK,
        "fd backward");
  CHECK(et_mesh_engine_get_local_grads(engine, analytic_g, fp) == ET_MESH_OK,
        "fd get grads");

  if (dump_dir != NULL) {
    char pbuf[512];
    snprintf(pbuf, sizeof(pbuf), "%s/step0_loss.f64", dump_dir);
    CHECK(write_exact_file(pbuf, &base_loss, sizeof(base_loss)) == 0,
          "dump step0 loss");
    snprintf(pbuf, sizeof(pbuf), "%s/step0_grads.f32", dump_dir);
    CHECK(write_exact_file(pbuf, analytic_g, sizeof(analytic_g)) == 0,
          "dump step0 grads");
    snprintf(pbuf, sizeof(pbuf), "%s/step0_dx.f32", dump_dir);
    CHECK(write_exact_file(pbuf, analytic_dx, sizeof(analytic_dx)) == 0,
          "dump step0 dx");
  }

  /* Exact power-of-2 perturbation h = 1/512 = 0.001953125 for minimal f32 rounding. */
  const double h = 1.0 / 512.0;
  memcpy(work_p, orig_p, sizeof(orig_p));
  double max_param_err = 0.0;
  for (size_t i = 0u; i < fp; ++i) {
    const float wp = (float)((double)orig_p[i] + h);
    const float wm = (float)((double)orig_p[i] - h);
    const double span = (double)wp - (double)wm;
    double lp = 0.0, lm = 0.0;
    work_p[i] = wp;
    CHECK(et_mesh_engine_load_full_params(engine, work_p, fp) == ET_MESH_OK,
          "fd load+");
    CHECK(et_mesh_engine_forward(engine, x, targets, 200u + i * 2u, kTimeoutMs,
                                 &lp) == ET_MESH_OK,
          "fd fwd+");
    work_p[i] = wm;
    CHECK(et_mesh_engine_load_full_params(engine, work_p, fp) == ET_MESH_OK,
          "fd load-");
    CHECK(et_mesh_engine_forward(engine, x, targets, 201u + i * 2u, kTimeoutMs,
                                 &lm) == ET_MESH_OK,
          "fd fwd-");
    work_p[i] = orig_p[i];
    const double err = fabs((lp - lm) / span - (double)analytic_g[i]);
    if (err > max_param_err) {
      max_param_err = err;
    }
  }

  /* Also central FD check on all 64 elements of dX! */
  CHECK(et_mesh_engine_load_full_params(engine, orig_p, fp) == ET_MESH_OK,
        "fd restore params");
  memcpy(work_x, x, sizeof(x));
  double max_dx_err = 0.0;
  for (size_t i = 0u; i < 64u; ++i) {
    const float xp = (float)((double)x[i] + h);
    const float xm = (float)((double)x[i] - h);
    const double span = (double)xp - (double)xm;
    double lp = 0.0, lm = 0.0;
    work_x[i] = xp;
    CHECK(et_mesh_engine_forward(engine, work_x, targets, 2000u + i * 2u,
                                 kTimeoutMs, &lp) == ET_MESH_OK,
          "fd x fwd+");
    work_x[i] = xm;
    CHECK(et_mesh_engine_forward(engine, work_x, targets, 2001u + i * 2u,
                                 kTimeoutMs, &lm) == ET_MESH_OK,
          "fd x fwd-");
    work_x[i] = x[i];
    const double err = fabs((lp - lm) / span - (double)analytic_dx[i]);
    if (err > max_dx_err) {
      max_dx_err = err;
    }
  }

  et_mesh_engine_destroy(engine);
  et_mesh_destroy(mesh);

  printf("NATIVE FD_CHECK params=454 max_param_err=%.3e dx=64 max_dx_err=%.3e PASS\n",
         max_param_err, max_dx_err);
  CHECK(max_param_err < 1e-5, "param FD error must be < 1e-5");
  CHECK(max_dx_err < 1e-5, "dX FD error must be < 1e-5");
  return 0;
}

/* Helper: run one topology (dp_size x tp_size) across real forked OS processes. */
static int run_topology_train10(uint32_t dp_size, uint32_t tp_size,
                                const char *work_root,
                                rank_train_result_t out_results[4]) {
  const uint32_t world = dp_size * tp_size;
  char sdir[512];
  snprintf(sdir, sizeof(sdir), "%s/sock_dp%u_tp%u", work_root, dp_size, tp_size);
  CHECK(mkdir(sdir, 0700) == 0 || errno == EEXIST, "mkdir sock dir");

  char rdir[512];
  snprintf(rdir, sizeof(rdir), "%s/res_dp%u_tp%u", work_root, dp_size, tp_size);
  CHECK(mkdir(rdir, 0700) == 0 || errno == EEXIST, "mkdir res dir");

  pid_t pids[4];
  /* Spawn in reverse rank order to stress AF_UNIX rendezvous retry/ordering. */
  for (int r = (int)world - 1; r >= 0; --r) {
    const pid_t pid = fork();
    CHECK(pid >= 0, "fork failed");
    if (pid == 0) {
      const et_mesh_model_config_v1 mcfg = make_mcfg();
      const et_mesh_optim_config_v1 ocfg = make_ocfg();
      et_mesh_t *mesh = NULL;
      CHILD_CHECK(et_mesh_open_unix(ET_MESH_ABI_MAJOR, ET_MESH_ABI_MINOR, sdir,
                                    (uint32_t)r, dp_size, tp_size, kTimeoutMs,
                                    &mesh) == ET_MESH_OK,
                  "open_unix");
      et_mesh_engine_t *eng = NULL;
      CHILD_CHECK(et_mesh_engine_create(mesh, &mcfg, &ocfg, &eng) == ET_MESH_OK,
                  "engine_create");

      float init_full[454];
      CHILD_CHECK(et_mesh_init_full_params(&mcfg, kModelSeed, init_full, 454u) ==
                      ET_MESH_OK,
                  "init_full");
      CHILD_CHECK(et_mesh_engine_load_full_params(eng, init_full, 454u) ==
                      ET_MESH_OK,
                  "load_full");

      rank_train_result_t res;
      memset(&res, 0, sizeof(res));
      CHILD_CHECK(et_mesh_get_coords(mesh, &res.coords) == ET_MESH_OK,
                  "get_coords");
      CHILD_CHECK(et_mesh_engine_get_memory_report(eng, &res.mem) == ET_MESH_OK,
                  "get_mem");

      /* Probe standalone forward & backward counters first, then reset stats. */
      float x[64];
      int32_t targets[8];
      CHILD_CHECK(et_mesh_make_microbatch(&mcfg, res.coords.dp_rank, dp_size,
                                          0u, x, targets) == ET_MESH_OK,
                  "make_microbatch probe");
      double probe_loss = 0.0;
      CHILD_CHECK(et_mesh_engine_forward(eng, x, targets, 50u, kTimeoutMs,
                                         &probe_loss) == ET_MESH_OK,
                  "probe forward");
      CHILD_CHECK(et_mesh_get_stats(mesh, &res.stats_after_fwd) == ET_MESH_OK,
                  "stats_after_fwd");
      CHILD_CHECK(et_mesh_engine_backward(eng, 51u, kTimeoutMs, NULL) ==
                      ET_MESH_OK,
                  "probe backward");
      CHILD_CHECK(et_mesh_get_stats(mesh, &res.stats_after_bwd) == ET_MESH_OK,
                  "stats_after_bwd");
      CHILD_CHECK(et_mesh_reset_stats(mesh) == ET_MESH_OK, "reset_stats");

      /* Run 10-step AdamW training on deterministic batch 0. */
      for (int s = 0; s < kTrainSteps; ++s) {
        CHILD_CHECK(et_mesh_make_microbatch(&mcfg, res.coords.dp_rank, dp_size,
                                            0u, x, targets) == ET_MESH_OK,
                    "make_microbatch step");
        et_mesh_step_info_v1 info;
        CHILD_CHECK(et_mesh_engine_train_step(
                        eng, x, targets, UINT64_C(1000) + (uint64_t)s * 16u,
                        kTimeoutMs, &info) == ET_MESH_OK,
                    "train_step");
        res.step_losses[s] = info.global_loss;
        res.step_grad_norms[s] = info.global_grad_norm;
        if (info.clipped) {
          res.any_clipped = 1u;
        }
      }

      CHILD_CHECK(et_mesh_get_stats(mesh, &res.stats_final) == ET_MESH_OK,
                  "stats_final");
      CHILD_CHECK(et_mesh_engine_local_state_sha256(eng, res.local_sha) ==
                      ET_MESH_OK,
                  "local_sha");
      CHILD_CHECK(et_mesh_engine_gather_full_params(
                      eng, res.final_full_params, 454u, 77777u, kTimeoutMs) ==
                      ET_MESH_OK,
                  "gather_full");
      CHILD_CHECK(et_mesh_sha256_bytes((const uint8_t *)res.final_full_params,
                                       sizeof(res.final_full_params),
                                       res.full_sha) == ET_MESH_OK,
                  "full_sha");

      char outp[512];
      snprintf(outp, sizeof(outp), "%s/rank_%d.bin", rdir, r);
      CHILD_CHECK(write_exact_file(outp, &res, sizeof(res)) == 0, "write res");

      et_mesh_engine_destroy(eng);
      et_mesh_destroy(mesh);
      _exit(0);
    }
    pids[r] = pid;
  }

  for (uint32_t r = 0u; r < world; ++r) {
    int status = 0;
    CHECK(waitpid(pids[r], &status, 0) == pids[r], "waitpid");
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child exited 0");
    char outp[512];
    snprintf(outp, sizeof(outp), "%s/rank_%u.bin", rdir, r);
    CHECK(read_exact_file(outp, &out_results[r], sizeof(rank_train_result_t)) ==
              0,
          "read res");
  }
  return 0;
}

/* 2. All 6 topologies (1x1, 2x1, 1x2, 2x2, 4x1, 1x4) + parity + replica identity. */
static int test_all_six_topologies(const char *work_root,
                                   const char *dump_dir,
                                   rank_train_result_t *out_ref_2x2) {
  static const uint32_t kTopos[6][2] = {
      {1u, 1u}, {2u, 1u}, {1u, 2u}, {2u, 2u}, {4u, 1u}, {1u, 4u}};

  rank_train_result_t baseline[4];
  for (size_t t = 0u; t < 6u; ++t) {
    const uint32_t dp = kTopos[t][0];
    const uint32_t tp = kTopos[t][1];
    const uint32_t world = dp * tp;
    rank_train_result_t res[4];
    CHECK(run_topology_train10(dp, tp, work_root, res) == 0,
          "run_topology_train10");

    if (t == 0u) {
      baseline[0] = res[0];
    }
    if (dp == 2u && tp == 2u && out_ref_2x2 != NULL) {
      for (uint32_t r = 0u; r < 4u; ++r) {
        out_ref_2x2[r] = res[r];
      }
    }

    /* Verify exact coordinates, sharding memory, zero DP comm in fwd/bwd, counters. */
    for (uint32_t r = 0u; r < world; ++r) {
      CHECK(res[r].coords.global_rank == r, "global_rank");
      CHECK(res[r].coords.dp_rank == r / tp, "dp_rank");
      CHECK(res[r].coords.tp_rank == r % tp, "tp_rank");
      CHECK(res[r].mem.full_sharded_weight_bytes == 1536u, "full_sharded=1536");
      CHECK(res[r].mem.rank_sharded_weight_bytes == 1536u / tp,
            "exact 1/tp sharded bytes");
      CHECK(res[r].mem.replicated_weight_bytes == 280u, "replicated=280");
      CHECK(fabs(res[r].mem.tp_sharding_ratio - (double)tp) < 1e-12,
            "tp_sharding_ratio");

      /* Forward MUST have 0 DP calls and 0 DP bytes; 1 TP AllReduce when tp>1. */
      CHECK(res[r].stats_after_fwd.dp_allreduce_calls == 0u,
            "fwd zero dp_allreduce");
      CHECK(res[r].stats_after_fwd.dp_bytes_sent == 0u, "fwd zero dp_bytes");
      CHECK(res[r].stats_after_fwd.tp_allreduce_calls == (tp > 1u ? 1u : 0u),
            "fwd exact tp_allreduce");

      /* Backward alone MUST also have 0 DP calls; +1 TP AllReduce when tp>1. */
      CHECK(res[r].stats_after_bwd.dp_allreduce_calls == 0u,
            "bwd zero dp_allreduce");
      CHECK(res[r].stats_after_bwd.tp_allreduce_calls == (tp > 1u ? 2u : 0u),
            "bwd exact tp_allreduce");

      /* Final 10-step counters:
       *   tp_allreduce_calls = (tp > 1 ? 30 : 0) [10 fwd + 10 bwd + 10 norm]
       *   dp_allreduce_calls = (dp > 1 ? 20 : 0) [10 grad + 10 loss] */
      CHECK(res[r].stats_final.tp_allreduce_calls == (tp > 1u ? 30u : 0u),
            "final tp_ar");
      CHECK(res[r].stats_final.dp_allreduce_calls == (dp > 1u ? 20u : 0u),
            "final dp_ar");
      if (tp == 1u) {
        CHECK(res[r].stats_final.tp_bytes_sent == 0u, "tp=1 zero tp_bytes");
      } else {
        CHECK(res[r].stats_final.tp_bytes_sent > 0u, "tp>1 positive tp_bytes");
      }
      if (dp == 1u) {
        CHECK(res[r].stats_final.dp_bytes_sent == 0u, "dp=1 zero dp_bytes");
      } else {
        CHECK(res[r].stats_final.dp_bytes_sent > 0u, "dp>1 positive dp_bytes");
      }

      /* Strictly decreasing loss and active clipping. */
      CHECK(res[r].any_clipped == 1u, "clipping exercised");
      for (int s = 1; s < kTrainSteps; ++s) {
        CHECK(res[r].step_losses[s] < res[r].step_losses[s - 1],
              "strictly decreasing loss");
      }

      /* Bitwise identical gathered full weights across ALL ranks in this mesh! */
      CHECK(memcmp(res[r].full_sha, res[0].full_sha, 32u) == 0,
            "all ranks bitwise identical gathered weights");
    }

    /* Bitwise identical local shard state AND global_grad_norm across DP replicas! */
    for (uint32_t d = 1u; d < dp; ++d) {
      for (uint32_t tr = 0u; tr < tp; ++tr) {
        const uint32_t r0 = tr;
        const uint32_t rd = d * tp + tr;
        CHECK(memcmp(res[r0].local_sha, res[rd].local_sha, 32u) == 0,
              "matching tp_rank DP replicas bitwise identical");
        for (int s = 0; s < kTrainSteps; ++s) {
          CHECK(memcmp(&res[r0].step_grad_norms[s],
                       &res[rd].step_grad_norms[s], sizeof(double)) == 0,
                "DP replicas bitwise identical global_grad_norm");
        }
      }
    }

    /* <= 1e-6 weight & loss parity against (1x1) dense baseline. */
    double max_loss_diff = 0.0;
    for (int s = 0; s < kTrainSteps; ++s) {
      const double ld = fabs(res[0].step_losses[s] - baseline[0].step_losses[s]);
      if (ld > max_loss_diff) {
        max_loss_diff = ld;
      }
    }
    double max_w_diff = 0.0;
    for (size_t i = 0u; i < 454u; ++i) {
      const double wd = fabs((double)res[0].final_full_params[i] -
                             (double)baseline[0].final_full_params[i]);
      if (wd > max_w_diff) {
        max_w_diff = wd;
      }
    }
    CHECK(max_loss_diff <= 1e-6, "loss parity <= 1e-6 vs 1x1");
    CHECK(max_w_diff <= 1e-6, "weight parity <= 1e-6 vs 1x1");

    if (dump_dir != NULL) {
      char pbuf[512];
      snprintf(pbuf, sizeof(pbuf), "%s/weights_dp%u_tp%u.f32", dump_dir, dp, tp);
      CHECK(write_exact_file(pbuf, res[0].final_full_params,
                             sizeof(res[0].final_full_params)) == 0,
            "dump weights");
      snprintf(pbuf, sizeof(pbuf), "%s/losses_dp%u_tp%u.f64", dump_dir, dp, tp);
      CHECK(write_exact_file(pbuf, res[0].step_losses,
                             sizeof(res[0].step_losses)) == 0,
            "dump losses");
    }

    printf("NATIVE MESH (%ux%u) init_loss=%.6f final_loss=%.6f "
           "loss_err=%.2e weight_err=%.2e sharded_b=%llu tp_ar=%llu dp_ar=%llu PASS\n",
           dp, tp, res[0].step_losses[0], res[0].step_losses[kTrainSteps - 1],
           max_loss_diff, max_w_diff,
           (unsigned long long)res[0].mem.rank_sharded_weight_bytes,
           (unsigned long long)res[0].stats_final.tp_allreduce_calls,
           (unsigned long long)res[0].stats_final.dp_allreduce_calls);
  }
  return 0;
}

/* 3. 5+5 fresh-process ET2D1000 checkpoint resume on (dp=2, tp=2) 4-process mesh. */
static int test_resume_5plus5_2x2(const char *work_root,
                                  const rank_train_result_t ref_2x2[4]) {
  char cdir[512], sdir1[512], sdir2[512], rdir[512];
  snprintf(cdir, sizeof(cdir), "%s/ckpt_2x2", work_root);
  snprintf(sdir1, sizeof(sdir1), "%s/sock_res1", work_root);
  snprintf(sdir2, sizeof(sdir2), "%s/sock_res2", work_root);
  snprintf(rdir, sizeof(rdir), "%s/res_res2", work_root);
  CHECK(mkdir(cdir, 0700) == 0 && mkdir(sdir1, 0700) == 0 &&
            mkdir(sdir2, 0700) == 0 && mkdir(rdir, 0700) == 0,
        "mkdir resume dirs");

  /* Phase A: steps 0..4 + save ET2D1000 shard checkpoint. */
  pid_t pids[4];
  for (int r = 3; r >= 0; --r) {
    const pid_t pid = fork();
    CHECK(pid >= 0, "fork resume A");
    if (pid == 0) {
      const et_mesh_model_config_v1 mcfg = make_mcfg();
      const et_mesh_optim_config_v1 ocfg = make_ocfg();
      et_mesh_t *mesh = NULL;
      et_mesh_engine_t *eng = NULL;
      CHILD_CHECK(et_mesh_open_unix(ET_MESH_ABI_MAJOR, ET_MESH_ABI_MINOR, sdir1,
                                    (uint32_t)r, 2u, 2u, kTimeoutMs, &mesh) ==
                      ET_MESH_OK,
                  "resume A open");
      CHILD_CHECK(et_mesh_engine_create(mesh, &mcfg, &ocfg, &eng) == ET_MESH_OK,
                  "resume A create");
      float init_full[454];
      CHILD_CHECK(et_mesh_init_full_params(&mcfg, kModelSeed, init_full, 454u) ==
                      ET_MESH_OK,
                  "resume A init");
      CHILD_CHECK(et_mesh_engine_load_full_params(eng, init_full, 454u) ==
                      ET_MESH_OK,
                  "resume A load");
      et_mesh_coords_v1 coords;
      CHILD_CHECK(et_mesh_get_coords(mesh, &coords) == ET_MESH_OK, "coords");
      float x[64];
      int32_t targets[8];
      for (int s = 0; s < 5; ++s) {
        CHILD_CHECK(et_mesh_make_microbatch(&mcfg, coords.dp_rank, 2u, 0u, x,
                                            targets) == ET_MESH_OK,
                    "batch A");
        CHILD_CHECK(et_mesh_engine_train_step(
                        eng, x, targets, UINT64_C(1000) + (uint64_t)s * 16u,
                        kTimeoutMs, NULL) == ET_MESH_OK,
                    "step A");
      }
      CHILD_CHECK(et_mesh_engine_get_step(eng) == 5u, "step==5");
      char cpath[512];
      snprintf(cpath, sizeof(cpath), "%s/rank_%d.et2d", cdir, r);
      CHILD_CHECK(et_mesh_engine_save_checkpoint(eng, cpath) == ET_MESH_OK,
                  "save ckpt");
      struct stat st;
      CHILD_CHECK(stat(cpath, &st) == 0 && (st.st_mode & 0777) == 0600,
                  "ckpt mode 0600");
      et_mesh_engine_destroy(eng);
      et_mesh_destroy(mesh);
      _exit(0);
    }
    pids[r] = pid;
  }
  for (int r = 0; r < 4; ++r) {
    int st = 0;
    CHECK(waitpid(pids[r], &st, 0) == pids[r] && WIFEXITED(st) &&
              WEXITSTATUS(st) == 0,
          "resume A child ok");
  }

  /* Phase B: fresh processes load ET2D1000 checkpoint, run steps 5..9. */
  for (int r = 3; r >= 0; --r) {
    const pid_t pid = fork();
    CHECK(pid >= 0, "fork resume B");
    if (pid == 0) {
      const et_mesh_model_config_v1 mcfg = make_mcfg();
      const et_mesh_optim_config_v1 ocfg = make_ocfg();
      et_mesh_t *mesh = NULL;
      et_mesh_engine_t *eng = NULL;
      CHILD_CHECK(et_mesh_open_unix(ET_MESH_ABI_MAJOR, ET_MESH_ABI_MINOR, sdir2,
                                    (uint32_t)r, 2u, 2u, kTimeoutMs, &mesh) ==
                      ET_MESH_OK,
                  "resume B open");
      CHILD_CHECK(et_mesh_engine_create(mesh, &mcfg, &ocfg, &eng) == ET_MESH_OK,
                  "resume B create");
      char cpath[512];
      snprintf(cpath, sizeof(cpath), "%s/rank_%d.et2d", cdir, r);
      CHILD_CHECK(et_mesh_engine_load_checkpoint(eng, cpath) == ET_MESH_OK,
                  "load ckpt");
      CHILD_CHECK(et_mesh_engine_get_step(eng) == 5u, "loaded step==5");
      et_mesh_coords_v1 coords;
      CHILD_CHECK(et_mesh_get_coords(mesh, &coords) == ET_MESH_OK, "coords B");
      float x[64];
      int32_t targets[8];
      for (int s = 5; s < 10; ++s) {
        CHILD_CHECK(et_mesh_make_microbatch(&mcfg, coords.dp_rank, 2u, 0u, x,
                                            targets) == ET_MESH_OK,
                    "batch B");
        CHILD_CHECK(et_mesh_engine_train_step(
                        eng, x, targets, UINT64_C(1000) + (uint64_t)s * 16u,
                        kTimeoutMs, NULL) == ET_MESH_OK,
                    "step B");
      }
      CHILD_CHECK(et_mesh_engine_get_step(eng) == 10u, "step==10");
      uint8_t digests[64];
      CHILD_CHECK(et_mesh_engine_gather_full_params_sha256(
                      eng, 88888u, kTimeoutMs, digests) == ET_MESH_OK,
                  "gather sha B");
      CHILD_CHECK(et_mesh_engine_local_state_sha256(eng, digests + 32u) ==
                      ET_MESH_OK,
                  "local sha B");
      char outp[512];
      snprintf(outp, sizeof(outp), "%s/dig_%d.bin", rdir, r);
      CHILD_CHECK(write_exact_file(outp, digests, sizeof(digests)) == 0,
                  "write digests B");
      et_mesh_engine_destroy(eng);
      et_mesh_destroy(mesh);
      _exit(0);
    }
    pids[r] = pid;
  }
  for (int r = 0; r < 4; ++r) {
    int st = 0;
    CHECK(waitpid(pids[r], &st, 0) == pids[r] && WIFEXITED(st) &&
              WEXITSTATUS(st) == 0,
          "resume B child ok");
    uint8_t digests[64];
    char outp[512];
    snprintf(outp, sizeof(outp), "%s/dig_%d.bin", rdir, r);
    CHECK(read_exact_file(outp, digests, sizeof(digests)) == 0, "read digests");
    CHECK(memcmp(digests, ref_2x2[r].full_sha, 32u) == 0,
          "5+5 resume full weights bitwise identical");
    CHECK(memcmp(digests + 32u, ref_2x2[r].local_sha, 32u) == 0,
          "5+5 resume local shard state bitwise identical");
  }
  printf("NATIVE RESUME_5PLUS5 (2x2) 4-process ET2D1000 bitwise parity: PASS\n");
  return 0;
}

/* 4. Atomic single-rank non-finite abort on rank 3 of (2x2) + clean recovery. */
static int test_nonfinite_atomic_2x2(const char *work_root) {
  char sdir[512];
  snprintf(sdir, sizeof(sdir), "%s/sock_nf_2x2", work_root);
  CHECK(mkdir(sdir, 0700) == 0, "mkdir nf sock dir");

  pid_t pids[4];
  for (int r = 3; r >= 0; --r) {
    const pid_t pid = fork();
    CHECK(pid >= 0, "fork nf");
    if (pid == 0) {
      const et_mesh_model_config_v1 mcfg = make_mcfg();
      const et_mesh_optim_config_v1 ocfg = make_ocfg();
      et_mesh_t *mesh = NULL;
      et_mesh_engine_t *eng = NULL;
      CHILD_CHECK(et_mesh_open_unix(ET_MESH_ABI_MAJOR, ET_MESH_ABI_MINOR, sdir,
                                    (uint32_t)r, 2u, 2u, kTimeoutMs, &mesh) ==
                      ET_MESH_OK,
                  "nf open");
      CHILD_CHECK(et_mesh_engine_create(mesh, &mcfg, &ocfg, &eng) == ET_MESH_OK,
                  "nf create");
      float init_full[454];
      CHILD_CHECK(et_mesh_init_full_params(&mcfg, kModelSeed, init_full, 454u) ==
                      ET_MESH_OK,
                  "nf init");
      CHILD_CHECK(et_mesh_engine_load_full_params(eng, init_full, 454u) ==
                      ET_MESH_OK,
                  "nf load");
      et_mesh_coords_v1 coords;
      CHILD_CHECK(et_mesh_get_coords(mesh, &coords) == ET_MESH_OK, "nf coords");

      float x[64];
      int32_t targets[8];
      for (int s = 0; s < 2; ++s) {
        CHILD_CHECK(et_mesh_make_microbatch(&mcfg, coords.dp_rank, 2u,
                                            (uint64_t)s, x, targets) ==
                        ET_MESH_OK,
                    "nf batch");
        CHILD_CHECK(et_mesh_engine_train_step(
                        eng, x, targets, UINT64_C(200) + (uint64_t)s * 16u,
                        kTimeoutMs, NULL) == ET_MESH_OK,
                    "nf valid step");
      }
      uint8_t mark[32], after[32];
      CHILD_CHECK(et_mesh_engine_local_state_sha256(eng, mark) == ET_MESH_OK,
                  "nf mark");

      CHILD_CHECK(et_mesh_make_microbatch(&mcfg, coords.dp_rank, 2u, 2u, x,
                                          targets) == ET_MESH_OK,
                  "nf batch 2");

      /* Inject qNaN on rank 3 ONLY -> ALL 4 ranks must abort atomically! */
      CHILD_CHECK(et_mesh_engine_train_step_inject_nonfinite(
                      eng, x, targets, 3u, 15u, UINT32_C(0x7fc00000), 500u,
                      kTimeoutMs, NULL) == ET_MESH_ERR_NONFINITE,
                  "qNaN abort");
      CHILD_CHECK(et_mesh_engine_local_state_sha256(eng, after) == ET_MESH_OK &&
                      memcmp(mark, after, 32u) == 0 &&
                      et_mesh_engine_get_step(eng) == 2u,
                  "state preserved after qNaN");

      /* Inject +Inf on rank 3 ONLY -> ALL 4 ranks must abort atomically! */
      CHILD_CHECK(et_mesh_engine_train_step_inject_nonfinite(
                      eng, x, targets, 3u, 90u, UINT32_C(0x7f800000), 600u,
                      kTimeoutMs, NULL) == ET_MESH_ERR_NONFINITE,
                  "+Inf abort");
      CHILD_CHECK(et_mesh_engine_local_state_sha256(eng, after) == ET_MESH_OK &&
                      memcmp(mark, after, 32u) == 0 &&
                      et_mesh_engine_get_step(eng) == 2u,
                  "state preserved after +Inf");

      /* Inject qNaN into forward input x_local on rank 3 ONLY -> atomic 2D abort! */
      float x_bad[64];
      memcpy(x_bad, x, sizeof(x));
      if (r == 3) {
        uint32_t qnan = UINT32_C(0x7fc00000);
        memcpy(&x_bad[0], &qnan, sizeof(float));
      }
      CHILD_CHECK(et_mesh_engine_train_step(eng, x_bad, targets, 700u,
                                            kTimeoutMs, NULL) ==
                      ET_MESH_ERR_NONFINITE,
                  "forward input qNaN abort");
      CHILD_CHECK(et_mesh_engine_local_state_sha256(eng, after) == ET_MESH_OK &&
                      memcmp(mark, after, 32u) == 0 &&
                      et_mesh_engine_get_step(eng) == 2u,
                  "state preserved after fwd input qNaN");

      /* Clean recovery step 2 succeeds across all 4 ranks! */
      et_mesh_step_info_v1 info;
      CHILD_CHECK(et_mesh_engine_train_step(eng, x, targets, 800u, kTimeoutMs,
                                            &info) == ET_MESH_OK,
                  "clean recovery step");
      CHILD_CHECK(et_mesh_engine_get_step(eng) == 3u &&
                      isfinite(info.global_loss),
                  "recovered step==3");

      et_mesh_engine_destroy(eng);
      et_mesh_destroy(mesh);
      _exit(0);
    }
    pids[r] = pid;
  }
  for (int r = 0; r < 4; ++r) {
    int st = 0;
    CHECK(waitpid(pids[r], &st, 0) == pids[r] && WIFEXITED(st) &&
              WEXITSTATUS(st) == 0,
          "nf child ok");
  }
  printf("NATIVE NONFINITE_ATOMIC (2x2) rank3 qNaN/+Inf/fwd_xNaN abort & recovery: PASS\n");
  return 0;
}

/* 5. Negative boundary, fault injection (mismatch/timeout/disconnect), ckpt, & FD leak tests. */
static int test_negative_and_fd_leaks(const char *work_root) {
  const int fds_before = count_open_fds();
  CHECK(fds_before > 0, "count_open_fds");

  /* Invalid topologies & ABI. */
  et_mesh_t *m = NULL;
  CHECK(et_mesh_open_unix(99u, 0u, "", 0u, 1u, 1u, 1000, &m) ==
            ET_MESH_ERR_INVALID_ARG,
        "bad abi major");
  CHECK(et_mesh_open_unix(1u, 1u, "", 0u, 1u, 1u, 1000, &m) ==
            ET_MESH_ERR_INVALID_ARG,
        "bad abi minor");
  CHECK(et_mesh_validate_topology(0u, 1u, 0u, NULL) == ET_MESH_ERR_INVALID_ARG,
        "dp=0");
  CHECK(et_mesh_validate_topology(1u, 0u, 0u, NULL) == ET_MESH_ERR_INVALID_ARG,
        "tp=0");
  CHECK(et_mesh_validate_topology(3u, 1u, 0u, NULL) == ET_MESH_ERR_INVALID_ARG,
        "dp=3");
  CHECK(et_mesh_validate_topology(1u, 3u, 0u, NULL) == ET_MESH_ERR_INVALID_ARG,
        "tp=3");
  CHECK(et_mesh_validate_topology(2u, 4u, 0u, NULL) == ET_MESH_ERR_INVALID_ARG,
        "dp*tp=8");
  CHECK(et_mesh_validate_topology(2u, 2u, 4u, NULL) == ET_MESH_ERR_INVALID_ARG,
        "rank>=world");

  /* Indivisible d_ff and global_batch. */
  et_mesh_model_config_v1 bad_m = make_mcfg();
  const et_mesh_optim_config_v1 ocfg = make_ocfg();
  bad_m.d_ff = 15u;
  CHECK(et_mesh_validate_configs(&bad_m, &ocfg, 1u, 2u) ==
            ET_MESH_ERR_DIVISIBILITY,
        "indivisible d_ff");
  bad_m = make_mcfg();
  bad_m.global_batch = 6u;
  CHECK(et_mesh_validate_configs(&bad_m, &ocfg, 4u, 1u) ==
            ET_MESH_ERR_DIVISIBILITY,
        "indivisible batch");

  /* Aliasing checks on collectives. */
  CHECK(et_mesh_open_unix(1u, 0u, "", 0u, 1u, 1u, 1000, &m) == ET_MESH_OK,
        "open 1x1 neg");
  float alias_buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  CHECK(et_mesh_tp_allreduce_sum_f32(m, alias_buf, alias_buf + 1, 4u, 1u,
                                     1000) == ET_MESH_ERR_ALIAS,
        "partial overlap allreduce");
  CHECK(et_mesh_tp_allgather_f32(m, alias_buf, 4u, alias_buf, 2u, 1000) ==
            ET_MESH_ERR_ALIAS,
        "overlap allgather");
  et_mesh_destroy(m);
  m = NULL;

  /* Timeout check when peer never connects. */
  char tdir[512];
  snprintf(tdir, sizeof(tdir), "%s/sock_timeout", work_root);
  CHECK(mkdir(tdir, 0700) == 0, "mkdir timeout dir");
  CHECK(et_mesh_open_unix(1u, 0u, tdir, 0u, 1u, 2u, 50, &m) ==
            ET_MESH_ERR_TIMEOUT &&
            m == NULL,
        "open_unix timeout");

  /* Bad fd passed to create_fds (non-socket pipe fd). */
  int pipefds[2];
  CHECK(pipe(pipefds) == 0, "pipe");
  CHECK(et_mesh_create_fds(1u, 0u, 0u, 1u, 2u, &pipefds[0], 1u, NULL, 0u, &m) ==
            ET_MESH_ERR_INVALID_ARG &&
            m == NULL,
        "reject pipe fd");
  close(pipefds[0]);
  close(pipefds[1]);

  /* Peer disconnect via socketpair. */
  int sp[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) == 0,
        "socketpair");
  CHECK(et_mesh_create_fds(1u, 0u, 0u, 1u, 2u, &sp[0], 1u, NULL, 0u, &m) ==
            ET_MESH_OK,
        "create_fds socketpair");
  close(sp[0]);
  close(sp[1]); /* peer disconnected! */
  float v1 = 1.0f, v2 = 0.0f;
  CHECK(et_mesh_tp_allreduce_sum_f32(m, &v1, &v2, 1u, 99u, 500) ==
            ET_MESH_ERR_PEER_DISCONNECTED,
        "peer disconnect detected");
  et_mesh_destroy(m);
  m = NULL;

  /* Sticky tag mismatch over real AF_UNIX (1x2). */
  char mdir[512];
  snprintf(mdir, sizeof(mdir), "%s/sock_mismatch", work_root);
  CHECK(mkdir(mdir, 0700) == 0, "mkdir mismatch dir");
  pid_t mpids[2];
  for (int r = 1; r >= 0; --r) {
    const pid_t pid = fork();
    CHECK(pid >= 0, "fork mismatch");
    if (pid == 0) {
      et_mesh_t *mm = NULL;
      CHILD_CHECK(et_mesh_open_unix(1u, 0u, mdir, (uint32_t)r, 1u, 2u,
                                    kTimeoutMs, &mm) == ET_MESH_OK,
                  "mismatch open");
      float s = 1.0f, o = 0.0f;
      CHILD_CHECK(et_mesh_tp_allreduce_sum_f32(mm, &s, &o, 1u,
                                               r == 0 ? 10u : 11u,
                                               kTimeoutMs) ==
                      ET_MESH_ERR_MISMATCH,
                  "mismatch 1");
      CHILD_CHECK(et_mesh_tp_allreduce_sum_f32(mm, &s, &o, 1u, 10u,
                                               kTimeoutMs) ==
                      ET_MESH_ERR_MISMATCH,
                  "mismatch sticky");
      et_mesh_destroy(mm);
      _exit(0);
    }
    mpids[r] = pid;
  }
  for (int r = 0; r < 2; ++r) {
    int st = 0;
    CHECK(waitpid(mpids[r], &st, 0) == mpids[r] && WIFEXITED(st) &&
              WEXITSTATUS(st) == 0,
          "mismatch child ok");
  }

  /* Checkpoint tamper / truncation / symlink / version / mesh-mismatch / config-mismatch. */
  const et_mesh_model_config_v1 mcfg = make_mcfg();
  et_mesh_t *m1 = NULL;
  et_mesh_engine_t *e1 = NULL;
  CHECK(et_mesh_open_unix(1u, 0u, "", 0u, 1u, 1u, kTimeoutMs, &m1) ==
            ET_MESH_OK,
        "ckpt neg open");
  CHECK(et_mesh_engine_create(m1, &mcfg, &ocfg, &e1) == ET_MESH_OK,
        "ckpt neg create");
  float init_full[454];
  CHECK(et_mesh_init_full_params(&mcfg, kModelSeed, init_full, 454u) ==
            ET_MESH_OK,
        "ckpt neg init");
  CHECK(et_mesh_engine_load_full_params(e1, init_full, 454u) == ET_MESH_OK,
        "ckpt neg load");
  uint8_t orig_sha[32], cur_sha[32];
  CHECK(et_mesh_engine_local_state_sha256(e1, orig_sha) == ET_MESH_OK,
        "ckpt orig sha");

  char p_good[512], p_magic[512], p_ver[512], p_sum[512], p_trunc[512],
      p_trail[512], p_sym[512];
  snprintf(p_good, sizeof(p_good), "%s/neg_good.et2d", work_root);
  snprintf(p_magic, sizeof(p_magic), "%s/neg_magic.et2d", work_root);
  snprintf(p_ver, sizeof(p_ver), "%s/neg_ver.et2d", work_root);
  snprintf(p_sum, sizeof(p_sum), "%s/neg_sum.et2d", work_root);
  snprintf(p_trunc, sizeof(p_trunc), "%s/neg_trunc.et2d", work_root);
  snprintf(p_trail, sizeof(p_trail), "%s/neg_trail.et2d", work_root);
  snprintf(p_sym, sizeof(p_sym), "%s/neg_sym.et2d", work_root);

  CHECK(et_mesh_engine_save_checkpoint(e1, p_good) == ET_MESH_OK, "save good");
  CHECK(et_mesh_engine_save_checkpoint(e1, p_magic) == ET_MESH_OK, "save magic");
  CHECK(et_mesh_engine_save_checkpoint(e1, p_ver) == ET_MESH_OK, "save ver");
  CHECK(et_mesh_engine_save_checkpoint(e1, p_sum) == ET_MESH_OK, "save sum");
  CHECK(et_mesh_engine_save_checkpoint(e1, p_trunc) == ET_MESH_OK, "save trunc");
  CHECK(et_mesh_engine_save_checkpoint(e1, p_trail) == ET_MESH_OK, "save trail");

  CHECK(et_mesh_aot_corrupt_file_byte(p_magic, 0) == ET_MESH_OK, "flip magic");
  CHECK(et_mesh_engine_load_checkpoint(e1, p_magic) == ET_MESH_ERR_BAD_MAGIC,
        "reject bad magic");

  CHECK(et_mesh_aot_corrupt_file_byte(p_ver, 8) == ET_MESH_OK, "flip version");
  CHECK(et_mesh_engine_load_checkpoint(e1, p_ver) ==
            ET_MESH_ERR_VERSION_MISMATCH,
        "reject bad version");

  CHECK(et_mesh_aot_corrupt_file_byte(p_sum, 200) == ET_MESH_OK, "flip payload");
  CHECK(et_mesh_engine_load_checkpoint(e1, p_sum) ==
            ET_MESH_ERR_CHECKSUM_MISMATCH,
        "reject bad checksum");

  CHECK(et_mesh_aot_truncate_file(p_trunc, 200) == ET_MESH_OK, "truncate");
  CHECK(et_mesh_engine_load_checkpoint(e1, p_trunc) == ET_MESH_ERR_TRUNCATED,
        "reject truncated");

  /* Append 1 trailing byte -> ET_MESH_ERR_TRUNCATED. */
  const int tfd = open(p_trail, O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC);
  CHECK(tfd >= 0 && write(tfd, "X", 1u) == 1, "append trailing byte");
  close(tfd);
  CHECK(et_mesh_engine_load_checkpoint(e1, p_trail) == ET_MESH_ERR_TRUNCATED,
        "reject trailing byte");

  /* Symlink rejection on both load and save. */
  CHECK(et_mesh_aot_make_symlink(p_good, p_sym) == ET_MESH_OK, "make symlink");
  CHECK(et_mesh_engine_load_checkpoint(e1, p_sym) == ET_MESH_ERR_IO,
        "reject symlink load");
  CHECK(et_mesh_engine_save_checkpoint(e1, p_sym) == ET_MESH_ERR_IO,
        "reject symlink save");

  /* Config mismatch: different lr. */
  et_mesh_optim_config_v1 ocfg2 = ocfg;
  ocfg2.lr = 0.01;
  et_mesh_engine_t *e_cfg = NULL;
  CHECK(et_mesh_engine_create(m1, &mcfg, &ocfg2, &e_cfg) == ET_MESH_OK,
        "create e_cfg");
  CHECK(et_mesh_engine_load_checkpoint(e_cfg, p_good) ==
            ET_MESH_ERR_CONFIG_MISMATCH,
        "reject config mismatch");
  et_mesh_engine_destroy(e_cfg);

  /* Mesh mismatch: load 1x1 checkpoint into a 1x2 engine. */
  int sp2[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp2) == 0,
        "socketpair mesh mismatch");
  et_mesh_t *m_tp2 = NULL;
  et_mesh_engine_t *e_tp2 = NULL;
  CHECK(et_mesh_create_fds(1u, 0u, 0u, 1u, 2u, &sp2[0], 1u, NULL, 0u, &m_tp2) ==
            ET_MESH_OK,
        "create 1x2 mesh");
  close(sp2[0]);
  close(sp2[1]);
  CHECK(et_mesh_engine_create(m_tp2, &mcfg, &ocfg, &e_tp2) == ET_MESH_OK,
        "create 1x2 engine");
  CHECK(et_mesh_engine_load_checkpoint(e_tp2, p_good) ==
            ET_MESH_ERR_MESH_MISMATCH,
        "reject mesh mismatch");
  et_mesh_engine_destroy(e_tp2);
  et_mesh_destroy(m_tp2);

  /* Verify live state untouched after every failed load. */
  CHECK(et_mesh_engine_local_state_sha256(e1, cur_sha) == ET_MESH_OK &&
            memcmp(orig_sha, cur_sha, 32u) == 0,
        "live state untouched after rejected loads");
  et_mesh_engine_destroy(e1);
  et_mesh_destroy(m1);

  const int fds_after = count_open_fds();
  CHECK(fds_after == fds_before, "zero FD leaks across all tests");
  printf("NATIVE NEGATIVE_AND_FD_LEAKS fds_before=%d fds_after=%d PASS\n",
         fds_before, fds_after);
  return 0;
}

int main(int argc, char **argv) {
  const char *dump_dir = NULL;
  if (argc == 3 && strcmp(argv[1], "--dump-dir") == 0) {
    dump_dir = argv[2];
  } else if (argc != 1) {
    fprintf(stderr, "Usage: %s [--dump-dir DIR]\n", argv[0]);
    return 2;
  }

  char work_tmpl[] = "/tmp/hybrid_mesh_native_XXXXXX";
  char *work_root = mkdtemp(work_tmpl);
  CHECK(work_root != NULL, "mkdtemp work_root");

  int rc = 0;
  rank_train_result_t ref_2x2[4];
  memset(ref_2x2, 0, sizeof(ref_2x2));

  if (rc == 0) {
    rc = test_central_fd_gradients(dump_dir);
  }
  if (rc == 0) {
    rc = test_all_six_topologies(work_root, dump_dir, ref_2x2);
  }
  if (rc == 0) {
    rc = test_resume_5plus5_2x2(work_root, ref_2x2);
  }
  if (rc == 0) {
    rc = test_nonfinite_atomic_2x2(work_root);
  }
  if (rc == 0) {
    rc = test_negative_and_fd_leaks(work_root);
  }

  /* Clean up temporary directory contents. */
  char cmd[640];
  snprintf(cmd, sizeof(cmd), "rm -rf -- '%s'", work_root);
  (void)system(cmd);

  if (rc == 0) {
    printf("HYBRID_MESH NATIVE SUITE: ALL PASS\n");
  }
  return rc;
}
