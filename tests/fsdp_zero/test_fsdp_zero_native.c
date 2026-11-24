#define _GNU_SOURCE
#include "eshkol_transformer/fsdp_zero_abi.h"
#include "fsdp_zero_eshkol_bridge.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define IN_DIM 8u
#define HIDDEN_DIM 17u
#define NUM_CLASSES 5u
#define NUM_PARAMS 243u
#define GLOBAL_BATCH 8u
#define TRAIN_STEPS 12u
#define MODEL_SEED UINT64_C(20261124)

#define LR_VAL 0.05
#define BETA1_VAL 0.9
#define BETA2_VAL 0.999
#define EPS_VAL 1e-8
#define WD_VAL 0.01
#define MAX_GN_VAL 0.5

static void die_msg(const char *msg, int line) {
  fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, line, msg);
  exit(1);
}

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      die_msg(#cond, __LINE__);                                                \
    }                                                                          \
  } while (0)

static int count_open_fds(void) {
  DIR *d = opendir("/proc/self/fd");
  if (d == NULL) {
    return -1;
  }
  int count = 0;
  struct dirent *ent;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] != '.') {
      ++count;
    }
  }
  closedir(d);
  return count;
}

static uint32_t f32_bits(float v) {
  uint32_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static float bits_f32(uint32_t b) {
  float v = 0.0f;
  memcpy(&v, &b, sizeof(v));
  return v;
}

static void put_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static et_fsdp_config_v1 make_cfg(uint32_t stage) {
  et_fsdp_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.stage = stage;
  cfg.reserved = 0u;
  cfg.num_params = NUM_PARAMS;
  cfg.lr = LR_VAL;
  cfg.beta1 = BETA1_VAL;
  cfg.beta2 = BETA2_VAL;
  cfg.eps = EPS_VAL;
  cfg.weight_decay = WD_VAL;
  cfg.max_grad_norm = MAX_GN_VAL;
  return cfg;
}

/* Helper to fork W ranks connected by real AF_UNIX socketpairs. */
typedef struct fork_result {
  double losses[TRAIN_STEPS];
  double grad_norms[TRAIN_STEPS];
  uint32_t clipped_flags[TRAIN_STEPS];
  float final_params[NUM_PARAMS];
  uint8_t final_sha256[32];
  et_fsdp_memory_report_v1 mem;
  et_fsdp_comm_stats_v1 comm_stats;
  int pad_zero_ok;
  int poison_ok;
} fork_result_t;

static void write_exact_fd(int fd, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, p + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else {
      _exit(91);
    }
  }
}

static void read_exact_fd(int fd, void *buf, size_t len) {
  uint8_t *p = (uint8_t *)buf;
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = read(fd, p + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else {
      die_msg("short read from child result pipe", __LINE__);
    }
  }
}

static void test_fd_gradient_check(void) {
  float params[NUM_PARAMS];
  float grads[NUM_PARAMS];
  float x[GLOBAL_BATCH * IN_DIM];
  int32_t y[GLOBAL_BATCH];
  double loss = 0.0;

  REQUIRE(et_fsdp_mlp_init_params(IN_DIM, HIDDEN_DIM, NUM_CLASSES, MODEL_SEED,
                                  params, NUM_PARAMS) == ET_FSDP_OK);
  REQUIRE(et_fsdp_mlp_make_batch(IN_DIM, NUM_CLASSES, GLOBAL_BATCH, 0u, 1u, 0u,
                                 x, y) == ET_FSDP_OK);
  REQUIRE(et_fsdp_mlp_forward_backward(IN_DIM, HIDDEN_DIM, NUM_CLASSES,
                                       GLOBAL_BATCH, params, x, y, &loss,
                                       grads) == ET_FSDP_OK);
  REQUIRE(isfinite(loss) && loss > 0.0);

  const double fd_eps = 1e-3;
  double max_err = 0.0;
  for (size_t i = 0u; i < NUM_PARAMS; ++i) {
    const float orig = params[i];
    const float p_plus = (float)((double)orig + fd_eps);
    const float p_minus = (float)((double)orig - fd_eps);
    const double span = (double)p_plus - (double)p_minus;
    double l_plus = 0.0;
    double l_minus = 0.0;

    params[i] = p_plus;
    REQUIRE(et_fsdp_mlp_forward_backward(IN_DIM, HIDDEN_DIM, NUM_CLASSES,
                                         GLOBAL_BATCH, params, x, y, &l_plus,
                                         NULL) == ET_FSDP_OK);
    params[i] = p_minus;
    REQUIRE(et_fsdp_mlp_forward_backward(IN_DIM, HIDDEN_DIM, NUM_CLASSES,
                                         GLOBAL_BATCH, params, x, y, &l_minus,
                                         NULL) == ET_FSDP_OK);
    params[i] = orig;

    const double fd = (l_plus - l_minus) / span;
    const double err = fabs((double)grads[i] - fd);
    if (err > max_err) {
      max_err = err;
    }
  }
  REQUIRE(max_err < 1e-4);
  printf("FSDP NATIVE FD_GRAD_CHECK params=%u max_err=%.3e PASS\n",
         (unsigned)NUM_PARAMS, max_err);
}

static void test_collectives_w4(void) {
  const int fds_before = count_open_fds();
  const uint32_t W = 4u;
  int sv[4][2];
  for (uint32_t r = 1u; r < W; ++r) {
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv[r]) == 0);
  }

  pid_t pids[4];
  for (uint32_t r = 0u; r < W; ++r) {
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      int my_fds[3];
      size_t n_fds = 0u;
      if (r == 0u) {
        for (uint32_t k = 1u; k < W; ++k) {
          close(sv[k][1]);
          my_fds[n_fds++] = sv[k][0];
        }
      } else {
        for (uint32_t k = 1u; k < W; ++k) {
          close(sv[k][0]);
          if (k != r) {
            close(sv[k][1]);
          }
        }
        my_fds[n_fds++] = sv[r][1];
      }

      /* Stagger workers in reversed rank order to prove arrival-order invariance. */
      if (r > 0u) {
        usleep((useconds_t)((W - r) * 3000u));
      }

      et_fsdp_comm_t *comm = NULL;
      if (et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, r, W,
                                  my_fds, n_fds, &comm) != ET_FSDP_OK) {
        _exit(10);
      }
      for (size_t i = 0u; i < n_fds; ++i) {
        close(my_fds[i]);
      }

      /* 1. AllGather f32 */
      float shard[3] = {(float)(r * 10u + 1u), (float)(r * 10u + 2u),
                        (float)(r * 10u + 3u)};
      float gathered[12];
      memset(gathered, 0, sizeof(gathered));
      if (et_fsdp_allgather_f32(comm, shard, 3u, gathered, 11u, 5000) !=
          ET_FSDP_OK) {
        _exit(11);
      }
      for (uint32_t k = 0u; k < W; ++k) {
        for (uint32_t j = 0u; j < 3u; ++j) {
          if (gathered[k * 3u + j] != (float)(k * 10u + j + 1u)) {
            _exit(12);
          }
        }
      }

      /* 2. ReduceScatterAvg f32: rank r sends [k*3+j + r*0.25] */
      float full_in[12];
      float scatter_out[3];
      for (uint32_t i = 0u; i < 12u; ++i) {
        full_in[i] = (float)(i + 1u) + 0.25f * (float)r;
      }
      if (et_fsdp_reduce_scatter_avg_f32(comm, full_in, scatter_out, 3u, 22u,
                                         5000) != ET_FSDP_OK) {
        _exit(13);
      }
      /* Mean of r=0,1,2,3 is 1.5 -> +0.375 */
      for (uint32_t j = 0u; j < 3u; ++j) {
        const float expected = (float)(r * 3u + j + 1u) + 0.375f;
        if (f32_bits(scatter_out[j]) != f32_bits(expected)) {
          _exit(14);
        }
      }

      /* 3. AllReduceSum f64 */
      double send_d[2] = {(double)(r + 1u) * 1.5, (double)(r + 1u) * (r + 1u)};
      double recv_d[2] = {0.0, 0.0};
      if (et_fsdp_allreduce_sum_f64(comm, send_d, recv_d, 2u, 33u, 5000) !=
          ET_FSDP_OK) {
        _exit(15);
      }
      /* sum(1..4)*1.5 = 15.0; sum(1,4,9,16) = 30.0 */
      if (recv_d[0] != 15.0 || recv_d[1] != 30.0) {
        _exit(16);
      }

      et_fsdp_comm_destroy(comm);
      _exit(0);
    }
    pids[r] = pid;
  }

  for (uint32_t r = 1u; r < W; ++r) {
    close(sv[r][0]);
    close(sv[r][1]);
  }
  for (uint32_t r = 0u; r < W; ++r) {
    int st = 0;
    REQUIRE(waitpid(pids[r], &st, 0) == pids[r]);
    REQUIRE(WIFEXITED(st) && WEXITSTATUS(st) == 0);
  }
  REQUIRE(count_open_fds() == fds_before);
  printf("FSDP NATIVE COLLECTIVES W=4 (AllGather/ReduceScatterAvg/AllReduceSum64): PASS\n");
}

static void run_training_group(uint32_t stage, uint32_t W, int use_unix_rendezvous,
                               const char *ckpt_dir, uint32_t start_step,
                               uint32_t end_step, fork_result_t *out_r0) {
  const int fds_before = count_open_fds();
  char sock_dir[128] = "/tmp/fsdp_native_sock_XXXXXX";
  if (use_unix_rendezvous && W > 1u) {
    REQUIRE(mkdtemp(sock_dir) != NULL);
  }

  int sv[4][2];
  if (!use_unix_rendezvous) {
    for (uint32_t r = 1u; r < W; ++r) {
      REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv[r]) == 0);
    }
  }

  int res_pipe[4][2];
  for (uint32_t r = 0u; r < W; ++r) {
    REQUIRE(pipe2(res_pipe[r], O_CLOEXEC) == 0);
  }

  pid_t pids[4];
  /* Fork in reversed rank order W-1..0 to exercise rendezvous retry & ordering. */
  for (int ri = (int)W - 1; ri >= 0; --ri) {
    const uint32_t r = (uint32_t)ri;
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      for (uint32_t k = 0u; k < W; ++k) {
        close(res_pipe[k][0]);
        if (k != r) {
          close(res_pipe[k][1]);
        }
      }

      et_fsdp_comm_t *comm = NULL;
      if (use_unix_rendezvous) {
        if (et_fsdp_comm_open_unix(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR,
                                   sock_dir, r, W, 10000, &comm) != ET_FSDP_OK) {
          _exit(20);
        }
      } else {
        int my_fds[3];
        size_t n_fds = 0u;
        if (W > 1u) {
          if (r == 0u) {
            for (uint32_t k = 1u; k < W; ++k) {
              close(sv[k][1]);
              my_fds[n_fds++] = sv[k][0];
            }
          } else {
            for (uint32_t k = 1u; k < W; ++k) {
              close(sv[k][0]);
              if (k != r) {
                close(sv[k][1]);
              }
            }
            my_fds[n_fds++] = sv[r][1];
          }
        }
        if (et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, r, W,
                                    n_fds > 0u ? my_fds : NULL, n_fds,
                                    &comm) != ET_FSDP_OK) {
          _exit(21);
        }
        for (size_t i = 0u; i < n_fds; ++i) {
          close(my_fds[i]);
        }
      }

      const et_fsdp_config_v1 cfg = make_cfg(stage);
      et_fsdp_engine_t *eng = NULL;
      if (et_fsdp_engine_create(comm, &cfg, &eng) != ET_FSDP_OK) {
        _exit(22);
      }

      char ckpt_path[256];
      if (ckpt_dir != NULL) {
        snprintf(ckpt_path, sizeof(ckpt_path), "%s/rank-%u.etfs", ckpt_dir, r);
      }

      if (start_step == 0u) {
        float init_p[NUM_PARAMS];
        if (et_fsdp_mlp_init_params(IN_DIM, HIDDEN_DIM, NUM_CLASSES, MODEL_SEED,
                                    init_p, NUM_PARAMS) != ET_FSDP_OK ||
            et_fsdp_engine_load_initial_params(eng, init_p, NUM_PARAMS) !=
                ET_FSDP_OK) {
          _exit(23);
        }
      } else {
        if (ckpt_dir == NULL ||
            et_fsdp_engine_load_checkpoint(eng, ckpt_path) != ET_FSDP_OK ||
            et_fsdp_engine_get_step(eng) != start_step) {
          _exit(24);
        }
      }

      fork_result_t fr;
      memset(&fr, 0, sizeof(fr));
      fr.poison_ok = 1;
      fr.pad_zero_ok = 1;

      const size_t local_b = GLOBAL_BATCH / (size_t)W;
      float x[GLOBAL_BATCH * IN_DIM];
      int32_t y[GLOBAL_BATCH];
      float grads[NUM_PARAMS];
      float eph_params[NUM_PARAMS];

      for (uint32_t s = start_step; s < end_step; ++s) {
        const uint64_t base_tag = (uint64_t)(s + 1u) * 100u;
        if (et_fsdp_engine_gather_params(eng, eph_params, NUM_PARAMS,
                                         base_tag + 1u, 10000) != ET_FSDP_OK) {
          _exit(25);
        }
        if (et_fsdp_mlp_make_batch(IN_DIM, NUM_CLASSES, GLOBAL_BATCH, r, W, s,
                                   x, y) != ET_FSDP_OK) {
          _exit(26);
        }
        double local_loss = 0.0;
        if (et_fsdp_mlp_forward_backward(IN_DIM, HIDDEN_DIM, NUM_CLASSES,
                                         local_b, eph_params, x, y, &local_loss,
                                         grads) != ET_FSDP_OK) {
          _exit(27);
        }
        if (et_fsdp_engine_poison_ephemeral(eph_params, NUM_PARAMS) !=
            ET_FSDP_OK) {
          _exit(28);
        }
        for (size_t k = 0u; k < NUM_PARAMS; ++k) {
          if (f32_bits(eph_params[k]) != ET_FSDP_POISON_F32_BITS) {
            fr.poison_ok = 0;
          }
        }
        double sum_loss = 0.0;
        if (et_fsdp_allreduce_sum_f64(comm, &local_loss, &sum_loss, 1u,
                                      base_tag + 2u, 10000) != ET_FSDP_OK) {
          _exit(29);
        }
        fr.losses[s] = sum_loss / (double)W;

        et_fsdp_step_info_v1 info;
        if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, base_tag + 3u, 10000,
                                &info) != ET_FSDP_OK) {
          _exit(30);
        }
        fr.grad_norms[s] = info.global_grad_norm;
        fr.clipped_flags[s] = info.clipped;
      }

      if (ckpt_dir != NULL && end_step < TRAIN_STEPS) {
        if (et_fsdp_engine_save_checkpoint(eng, ckpt_path) != ET_FSDP_OK) {
          _exit(31);
        }
        struct stat cst;
        if (stat(ckpt_path, &cst) != 0 || (cst.st_mode & 0777) != 0600) {
          _exit(32);
        }
      }

      if (et_fsdp_engine_get_memory_report(eng, &fr.mem) != ET_FSDP_OK ||
          et_fsdp_comm_get_stats(comm, &fr.comm_stats) != ET_FSDP_OK) {
        _exit(33);
      }

      /* Verify trailing pad slots on rank r are strictly 0.0f. */
      const size_t slen = (size_t)fr.mem.shard_len;
      float *ps = (float *)malloc(slen * sizeof(float));
      float *ms = (float *)malloc(slen * sizeof(float));
      float *vs = (float *)malloc(slen * sizeof(float));
      float *gs = (float *)malloc(slen * sizeof(float));
      if (ps == NULL || ms == NULL || vs == NULL || gs == NULL ||
          et_fsdp_engine_get_param_shard(eng, ps, slen) != ET_FSDP_OK ||
          et_fsdp_engine_get_moments_shard(eng, ms, vs, slen) != ET_FSDP_OK ||
          et_fsdp_engine_get_grad_shard(eng, gs, slen) != ET_FSDP_OK) {
        _exit(34);
      }
      const uint64_t gbase = (uint64_t)r * fr.mem.shard_len;
      for (size_t i = 0u; i < slen; ++i) {
        if (gbase + i >= NUM_PARAMS) {
          if (f32_bits(ps[i]) != 0u || f32_bits(ms[i]) != 0u ||
              f32_bits(vs[i]) != 0u || f32_bits(gs[i]) != 0u) {
            fr.pad_zero_ok = 0;
          }
        }
      }
      free(ps);
      free(ms);
      free(vs);
      free(gs);

      if (et_fsdp_engine_gather_params(eng, fr.final_params, NUM_PARAMS,
                                       90001u, 10000) != ET_FSDP_OK ||
          et_fsdp_engine_gather_full_params_sha256(eng, 90002u, 10000,
                                                   fr.final_sha256) !=
              ET_FSDP_OK) {
        _exit(35);
      }

      et_fsdp_engine_destroy(eng);
      et_fsdp_comm_destroy(comm);

      write_exact_fd(res_pipe[r][1], &fr, sizeof(fr));
      close(res_pipe[r][1]);
      _exit(0);
    }
    pids[r] = pid;
  }

  if (!use_unix_rendezvous) {
    for (uint32_t r = 1u; r < W; ++r) {
      close(sv[r][0]);
      close(sv[r][1]);
    }
  }

  fork_result_t rank_res[4];
  for (uint32_t r = 0u; r < W; ++r) {
    close(res_pipe[r][1]);
    read_exact_fd(res_pipe[r][0], &rank_res[r], sizeof(fork_result_t));
    close(res_pipe[r][0]);
  }
  for (uint32_t r = 0u; r < W; ++r) {
    int st = 0;
    REQUIRE(waitpid(pids[r], &st, 0) == pids[r]);
    REQUIRE(WIFEXITED(st) && WEXITSTATUS(st) == 0);
  }
  if (use_unix_rendezvous && W > 1u) {
    (void)rmdir(sock_dir);
  }

  /* Verify all ranks agree bitwise on final params, SHA-256, and step losses. */
  for (uint32_t r = 0u; r < W; ++r) {
    REQUIRE(rank_res[r].poison_ok == 1);
    REQUIRE(rank_res[r].pad_zero_ok == 1);
    REQUIRE(memcmp(rank_res[0].final_sha256, rank_res[r].final_sha256, 32u) ==
            0);
    REQUIRE(memcmp(rank_res[0].final_params, rank_res[r].final_params,
                   sizeof(rank_res[0].final_params)) == 0);
    for (uint32_t s = start_step; s < end_step; ++s) {
      REQUIRE(rank_res[0].losses[s] == rank_res[r].losses[s]);
      REQUIRE(rank_res[0].grad_norms[s] == rank_res[r].grad_norms[s]);
    }
  }

  REQUIRE(count_open_fds() == fds_before);
  *out_r0 = rank_res[0];
}

static void check_memory_invariants(const et_fsdp_memory_report_v1 *m,
                                    uint32_t stage, uint32_t W) {
  const uint64_t expected_shard = (NUM_PARAMS + W - 1u) / W;
  const uint64_t expected_padded = expected_shard * W;
  REQUIRE(m->stage == stage);
  REQUIRE(m->world_size == W);
  REQUIRE(m->num_params == NUM_PARAMS);
  REQUIRE(m->padded_params == expected_padded);
  REQUIRE(m->shard_len == expected_shard);
  REQUIRE(m->persistent_opt_bytes == 2u * expected_shard * 4u);
  REQUIRE(fabs(m->optimizer_reduction_ratio - (double)W) < 1e-12);

  if (stage == ET_ZERO_STAGE_1) {
    REQUIRE(m->persistent_param_bytes == expected_padded * 4u);
    REQUIRE(m->persistent_grad_bytes == expected_padded * 4u);
  } else if (stage == ET_ZERO_STAGE_2) {
    REQUIRE(m->persistent_param_bytes == expected_padded * 4u);
    REQUIRE(m->persistent_grad_bytes == expected_shard * 4u);
  } else {
    REQUIRE(m->persistent_param_bytes == expected_shard * 4u);
    REQUIRE(m->persistent_grad_bytes == expected_shard * 4u);
    REQUIRE(m->persistent_total_bytes == 4u * expected_shard * 4u);
    REQUIRE(m->unsharded_baseline_bytes == 4u * expected_padded * 4u);
    REQUIRE(fabs(m->total_persistent_reduction_ratio - (double)W) < 1e-12);
    REQUIRE(m->ephemeral_peak_bytes == expected_padded * 4u);
  }
}

static void write_oracle_dump(const char *dump_dir, uint32_t stage, uint32_t W,
                              const fork_result_t *res) {
  char path[512];
  snprintf(path, sizeof(path), "%s/stage%u_w%u.bin", dump_dir, stage, W);
  FILE *fp = fopen(path, "wb");
  REQUIRE(fp != NULL);
  uint64_t hdr[8] = {stage,
                     W,
                     NUM_PARAMS,
                     TRAIN_STEPS,
                     res->mem.persistent_param_bytes,
                     res->mem.persistent_grad_bytes,
                     res->mem.persistent_opt_bytes,
                     res->mem.persistent_total_bytes};
  REQUIRE(fwrite(hdr, sizeof(uint64_t), 8u, fp) == 8u);
  REQUIRE(fwrite(res->losses, sizeof(double), TRAIN_STEPS, fp) == TRAIN_STEPS);
  REQUIRE(fwrite(res->grad_norms, sizeof(double), TRAIN_STEPS, fp) ==
          TRAIN_STEPS);
  REQUIRE(fwrite(res->final_params, sizeof(float), NUM_PARAMS, fp) ==
          NUM_PARAMS);
  REQUIRE(fclose(fp) == 0);
}

static void test_all_stages_and_worlds(const char *oracle_dump_dir) {
  const uint32_t worlds[3] = {1u, 2u, 4u};
  fork_result_t results[4][3]; /* [stage 1..3][w_idx 0..2] */

  for (uint32_t wi = 0u; wi < 3u; ++wi) {
    const uint32_t W = worlds[wi];
    for (uint32_t stage = 1u; stage <= 3u; ++stage) {
      /* Alternate socketpair vs AF_UNIX rendezvous across stages so both
       * bootstrap paths are thoroughly tested! */
      const int use_unix = (stage == ET_ZERO_STAGE_3_FSDP) ? 1 : 0;
      run_training_group(stage, W, use_unix, NULL, 0u, TRAIN_STEPS,
                         &results[stage][wi]);
      check_memory_invariants(&results[stage][wi].mem, stage, W);

      /* Strictly decreasing loss across all 12 steps + active global clipping. */
      uint32_t any_clipped = 0u;
      for (uint32_t s = 0u; s < TRAIN_STEPS; ++s) {
        REQUIRE(isfinite(results[stage][wi].losses[s]));
        if (s > 0u) {
          REQUIRE(results[stage][wi].losses[s] <
                  results[stage][wi].losses[s - 1u]);
        }
        any_clipped |= results[stage][wi].clipped_flags[s];
      }
      REQUIRE(any_clipped == 1u);

      if (oracle_dump_dir != NULL) {
        write_oracle_dump(oracle_dump_dir, stage, W, &results[stage][wi]);
      }
    }

    /* Stages 1, 2, and 3 MUST be bitwise identical for the same W! */
    for (uint32_t stage = 2u; stage <= 3u; ++stage) {
      REQUIRE(memcmp(results[1][wi].final_sha256,
                     results[stage][wi].final_sha256, 32u) == 0);
      REQUIRE(memcmp(results[1][wi].final_params,
                     results[stage][wi].final_params,
                     NUM_PARAMS * sizeof(float)) == 0);
      for (uint32_t s = 0u; s < TRAIN_STEPS; ++s) {
        REQUIRE(results[1][wi].losses[s] == results[stage][wi].losses[s]);
      }
    }
  }

  /* Verify tight <= 1e-6 weight and loss parity across W=1, W=2, W=4! */
  for (uint32_t wi = 1u; wi < 3u; ++wi) {
    const uint32_t W = worlds[wi];
    double max_w_diff = 0.0;
    double max_l_diff = 0.0;
    for (size_t i = 0u; i < NUM_PARAMS; ++i) {
      const double d = fabs((double)results[3][wi].final_params[i] -
                            (double)results[3][0].final_params[i]);
      if (d > max_w_diff) {
        max_w_diff = d;
      }
    }
    for (uint32_t s = 0u; s < TRAIN_STEPS; ++s) {
      const double d =
          fabs(results[3][wi].losses[s] - results[3][0].losses[s]);
      if (d > max_l_diff) {
        max_l_diff = d;
      }
    }
    printf("FSDP NATIVE PARITY W=%u vs W=1: max_weight_diff=%.3e max_loss_diff=%.3e "
           "loss0=%.8f loss11=%.8f mem_total=%lluB (%.1fx) PASS\n",
           W, max_w_diff, max_l_diff, results[3][wi].losses[0],
           results[3][wi].losses[TRAIN_STEPS - 1u],
           (unsigned long long)results[3][wi].mem.persistent_total_bytes,
           results[3][wi].mem.total_persistent_reduction_ratio);
    REQUIRE(max_w_diff <= 1e-6);
    REQUIRE(max_l_diff <= 1e-6);
  }

  /* Verify exact 6+6 fresh-process sharded checkpoint resume for Stage 3
   * across W = 1, 2, 4 AND Stage 2 for W = 2! */
  for (uint32_t wi = 0u; wi < 3u; ++wi) {
    const uint32_t W = worlds[wi];
    char ckpt_dir[128] = "/tmp/fsdp_ckpt_XXXXXX";
    REQUIRE(mkdtemp(ckpt_dir) != NULL);

    fork_result_t part1;
    fork_result_t part2;
    run_training_group(ET_ZERO_STAGE_3_FSDP, W, 1, ckpt_dir, 0u, 6u, &part1);
    run_training_group(ET_ZERO_STAGE_3_FSDP, W, 1, ckpt_dir, 6u, TRAIN_STEPS,
                       &part2);

    /* Bitwise identical final weights and step 6..11 losses! */
    REQUIRE(memcmp(part2.final_sha256, results[3][wi].final_sha256, 32u) == 0);
    REQUIRE(memcmp(part2.final_params, results[3][wi].final_params,
                   NUM_PARAMS * sizeof(float)) == 0);
    for (uint32_t s = 6u; s < TRAIN_STEPS; ++s) {
      REQUIRE(part2.losses[s] == results[3][wi].losses[s]);
    }

    for (uint32_t r = 0u; r < W; ++r) {
      char p[256];
      snprintf(p, sizeof(p), "%s/rank-%u.etfs", ckpt_dir, r);
      REQUIRE(unlink(p) == 0);
    }
    REQUIRE(rmdir(ckpt_dir) == 0);
    printf("FSDP NATIVE RESUME 6+6 Stage=3 W=%u: bitwise_exact=PASS\n", W);
  }

  /* Verify Stage 2 W=2 6+6 fresh-process checkpoint resume (exercises
   * params_full serialization/deserialization alongside m_shard/v_shard). */
  {
    const uint32_t W = 2u;
    char ckpt_dir[128] = "/tmp/fsdp_ckpt_s2_XXXXXX";
    REQUIRE(mkdtemp(ckpt_dir) != NULL);

    fork_result_t part1;
    fork_result_t part2;
    run_training_group(ET_ZERO_STAGE_2, W, 1, ckpt_dir, 0u, 6u, &part1);
    run_training_group(ET_ZERO_STAGE_2, W, 1, ckpt_dir, 6u, TRAIN_STEPS, &part2);

    REQUIRE(memcmp(part2.final_sha256, results[2][1].final_sha256, 32u) == 0);
    REQUIRE(memcmp(part2.final_params, results[2][1].final_params,
                   NUM_PARAMS * sizeof(float)) == 0);
    for (uint32_t s = 6u; s < TRAIN_STEPS; ++s) {
      REQUIRE(part2.losses[s] == results[2][1].losses[s]);
    }
    for (uint32_t r = 0u; r < W; ++r) {
      char p[256];
      snprintf(p, sizeof(p), "%s/rank-%u.etfs", ckpt_dir, r);
      REQUIRE(unlink(p) == 0);
    }
    REQUIRE(rmdir(ckpt_dir) == 0);
    printf("FSDP NATIVE RESUME 6+6 Stage=2 W=2: bitwise_exact=PASS\n");
  }
}

static void test_atomic_nonfinite_abort_w4(void) {
  const int fds_before = count_open_fds();
  const uint32_t W = 4u;
  int sv[4][2];
  for (uint32_t r = 1u; r < W; ++r) {
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv[r]) == 0);
  }

  pid_t pids[4];
  for (uint32_t r = 0u; r < W; ++r) {
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      int my_fds[3];
      size_t n_fds = 0u;
      if (r == 0u) {
        for (uint32_t k = 1u; k < W; ++k) {
          close(sv[k][1]);
          my_fds[n_fds++] = sv[k][0];
        }
      } else {
        for (uint32_t k = 1u; k < W; ++k) {
          close(sv[k][0]);
          if (k != r) {
            close(sv[k][1]);
          }
        }
        my_fds[n_fds++] = sv[r][1];
      }

      et_fsdp_comm_t *comm = NULL;
      if (et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, r, W,
                                  my_fds, n_fds, &comm) != ET_FSDP_OK) {
        _exit(40);
      }
      for (size_t i = 0u; i < n_fds; ++i) {
        close(my_fds[i]);
      }

      const et_fsdp_config_v1 cfg = make_cfg(ET_ZERO_STAGE_3_FSDP);
      et_fsdp_engine_t *eng = NULL;
      float init_p[NUM_PARAMS];
      if (et_fsdp_engine_create(comm, &cfg, &eng) != ET_FSDP_OK ||
          et_fsdp_mlp_init_params(IN_DIM, HIDDEN_DIM, NUM_CLASSES, MODEL_SEED,
                                  init_p, NUM_PARAMS) != ET_FSDP_OK ||
          et_fsdp_engine_load_initial_params(eng, init_p, NUM_PARAMS) !=
              ET_FSDP_OK) {
        _exit(41);
      }

      /* Run 2 valid steps first so m_shard and v_shard are non-zero. */
      float grads[NUM_PARAMS];
      for (size_t i = 0u; i < NUM_PARAMS; ++i) {
        grads[i] = 0.01f * (float)((i % 7u) + 1u);
      }
      et_fsdp_step_info_v1 info;
      if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, 1u, 5000, &info) !=
              ET_FSDP_OK ||
          et_fsdp_engine_step(eng, grads, NUM_PARAMS, 2u, 5000, &info) !=
              ET_FSDP_OK) {
        _exit(42);
      }

      /* Snapshot pre-abort shard state on EVERY rank. */
      const size_t slen = 61u;
      float p0[61], m0[61], v0[61];
      float p1[61], m1[61], v1[61];
      if (et_fsdp_engine_get_param_shard(eng, p0, slen) != ET_FSDP_OK ||
          et_fsdp_engine_get_moments_shard(eng, m0, v0, slen) != ET_FSDP_OK) {
        _exit(43);
      }

      /* Inject qNaN on rank 3 ONLY. */
      if (r == 3u) {
        grads[17] = bits_f32(UINT32_C(0x7fc00000));
      }
      if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, 3u, 5000, &info) !=
          ET_FSDP_ERR_NONFINITE) {
        _exit(44);
      }
      if (et_fsdp_engine_get_step(eng) != 2u ||
          et_fsdp_engine_get_param_shard(eng, p1, slen) != ET_FSDP_OK ||
          et_fsdp_engine_get_moments_shard(eng, m1, v1, slen) != ET_FSDP_OK ||
          memcmp(p0, p1, sizeof(p0)) != 0 || memcmp(m0, m1, sizeof(m0)) != 0 ||
          memcmp(v0, v1, sizeof(v0)) != 0) {
        _exit(45);
      }

      /* Restore rank 3 and inject +Inf on rank 2 ONLY. */
      grads[17] = 0.01f * (float)((17u % 7u) + 1u);
      if (r == 2u) {
        grads[200] = bits_f32(UINT32_C(0x7f800000));
      }
      if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, 4u, 5000, &info) !=
          ET_FSDP_ERR_NONFINITE) {
        _exit(46);
      }
      if (et_fsdp_engine_get_step(eng) != 2u ||
          et_fsdp_engine_get_param_shard(eng, p1, slen) != ET_FSDP_OK ||
          et_fsdp_engine_get_moments_shard(eng, m1, v1, slen) != ET_FSDP_OK ||
          memcmp(p0, p1, sizeof(p0)) != 0 || memcmp(m0, m1, sizeof(m0)) != 0 ||
          memcmp(v0, v1, sizeof(v0)) != 0) {
        _exit(47);
      }

      /* Recover cleanly with finite gradients on step 3! */
      grads[200] = 0.01f * (float)((200u % 7u) + 1u);
      if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, 5u, 5000, &info) !=
              ET_FSDP_OK ||
          et_fsdp_engine_get_step(eng) != 3u) {
        _exit(48);
      }

      et_fsdp_engine_destroy(eng);
      et_fsdp_comm_destroy(comm);
      _exit(0);
    }
    pids[r] = pid;
  }

  for (uint32_t r = 1u; r < W; ++r) {
    close(sv[r][0]);
    close(sv[r][1]);
  }
  for (uint32_t r = 0u; r < W; ++r) {
    int st = 0;
    REQUIRE(waitpid(pids[r], &st, 0) == pids[r]);
    REQUIRE(WIFEXITED(st) && WEXITSTATUS(st) == 0);
  }
  REQUIRE(count_open_fds() == fds_before);
  printf("FSDP NATIVE ATOMIC_NONFINITE_ABORT W=4 (NaN + Inf + recovery): PASS\n");
}

static void test_atomic_adamw_shard_overflow_w4(et_zero_stage_t stage) {
  const int fds_before = count_open_fds();
  const uint32_t W = 4u;
  int sv[4][2];
  for (uint32_t r = 1u; r < W; ++r) {
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv[r]) == 0);
  }

  pid_t pids[4];
  for (uint32_t r = 0u; r < W; ++r) {
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      int my_fds[3];
      size_t n_fds = 0u;
      if (r == 0u) {
        for (uint32_t k = 1u; k < W; ++k) {
          close(sv[k][1]);
          my_fds[n_fds++] = sv[k][0];
        }
      } else {
        for (uint32_t k = 1u; k < W; ++k) {
          close(sv[k][0]);
          if (k != r) {
            close(sv[k][1]);
          }
        }
        my_fds[n_fds++] = sv[r][1];
      }

      et_fsdp_comm_t *comm = NULL;
      if (et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, r, W,
                                  my_fds, n_fds, &comm) != ET_FSDP_OK) {
        _exit(70);
      }
      for (size_t i = 0u; i < n_fds; ++i) {
        close(my_fds[i]);
      }

      /* Disable global clipping (max_grad_norm = 0.0, explicitly supported) so
       * a finite gradient 1.0e21f inside rank 1's shard passes gradient sync
       * and binary64 norm reduction, yet overflows binary32 v_new = 1e39 > FLT_MAX
       * ONLY on rank 1's AdamW shard update. */
      et_fsdp_config_v1 cfg = make_cfg(stage);
      cfg.max_grad_norm = 0.0;
      et_fsdp_engine_t *eng = NULL;
      float init_p[NUM_PARAMS];
      if (et_fsdp_engine_create(comm, &cfg, &eng) != ET_FSDP_OK ||
          et_fsdp_mlp_init_params(IN_DIM, HIDDEN_DIM, NUM_CLASSES, MODEL_SEED,
                                  init_p, NUM_PARAMS) != ET_FSDP_OK ||
          et_fsdp_engine_load_initial_params(eng, init_p, NUM_PARAMS) !=
              ET_FSDP_OK) {
        _exit(71);
      }

      float grads[NUM_PARAMS];
      for (size_t i = 0u; i < NUM_PARAMS; ++i) {
        grads[i] = 0.01f * (float)((i % 7u) + 1u);
      }
      et_fsdp_step_info_v1 info;
      if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, 10u, 5000, &info) !=
          ET_FSDP_OK) {
        _exit(72);
      }

      const size_t slen = 61u;
      float p0[61], m0[61], v0[61];
      float p1[61], m1[61], v1[61];
      float full0[NUM_PARAMS], full1[NUM_PARAMS];
      if (et_fsdp_engine_get_param_shard(eng, p0, slen) != ET_FSDP_OK ||
          et_fsdp_engine_get_moments_shard(eng, m0, v0, slen) != ET_FSDP_OK ||
          et_fsdp_engine_gather_params(eng, full0, NUM_PARAMS, 11u, 5000) !=
              ET_FSDP_OK) {
        _exit(73);
      }

      /* Index 61 belongs exclusively to rank 1 (shard_len = 61). All ranks
       * supply finite 1.0e21f at index 61 so ReduceScatterAvg/AllReduceAvg
       * yields finite 1.0e21f and binary64 norm is finite (~1e21), while
       * rank 1's binary32 AdamW v_new overflows FLT_MAX. */
      grads[61] = 1.0e21f;
      if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, 12u, 5000, &info) !=
          ET_FSDP_ERR_NONFINITE) {
        _exit(74);
      }

      /* Every rank must preserve step, shard state, and gathered full params
       * bitwise (no Stage 3 split-brain commit, no Stage 1/2 hang/teardown). */
      if (et_fsdp_engine_get_step(eng) != 1u ||
          et_fsdp_engine_get_param_shard(eng, p1, slen) != ET_FSDP_OK ||
          et_fsdp_engine_get_moments_shard(eng, m1, v1, slen) != ET_FSDP_OK ||
          et_fsdp_engine_gather_params(eng, full1, NUM_PARAMS, 13u, 5000) !=
              ET_FSDP_OK ||
          memcmp(p0, p1, sizeof(p0)) != 0 || memcmp(m0, m1, sizeof(m0)) != 0 ||
          memcmp(v0, v1, sizeof(v0)) != 0 ||
          memcmp(full0, full1, sizeof(full0)) != 0) {
        _exit(75);
      }

      /* Communicator and engine must recover cleanly on the next valid step. */
      grads[61] = 0.01f * (float)((61u % 7u) + 1u);
      if (et_fsdp_engine_step(eng, grads, NUM_PARAMS, 14u, 5000, &info) !=
              ET_FSDP_OK ||
          et_fsdp_engine_get_step(eng) != 2u) {
        _exit(76);
      }

      et_fsdp_engine_destroy(eng);
      et_fsdp_comm_destroy(comm);
      _exit(0);
    }
    pids[r] = pid;
  }

  for (uint32_t r = 1u; r < W; ++r) {
    close(sv[r][0]);
    close(sv[r][1]);
  }
  for (uint32_t r = 0u; r < W; ++r) {
    int st = 0;
    REQUIRE(waitpid(pids[r], &st, 0) == pids[r]);
    REQUIRE(WIFEXITED(st) && WEXITSTATUS(st) == 0);
  }
  REQUIRE(count_open_fds() == fds_before);
  printf("FSDP NATIVE ATOMIC_ADAMW_SHARD_OVERFLOW Stage=%u W=4 (abort + recovery): PASS\n",
         (unsigned)stage);
}

static void test_negative_and_faults(void) {
  const int fds_before = count_open_fds();

  /* 1. Invalid communicator / engine configs. */
  et_fsdp_comm_t *bad_comm = NULL;
  REQUIRE(et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 0u, 3u,
                                  NULL, 0u, &bad_comm) ==
          ET_FSDP_ERR_INVALID_ARG);
  REQUIRE(et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 0u, 0u,
                                  NULL, 0u, &bad_comm) ==
          ET_FSDP_ERR_INVALID_ARG);
  REQUIRE(et_fsdp_comm_create_fds(99u, 0u, 0u, 1u, NULL, 0u, &bad_comm) ==
          ET_FSDP_ERR_INVALID_ARG);
  REQUIRE(et_fsdp_comm_open_unix(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR,
                                 "/nonexistent_dir_fsdp_xyz", 0u, 2u, 100,
                                 &bad_comm) == ET_FSDP_ERR_INVALID_ARG);

  et_fsdp_comm_t *c1 = NULL;
  REQUIRE(et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 0u, 1u,
                                  NULL, 0u, &c1) == ET_FSDP_OK);

  /* Overflow-guarded batch_size * in_dim check in MLP forward/backward. */
  {
    float dummy_p[NUM_PARAMS] = {0.0f};
    float dummy_x[IN_DIM] = {0.0f};
    int32_t dummy_t[1] = {0};
    double dummy_loss = 0.0;
    REQUIRE(et_fsdp_mlp_forward_backward(IN_DIM, HIDDEN_DIM, NUM_CLASSES,
                                         ((size_t)ET_FSDP_MAX_PARAMS / IN_DIM) + 1u,
                                         dummy_p, dummy_x, dummy_t, &dummy_loss,
                                         NULL) == ET_FSDP_ERR_INVALID_ARG);
    REQUIRE(et_fsdp_mlp_forward_backward(IN_DIM, HIDDEN_DIM, NUM_CLASSES,
                                         SIZE_MAX / 2u, dummy_p, dummy_x,
                                         dummy_t, &dummy_loss,
                                         NULL) == ET_FSDP_ERR_INVALID_ARG);
  }

  et_fsdp_config_v1 cfg = make_cfg(ET_ZERO_STAGE_3_FSDP);
  et_fsdp_engine_t *eng = NULL;
  cfg.stage = 0u;
  REQUIRE(et_fsdp_engine_create(c1, &cfg, &eng) == ET_FSDP_ERR_INVALID_ARG);
  cfg.stage = 4u;
  REQUIRE(et_fsdp_engine_create(c1, &cfg, &eng) == ET_FSDP_ERR_INVALID_ARG);
  cfg = make_cfg(ET_ZERO_STAGE_3_FSDP);
  cfg.lr = -0.1;
  REQUIRE(et_fsdp_engine_create(c1, &cfg, &eng) == ET_FSDP_ERR_INVALID_ARG);
  cfg = make_cfg(ET_ZERO_STAGE_3_FSDP);
  cfg.beta1 = NAN;
  REQUIRE(et_fsdp_engine_create(c1, &cfg, &eng) == ET_FSDP_ERR_NONFINITE);
  cfg = make_cfg(ET_ZERO_STAGE_3_FSDP);
  cfg.eps = 0.0;
  REQUIRE(et_fsdp_engine_create(c1, &cfg, &eng) == ET_FSDP_ERR_INVALID_ARG);

  /* Overlapping buffers rejected in allgather / reduce_scatter. */
  float buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  REQUIRE(et_fsdp_allgather_f32(c1, buf, 4u, buf + 2, 1u, 1000) ==
          ET_FSDP_ERR_INVALID_ARG);
  REQUIRE(et_fsdp_reduce_scatter_avg_f32(c1, buf, buf + 2, 4u, 1u, 1000) ==
          ET_FSDP_ERR_INVALID_ARG);

  /* 2. Corrupt / tampered / truncated / mismatched / symlink checkpoint tests. */
  cfg = make_cfg(ET_ZERO_STAGE_3_FSDP);
  REQUIRE(et_fsdp_engine_create(c1, &cfg, &eng) == ET_FSDP_OK);
  float init_p[NUM_PARAMS];
  REQUIRE(et_fsdp_mlp_init_params(IN_DIM, HIDDEN_DIM, NUM_CLASSES, MODEL_SEED,
                                  init_p, NUM_PARAMS) == ET_FSDP_OK);
  REQUIRE(et_fsdp_engine_load_initial_params(eng, init_p, NUM_PARAMS) ==
          ET_FSDP_OK);
  float g[NUM_PARAMS];
  for (size_t i = 0u; i < NUM_PARAMS; ++i) {
    g[i] = 0.02f;
  }
  et_fsdp_step_info_v1 info;
  REQUIRE(et_fsdp_engine_step(eng, g, NUM_PARAMS, 1u, 5000, &info) ==
          ET_FSDP_OK);

  char tdir[128] = "/tmp/fsdp_neg_XXXXXX";
  REQUIRE(mkdtemp(tdir) != NULL);
  char good_ckpt[256], bad_ckpt[256], link_ckpt[256];
  snprintf(good_ckpt, sizeof(good_ckpt), "%s/good.etfs", tdir);
  snprintf(bad_ckpt, sizeof(bad_ckpt), "%s/bad.etfs", tdir);
  snprintf(link_ckpt, sizeof(link_ckpt), "%s/link.etfs", tdir);

  REQUIRE(et_fsdp_engine_save_checkpoint(eng, good_ckpt) == ET_FSDP_OK);

  /* Read good checkpoint into memory to craft negative variants. */
  FILE *fp = fopen(good_ckpt, "rb");
  REQUIRE(fp != NULL);
  REQUIRE(fseek(fp, 0, SEEK_END) == 0);
  const long sz = ftell(fp);
  REQUIRE(sz > 160);
  rewind(fp);
  uint8_t *raw = (uint8_t *)malloc((size_t)sz + 4u);
  REQUIRE(raw != NULL);
  REQUIRE(fread(raw, 1u, (size_t)sz, fp) == (size_t)sz);
  fclose(fp);

  /* 2a. Symlink refused on both load and save. */
  REQUIRE(symlink(good_ckpt, link_ckpt) == 0);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng, link_ckpt) == ET_FSDP_ERR_IO);
  REQUIRE(et_fsdp_engine_save_checkpoint(eng, link_ckpt) == ET_FSDP_ERR_IO);
  REQUIRE(unlink(link_ckpt) == 0);

  /* 2b. Bad magic */
  uint8_t *mod = (uint8_t *)malloc((size_t)sz + 4u);
  REQUIRE(mod != NULL);
  memcpy(mod, raw, (size_t)sz);
  mod[0] ^= 0xffu;
  fp = fopen(bad_ckpt, "wb");
  REQUIRE(fp != NULL && fwrite(mod, 1u, (size_t)sz, fp) == (size_t)sz);
  fclose(fp);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng, bad_ckpt) ==
          ET_FSDP_ERR_BAD_MAGIC);

  /* 2c. Version mismatch */
  memcpy(mod, raw, (size_t)sz);
  mod[8] = 9u;
  fp = fopen(bad_ckpt, "wb");
  REQUIRE(fp != NULL && fwrite(mod, 1u, (size_t)sz, fp) == (size_t)sz);
  fclose(fp);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng, bad_ckpt) ==
          ET_FSDP_ERR_VERSION_MISMATCH);

  /* 2d. Tampered payload byte -> checksum mismatch */
  memcpy(mod, raw, (size_t)sz);
  mod[ET_FSDP_CKPT_HEADER_BYTES + 12u] ^= 0x01u;
  fp = fopen(bad_ckpt, "wb");
  REQUIRE(fp != NULL && fwrite(mod, 1u, (size_t)sz, fp) == (size_t)sz);
  fclose(fp);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng, bad_ckpt) ==
          ET_FSDP_ERR_CHECKSUM_MISMATCH);

  /* 2e. Truncated and trailing bytes */
  fp = fopen(bad_ckpt, "wb");
  REQUIRE(fp != NULL && fwrite(raw, 1u, (size_t)sz - 1u, fp) == (size_t)sz - 1u);
  fclose(fp);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng, bad_ckpt) ==
          ET_FSDP_ERR_TRUNCATED);
  memcpy(mod, raw, (size_t)sz);
  mod[sz] = 0x42u;
  fp = fopen(bad_ckpt, "wb");
  REQUIRE(fp != NULL && fwrite(mod, 1u, (size_t)sz + 1u, fp) == (size_t)sz + 1u);
  fclose(fp);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng, bad_ckpt) ==
          ET_FSDP_ERR_TRUNCATED);

  /* 2f. Mismatched stage / world_size / hyperparam rejected */
  et_fsdp_config_v1 cfg_s2 = make_cfg(ET_ZERO_STAGE_2);
  et_fsdp_engine_t *eng_s2 = NULL;
  REQUIRE(et_fsdp_engine_create(c1, &cfg_s2, &eng_s2) == ET_FSDP_OK);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng_s2, good_ckpt) ==
          ET_FSDP_ERR_CONFIG_MISMATCH);
  et_fsdp_engine_destroy(eng_s2);

  /* 2g. Re-signed negative v_shard rejected atomically */
  memcpy(mod, raw, (size_t)sz);
  const size_t v_off = ET_FSDP_CKPT_HEADER_BYTES + (2u * NUM_PARAMS) * 4u;
  put_u32_le(mod + v_off, f32_bits(-1.0f));
  uint8_t resign[32];
  REQUIRE(et_fsdp_sha256_bytes(mod, (size_t)sz - 32u, resign) == ET_FSDP_OK);
  memcpy(mod + (size_t)sz - 32u, resign, 32u);
  fp = fopen(bad_ckpt, "wb");
  REQUIRE(fp != NULL && fwrite(mod, 1u, (size_t)sz, fp) == (size_t)sz);
  fclose(fp);
  REQUIRE(et_fsdp_engine_load_checkpoint(eng, bad_ckpt) ==
          ET_FSDP_ERR_NONFINITE);
  REQUIRE(et_fsdp_engine_get_step(eng) == 1u);

  free(raw);
  free(mod);
  REQUIRE(unlink(good_ckpt) == 0);
  REQUIRE(unlink(bad_ckpt) == 0);
  REQUIRE(rmdir(tdir) == 0);

  et_fsdp_engine_destroy(eng);
  et_fsdp_comm_destroy(c1);

  /* 3. Tag mismatch (W=2) -> sticky ET_FSDP_ERR_MISMATCH on both ranks. */
  {
    int sp[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) == 0);
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      close(sp[0]);
      et_fsdp_comm_t *c = NULL;
      if (et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 1u, 2u,
                                  &sp[1], 1u, &c) != ET_FSDP_OK) {
        _exit(50);
      }
      close(sp[1]);
      double a = 1.0, b = 0.0;
      const et_fsdp_status_t st1 =
          et_fsdp_allreduce_sum_f64(c, &a, &b, 1u, 999u, 3000);
      const et_fsdp_status_t st2 =
          et_fsdp_allreduce_sum_f64(c, &a, &b, 1u, 111u, 3000);
      et_fsdp_comm_destroy(c);
      _exit((st1 == ET_FSDP_ERR_MISMATCH && st2 == ET_FSDP_ERR_MISMATCH) ? 0
                                                                         : 51);
    }
    close(sp[1]);
    et_fsdp_comm_t *c0 = NULL;
    REQUIRE(et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 0u,
                                    2u, &sp[0], 1u, &c0) == ET_FSDP_OK);
    close(sp[0]);
    double a = 1.0, b = 0.0;
    REQUIRE(et_fsdp_allreduce_sum_f64(c0, &a, &b, 1u, 111u, 3000) ==
            ET_FSDP_ERR_MISMATCH);
    REQUIRE(et_fsdp_allreduce_sum_f64(c0, &a, &b, 1u, 111u, 3000) ==
            ET_FSDP_ERR_MISMATCH);
    et_fsdp_comm_destroy(c0);
    int wst = 0;
    REQUIRE(waitpid(pid, &wst, 0) == pid && WIFEXITED(wst) &&
            WEXITSTATUS(wst) == 0);
  }

  /* 4. Peer disconnect & monotonic timeout with zero FD leaks. */
  {
    int sp[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) == 0);
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      close(sp[0]);
      close(sp[1]);
      _exit(0);
    }
    close(sp[1]);
    et_fsdp_comm_t *c0 = NULL;
    REQUIRE(et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 0u,
                                    2u, &sp[0], 1u, &c0) == ET_FSDP_OK);
    close(sp[0]);
    double a = 1.0, b = 0.0;
    REQUIRE(et_fsdp_allreduce_sum_f64(c0, &a, &b, 1u, 55u, 2000) ==
            ET_FSDP_ERR_PEER_DISCONNECTED);
    et_fsdp_comm_destroy(c0);
    int wst = 0;
    REQUIRE(waitpid(pid, &wst, 0) == pid);
  }
  {
    int sp[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) == 0);
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      close(sp[0]);
      usleep(250000);
      close(sp[1]);
      _exit(0);
    }
    close(sp[1]);
    et_fsdp_comm_t *c0 = NULL;
    REQUIRE(et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 0u,
                                    2u, &sp[0], 1u, &c0) == ET_FSDP_OK);
    close(sp[0]);
    double a = 1.0, b = 0.0;
    REQUIRE(et_fsdp_allreduce_sum_f64(c0, &a, &b, 1u, 66u, 50) ==
            ET_FSDP_ERR_TIMEOUT);
    et_fsdp_comm_destroy(c0);
    int wst = 0;
    REQUIRE(waitpid(pid, &wst, 0) == pid);
  }

  /* 5. Duplicate rank rejection in AF_UNIX rendezvous. */
  {
    char sdir[128] = "/tmp/fsdp_dup_XXXXXX";
    REQUIRE(mkdtemp(sdir) != NULL);
    pid_t p1 = fork();
    REQUIRE(p1 >= 0);
    if (p1 == 0) {
      et_fsdp_comm_t *c = NULL;
      const et_fsdp_status_t st = et_fsdp_comm_open_unix(
          ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, sdir, 1u, 4u, 3000, &c);
      et_fsdp_comm_destroy(c);
      _exit(st != ET_FSDP_OK ? 0 : 61);
    }
    pid_t p2 = fork();
    REQUIRE(p2 >= 0);
    if (p2 == 0) {
      et_fsdp_comm_t *c = NULL;
      const et_fsdp_status_t st = et_fsdp_comm_open_unix(
          ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, sdir, 1u, 4u, 3000, &c);
      et_fsdp_comm_destroy(c);
      _exit(st != ET_FSDP_OK ? 0 : 62);
    }
    et_fsdp_comm_t *c0 = NULL;
    const et_fsdp_status_t st0 = et_fsdp_comm_open_unix(
        ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, sdir, 0u, 4u, 3000, &c0);
    REQUIRE(st0 == ET_FSDP_ERR_MISMATCH && c0 == NULL);
    int wst = 0;
    REQUIRE(waitpid(p1, &wst, 0) == p1 && WIFEXITED(wst) &&
            WEXITSTATUS(wst) == 0);
    REQUIRE(waitpid(p2, &wst, 0) == p2 && WIFEXITED(wst) &&
            WEXITSTATUS(wst) == 0);
    REQUIRE(rmdir(sdir) == 0);
  }

  REQUIRE(count_open_fds() == fds_before);
  printf("FSDP NATIVE NEGATIVE_AND_FAULTS (ckpt/mismatch/disconnect/timeout/dup/FD=0): PASS\n");
}

int main(int argc, char **argv) {
  const char *oracle_dump_dir = NULL;
  if (argc == 3 && strcmp(argv[1], "--oracle-dump") == 0) {
    oracle_dump_dir = argv[2];
  } else if (argc != 1) {
    fprintf(stderr, "usage: %s [--oracle-dump DIR]\n", argv[0]);
    return 2;
  }

  test_fd_gradient_check();
  test_collectives_w4();
  test_all_stages_and_worlds(oracle_dump_dir);
  test_atomic_nonfinite_abort_w4();
  test_atomic_adamw_shard_overflow_w4(ET_ZERO_STAGE_2);
  test_atomic_adamw_shard_overflow_w4(ET_ZERO_STAGE_3_FSDP);
  test_negative_and_faults();

  printf("FSDP NATIVE ALL TESTS PASS\n");
  return 0;
}
