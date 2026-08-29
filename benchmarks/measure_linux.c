#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef __linux__
#error "B0 peak-RSS measurement is supported only on Linux"
#endif
#ifndef SYS_pidfd_open
#error "B0 completion measurement requires Linux pidfd_open"
#endif

static void fail(const char *message) {
  fprintf(stderr, "error: %s: %s\n", message, strerror(errno));
  exit(2);
}

static int64_t monotonic_ns(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    fail("CLOCK_MONOTONIC is unavailable");
  }
  return (int64_t)now.tv_sec * 1000000000LL + now.tv_nsec;
}

static long parse_timeout_ms(const char *text) {
  char *end = NULL;
  errno = 0;
  long value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > 3600000) {
    fprintf(stderr, "error: timeout must be an integer from 1 through 3600000 ms\n");
    exit(2);
  }
  return value;
}

int main(int argc, char **argv) {
  if (argc < 6 || strcmp(argv[4], "--") != 0) {
    fprintf(stderr,
            "usage: measure-linux TIMEOUT_MS STDOUT_PATH STDERR_PATH -- COMMAND [ARG ...]\n");
    return 2;
  }

  const long timeout_ms = parse_timeout_ms(argv[1]);
  const int stdout_fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (stdout_fd < 0) {
    fail("cannot open captured stdout");
  }
  const int stderr_fd = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (stderr_fd < 0) {
    fail("cannot open captured stderr");
  }

  const int64_t start_ns = monotonic_ns();
  const pid_t child = fork();
  if (child < 0) {
    fail("fork failed");
  }
  if (child == 0) {
    (void)setpgid(0, 0);
    if (dup2(stdout_fd, STDOUT_FILENO) < 0 || dup2(stderr_fd, STDERR_FILENO) < 0) {
      _exit(126);
    }
    close(stdout_fd);
    close(stderr_fd);
    execv(argv[5], &argv[5]);
    dprintf(STDERR_FILENO, "error: benchmark exec failed: %s\n", strerror(errno));
    _exit(127);
  }

  close(stdout_fd);
  close(stderr_fd);
  (void)setpgid(child, child);

  const int pidfd = (int)syscall(SYS_pidfd_open, child, 0);
  if (pidfd < 0) {
    const int saved_errno = errno;
    (void)kill(-child, SIGKILL);
    (void)kill(child, SIGKILL);
    (void)waitpid(child, NULL, 0);
    errno = saved_errno;
    fail("pidfd_open is unavailable; no completion-timing fallback is permitted");
  }

  int status = 0;
  int timed_out = 0;
  int64_t end_ns = 0;
  struct rusage usage;
  memset(&usage, 0, sizeof(usage));
  const int64_t deadline_ns = start_ns + (int64_t)timeout_ms * 1000000LL;
  struct pollfd completion = {.fd = pidfd, .events = POLLIN, .revents = 0};
  for (;;) {
    const int64_t remaining_ns = deadline_ns - monotonic_ns();
    if (remaining_ns <= 0) {
      timed_out = 1;
      break;
    }
    const int64_t remaining_ms64 = (remaining_ns + 999999LL) / 1000000LL;
    const int remaining_ms =
        remaining_ms64 > INT_MAX ? INT_MAX : (int)remaining_ms64;
    const int polled = poll(&completion, 1, remaining_ms);
    if (polled > 0) {
      if ((completion.revents & POLLIN) == 0) {
        (void)kill(-child, SIGKILL);
        (void)kill(child, SIGKILL);
        (void)waitpid(child, NULL, 0);
        close(pidfd);
        errno = EIO;
        fail("pidfd poll returned no completion event");
      }
      end_ns = monotonic_ns();
      break;
    }
    if (polled == 0) {
      timed_out = 1;
      break;
    }
    if (errno != EINTR) {
      const int saved_errno = errno;
      (void)kill(-child, SIGKILL);
      (void)kill(child, SIGKILL);
      (void)waitpid(child, NULL, 0);
      close(pidfd);
      errno = saved_errno;
      fail("pidfd poll failed");
    }
  }
  if (timed_out) {
    if (kill(-child, SIGKILL) != 0 && errno != ESRCH) {
      fail("cannot terminate timed-out benchmark process group");
    }
    if (kill(child, SIGKILL) != 0 && errno != ESRCH) {
      fail("cannot terminate timed-out benchmark process");
    }
  }
  if (wait4(child, &status, 0, &usage) != child) {
    fail("wait4 failed");
  }
  if (timed_out) {
    end_ns = monotonic_ns();
  }
  close(pidfd);

  const int64_t elapsed_ns = end_ns - start_ns;
  const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  const int term_signal = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
  printf("format_version\t1\n");
  printf("elapsed_ns\t%lld\n", (long long)elapsed_ns);
  printf("peak_rss_kib\t%ld\n", usage.ru_maxrss);
  printf("exit_code\t%d\n", exit_code);
  printf("term_signal\t%d\n", term_signal);
  printf("timed_out\t%s\n", timed_out ? "true" : "false");
  return 0;
}
