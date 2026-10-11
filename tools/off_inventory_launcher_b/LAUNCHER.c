#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <linux/if_alg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/random.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "PINS.h"

#define BASE "/home/gabe/.codex/evidence/eshkol-transformer"
#define I_RUNNER BASE "/g3r-cold-s-dual64-tagged-native-off-inventory-runner-20261011-i/RUNNER.py"
#define I_RUNNER_SHA "dd5954dd713653c3e0f18fd884a31d1a644c4bd1a1a1c23e1bfb5e8b7939872a"
#define I_RELEASE BASE "/g3r-cold-s-dual64-tagged-native-off-inventory-runner-release-20261011-i/RELEASE.json"
#define I_BINDING BASE "/g3r-cold-s-dual64-tagged-native-off-inventory-runner-root-binding-20261011-i/PINS.json"
#define AUTH_DIR "/var/lib/eshkol-off-inventory-20261011-i"
#define AUTH_FILE "AUTHORIZED-BINDING.sha256"
#define LEDGER_FILE "RUNNER-I-one-use.consumed"

static int protected_path(const char *path, int final_regular) {
    char part[4096];
    size_t n = strlen(path);
    if (n < 2 || n >= sizeof(part) || path[0] != '/') return -1;
    struct stat st;
    if (lstat("/", &st) || st.st_uid || (st.st_mode & 022) || !S_ISDIR(st.st_mode)) return -1;
    size_t used = 0;
    for (size_t i = 1; i <= n; ++i) {
        if (path[i] != '/' && path[i] != '\0') continue;
        if (i == 1 || (i < n && path[i + 1] == '/') ||
            (i - used == 2 && path[used + 1] == '.') ||
            (i - used == 3 && path[used + 1] == '.' && path[used + 2] == '.')) return -1;
        memcpy(part, path, i);
        part[i] = '\0';
        if (lstat(part, &st) || st.st_uid || (st.st_mode & 022)) return -1;
        if (i == n ? (final_regular ? !S_ISREG(st.st_mode) : !S_ISDIR(st.st_mode))
                   : !S_ISDIR(st.st_mode)) return -1;
        used = i;
    }
    return 0;
}

static int absent_zip(const char *path) {
    struct stat st;
    if (!lstat(path, &st) || errno != ENOENT) return -1;
    char parent[4096];
    size_t n = strlen(path);
    if (n >= sizeof(parent)) return -1;
    memcpy(parent, path, n + 1);
    char *slash = strrchr(parent, '/');
    if (!slash || slash == parent) return -1;
    *slash = '\0';
    return protected_path(parent, 0);
}

static int protected_self(void) {
    char path[4096];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0 || n >= (ssize_t)sizeof(path) - 1) return -1;
    path[n] = '\0';
    struct stat st;
    if (protected_path(path, 1) || lstat(path, &st) ||
        st.st_uid || !(st.st_mode & S_ISUID) || (st.st_mode & 022)) return -1;
    return 0;
}

static int digest_file(const char *path, char hex[65]) {
    int fd = -1, alg = -1, op = -1, ok = -1;
    unsigned char buf[65536], digest[32];
    struct stat st;
    fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0 || fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_nlink != 1 || st.st_size < 0) goto done;
    alg = socket(AF_ALG, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (alg < 0) goto done;
    struct sockaddr_alg sa = {.salg_family = AF_ALG, .salg_type = "hash", .salg_name = "sha256"};
    if (bind(alg, (struct sockaddr *)&sa, sizeof(sa))) goto done;
    op = accept4(alg, NULL, NULL, SOCK_CLOEXEC);
    if (op < 0) goto done;
    off_t remaining = st.st_size;
    if (!remaining && send(op, "", 0, 0) < 0) goto done;
    while (remaining) {
        size_t want = remaining < (off_t)sizeof(buf) ? (size_t)remaining : sizeof(buf);
        ssize_t got = read(fd, buf, want);
        if (got <= 0 || send(op, buf, (size_t)got, remaining > got ? MSG_MORE : 0) != got) goto done;
        remaining -= got;
    }
    if (read(op, digest, sizeof(digest)) != (ssize_t)sizeof(digest)) goto done;
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(digest); ++i) {
        hex[2*i] = digits[digest[i] >> 4];
        hex[2*i+1] = digits[digest[i] & 15];
    }
    hex[64] = '\0';
    ok = 0;
done:
    if (op >= 0) close(op);
    if (alg >= 0) close(alg);
    if (fd >= 0) close(fd);
    return ok;
}

static int exact_digest(const char *path, const char *expected) {
    char actual[65];
    return digest_file(path, actual) || strcmp(actual, expected) ? -1 : 0;
}

static int parse_authorization(int dirfd, char digest[65]) {
    int fd = openat(dirfd, AUTH_FILE, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    struct stat st;
    char line[66];
    if (fd < 0 || fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_uid ||
        (st.st_mode & 077) || st.st_nlink != 1) { if (fd >= 0) close(fd); return -1; }
    ssize_t n = read(fd, line, sizeof(line));
    close(fd);
    if (n != 65 || line[64] != '\n') return -1;
    for (int i = 0; i < 64; ++i)
        if (!((line[i] >= '0' && line[i] <= '9') || (line[i] >= 'a' && line[i] <= 'f'))) return -1;
    memcpy(digest, line, 64);
    digest[64] = '\0';
    return 0;
}

static int claim_ledger(int dirfd) {
    int fd = openat(dirfd, LEDGER_FILE, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return -1;
    if (fsync(fd) || fsync(dirfd)) { close(fd); return -1; }
    return fd; /* Any later failure leaves the claim consumed. */
}

static int process_starttime(char out[32]) {
    char text[4096];
    int fd = open("/proc/self/stat", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    ssize_t n = read(fd, text, sizeof(text) - 1);
    close(fd);
    if (n <= 0 || n >= (ssize_t)sizeof(text) - 1) return -1;
    text[n] = '\0';
    char *tail = strrchr(text, ')');
    if (!tail || tail[1] != ' ') return -1;
    unsigned field = 3;
    char *save = NULL;
    for (char *tok = strtok_r(tail + 2, " \n", &save); tok;
         tok = strtok_r(NULL, " \n", &save), ++field) {
        if (field == 22) {
            if (strlen(tok) >= 32) return -1;
            strcpy(out, tok);
            return 0;
        }
    }
    return -1;
}

static int make_owned_child(int basefd, const char *name, uid_t uid, gid_t gid,
                            unsigned long long *dev, unsigned long long *ino) {
    if (!*name || strchr(name, '/')) return -1;
    if (mkdirat(basefd, name, 0700)) return -1;
    int outfd = openat(basefd, name,
                       O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (outfd < 0) return -1;
    struct stat output;
    int ok = !fchown(outfd, uid, gid) && !fchmod(outfd, 0700) &&
             !fstat(outfd, &output) && S_ISDIR(output.st_mode) &&
             output.st_uid == uid && output.st_gid == gid &&
             (output.st_mode & 0777) == 0700 && !fsync(outfd) && !fsync(basefd);
    if (ok) { *dev = (unsigned long long)output.st_dev; *ino = (unsigned long long)output.st_ino; }
    close(outfd);
    return ok ? 0 : -1;
}

static int prepare_output(const char *path, uid_t uid, gid_t gid,
                          unsigned long long *dev, unsigned long long *ino) {
    const char *prefix = BASE "/g3r-cold-s-dual64-tagged-native-off-inventory-run-";
    size_t n = strlen(prefix);
    if (strlen(path) <= n || strncmp(path, prefix, n) ||
        strchr(path + n, '/')) return -1;
    int basefd = open(BASE, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (basefd < 0) return -1;
    struct stat base;
    if (fstat(basefd, &base) || base.st_uid || (base.st_mode & 022) ||
        !S_ISDIR(base.st_mode)) { close(basefd); return -1; }
    int result = make_owned_child(basefd, path + strlen(BASE) + 1, uid, gid, dev, ino);
    close(basefd);
    return result;
}

static int sealed_handoff(const char *binding, uid_t uid, pid_t pid,
                          unsigned long long dev, unsigned long long ino) {
    int fd = memfd_create("eshkol-off-inventory-handoff", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0 || fchmod(fd, 0600)) { if (fd >= 0) close(fd); return -1; }
    unsigned char nonce[16];
    char starttime[32];
    if (getrandom(nonce, sizeof(nonce), 0) != (ssize_t)sizeof(nonce) ||
        process_starttime(starttime)) { close(fd); return -1; }
    char nonce_hex[33], record[512];
    static const char digits[] = "0123456789abcdef";
    for (unsigned i = 0; i < sizeof(nonce); ++i) {
        nonce_hex[2*i] = digits[nonce[i] >> 4];
        nonce_hex[2*i+1] = digits[nonce[i] & 15];
    }
    nonce_hex[32] = '\0';
    int n = snprintf(record, sizeof(record),
                     "eshkol-off-inventory-handoff-v2\n%s\n%u\n%ld\n%s\n%llu\n%llu\n%s\n",
                     binding, (unsigned)uid, (long)pid, starttime, dev, ino, nonce_hex);
    if (n <= 0 || n >= (int)sizeof(record) || write(fd, record, (size_t)n) != n ||
        fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)) {
        close(fd);
        return -1;
    }
    return fd;
}

#ifndef LAUNCHER_TEST
static void hold(const char *message) {
    fprintf(stderr, "HOLD: %s\n", message);
    _exit(2);
}

int main(int argc, char **argv) {
    if (argc != 2 || getuid() == 0 || geteuid() != 0) hold("setuid-root launch contract absent");
    if (protected_self()) hold("launcher installation is not protected");
    if (absent_zip("/usr/lib/python314.zip")) hold("stdlib zip present or parent untrusted");
    if (protected_path(I_RUNNER, 1)) hold("I source path is not protected from same UID");
    if (exact_digest(I_RUNNER, I_RUNNER_SHA)) hold("I source digest drift");
    for (unsigned i = 0; i < PIN_COUNT; ++i)
        if (exact_digest(PINS[i].path, PINS[i].sha256)) hold("accepted source or input seal drift");
    if (protected_path(AUTH_DIR, 0)) hold("root authorization directory unavailable");
    int dirfd = open(AUTH_DIR, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    struct stat dir;
    if (dirfd < 0 || fstat(dirfd, &dir) || dir.st_uid || (dir.st_mode & 0777) != 0700)
        hold("root authorization directory not private");
    char binding[65];
    if (parse_authorization(dirfd, binding) || exact_digest(I_BINDING, binding))
        hold("authorized external binding absent or drifted");
    int ledgerfd = claim_ledger(dirfd);
    if (ledgerfd < 0) hold("one-use ledger already consumed or failed");
    if (write(ledgerfd, binding, 64) != 64 || write(ledgerfd, "\n", 1) != 1 ||
        fsync(ledgerfd)) hold("one-use ledger write failed; attempt consumed");
    close(ledgerfd);
    close(dirfd);
    uid_t caller_uid = getuid();
    gid_t caller_gid = getgid();
    unsigned long long output_dev, output_ino;
    if (prepare_output(argv[1], caller_uid, caller_gid, &output_dev, &output_ino))
        hold("launcher output or handoff failed; attempt consumed");
    int handofffd = sealed_handoff(binding, caller_uid, getpid(), output_dev, output_ino);
    if (handofffd < 0) hold("sealed handoff failed; attempt consumed");
    if (handofffd != 3 && dup2(handofffd, 3) != 3)
        hold("handoff descriptor transfer failed; attempt consumed");
    if (handofffd != 3) close(handofffd);
    if (fcntl(3, F_SETFD, 0)) hold("handoff descriptor inheritance failed; attempt consumed");
    if (setgroups(0, NULL) || setresgid(caller_gid, caller_gid, caller_gid) ||
        setresuid(caller_uid, caller_uid, caller_uid)) hold("privilege drop failed; attempt consumed");
    char *const args[] = {"/usr/bin/python3", "-I", "-S", I_RUNNER, "--run",
                          I_RELEASE, argv[1], binding, "3", NULL};
    char *const env[] = {"LC_ALL=C", "LANG=C", "PATH=/usr/bin:/bin",
                         "HOME=/nonexistent", "PYTHONNOUSERSITE=1", NULL};
    execve(args[0], args, env);
    hold("Python exec failed; attempt consumed");
}
#endif
