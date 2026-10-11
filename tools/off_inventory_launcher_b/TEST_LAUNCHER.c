#define LAUNCHER_TEST
#include "LAUNCHER.c"
#include <assert.h>
#include <sys/wait.h>

int main(void) {
    assert(protected_self() != 0); /* Synthetic build is deliberately not installed. */
    assert(absent_zip("/usr/lib/python314.zip") == 0);
    assert(protected_path("/usr/lib", 0) == 0);
    assert(protected_path("/usr/lib/../lib", 0) != 0);
    char root[] = "/tmp/eshkol-off-launcher-test-XXXXXX";
    assert(mkdtemp(root));
    char path[4096], target[4096], ledger[4096], output[4096];
    snprintf(path, sizeof(path), "%s/python314.zip", root);
    snprintf(target, sizeof(target), "%s/target", root);
    int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0600);
    assert(fd >= 0 && write(fd, "abc", 3) == 3);
    close(fd);
    assert(absent_zip(path) != 0);
    assert(protected_path(path, 1) != 0); /* Caller-writable code is never an install. */
    char digest[65];
    assert(digest_file(path, digest) == 0);
    assert(!strcmp(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    assert(exact_digest(path, digest) == 0);
    assert(exact_digest(path, "0000000000000000000000000000000000000000000000000000000000000000") != 0);
    fd = open(path, O_WRONLY | O_TRUNC);
    assert(fd >= 0);
    char block[1000];
    memset(block, 'x', sizeof(block));
    for (int i = 0; i < 100; ++i) assert(write(fd, block, sizeof(block)) == (ssize_t)sizeof(block));
    close(fd);
    assert(digest_file(path, digest) == 0);
    assert(!strcmp(digest, "d69e68988157833272305aaf21f453c800346e8a3640db6578e260215542e5d4"));
    assert(unlink(path) == 0 && symlink(target, path) == 0);
    assert(absent_zip(path) != 0); /* Dangling symlink is present. */
    assert(unlink(path) == 0 && absent_zip(path) != 0); /* Mutable parent. */
    assert(protected_path(root, 0) != 0);
    int dirfd = open(root, O_RDONLY | O_DIRECTORY);
    assert(dirfd >= 0);
    unsigned long long dev = 0, ino = 0;
    assert(make_owned_child(dirfd, "output", getuid(), getgid(), &dev, &ino) == 0);
    snprintf(output, sizeof(output), "%s/output", root);
    struct stat created;
    assert(lstat(output, &created) == 0 && S_ISDIR(created.st_mode));
    assert(created.st_uid == getuid() && created.st_gid == getgid() &&
           (created.st_mode & 0777) == 0700);
    assert(dev == (unsigned long long)created.st_dev && ino == (unsigned long long)created.st_ino);
    assert(make_owned_child(dirfd, "output", getuid(), getgid(), &dev, &ino) != 0);
    assert(make_owned_child(dirfd, "../escape", getuid(), getgid(), &dev, &ino) != 0);
    assert(prepare_output(root, getuid(), getgid(), &dev, &ino) != 0);
    int handoff = sealed_handoff(digest, getuid(), getpid(), dev, ino);
    assert(handoff >= 0);
    struct stat handoff_stat;
    assert(fstat(handoff, &handoff_stat) == 0 && handoff_stat.st_nlink == 0 &&
           (handoff_stat.st_mode & 0777) == 0600);
    int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    assert((fcntl(handoff, F_GET_SEALS) & seals) == seals);
    char handoff_text[512];
    ssize_t handoff_n = pread(handoff, handoff_text, sizeof(handoff_text) - 1, 0);
    assert(handoff_n > 0);
    handoff_text[handoff_n] = '\0';
    assert(strstr(handoff_text, "eshkol-off-inventory-handoff-v2\n") == handoff_text &&
           strstr(handoff_text, digest));
    close(handoff);
    int winners = 0;
    for (int i = 0; i < 8; ++i) {
        pid_t child = fork();
        assert(child >= 0);
        if (!child) { int claim = claim_ledger(dirfd); if (claim >= 0) close(claim);
                      _exit(claim >= 0 ? 0 : 1); }
    }
    for (int i = 0; i < 8; ++i) {
        int status;
        assert(wait(&status) > 0 && WIFEXITED(status));
        winners += WEXITSTATUS(status) == 0;
    }
    assert(winners == 1);
    assert(make_owned_child(dirfd, "output", getuid(), getgid(), &dev, &ino) != 0);
    assert(claim_ledger(dirfd) != 0); /* Failed output setup cannot restore the claim. */
    snprintf(ledger, sizeof(ledger), "%s/%s", root, LEDGER_FILE);
    assert(unlink(ledger) == 0);
    assert(symlink(target, ledger) == 0);
    assert(claim_ledger(dirfd) != 0);
    assert(access(target, F_OK) != 0);
    assert(unlink(ledger) == 0);
    assert(parse_authorization(dirfd, digest) != 0); /* No ROOT authorization. */
    close(dirfd);
    assert(rmdir(output) == 0);
    assert(rmdir(root) == 0);
    puts("synthetic zip, digest, protected path, output, and 8-way replay tests: PASS");
    return 0;
}
