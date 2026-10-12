#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static void *thread_work(void *unused) {
    (void)unused;
    return NULL;
}
static void *thread_exec(void *unused) {
    (void)unused;
    char *args[] = {"/bin/true", NULL};
    execve(args[0], args, environ);
    return NULL;
}

static int touch_child(void *path) {
    char *args[] = {"/bin/touch", path, NULL};
    execve(args[0], args, environ);
    _exit(127);
}

static int await(pid_t child) {
    int status;
    if (waitpid(child, &status, 0) != child) return 1;
    return !WIFEXITED(status) || WEXITSTATUS(status) != 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    if (strcmp(argv[1], "faultchild") == 0) {
        if (argc != 3) return 2;
        pid_t child = fork();
        if (child < 0) return 11;
        if (child == 0) {
            char *args[] = {"/bin/touch", argv[2], NULL};
            execve(args[0], args, environ);
            _exit(127);
        }
        return await(child) ? 12 : 0;
    }
    if (strcmp(argv[1], "untraced") == 0) {
        if (argc != 3) return 2;
        void *stack = malloc(65536);
        if (!stack) return 13;
        pid_t child = clone(touch_child, (char *)stack + 65536,
                            CLONE_UNTRACED | SIGCHLD, argv[2]);
        if (child > 0) (void)await(child);
        free(stack);
        /* The target ignores a denied clone. The supervisor still must HOLD. */
        return 0;
    }
    if (argc != 2) return 2;
    if (strcmp(argv[1], "fast") == 0) {
        pid_t children[32];
        for (size_t i = 0; i < 32; ++i) {
            pid_t child = fork();
            if (child < 0) return 3;
            if (child == 0) {
                char *args[] = {"/bin/true", NULL};
                execve(args[0], args, environ);
                _exit(127);
            }
            children[i] = child;
        }
        for (size_t i = 0; i < 32; ++i) if (await(children[i])) return 4;
        return 0;
    }
    if (strcmp(argv[1], "spawn") == 0) {
        pid_t child;
        char *args[] = {"/bin/true", NULL};
        if (posix_spawn(&child, args[0], NULL, NULL, args, environ) != 0) return 5;
        return await(child) ? 6 : 0;
    }
    if (strcmp(argv[1], "system") == 0) {
        int status = system("/bin/true");
        return status == 0 ? 0 : 7;
    }
    if (strcmp(argv[1], "thread") == 0) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, thread_work, NULL) != 0) return 9;
        return pthread_join(thread, NULL) == 0 ? 0 : 10;
    }
    if (strcmp(argv[1], "threadexec") == 0) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, thread_exec, NULL) != 0) return 14;
        return pthread_join(thread, NULL) == 0 ? 15 : 16;
    }
    return 8;
}
