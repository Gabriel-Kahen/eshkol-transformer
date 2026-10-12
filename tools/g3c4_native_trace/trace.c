#define _GNU_SOURCE
#include "sha256.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/sched.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

enum { MAX_TASKS=4096, MAX_BLOB=4*1024*1024, MAX_LOG=32*1024*1024 };
#if !defined(__x86_64__) || !defined(SYS_clone3)
#error "This prototype's seccomp filter is x86_64-only and needs SYS_clone3"
#endif
typedef struct { pid_t pid, parent; bool live, initial_stop; } task;
static task tasks[MAX_TASKS];
static size_t live_count, log_size;
static int log_fd=-1, root_status=86;
static pid_t root_pid;
static unsigned spawn_count, exec_count, exit_count, merge_count, fault_after_exec;

static void fail(const char *what) {
    dprintf(STDERR_FILENO,"native trace HOLD: %s: %s\n",what,strerror(errno));
    /* EXITKILL is set before any tracee is resumed. The outer cgroup is a
       second kill boundary if the supervisor itself is terminated. */
    _exit(86);
}
static void log_bytes(const void *data, size_t size) {
    if (size>MAX_LOG-log_size) { errno=EFBIG; fail("event log cap"); }
    const unsigned char *p=data;
    while (size) {
        ssize_t n=write(log_fd,p,size);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) fail("write event log");
        p+=n;size-=(size_t)n;log_size+=(size_t)n;
    }
}
static void logf_checked(const char *fmt, ...) {
    char buffer[512];va_list args;va_start(args,fmt);
    int n=vsnprintf(buffer,sizeof(buffer),fmt,args);va_end(args);
    if (n<0 || (size_t)n>=sizeof(buffer)) { errno=EOVERFLOW;fail("format event"); }
    log_bytes(buffer,(size_t)n);
}
static void log_hex(const unsigned char *data,size_t size) {
    static const char hex[]="0123456789abcdef";
    char buffer[4096];size_t used=0;
    for(size_t i=0;i<size;++i) {
        if (used+2>sizeof(buffer)) { log_bytes(buffer,used);used=0; }
        buffer[used++]=hex[data[i]>>4];buffer[used++]=hex[data[i]&15];
    }
    if (used) log_bytes(buffer,used);
}
static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC,&ts)!=0) fail("monotonic clock");
    return (uint64_t)ts.tv_sec*1000000000u+(uint64_t)ts.tv_nsec;
}
static task *lookup(pid_t pid) {
    for(size_t i=0;i<MAX_TASKS;++i) if(tasks[i].live && tasks[i].pid==pid) return &tasks[i];
    return NULL;
}
static task *add_task(pid_t pid,pid_t parent,bool initial_stop) {
    if (pid<=0 || lookup(pid)) { errno=EINVAL;fail("duplicate task"); }
    for(size_t i=0;i<MAX_TASKS;++i) if(!tasks[i].live) {
        tasks[i]=(task){.pid=pid,.parent=parent,.live=true,.initial_stop=initial_stop};
        ++live_count;return &tasks[i];
    }
    errno=E2BIG;fail("task cap");
    __builtin_unreachable();
}
static unsigned char *read_proc(pid_t pid,const char *name,size_t *length) {
    char path[96];snprintf(path,sizeof(path),"/proc/%ld/%s",(long)pid,name);
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if(fd<0) fail("open proc file");
    unsigned char *buffer=malloc(MAX_BLOB+1);
    if(!buffer) fail("allocate proc input");
    size_t used=0;
    for(;;) {
        if(used==MAX_BLOB) { errno=EFBIG;fail("proc input cap"); }
        ssize_t n=read(fd,buffer+used,MAX_BLOB-used);
        if(n<0 && errno==EINTR) continue;
        if(n<0) fail("read proc file");
        if(n==0) break;
        used+=(size_t)n;
    }
    if(close(fd)!=0) fail("close proc file");
    buffer[used]=0;*length=used;return buffer;
}
static size_t proc_link(pid_t pid,const char *name,char out[PATH_MAX]) {
    char path[96];snprintf(path,sizeof(path),"/proc/%ld/%s",(long)pid,name);
    ssize_t n=readlink(path,out,PATH_MAX);
    if(n<0 || n==PATH_MAX) { errno=n<0?errno:ENAMETOOLONG;fail("read proc link"); }
    return (size_t)n;
}
static unsigned long long start_ticks(pid_t pid) {
    size_t length;unsigned char *stat=read_proc(pid,"stat",&length);
    unsigned char *end=(unsigned char*)strrchr((char*)stat,')');
    if(!end || end+2>=stat+length) { errno=EINVAL;fail("parse proc stat"); }
    char *save=NULL,*word=strtok_r((char*)end+2," ",&save);
    for(unsigned i=3;i<22 && word;++i) word=strtok_r(NULL," ",&save);
    if(!word) { errno=EINVAL;fail("missing proc starttime"); }
    errno=0;unsigned long long value=strtoull(word,NULL,10);
    if(errno) fail("parse proc starttime");
    free(stat);return value;
}
static void record_exec(task *item) {
    size_t argv_length,env_length;
    unsigned char *argv=read_proc(item->pid,"cmdline",&argv_length);
    unsigned char *env=read_proc(item->pid,"environ",&env_length);
    char exe[PATH_MAX],cwd[PATH_MAX],proc_exe[96];
    size_t exe_length=proc_link(item->pid,"exe",exe);
    size_t cwd_length=proc_link(item->pid,"cwd",cwd);
    snprintf(proc_exe,sizeof(proc_exe),"/proc/%ld/exe",(long)item->pid);
    struct stat identity;
    if(stat(proc_exe,&identity)!=0) fail("stat executable identity");
    if(argv_length==0 || argv[argv_length-1]!=0) { errno=EINVAL;fail("empty or malformed argv"); }
    sha256_state hash;uint8_t digest[32];sha256_init(&hash);
    sha256_update(&hash,env,env_length);sha256_final(&hash,digest);
    logf_checked("exec\t%llu\t%ld\t%ld\t%llu\t%llu\t%llu\t",
                 (unsigned long long)now_ns(),(long)item->pid,(long)item->parent,
                 start_ticks(item->pid),(unsigned long long)identity.st_dev,
                 (unsigned long long)identity.st_ino);
    log_hex(argv,argv_length);log_bytes("\t",1);
    log_hex((const unsigned char*)exe,exe_length);log_bytes("\t",1);
    log_hex((const unsigned char*)cwd,cwd_length);log_bytes("\t",1);
    log_hex(digest,32);log_bytes("\n",1);
    free(argv);free(env);
    ++exec_count;
    if(fault_after_exec && exec_count>=fault_after_exec) {
        errno=ECANCELED;fail("injected supervisor failure");
    }
}
static void resume(pid_t pid,int signal_number) {
    if(ptrace(PTRACE_CONT,pid,0,signal_number)<0) fail("resume tracee");
}
static int install_child_filter(void) {
    /* An untraced clone would silently escape the process inventory. Trace
       any such attempt before execution. clone3's flags live behind a pointer;
       deny clone3 with ENOSYS so libc can use traceable legacy clone. */
    struct sock_filter code[]={
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,arch)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AUDIT_ARCH_X86_64,0,10),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
        BPF_JUMP(BPF_JMP|BPF_JSET|BPF_K,0x40000000u,5,0),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_clone3,6,0),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,SYS_clone,0,4),
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[0])),
        BPF_STMT(BPF_ALU|BPF_AND|BPF_K,CLONE_UNTRACED),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,0,1,0),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_TRACE),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|ENOSYS),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
    };
    struct sock_fprog program={.len=(unsigned short)(sizeof(code)/sizeof(code[0])),
                               .filter=code};
    if(prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)!=0) return -1;
    return prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&program);
}
static void trace_loop(void) {
    while(live_count) {
        int status=0;pid_t pid=waitpid(-1,&status,__WALL);
        if(pid<0 && errno==EINTR) continue;
        if(pid<0) fail("wait traced task");
        task *item=lookup(pid);
        if(!item) { errno=ESRCH;fail("unattributed traced task"); }
        if(WIFEXITED(status)||WIFSIGNALED(status)) {
            int signal_number=WIFSIGNALED(status)?WTERMSIG(status):0;
            int value=signal_number?128+signal_number:WEXITSTATUS(status);
            logf_checked("exit\t%llu\t%ld\t%d\t%d\n",
                         (unsigned long long)now_ns(),(long)pid,signal_number,value);
            ++exit_count;
            if(pid==root_pid) root_status=value;
            item->live=false;--live_count;continue;
        }
        if(!WIFSTOPPED(status)) { errno=EINVAL;fail("unexpected wait state"); }
        unsigned event=(unsigned)status>>16;
        if(event==PTRACE_EVENT_SECCOMP) {
            errno=EPERM;fail("forbidden clone route");
        }
        if(event==PTRACE_EVENT_FORK || event==PTRACE_EVENT_VFORK ||
           event==PTRACE_EVENT_CLONE) {
            unsigned long child=0;
            if(ptrace(PTRACE_GETEVENTMSG,pid,0,&child)<0 || child==0 || child>INT_MAX)
                fail("read child event");
            add_task((pid_t)child,pid,true);
            const char *kind=event==PTRACE_EVENT_FORK?"fork":
                             event==PTRACE_EVENT_VFORK?"vfork":"clone";
            logf_checked("spawn\t%llu\t%ld\t%ld\t%s\n",
                         (unsigned long long)now_ns(),(long)pid,(long)child,kind);
            ++spawn_count;
            resume(pid,0);continue;
        }
        if(event==PTRACE_EVENT_EXEC) {
            unsigned long former=0;
            if(ptrace(PTRACE_GETEVENTMSG,pid,0,&former)<0 || former==0 ||
               former>INT_MAX) fail("read exec former tid");
            if((pid_t)former!=pid) {
                task *old=lookup((pid_t)former);
                if(!old || old==item) { errno=EINVAL;fail("unmapped thread exec"); }
                old->live=false;--live_count;++merge_count;
                logf_checked("merge\t%llu\t%ld\t%ld\n",
                             (unsigned long long)now_ns(),(long)former,(long)pid);
            }
            record_exec(item);resume(pid,0);continue;
        }
        if(event==PTRACE_EVENT_EXIT) { resume(pid,0);continue; }
        if(event!=0) { errno=EINVAL;fail("unknown ptrace event"); }
        int signal_number=WSTOPSIG(status);
        if(item->initial_stop && signal_number==SIGSTOP) {
            item->initial_stop=false;resume(pid,0);
        } else {
            item->initial_stop=false;resume(pid,signal_number);
        }
    }
}
int main(int argc,char **argv) {
    const char *log_path=NULL;int command_index=0;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--log") && i+1<argc) log_path=argv[++i];
        else if(!strcmp(argv[i],"--fault-after-exec") && i+1<argc) {
            char *end=NULL;errno=0;unsigned long value=strtoul(argv[++i],&end,10);
            if(errno || !end || *end || value==0 || value>UINT_MAX) return 86;
            fault_after_exec=(unsigned)value;
        } else if(!strcmp(argv[i],"--") && i+1<argc) { command_index=i+1;break; }
        else return 86;
    }
    if(!log_path || !command_index) return 86;
    log_fd=open(log_path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(log_fd<0) fail("create private event log");
    static const char header[]="g3c4-native-exec-v1\n";
    log_bytes(header,sizeof(header)-1);
    root_pid=fork();
    if(root_pid<0) fail("fork root tracee");
    if(root_pid==0) {
        close(log_fd);
        if(ptrace(PTRACE_TRACEME,0,0,0)<0) _exit(86);
        raise(SIGSTOP);
        if(install_child_filter()!=0) _exit(86);
        execvp(argv[command_index],&argv[command_index]);
        _exit(127);
    }
    add_task(root_pid,0,false);
    int status=0;
    if(waitpid(root_pid,&status,WUNTRACED)!=root_pid || !WIFSTOPPED(status) ||
       WSTOPSIG(status)!=SIGSTOP) {
        kill(root_pid,SIGKILL);waitpid(root_pid,NULL,0);errno=EINVAL;fail("initial trace stop");
    }
    long options=PTRACE_O_TRACEFORK|PTRACE_O_TRACEVFORK|PTRACE_O_TRACECLONE|
                 PTRACE_O_TRACEEXEC|PTRACE_O_TRACEEXIT|PTRACE_O_TRACESECCOMP|
                 PTRACE_O_EXITKILL;
    if(ptrace(PTRACE_SETOPTIONS,root_pid,0,options)<0) {
        kill(root_pid,SIGKILL);waitpid(root_pid,NULL,0);fail("enable ptrace events");
    }
    resume(root_pid,0);trace_loop();
    if(root_status==0) {
        if((unsigned long long)spawn_count+1 !=
           (unsigned long long)exit_count+merge_count) {
            errno=EINVAL;fail("task ledger mismatch");
        }
        logf_checked("complete\t%llu\t%ld\t%d\t%u\t%u\t%u\t%u\n",
                     (unsigned long long)now_ns(),(long)root_pid,root_status,
                     spawn_count,exec_count,exit_count,merge_count);
    }
    if(fsync(log_fd)!=0 || close(log_fd)!=0) fail("seal event log");
    return root_status;
}
