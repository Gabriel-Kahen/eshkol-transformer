/* SysV x86-64 test-only stack switch. r12 preserves the original caller stack. */
.text
.globl g3s_test_on_stack
.type g3s_test_on_stack,@function
g3s_test_on_stack:
 pushq %r12
 movq %rsp,%r12
 movq %rdi,%rsp
 andq $-16,%rsp
 movq %rdx,%rdi
 call *%rsi
 movq %r12,%rsp
 popq %r12
 ret
.size g3s_test_on_stack,.-g3s_test_on_stack
.section .note.GNU-stack,"",@progbits
