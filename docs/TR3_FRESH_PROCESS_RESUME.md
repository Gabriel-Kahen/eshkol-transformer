# TR3 private fresh-process checkpoint continuation

`scripts/test-tr3-fresh-process-resume.sh` compiles the accepted private TR3/C2
source composition under the pinned f31/LLVM21 network-disabled gate. It runs
one executable in nine distinct OS processes: uninterrupted `K+R`, checkpoint
producer `K`, and fresh receiver `LOAD+R` for each accumulation count
`A=1,2,3`, with `K=1` and `R=3`.

The producer emits an authentic versioned, checksummed C2 checkpoint. The
receiver starts from an independently constructed trainer and loads it through
the accepted C2 LOAD and TR3 joint restore. The gate compares canonical C2
bytes at K, immediately after restore, and after each suffix update: five
comparisons per profile. Those bytes encode all 14 parameters, 28 O2 moments,
RNG words, current and epoch-start D2 cursors, X1/tokenizer identity, and
trainer/O2 token, epoch, and update controls. Each process also checks its
reachable detached 42-tensor image and live O2/trainer counter agreement.
The `A=1` and `A=3` uninterrupted runs verify unequal active-token counts;
all resumed suffixes cross finite D2 EOS.

Each fresh receiver rejects a checksum-corrupted K file and an authentic X1
run-seed mismatch without changing the receiver image. After successful load,
an injected `precommit` failure before the first parameter write preserves
the exact K image and controls; the same trainer then retries all three
updates and produces the same final C2 bytes as uninterrupted execution.

This remains a test-only private CPU f32 source composition. The current step
leaf has no public next-step metrics or effective-learning-rate observation,
so those two TR3-C §13 comparisons remain pending. Public trainer packaging,
full failure-prefix/environment acceptance, and other model profiles remain
separate gates.
