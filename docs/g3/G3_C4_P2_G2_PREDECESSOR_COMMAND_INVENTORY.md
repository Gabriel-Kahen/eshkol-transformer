# P2/G2 terminal predecessor command inventory

Status: source-derived inventory at `309da811` (tree `99ef18e1`); no
predecessor runner changed here. Line numbers below refer to that tree.
The terminal runner records the **prefix** and **clone** predecessor Bash
processes as two aggregate commands with argv, cwd, numeric exit and raw
stdout/stderr. The eight scripts below contain further commands without
equivalent individual receipts. A top-level zero exit establishes their
script assertions, not complete per-command evidence.

## Actual transitive call graph

```text
test-g3c4-p2g2-terminal-composer.sh
├─ test-g3c4-p2g2-t1-prefix-coordinator.sh
│  ├─ test-g3c4-p2g1-pending.sh
│  └─ test-g3c4-p2g2-protected-scope.sh
│     ├─ test-g3c4-p2g1-pending.sh   (second invocation)
│     └─ test-g3c4-p2g2-carrier-bridge.sh
│        └─ test-g3c4-p2g2-prefix-commit.sh
│           └─ test-g3c4-p2g2-first-frame.sh
└─ test-g3c4-p2g2-terminal-ids-clone.sh
   └─ test-g3c4-p2g2-terminal.sh
```

The two root calls start at terminal-runner lines 31 and 34. All eight scripts
call `verify_toolchain` and/or `resolve_provenance_compilers` before or
around their evidence setup; these nested setup commands have no
individual receipt. Every `-o` below names a compiled output, but the
compiler/linker argv, cwd, numeric exit and separate raw streams are not
sealed together. `set -e` propagates unexpected failure to the parent;
expected negative compiles run in `if` and their actual nonzero codes are
not retained. A `grep`, `cmp`, oracle or runtime check may establish a
predicate while still lacking its own numeric exit receipt.

## Command sites and retained streams

| Script (under `scripts/`) | Child calls and checker/oracle sites | Compiler, linker and run sites | Present evidence and missing receipt |
| --- | --- | --- | --- |
| `test-g3c4-p2g2-t1-prefix-coordinator.sh` | static 15; Q0 17; pending 19; protected 21; prefix oracle 150; `git diff --check` 166; closure hash 167 | 17 native C/C++ object commands at 43–88 (9-name provider loop at 72), three `nm`/`grep` symbol predicates 89–94, `ar` 95; generated `cxx-wrap` `exec` 101; Eshkol compile/link 128–136 and fixture run 138–142 per fixture/mode; repeat 152–156; parity 160–165 | Eshkol compile/run streams copied from temporary build to named evidence at 143–144; static/Q0/oracle and aggregate child stdout have named files. Native compile, archive, wrapper clang++, symbol checks, repeat and comparison commands have no per-command raw+numeric receipt. Successful trap removes temporary objects and wrapper. |
| `test-g3c4-p2g1-pending.sh` | static 13; per-mode P2/G1 oracle 91; `git diff --check` 205; no child script | owner on/off C compiles 52–58; `nm`/`cmp` 60–63; intentionally failing isolated C compile 64–70; native compile/run 81–90; optional feature-off compile/run 95–103; per-mode native provider compiles 108–145 and archive 148; generated C++ wrapper `exec` 153 and Eshkol compile/link/run 164–179; repeat 183–194 | Fixture stdout/stderr and oracle stdout are copied to named evidence at 196–203. `if` proves isolated compile failed but retains only stderr, not numeric exit. Provider compiles, archive, wrapper clang++, repeats and comparisons lack individual receipts; temporary build is removed on successful exit. |
| `test-g3c4-p2g2-protected-scope.sh` | static 15; Q0 17; pending 19; carrier bridge 21; `git diff --check` 148; closure hash 149 | native object compiles 43–80, archive 83; generated C++ wrapper `exec` 89; Eshkol compile/link 112–120 and fixture run 122–126 per fixture/mode; repeat 134–140; parity 142–147 | Fixture compile/run streams copied to named evidence at 127–128. Native objects, archive, wrapper clang++, repeat and comparison commands lack individual numeric/raw receipts; successful trap removes temporary build. |
| `test-g3c4-p2g2-carrier-bridge.sh` | static 11; prefix commit 13; P2/G1 oracle 92 | feature on/off C compiles and `nm`/`cmp` 53–63; intentionally failing isolated compile 64–71; normal/sanitize C compile+link and fixture run 82–90; repeat/parity 94–97 | Named normal/sanitize **run** files exist; compiles at 82 have no raw redirection. Isolated negative compile has stderr but no exact exit; symbol predicates, repeat and comparison have no individual receipts. |
| `test-g3c4-p2g2-prefix-commit.sh` | static 11; first frame 13; P2/G1 oracle 91 | feature on/off C compiles and `nm`/`cmp` 51–61; intentionally failing isolated compile 62–69; normal/sanitize C compile+link and fixture run 80–89; repeat/parity 93–96 | Named **run** files exist; compiles at 80 have no raw redirection. Isolated negative compile has stderr only; symbol predicates, repeat and comparisons lack individual numeric/raw receipts. |
| `test-g3c4-p2g2-first-frame.sh` | static 12; P2/G1 oracle 103 | on/off C compiles and `nm`/`cmp` 57–63; intentionally failing invalid compile 64–71; feature-off C compile+link/run 75–83; normal/sanitize C compile+link/run 93–101; repeat/parity 105–108 | Named fixture **run** files exist; compiles at 75 and 93 have no raw redirection. Negative compile has stderr only; symbol predicates, repeat and comparisons have no individual numeric/raw receipts. |
| `test-g3c4-p2g2-terminal-ids-clone.sh` | static 11; native terminal predecessor 115 | owner on/off C compile 42–44; `nm`/`awk`/`sort`/`comm`/`cmp` 45–53; intentionally failing isolated C compile 54–60; normal/sanitize C compile+link 87–91 and run 92–96; repeat/parity 98–101; coexistence compile+link/run 103–113 | Named compile/run files for main/coexist and isolated stderr exist. Expected isolated failure is not a numeric receipt. Owner compiles, symbol predicates, repeat and child terminal commands lack individual raw+exit receipts. |
| `test-g3c4-p2g2-terminal.sh` | static 11; second-output oracle 113 and 156 | owner/A2 on/off C compiles 43–54; `nm`/`awk`/`sort`/`comm`/`cmp` 49–62; two intentionally failing isolated compiles 63–76; normal/sanitize C compile+link/run 103–116; repeat/parity 118–121; coexistence C compile+link/run 124–134; three predecessor C compile+link/run controls 144–154 | Named main/coexist/predecessor compile and run files plus negative stderr exist. Actual negative exit numbers, symbol pipeline exits, repeat and oracle raw streams/exit are not complete receipts. |

The table is a grouped source-site inventory, not an exact executable
invocation manifest or a single execution count: loops
run in two modes, the pending runner is invoked twice, and feature branches
affect some tests. Generated C++ wrappers in prefix, pending and protected
`exec` the linker; their parent Eshkol stdout/stderr does not give the
linker's exact argv/cwd/exit/raw streams. The terminal runner's own C++
wrapper has a separate nested receipt, but that does not instrument these
predecessors.

## Completion gate

Keep complete per-command predecessor evidence **HOLD**. A separately
reviewed slice must instrument every listed command, including actual C++
wrapper invocations and expected failing compiles, preserve independent
raw streams/numeric exits on abnormal termination, and enforce a source
derived success/postmortem receipt manifest. The outer one-use launcher
must then prove no missing child/pipe/writer/container evidence before any
physical seal. No compiled run or claim was made for this inventory.
