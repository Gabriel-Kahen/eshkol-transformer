"""Exact actual-witness record check only; synthetic tests are not execution."""
from pathlib import Path
import sys

EXPECTED = ''.join(
    f'CPU4-BINDING-WITNESS case={case} pushes=1 condition5-reads={reads} '
    f'failure-create-attempts=0 pending={pending} retry-blocked={pending} '
    f'retry-create={created} prepare={created} commit={created} dead-check={created}\n'
    for case, reads, pending, created in ((1,1,1,0),(2,1,0,1),(3,2,0,1)))
EXPECTED += 'CPU4-BINDING-WITNESS-PASS relocation=UNPROVEN retained-bytes=UNMEASURED\n'


def check(text):
    if text != EXPECTED:
        raise ValueError('missing, reordered, extra or incorrect causal witness record')
    return True


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('check ACTUAL_WITNESS_STDOUT')
    check(Path(sys.argv[1]).read_bytes().decode('utf-8'))
    print('CPU4-BINDING-WITNESS-CHECK-PASS scope=handler-cause-and-native-lifecycle relocation=UNPROVEN')
