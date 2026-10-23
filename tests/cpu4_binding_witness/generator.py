"""Extract one immutable historical constructor, never a historical union."""
import hashlib
from pathlib import Path
import subprocess
import sys
from tests.cpu4_trainer_control.test_source_contract import parse

BASE = '79ea814d994417d646835f5d09f26459869bf4a7'
SOURCE = 'native/cpu4_model_binding_extension.esk'
ORIGINAL = 'cpu4-model-binding-create-internal'
RENAMED = 'cpu4bw-baseline-create'


def extract(source):
    start = source.index('(define (' + ORIGINAL + ' model resolved)')
    depth, quoted, escaped, comment = 0, False, False, False
    for index in range(start, len(source)):
        char = source[index]
        if comment:
            if char == '\n': comment = False
            continue
        if quoted:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
            continue
        if char == ';': comment = True
        elif char == '"': quoted = True
        elif char == '(': depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return source[start:index+1]
    raise ValueError('unterminated baseline constructor')


def generated(source):
    original = extract(source)
    renamed = original.replace('(define (' + ORIGINAL, '(define (' + RENAMED, 1)
    before, after = parse(original), parse(renamed)
    assert len(before) == len(after) == 1
    assert after[0][1] == [RENAMED, 'model', 'resolved']
    after[0][1][0] = ORIGINAL
    assert before == after, 'baseline body must be exact after header rename'
    digest = hashlib.sha256(source.encode()).hexdigest()
    return f';;; Constructor only from {BASE}; source SHA256 {digest}.\n' + renamed + '\n'


def generate(repo, output):
    if output.exists() or output.is_symlink():
        raise ValueError('fresh output file required')
    source = subprocess.check_output(['git','show',BASE+':'+SOURCE],cwd=repo,text=True)
    output.write_text(generated(source))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('generator OUTPUT_BASELINE_SOURCE')
    generate(Path(__file__).resolve().parents[2], Path(sys.argv[1]))
