"""Exact-source host Guile receipt; guest never substitutes a reader or installs one."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FILES=('native/e3_cpu4_private_root.esk','native/e3_cpu4_private_extension.esk',
       'native/e3_cpu4_boundary.esk','internal/e3/lib/e3_cpu4_d2_restore.esk',
       'tests/e3_cpu4_consumer/runtime.esk')
CODE=('tests/e3_cpu4_consumer/test_source.py','tests/e3_cpu4_consumer/host_reader.py')
EXPRESSION='(for-each (lambda (p) (call-with-input-file p (lambda (s) (let loop () (if (not (eof-object? (read s))) (loop)))))) (cdr (command-line)))'
ARGUMENTS=['--no-auto-compile','-c',EXPRESSION,*FILES]


def identity(root=ROOT):
    head,tree=subprocess.check_output(['git','rev-parse','HEAD','HEAD^{tree}'],cwd=root,text=True).splitlines()
    return {'schema':'cpu4-consumer-host-reader-v1','head':head,'tree':tree,
            'sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES},
            'code':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CODE}}


def validate(receipt,root=ROOT):
    expected=identity(root)
    if set(receipt)!=set(expected)|{'reader'}:raise ValueError('reader receipt schema differs')
    for key,value in expected.items():
        if receipt[key]!=value:raise ValueError('reader receipt '+key+' differs')
    reader=receipt['reader']
    if (set(reader)!={'executable','sha256','version','arguments','exit','stdout','stderr'} or
        not isinstance(reader['executable'],str) or not reader['executable'].startswith('/') or
        not re.fullmatch('[0-9a-f]{64}',reader['sha256']) or
        not isinstance(reader['version'],str) or 'GNU Guile' not in reader['version'] or
        reader['arguments']!=ARGUMENTS or type(reader['exit']) is not int or
        reader['exit']!=0 or reader['stdout']!='' or reader['stderr']!=''):
        raise ValueError('host reader did not complete exact input successfully')
    return receipt


def create(output,root=ROOT):
    if not output.is_absolute():raise ValueError('host receipt output must be absolute')
    if subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=root):
        raise ValueError('host reader requires clean committed source')
    before=identity(root)
    binary=shutil.which('guile')
    if not binary:raise ValueError('host Guile is required')
    executable=Path(binary).resolve(strict=True)
    version=subprocess.run([str(executable),'--version'],capture_output=True,text=True,check=True,timeout=30)
    if version.stderr:raise ValueError('host Guile version emitted diagnostics')
    result=subprocess.run([str(executable),*ARGUMENTS],cwd=root,capture_output=True,text=True,timeout=30)
    receipt={**before,'reader':{'executable':str(executable),
             'sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
             'version':version.stdout,'arguments':ARGUMENTS,'exit':result.returncode,
             'stdout':result.stdout,'stderr':result.stderr}}
    validate(receipt,root)
    if identity(root)!=before:raise ValueError('source changed during host reader')
    with output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in ('create','validate'):
        raise SystemExit('usage: host_reader.py create|validate RECEIPT')
    path=Path(sys.argv[2])
    if sys.argv[1]=='create':create(path)
    else:
        validate(json.loads(path.read_text()))
        print('HOST GUILE VERIFIED / GUEST GUILE NOT RUN')
