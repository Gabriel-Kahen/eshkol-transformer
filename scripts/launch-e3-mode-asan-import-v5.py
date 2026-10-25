#!/usr/bin/env python3
"""Run the frozen E3 mode gate after acquiring the shared heavy-build lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time

source = Path('/home/gabe/.codex/worktrees/e3-mode-asan-import-repair-20261025')
evidence = Path('/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/e3/mode-dc565e1-attempt5')
deps = Path('/home/gabe/.codex/evidence/eshkol-transformer/e3-retention-teardown-gate-d985917-20261014-a/deps')
venv = Path('/home/gabe/.codex/venvs/eshkol-oracle-py3146')
lock = Path('/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/heavy.lock')
image = 'sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6'
container_name = 'finish-e3-mode-dc565e1-v5-20261025'
run_id = secrets.token_hex(16)

def git(*args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()

def tracked_digest():
    result = hashlib.sha256()
    paths = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).split(b'\0')
    for raw in filter(None, paths):
        path = source / os.fsdecode(raw)
        result.update(raw + b'\0')
        if path.is_symlink():
            result.update(b'L' + os.fsencode(os.readlink(path)) + b'\0')
        else:
            result.update(b'F' + hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()

def check_deps_mountpoint():
    mountpoint = source / '.deps'
    assert mountpoint.is_dir() and not mountpoint.is_symlink(), 'source .deps mountpoint missing or symlinked'
    assert not any(mountpoint.iterdir()), 'source .deps mountpoint must be empty'
    ignored = subprocess.run(['git', '-C', str(source), 'check-ignore', '-q', '.deps/'],
                             check=False)
    assert ignored.returncode == 0, 'source .deps mountpoint is not ignored'

def inspect_container(target):
    inspected = subprocess.run(['docker', 'inspect', target],
                               capture_output=True, text=True, timeout=30,
                               check=False)
    if inspected.returncode != 0:
        if 'no such object' in inspected.stderr.lower():
            return None
        raise RuntimeError(f'Docker inspect failed: {inspected.returncode}: '
                           f'{inspected.stderr.strip()}')
    records = json.loads(inspected.stdout)
    if len(records) != 1:
        raise RuntimeError('Docker inspect returned an unexpected record count')
    return records[0]

def record_retry(message):
    try:
        with (evidence / 'container-custody-retry.log').open('a') as retry:
            retry.write(f'{time.time()} {message}\n')
    except OSError:
        pass

def ensure_container_gone():
    """Keep the heavy lock until our labeled CID is absent."""
    force_stopped = False
    exit_state = None
    nameless_absent_checks = 0
    while True:
        try:
            cid_file = evidence / 'container.cid'
            cid = cid_file.read_text().strip() if cid_file.exists() else None
            target = cid or container_name
            inspected = inspect_container(target)
            if inspected is None:
                if cid is None:
                    nameless_absent_checks += 1
                    if nameless_absent_checks < 3:
                        time.sleep(5)
                        continue
                custody = {'container': container_name, 'cid': cid,
                           'owned_absent': True, 'force_stopped': force_stopped,
                           'exit_state': exit_state,
                           'foreign_name': False,
                           'nameless_absent_checks': nameless_absent_checks}
                (evidence / 'container-custody.json').write_text(
                    json.dumps(custody, indent=2) + '\n')
                return custody
            actual_id = inspected['Id']
            nameless_absent_checks = 0
            labels = inspected['Config'].get('Labels') or {}
            if labels.get('codex.e3.run') != run_id:
                if cid is None:
                    custody = {'container': container_name, 'cid': None,
                               'owned_absent': True, 'force_stopped': force_stopped,
                               'exit_state': exit_state,
                               'foreign_name': True}
                    (evidence / 'container-custody.json').write_text(
                        json.dumps(custody, indent=2) + '\n')
                    return custody
                record_retry(f'CID ownership mismatch: {actual_id}')
                time.sleep(10)
                continue
            state = inspected['State']
            if not state['Running'] and exit_state is None:
                exit_state = {'exit_code': state['ExitCode'],
                              'oom_killed': state['OOMKilled'],
                              'status': state['Status']}
            if state['Running']:
                force_stopped = True
                subprocess.run(['docker', 'stop', '--time', '30', actual_id],
                               capture_output=True, timeout=75, check=False)
            subprocess.run(['docker', 'rm', '--force', actual_id],
                           capture_output=True, timeout=45, check=False)
            record_retry(f'cleaned owned CID {actual_id}')
        except Exception as exc:
            record_retry(f'custody retry {type(exc).__name__}: {exc}')
        time.sleep(10)

def interrupted(signum, _frame):
    raise InterruptedError(f'launcher signal {signum}')

cancel_requested = False
def defer_create_signal(_signum, _frame):
    global cancel_requested
    cancel_requested = True

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

if sys.argv[1:] not in ([], ['--preflight']):
    raise SystemExit('usage: launch-e3-mode-asan-import-v5.py [--preflight]')
assert git('rev-parse', 'HEAD') == 'dc565e1d259e3f296521a8c1dac91d8e8e95f0a2'
assert git('rev-parse', 'HEAD^{tree}') == '67bee60a6ba24e101a85bf7926da5735ca422185'
assert not git('status', '--porcelain')
assert deps.is_dir() and (venv / 'bin/python').is_file()
check_deps_mountpoint()
source_digest = tracked_digest()
if sys.argv[1:] == ['--preflight']:
    assert not evidence.exists(), 'one-use evidence directory already exists'
    assert inspect_container(container_name) is None, 'container name already occupied'
    print(json.dumps({'source_head': git('rev-parse', 'HEAD'),
                      'source_tree': git('rev-parse', 'HEAD^{tree}'),
                      'tracked_digest': source_digest,
                      'gate_sha256': hashlib.sha256((source / 'scripts/test-e3-mode-graph.sh').read_bytes()).hexdigest(),
                      'deps_mountpoint': str(source / '.deps'),
                      'deps_mountpoint_ignored_empty': True,
                      'image': image, 'evidence_absent': True,
                      'container_name_free': True}, indent=2))
    raise SystemExit(0)
evidence.mkdir(exist_ok=False)
(evidence / 'tmp').mkdir()
create_argv = [
    'docker', 'create', '--pull', 'never', '--network', 'none', '--read-only',
    '--cpus=3', '--memory=12g', '--pids-limit=256', '--ulimit', 'core=0',
    '--cidfile', str(evidence / 'container.cid'),
    '--name', container_name,
    '--label', f'codex.e3.run={run_id}',
    '--user', '0:0', '--workdir', '/workspace',
    '-e', 'GIT_CONFIG_COUNT=2', '-e', 'GIT_CONFIG_KEY_0=safe.directory',
    '-e', 'GIT_CONFIG_VALUE_0=/workspace', '-e', 'GIT_CONFIG_KEY_1=safe.directory',
    '-e', 'GIT_CONFIG_VALUE_1=/workspace/.deps/eshkol-src',
    '-e', 'TMPDIR=/out/tmp', '-e', 'CC=clang-21', '-e', 'CXX=clang++-21',
    '-e', 'CMAKE_BUILD_PARALLEL_LEVEL=3', '-e', 'MAKEFLAGS=-j3',
    '-e', 'Q0_PYTHON=/home/gabe/.codex/venvs/eshkol-oracle-py3146/bin/python',
    '-e', 'GIT_OPTIONAL_LOCKS=0',
    '-v', f'{source}:/workspace:ro', '-v', f'{deps}:/workspace/.deps:ro',
    '-v', f'{evidence}:/out:rw', '-v', f'{venv}:{venv}:ro',
    '-v', '/home/gabe/.local/share/uv/python:/home/gabe/.local/share/uv/python:ro',
    image, '/usr/bin/bash', '-c',
    'set -euo pipefail; source scripts/common.sh; verify_toolchain; '
    '/usr/bin/bash scripts/test-e3-mode-graph.sh /out/gate',
]
(evidence / 'argv.json').write_text(json.dumps({'create': create_argv,
    'start_template': ['timeout', '--foreground', '--signal=TERM',
                       '--kill-after=30s', '10800s', 'docker', 'start',
                       '--attach', '<CID>']}, indent=2) + '\n')
(evidence / 'source.json').write_text(json.dumps({
    'head': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
    'script_sha256': hashlib.sha256((source / 'scripts/test-e3-mode-graph.sh').read_bytes()).hexdigest(),
    'image': image, 'dependencies': str(deps), 'tracked_digest': source_digest,
    'run_id': run_id,
}, indent=2) + '\n')
with lock.open('a+b') as held:
    fcntl.flock(held, fcntl.LOCK_EX)
    assert git('rev-parse', 'HEAD') == 'dc565e1d259e3f296521a8c1dac91d8e8e95f0a2'
    assert git('rev-parse', 'HEAD^{tree}') == '67bee60a6ba24e101a85bf7926da5735ca422185'
    assert not git('status', '--porcelain')
    check_deps_mountpoint()
    assert hashlib.sha256((source / 'scripts/test-e3-mode-graph.sh').read_bytes()).hexdigest() == json.loads((evidence / 'source.json').read_text())['script_sha256']
    assert tracked_digest() == source_digest
    assert inspect_container(container_name) is None, 'container name already occupied'
    (evidence / 'lock-acquired').write_text(str(time.time()) + '\n')
    child = None
    create_status = None
    start_status = None
    cid = None
    start = time.monotonic()
    try:
        signal.signal(signal.SIGTERM, defer_create_signal)
        signal.signal(signal.SIGINT, defer_create_signal)
        with (evidence / 'create.stdout').open('wb') as stdout, \
                (evidence / 'create.stderr').open('wb') as stderr:
            created = subprocess.run(create_argv, stdout=stdout, stderr=stderr,
                                     check=False)
        create_status = created.returncode
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        cid_path = evidence / 'container.cid'
        cid = cid_path.read_text().strip() if cid_path.exists() else None
        if create_status == 0 and cid and re.fullmatch(r'[0-9a-f]{64}', cid):
            inspected = inspect_container(cid)
            if inspected is None or (inspected['Config'].get('Labels') or {}).get('codex.e3.run') != run_id:
                raise RuntimeError('created CID lacks the pinned run label')
            if not cancel_requested:
                start_argv = ['timeout', '--foreground', '--signal=TERM',
                              '--kill-after=30s', '10800s', 'docker', 'start',
                              '--attach', cid]
                (evidence / 'start.argv.json').write_text(json.dumps(start_argv, indent=2) + '\n')
                with (evidence / 'stdout').open('wb') as stdout, \
                        (evidence / 'stderr').open('wb') as stderr:
                    child = subprocess.Popen(start_argv, stdout=stdout,
                                             stderr=stderr, start_new_session=True)
                    start_status = child.wait()
        elif create_status == 0:
            raise RuntimeError('Docker create returned success without a valid CID')
    except BaseException:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        custody = ensure_container_gone()
    elapsed = time.monotonic() - start
    source_post_head = git('rev-parse', 'HEAD')
    source_post_tree = git('rev-parse', 'HEAD^{tree}')
    source_post_clean = not bool(git('status', '--porcelain'))
    source_post_digest = tracked_digest()
    eligible = (create_status == 0 and start_status == 0 and
                custody['exit_state'] is not None and
                custody['exit_state']['exit_code'] == 0 and
                not custody['exit_state']['oom_killed'] and
                not custody['force_stopped'] and
                not custody['foreign_name'] and source_post_clean and
                source_post_head == 'dc565e1d259e3f296521a8c1dac91d8e8e95f0a2' and
                source_post_tree == '67bee60a6ba24e101a85bf7926da5735ca422185' and
                source_post_digest == source_digest)
    (evidence / 'result.json').write_text(json.dumps({
        'create_exit': create_status, 'start_exit': start_status,
        'cancel_requested_during_create': cancel_requested,
        'cid': cid, 'elapsed_seconds': elapsed,
        'source_post_head': source_post_head,
        'source_post_tree': source_post_tree,
        'source_post_clean': source_post_clean,
        'source_post_tracked_digest': source_post_digest,
        'source_pre_tracked_digest': source_digest,
        'container_custody': custody,
        'eligible_for_result_review': eligible,
    }, indent=2) + '\n')
    if not eligible:
        raise SystemExit('E3 mode gate or custody failed; inspect result.json')
