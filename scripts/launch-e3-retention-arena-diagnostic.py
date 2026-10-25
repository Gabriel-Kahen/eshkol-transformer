#!/usr/bin/env python3
"""Host-only diagnostic launcher; retain heavy.lock until the owned CID is gone."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[1]
LOCK = Path('/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/heavy.lock')
DEPS = Path('/home/gabe/.codex/evidence/eshkol-transformer/e3-retention-teardown-gate-d985917-20261014-a/deps')
VENV = Path('/home/gabe/.codex/venvs/eshkol-oracle-py3146')
IMAGE = 'sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6'
BASE = '44c0418d2d2d8f3c1354c922d3eeee39b0d7e6ef'
LABEL = 'codex.e3.retention.arena.diagnostic.run'


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()


def tracked_digest():
    digest = hashlib.sha256()
    paths = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', '-z']).split(b'\0')
    for raw in filter(None, paths):
        path = SOURCE / os.fsdecode(raw)
        digest.update(raw + b'\0')
        if path.is_symlink():
            digest.update(b'L' + os.fsencode(os.readlink(path)) + b'\0')
        else:
            digest.update(b'F' + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare_dependency_mountpoint():
    mountpoint = SOURCE / '.deps'
    if not mountpoint.exists() and not mountpoint.is_symlink():
        mountpoint.mkdir(mode=0o700)
    if mountpoint.is_symlink() or not mountpoint.is_dir() or any(mountpoint.iterdir()):
        raise RuntimeError('source .deps mountpoint must be an empty real directory')
    return mountpoint


def inspect_container(target):
    result = subprocess.run(['docker', 'inspect', target], capture_output=True,
                            text=True, timeout=30, check=False)
    if result.returncode != 0:
        if 'no such object' in result.stderr.lower():
            return None
        raise RuntimeError(f'Docker inspect failed: {result.returncode}: {result.stderr.strip()}')
    records = json.loads(result.stdout)
    if len(records) != 1:
        raise RuntimeError('Docker inspect returned an unexpected record count')
    return records[0]


def record_retry(evidence, message):
    try:
        with (evidence / 'container-custody-retry.log').open('a') as retry:
            retry.write(f'{time.time()} {message}\n')
    except OSError:
        pass


def ensure_container_gone(evidence, name, run_id):
    """Inspect by CID, remove only our label, then prove absence before unlock."""
    force_stopped = False
    exit_state = None
    nameless_absent_checks = 0
    identity_mismatch = False
    while True:
        try:
            cid_file = evidence / 'container.cid'
            raw_cid = cid_file.read_text().strip() if cid_file.exists() else None
            cid = raw_cid if raw_cid and re.fullmatch(r'[0-9a-f]{64}', raw_cid) else None
            if raw_cid and cid is None:
                identity_mismatch = True
                record_retry(evidence, f'invalid CID file: {raw_cid!r}')
            inspected = inspect_container(cid or name)
            if inspected is None and cid:
                named = inspect_container(name)
                if named is not None:
                    if (named['Config'].get('Labels') or {}).get(LABEL) != run_id:
                        custody = {'container': name, 'cid': cid, 'owned_absent': True,
                                   'force_stopped': force_stopped, 'exit_state': exit_state,
                                   'foreign_name': True,
                                   'identity_mismatch': identity_mismatch,
                                   'nameless_absent_checks': 0}
                        (evidence / 'container-custody.json').write_text(
                            json.dumps(custody, indent=2) + '\n')
                        return custody
                    identity_mismatch = True
                    record_retry(evidence, f'CID absent but labeled name holds {named["Id"]}')
                    inspected = named
            if inspected is None:
                if cid is None:
                    nameless_absent_checks += 1
                    if nameless_absent_checks < 3:
                        time.sleep(5)
                        continue
                custody = {'container': name, 'cid': raw_cid, 'owned_absent': True,
                           'force_stopped': force_stopped, 'exit_state': exit_state,
                           'foreign_name': False, 'identity_mismatch': identity_mismatch,
                           'nameless_absent_checks': nameless_absent_checks}
                (evidence / 'container-custody.json').write_text(json.dumps(custody, indent=2) + '\n')
                return custody
            actual_id = inspected['Id']
            nameless_absent_checks = 0
            labels = inspected['Config'].get('Labels') or {}
            if labels.get(LABEL) != run_id:
                if cid is None:
                    custody = {'container': name, 'cid': raw_cid, 'owned_absent': True,
                               'force_stopped': force_stopped, 'exit_state': exit_state,
                               'foreign_name': True,
                               'identity_mismatch': identity_mismatch,
                               'nameless_absent_checks': 0}
                    (evidence / 'container-custody.json').write_text(json.dumps(custody, indent=2) + '\n')
                    return custody
                record_retry(evidence, f'CID ownership mismatch: {actual_id}')
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
            record_retry(evidence, f'cleaned owned CID {actual_id}')
        except Exception as exc:
            record_retry(evidence, f'custody retry {type(exc).__name__}: {exc}')
        time.sleep(10)


def interrupted(signum, _frame):
    raise InterruptedError(f'launcher signal {signum}')


def stop_child(child):
    if child is None or child.poll() is not None:
        return
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


def seal_evidence(evidence):
    rows = []
    for path in sorted(evidence.rglob('*')):
        if path.is_symlink():
            raise RuntimeError(f'evidence contains an unsealed symlink: {path}')
        if path.is_file():
            relative = path.relative_to(evidence).as_posix()
            if relative == 'POSTEXIT-SHA256SUMS':
                continue
            digest_state = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest_state.update(chunk)
            digest = digest_state.hexdigest()
            rows.append(f'{digest}  ./{relative}\n')
    (evidence / 'POSTEXIT-SHA256SUMS').write_text(''.join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected_head', help='reviewed gate commit SHA-1')
    parser.add_argument('expected_tree', help='reviewed gate tree SHA-1')
    parser.add_argument('evidence', type=Path, help='fresh absolute evidence directory')
    args = parser.parse_args()
    if not all(re.fullmatch(r'[0-9a-f]{40}', value)
               for value in (args.expected_head, args.expected_tree)):
        parser.error('expected HEAD and tree must be full lowercase SHA-1 values')
    evidence = args.evidence
    if (not evidence.is_absolute() or evidence.exists() or
            evidence.resolve().is_relative_to(SOURCE.resolve())):
        parser.error('evidence must be a fresh absolute path outside the source tree')
    if not LOCK.is_file() or LOCK.is_symlink() or not DEPS.is_dir() or not (VENV / 'bin/python').is_file():
        parser.error('lock or pinned dependency mount is missing')

    run_id = secrets.token_hex(16)
    name = f'e3-retention-arena-diagnostic-{args.expected_head[:8]}-{run_id[:8]}'
    source_digest = tracked_digest()
    with LOCK.open('a+b') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        if (git('rev-parse', 'HEAD') != args.expected_head or
                git('rev-parse', 'HEAD^{tree}') != args.expected_tree or
                git('status', '--porcelain') or tracked_digest() != source_digest or
                subprocess.run(['git', '-C', str(SOURCE), 'merge-base',
                                '--is-ancestor', BASE, args.expected_head],
                               check=False).returncode != 0):
            raise SystemExit('locked source differs from reviewed gate HEAD/tree or 44c base')
        prepare_dependency_mountpoint()
        if inspect_container(name) is not None:
            raise SystemExit('generated container name is occupied')
        git_dir = Path(git('rev-parse', '--git-dir'))
        common_git = Path(git('rev-parse', '--git-common-dir'))
        if not git_dir.is_relative_to(common_git):
            raise SystemExit('worktree gitdir is outside the mounted common Git directory')
        evidence.mkdir()
        (evidence / 'tmp').mkdir()
        create_argv = [
            'docker', 'create', '--pull', 'never', '--network', 'none', '--read-only',
            '--cpus=3', '--memory=12g', '--pids-limit=256', '--ulimit', 'core=0',
            '--cidfile', str(evidence / 'container.cid'), '--name', name,
            '--label', f'{LABEL}={run_id}',
            '--user', '0:0', '--workdir', '/workspace',
            '-e', 'GIT_CONFIG_COUNT=2', '-e', 'GIT_CONFIG_KEY_0=safe.directory',
            '-e', 'GIT_CONFIG_VALUE_0=/workspace',
            '-e', 'GIT_CONFIG_KEY_1=safe.directory',
            '-e', 'GIT_CONFIG_VALUE_1=/workspace/.deps/eshkol-src',
            '-e', 'GIT_OPTIONAL_LOCKS=0', '-e', 'TMPDIR=/out/tmp',
            '-e', 'CC=clang-21', '-e', 'CXX=clang++-21',
            '-e', 'CMAKE_BUILD_PARALLEL_LEVEL=3', '-e', 'MAKEFLAGS=-j3',
            '-e', f'Q0_PYTHON={VENV / "bin/python"}',
            '-v', f'{SOURCE}:/workspace:ro', '-v', f'{common_git}:{common_git}:ro',
            '-v', f'{DEPS}:/workspace/.deps:ro', '-v', f'{evidence}:/out:rw',
            '-v', f'{VENV}:{VENV}:ro',
            '-v', '/home/gabe/.local/share/uv/python:/home/gabe/.local/share/uv/python:ro',
            IMAGE, '/usr/bin/bash', 'scripts/test-e3-complete-call-retention.sh',
            '/out/gate', 'diagnostic', args.expected_head, args.expected_tree,
        ]
        (evidence / 'argv.json').write_text(json.dumps({
            'create': create_argv,
            'start_template': ['timeout', '--foreground', '--signal=TERM',
                               '--kill-after=30s', '14400s', 'docker', 'start',
                               '--attach', '<CID>'],
        }, indent=2) + '\n')
        (evidence / 'source.json').write_text(json.dumps({
            'head': args.expected_head, 'tree': args.expected_tree, 'base': BASE,
            'guest_script_sha256': hashlib.sha256(
                (SOURCE / 'scripts/test-e3-complete-call-retention.sh').read_bytes()).hexdigest(),
            'image': IMAGE, 'dependencies': str(DEPS),
            'tracked_digest': source_digest, 'run_id': run_id,
        }, indent=2) + '\n')
        (evidence / 'lock-acquired').write_text(str(time.time()) + '\n')
        child = None
        create_status = None
        start_status = None
        cid = None
        cancelled = False
        failure = None
        start = time.monotonic()
        def defer_create_signal(_signum, _frame):
            nonlocal cancelled
            cancelled = True
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
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
            cid_file = evidence / 'container.cid'
            cid = cid_file.read_text().strip() if cid_file.exists() else None
            if create_status != 0:
                raise RuntimeError(f'Docker create failed: {create_status}')
            if not cid or not re.fullmatch(r'[0-9a-f]{64}', cid):
                raise RuntimeError('Docker create returned success without a valid CID')
            inspected = inspect_container(cid)
            if inspected is None or inspected['Id'] != cid or \
                    (inspected['Config'].get('Labels') or {}).get(LABEL) != run_id:
                raise RuntimeError('created CID lacks the pinned run label')
            if not cancelled:
                start_argv = ['timeout', '--foreground', '--signal=TERM',
                              '--kill-after=30s', '14400s', 'docker', 'start',
                              '--attach', cid]
                (evidence / 'start.argv.json').write_text(json.dumps(start_argv, indent=2) + '\n')
                with (evidence / 'stdout').open('wb') as stdout, \
                        (evidence / 'stderr').open('wb') as stderr:
                    child = subprocess.Popen(start_argv, stdout=stdout,
                                             stderr=stderr, start_new_session=True)
                    start_status = child.wait()
        except BaseException as exc:
            failure = f'{type(exc).__name__}: {exc}'
            stop_child(child)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            custody = ensure_container_gone(evidence, name, run_id)
        post_head = git('rev-parse', 'HEAD')
        post_tree = git('rev-parse', 'HEAD^{tree}')
        post_clean = not bool(git('status', '--porcelain'))
        post_digest = tracked_digest()
        eligible = (failure is None and not cancelled and create_status == 0 and
                    start_status == 0 and custody['owned_absent'] and
                    custody['exit_state'] is not None and
                    custody['exit_state']['exit_code'] == 0 and
                    not custody['exit_state']['oom_killed'] and
                    not custody['force_stopped'] and not custody['foreign_name'] and
                    not custody['identity_mismatch'] and
                    post_head == args.expected_head and post_tree == args.expected_tree and
                    post_clean and post_digest == source_digest)
        (evidence / 'result.json').write_text(json.dumps({
            'create_exit': create_status, 'start_exit': start_status,
            'cancel_requested_during_create': cancelled, 'failure': failure,
            'cid': cid, 'elapsed_seconds': time.monotonic() - start,
            'source_post_head': post_head, 'source_post_tree': post_tree,
            'source_post_clean': post_clean,
            'source_pre_tracked_digest': source_digest,
            'source_post_tracked_digest': post_digest,
            'container_custody': custody, 'diagnostic_only': True,
            'eligible_for_diagnostic_review': eligible,
        }, indent=2) + '\n')
        seal_evidence(evidence)
        if not eligible:
            raise SystemExit('E3 retention arena diagnostic or custody failed; inspect result.json')


if __name__ == '__main__':
    main()
