#!/usr/bin/env python3
"""Host-only f31 launcher. Its file lock covers the complete Docker run."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[1]
LOCK = Path('/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/heavy.lock')
DEPS = Path('/home/gabe/.codex/evidence/eshkol-transformer/e3-retention-teardown-gate-d985917-20261014-a/deps')
VENV = Path('/home/gabe/.codex/venvs/eshkol-oracle-py3146')
IMAGE = 'sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6'
BASE = '5e29debd48762648207bb087c5fd0567e3828c6d'


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected_head', help='reviewed repair commit SHA-1')
    parser.add_argument('expected_tree', help='reviewed repair tree SHA-1')
    parser.add_argument('evidence', type=Path, help='fresh absolute evidence directory')
    args = parser.parse_args()
    if not all(re.fullmatch(r'[0-9a-f]{40}', value)
               for value in (args.expected_head, args.expected_tree)):
        parser.error('expected HEAD and tree must be full lowercase SHA-1 values')
    evidence = args.evidence
    if not evidence.is_absolute() or evidence.exists():
        parser.error('evidence must be a fresh absolute path')
    if not LOCK.is_file() or LOCK.is_symlink() or not DEPS.is_dir() or not (VENV / 'bin/python').is_file():
        parser.error('lock or pinned dependency mount is missing')

    with LOCK.open('a+b') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        if (git('rev-parse', 'HEAD') != args.expected_head or
                git('rev-parse', 'HEAD^{tree}') != args.expected_tree or
                git('status', '--porcelain') or
                subprocess.run(['git', '-C', str(SOURCE), 'merge-base',
                                '--is-ancestor', BASE, args.expected_head],
                               check=False).returncode != 0):
            raise SystemExit('locked source differs from reviewed repair HEAD/tree or 5e29 base')
        git_dir = Path(git('rev-parse', '--git-dir'))
        common_git = Path(git('rev-parse', '--git-common-dir'))
        if not git_dir.is_relative_to(common_git):
            raise SystemExit('worktree gitdir is outside the mounted common Git directory')
        evidence.mkdir()
        (evidence / 'tmp').mkdir()
        (evidence / 'gate').mkdir()
        argv = [
            'timeout', '--foreground', '--signal=TERM', '--kill-after=30s', '14400s',
            'docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--read-only',
            '--cpus=3', '--memory=12g', '--pids-limit=256', '--ulimit', 'core=0',
            '--cidfile', str(evidence / 'container.cid'),
            '--user', '0:0', '--workdir', '/workspace',
            '-e', 'GIT_CONFIG_COUNT=2', '-e', 'GIT_CONFIG_KEY_0=safe.directory',
            '-e', 'GIT_CONFIG_VALUE_0=/workspace',
            '-e', 'GIT_CONFIG_KEY_1=safe.directory',
            '-e', 'GIT_CONFIG_VALUE_1=/workspace/.deps/eshkol-src',
            '-e', f'GIT_DIR={git_dir}', '-e', 'GIT_WORK_TREE=/workspace',
            '-e', 'GIT_OPTIONAL_LOCKS=0', '-e', 'TMPDIR=/out/tmp',
            '-e', 'CC=clang-21', '-e', 'CXX=clang++-21',
            '-e', 'CMAKE_BUILD_PARALLEL_LEVEL=3', '-e', 'MAKEFLAGS=-j3',
            '-e', f'Q0_PYTHON={VENV / "bin/python"}',
            '-v', f'{SOURCE}:/workspace:ro', '-v', f'{common_git}:{common_git}:ro',
            '-v', f'{DEPS}:/workspace/.deps:ro', '-v', f'{evidence}:/out:rw',
            '-v', f'{VENV}:{VENV}:ro',
            '-v', '/home/gabe/.local/share/uv/python:/home/gabe/.local/share/uv/python:ro',
            IMAGE, '/usr/bin/bash', 'scripts/test-e3-r1-stage8-compiled.sh',
            '/out/gate', args.expected_head, args.expected_tree,
        ]
        (evidence / 'argv.json').write_text(json.dumps(argv, indent=2) + '\n')
        (evidence / 'source.json').write_text(json.dumps({
            'head': args.expected_head, 'tree': args.expected_tree, 'base': BASE,
            'guest_script_sha256': hashlib.sha256(
                (SOURCE / 'scripts/test-e3-r1-stage8-compiled.sh').read_bytes()).hexdigest(),
            'image': IMAGE, 'dependencies': str(DEPS),
        }, indent=2) + '\n')
        (evidence / 'lock-acquired').write_text(str(time.time()) + '\n')
        start = time.monotonic()
        with (evidence / 'stdout').open('wb') as stdout, (evidence / 'stderr').open('wb') as stderr:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
        post_head = git('rev-parse', 'HEAD')
        post_tree = git('rev-parse', 'HEAD^{tree}')
        post_clean = not bool(git('status', '--porcelain'))
        (evidence / 'result.json').write_text(json.dumps({
            'exit': result.returncode, 'elapsed_seconds': time.monotonic() - start,
            'source_post_head': post_head, 'source_post_tree': post_tree,
            'source_post_clean': post_clean,
        }, indent=2) + '\n')
        if result.returncode != 0 or post_head != args.expected_head or post_tree != args.expected_tree or not post_clean:
            raise SystemExit('E3 stage-8 gate failed or locked source changed; inspect result.json')


if __name__ == '__main__':
    main()
