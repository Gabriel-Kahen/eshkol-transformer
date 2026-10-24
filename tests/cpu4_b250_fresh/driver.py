"""Bounded producer/live-final-B to fresh FILE factory receipt driver."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tests.cpu4_b250_fresh.witness import compare, seal
from tests.cpu4_trained_g2.normal_driver import (EXPECTED_B125_SHA256,
                                                   hashes, inventory_depfiles,
                                                   write_resources)
from tests.cpu4_v1_aggregate.closure import ROOT, load_roster, manifest

PRODUCER = ROOT / 'tests/cpu4_trained_g2/runtime.esk'
FRESH = ROOT / 'tests/cpu4_b250_fresh/runtime.esk'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gate(out: Path, prior: Path, expected_b250: str, expected_archive: str,
         runner: Path, cxx: Path) -> int:
    require(out.is_absolute() and not out.exists(), 'fresh absolute output required')
    out.mkdir(mode=0o700)
    program, custody, before = 1, 0, None
    source = ROOT

    def run(label, args, env=None):
        args = [str(x) for x in args]
        (out / (label + '.argv.json')).write_text(json.dumps(args) + '\n')
        with (out / (label + '.stdout')).open('wb') as stdout, (out / (label + '.stderr')).open('wb') as stderr:
            try:
                status = subprocess.run(args, cwd=source, env=env, stdout=stdout, stderr=stderr).returncode
            except OSError as error:
                stderr.write((repr(error) + '\n').encode())
                status = 127
        (out / (label + '.exit')).write_text(str(status) + '\n')
        if status:
            raise subprocess.CalledProcessError(status, args)
        return (out / (label + '.stdout')).read_bytes()

    try:
        before = run('source.before', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
        require(not run('source.before.status', ['git', 'status', '--porcelain', '--untracked-files=all']),
                'source tree is dirty')
        roster = load_roster()
        require(manifest(ROOT, roster) == (ROOT / 'native/cpu4_v1_source_members.sha256').read_text(),
                'CPU4 native source closure differs from sealed archive source')
        archive = prior / 'combined/libcpu4_resume.a'
        checkpoint = prior / 'combined/two-token.final-b250.1.c2'
        require((os.statvfs(checkpoint).f_flag & os.ST_RDONLY) and
                (os.statvfs(archive).f_flag & os.ST_RDONLY) and
                (os.statvfs(ROOT).f_flag & os.ST_RDONLY),
                'source, prior B250 and archive must be mounted read-only')
        require(len(expected_b250) == 64 and len(expected_archive) == 64 and
                checkpoint.stat().st_size == 25504 and sha(checkpoint) == expected_b250 and
                sha(archive) == expected_archive and expected_b250 != EXPECTED_B125_SHA256,
                'prior B250 or native archive identity changed')
        (out / 'prior-pins.json').write_text(json.dumps({
            'source': before.decode().splitlines(), 'prior_b250_sha256': expected_b250,
            'prior_b250_size': 25504, 'prior_archive_sha256': expected_archive}, indent=2) + '\n')
        train, heldout = write_resources(out)
        cache = out / 'cache'
        cache.mkdir()
        (out / 'cxx-link').write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(str(cxx)) + ' "$@"\n')
        (out / 'cxx-link').chmod(0o500)
        env = dict(os.environ)
        for name in ('ESHKOL_PATH', 'ESHKOL_JIT_CACHE_DIR'):
            env.pop(name, None)
        env.update(ESHKOL_JIT_CACHE='0', XDG_CACHE_HOME=str(cache),
                   PYTHONDONTWRITEBYTECODE='1', ESHKOL_ARENA_POISON='1',
                   ESHKOL_CXX_COMPILER=str(out / 'cxx-link'),
                   ESHKOL_LIB_DIR=str(ROOT / 'lib'))
        includes = [arg for directory in roster['source_include_dirs']
                    for arg in ('-I', str(ROOT / directory))]
        for label, source_file in (('producer', PRODUCER), ('fresh', FRESH)):
            raw = out / (label + '.raw.o')
            run(label + '.compile', ['timeout', '--foreground', '--signal=TERM',
                '--kill-after=10s', '1200s', runner, '--strict-types', '--no-stdlib',
                '--optimize', '0', *includes, '-L', archive.parent, '--lib', 'cpu4_resume',
                '--emit-object', '--dump-ir', '--emit-depfile', out / (label + '.d'),
                source_file, '-o', raw], env)
            for suffix in ('', '.ll', '.bc'):
                require(Path(str(raw) + suffix).stat().st_size > 0, 'compiler artifact missing')
            run(label + '.link', ['timeout', '--foreground', '--signal=TERM',
                '--kill-after=10s', '120s', runner, '--no-stdlib', '-L', archive.parent,
                '--lib', 'cpu4_resume', raw, '-o', out / label], env)
        produced_b125 = out / 'producer.b125.c2'
        produced_b250 = out / 'producer.b250.c2'
        producer_stdout = run('producer.run', ['timeout', '--foreground',
            '--signal=TERM', '--kill-after=5s', '900s', out / 'producer', train,
            heldout, produced_b125, produced_b250, 'second-eos'], env)
        require(not (out / 'producer.run.stderr').read_bytes(), 'producer diagnostics')
        require(produced_b125.stat().st_size == 25504 and
                sha(produced_b125) == EXPECTED_B125_SHA256, 'producer B125 identity')
        require(produced_b250.stat().st_size == 25504 and
                sha(produced_b250) == expected_b250 and
                produced_b250.read_bytes() == checkpoint.read_bytes(),
                'producer B250 differs from prior sealed model')
        filtered = b'\n'.join(line for line in producer_stdout.splitlines()
                              if not line.startswith(b'CPU4-LIVE-B250-PLANE ')) + b'\n'
        (out / 'producer.standard.stdout').write_bytes(filtered)
        run('producer.check', [sys.executable, ROOT / 'tests/cpu4_trained_g2/check_output.py',
                               out / 'producer.standard.stdout', 'second-eos'], env)
        witness_dir = out / 'live-witness'
        seal(producer_stdout, 'CPU4-LIVE-B250-PLANE', witness_dir)
        for mode in ('inspect', 'handoff'):
            require(sha(checkpoint) == expected_b250 and
                    sha(archive) == expected_archive,
                    'read-only B250 or archive changed before fresh process')
            stdout = run('fresh.' + mode, ['timeout', '--foreground', '--signal=TERM',
                '--kill-after=5s', '900s', out / 'fresh', train, checkpoint, mode], env)
            require(not (out / ('fresh.' + mode + '.stderr')).read_bytes(),
                    'fresh FILE diagnostics')
            compare(witness_dir, stdout, 'CPU4-FRESH-B250-PLANE')
            require(stdout.splitlines()[-1].startswith(
                ('CPU4-B250-FRESH-PASS ' + mode + ' ').encode()),
                'fresh FILE result line missing')
            require(sha(checkpoint) == expected_b250 and
                    sha(archive) == expected_archive,
                    'read-only B250 or archive changed after fresh process')
        program = 0
    except subprocess.CalledProcessError as error:
        program = error.returncode
    except Exception as error:
        (out / 'driver-error.txt').write_text(repr(error) + '\n')
    finally:
        try:
            inventory_depfiles(out)
            after = run('source.after', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
            require(before is not None and before == after, 'source revision changed')
            require(not run('source.after.status', ['git', 'status', '--porcelain',
                                                    '--untracked-files=all']), 'source dirty after gate')
        except Exception as error:
            custody = 74
            (out / 'custody-error.txt').write_text(repr(error) + '\n')
        (out / 'driver-program.exit').write_text(str(program) + '\n')
        (out / 'driver-custody.exit').write_text(str(custody) + '\n')
        hashes(out, out / 'SHA256SUMS')
        (out / 'SHA256SUMS.sha256').write_text(
            hashlib.sha256((out / 'SHA256SUMS').read_bytes()).hexdigest() + '  SHA256SUMS\n')
    return custody or program


if __name__ == '__main__':
    if len(sys.argv) != 7:
        raise SystemExit('driver OUT PRIOR EXPECTED_B250_SHA EXPECTED_ARCHIVE_SHA RUNNER CXX')
    raise SystemExit(gate(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
                          sys.argv[4], Path(sys.argv[5]), Path(sys.argv[6])))
