"""Sealed two-process B250/P3 FILE continuation gate; never runs implicitly."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tests.cpu4_c4_p3_fresh.contract import (normal_overlay, units,
    validate_reference, validate_resumed, validate_wire)
from tests.cpu4_trained_g2.normal_driver import (EXPECTED_B125_SHA256,
    hashes, inventory_depfiles, write_resources)
from tests.cpu4_v1_aggregate.closure import ROOT, load_roster, manifest

CALLER = ROOT / 'tests/cpu4_c4_p3_fresh/runtime.esk'
OVERLAY = ROOT / 'tests/cpu4_trained_g2/normal_native_overlay.json'
PRIOR_SOURCE = ('c41c6403683eb2474d60a1ec81a1609e2d6377f6',
                '2c20653a75e0a58a5152221a9b55f6139682917d')
PRIOR_RECIPE_SHA = 'b8fafe11bd5c6ad7c31444ace15161f86af037ee47d767d86a27edaa58b2d159'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prior_receipt(prior: Path, seal: str, expected: dict[str, str]) -> None:
    combined = prior / 'combined'
    sums = combined / 'SHA256SUMS'
    if (len(seal) != 64 or sha(sums) != seal or
            (combined / 'SHA256SUMS.sha256').read_text() !=
            seal + '  SHA256SUMS\n' or
            (combined / 'driver-program.exit').read_text() != '0\n' or
            (combined / 'driver-custody.exit').read_text() != '0\n' or
            (combined / 'source.before.stdout').read_text().splitlines() !=
            list(PRIOR_SOURCE) or
            (combined / 'source.after.stdout').read_text().splitlines() !=
            list(PRIOR_SOURCE)):
        raise ValueError('prior B250 receipt seal or exit changed')
    rows = sums.read_text().splitlines()
    for name, digest in expected.items():
        if rows.count(digest + '  ' + name) != 1:
            raise ValueError('prior receipt omits or changes ' + name)


def outer_receipt(prior: Path, seal: str) -> None:
    sums = prior / 'SHA256SUMS-RUN'
    require(len(seal) == 64 and sha(sums) == seal,
            'prior outer receipt seal changed')
    listed = set()
    for row in sums.read_text().splitlines():
        digest, separator, name = row.partition('  ')
        parts = Path(name).parts
        require(separator and len(digest) == 64 and
                all(ch in '0123456789abcdef' for ch in digest) and
                name.startswith('./') and parts and '..' not in parts and
                name not in listed and name != './SHA256SUMS-RUN',
                'prior outer receipt row is malformed')
        listed.add(name)
        target = prior / name
        require(target.is_file() and not target.is_symlink() and sha(target) == digest,
                'prior outer receipt file changed: ' + name)
    actual = {'./' + path.relative_to(prior).as_posix()
              for path in prior.rglob('*') if path.is_file() and path != sums}
    require(listed == actual, 'prior outer receipt file inventory changed')
    for name in ('docker.exit', 'launcher.exit', 'inside-program.exit',
                 'inside-custody.exit', 'inside.exit', 'native.exit'):
        require((prior / name).read_text() == '0\n',
                'prior outer launch did not exit zero: ' + name)
    require((prior / 'state').read_text() == 'complete\n' and
            (prior / 'started.utc').is_file() and
            (prior / 'finished.utc').is_file(),
            'prior outer launch was not complete')
    for name in ('recipe-precheck', 'recipe-after-lock', 'preflight',
                 'recipe-postcheck', 'postflight'):
        require(not (prior / (name + '.stderr')).read_bytes(),
                'prior outer custody diagnostics: ' + name)
    for name in ('preflight', 'postflight'):
        evidence = json.loads((prior / (name + '.stdout')).read_text())
        require(evidence['preflight'] == 'PASS' and
                evidence['source'] == PRIOR_SOURCE[0],
                'prior outer preflight/postflight did not pass')
    recipe_pin = (prior / 'recipe.expected-sha256').read_text().strip()
    require(recipe_pin == PRIOR_RECIPE_SHA,
            'prior outer recipe pin missing')


def gate(out: Path, prior: Path, expected_outer_seal: str,
         expected_prior_seal: str,
         expected_b250: str, expected_reference: str,
         runner: Path, cc: Path, cxx: Path, compiler_source: Path) -> int:
    require(out.is_absolute() and not out.exists() and not out.is_symlink(),
            'fresh absolute output directory required')
    out.mkdir(mode=0o700)
    program, custody, before = 1, 0, None
    checkpoint = prior / 'combined/two-token.final-b250.1.c2'
    reference = prior / 'combined/second-eos.run1.stdout'

    def run(label, argv, env=None):
        argv = [str(x) for x in argv]
        (out / (label + '.argv.json')).write_text(json.dumps(argv) + '\n')
        with (out / (label + '.stdout')).open('wb') as stdout, \
             (out / (label + '.stderr')).open('wb') as stderr:
            try:
                code = subprocess.run(argv, cwd=ROOT, env=env,
                                      stdout=stdout, stderr=stderr).returncode
            except OSError as error:
                stderr.write((repr(error) + '\n').encode())
                code = 127
        (out / (label + '.exit')).write_text(str(code) + '\n')
        if code:
            raise subprocess.CalledProcessError(code, argv)
        return (out / (label + '.stdout')).read_bytes()

    try:
        before = run('source.before', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
        require(not run('source.before.status',
                        ['git', 'status', '--porcelain', '--untracked-files=all']),
                'source tree is dirty')
        roster = load_roster()
        require(manifest(ROOT, roster) ==
                (ROOT / 'native/cpu4_v1_source_members.sha256').read_text(),
                'CPU4 source closure changed')
        selected = units(roster, normal_overlay(OVERLAY))
        require((os.statvfs(ROOT).f_flag & os.ST_RDONLY) and
                (os.statvfs(checkpoint).f_flag & os.ST_RDONLY) and
                (os.statvfs(reference).f_flag & os.ST_RDONLY),
                'source, B250 C2 and uninterrupted reference must be read-only')
        require(len(expected_b250) == 64 and len(expected_reference) == 64 and
                checkpoint.stat().st_size == 25504 and
                sha(checkpoint) == expected_b250 and
                expected_b250 != EXPECTED_B125_SHA256 and
                sha(reference) == expected_reference,
                'sealed B250 or trained reference identity changed')
        outer_receipt(prior, expected_outer_seal)
        prior_receipt(prior, expected_prior_seal, {
            'two-token.final-b250.1.c2': expected_b250,
            'second-eos.run1.stdout': expected_reference})
        validate_reference(reference.read_bytes())
        for mode in ('first-eos', 'second-eos', 'two-token'):
            other = prior / ('combined/' + mode + '.final-b250.1.c2')
            require(other.stat().st_size == 25504 and sha(other) == expected_b250,
                    'prior modes use different B250 checkpoints')
        (out / 'prior-pins.json').write_text(json.dumps({
            'source': before.decode().splitlines(),
            'prior_outer_sha256': expected_outer_seal,
            'prior_combined_sha256': expected_prior_seal,
            'prior_b250_sha256': expected_b250,
            'prior_reference_sha256': expected_reference,
            'native_features': selected[[u['source'] for u in selected].index(
                'src/eshkol_transformer/g3c4_model_owner.c')]['defines']},
            sort_keys=True, indent=2) + '\n')
        (out / 'selected-native-inputs.json').write_text(
            json.dumps(selected, sort_keys=True, indent=2) + '\n')
        train, unused_heldout = write_resources(out)
        cache, objects = out / 'cache', out / 'objects'
        cache.mkdir()
        objects.mkdir()
        flags = roster['cflags'] + ['-O2']
        for directory in roster['native_include_dirs']:
            flags += ['-I', str(ROOT / directory)]
        flags += ['-I', str(compiler_source / 'inc')]
        flags += ['-D' + feature for feature in roster['common_defines']]
        built = []
        for index, unit in enumerate(selected):
            stem = f'{index:02d}-' + unit['source'].replace('/', '_')
            target = objects / (stem + '.o')
            run('native.' + stem, ['timeout', '--foreground', '--signal=TERM',
                '--kill-after=5s', '120s', cc, *flags,
                *['-D' + feature for feature in unit['defines']], '-MMD', '-MF',
                objects / (stem + '.d'), '-c', ROOT / unit['source'], '-o', target])
            require(target.stat().st_size > 0 and
                    (objects / (stem + '.d')).stat().st_size > 0,
                    'native object or depfile missing')
            built.append(target)
        archive = out / 'libcpu4_p3_fresh.a'
        run('archive', ['ar', 'rcsD', archive, *built])
        run('archive.imports', ['nm', '-u', '--format=posix', archive])
        run('archive.symbols', ['nm', '-a', '--format=posix', archive])
        (out / 'cxx-link').write_text('#!/usr/bin/env bash\nexec ' +
                                      shlex.quote(str(cxx)) + ' "$@"\n')
        (out / 'cxx-link').chmod(0o500)
        env = dict(os.environ)
        for key in ('ESHKOL_PATH', 'ESHKOL_JIT_CACHE_DIR'):
            env.pop(key, None)
        env.update(ESHKOL_JIT_CACHE='0', XDG_CACHE_HOME=str(cache),
                   PYTHONDONTWRITEBYTECODE='1', ESHKOL_ARENA_POISON='1',
                   ESHKOL_CXX_COMPILER=str(out / 'cxx-link'),
                   ESHKOL_LIB_DIR=str(ROOT / 'lib'))
        includes = [arg for directory in roster['source_include_dirs']
                    for arg in ('-I', str(ROOT / directory))]
        raw = out / 'runtime.raw.o'
        run('compile', ['timeout', '--foreground', '--signal=TERM',
            '--kill-after=10s', '1200s', runner, '--strict-types', '--no-stdlib',
            '--optimize', '0', *includes, '-L', out, '--lib', 'cpu4_p3_fresh',
            '--emit-object', '--dump-ir', '--emit-depfile', out / 'runtime.d',
            CALLER, '-o', raw], env)
        for suffix in ('', '.ll', '.bc'):
            require(Path(str(raw) + suffix).stat().st_size > 0,
                    'compiler artifact missing')
        run('object.imports', ['nm', '-u', '--format=posix', raw])
        run('link', ['timeout', '--foreground', '--signal=TERM',
            '--kill-after=10s', '120s', runner, '--no-stdlib', '-L', out,
            '--lib', 'cpu4_p3_fresh', raw, '-o', out / 'runtime'], env)
        run('elf.imports', ['nm', '-u', '--format=posix', out / 'runtime'])
        expected_file = out / 'expected-b250.sha256.bin'
        expected_file.write_bytes(bytes.fromhex(expected_b250))
        wire, values = out / 'p3.c4grcv1', out / 'p3.values.f32le'
        args = [train, checkpoint, expected_file, wire, values]
        require(sha(checkpoint) == expected_b250 and
                sha(reference) == expected_reference,
                'prior inputs changed before producer')
        produced = run('producer.run', ['timeout', '--foreground', '--signal=TERM',
            '--kill-after=5s', '900s', out / 'runtime', *args, 'produce'], env)
        require(not (out / 'producer.run.stderr').read_bytes() and
                produced.splitlines()[-1].startswith(b'CPU4-P3-FRESH-PASS produce ') and
                produced.splitlines().count(b'CPU4-P3-PRODUCED 3 1 41 0 0') == 1,
                'fresh trained P3 producer failed')
        wire_bytes, value_bytes = wire.read_bytes(), values.read_bytes()
        validate_wire(wire_bytes, value_bytes, bytes.fromhex(expected_b250))
        producer_pins = (sha(wire), sha(values))
        (out / 'producer-file-pins.json').write_text(json.dumps({
            'wire_sha256': producer_pins[0], 'values_sha256': producer_pins[1],
            'wire_bytes': 400, 'values_bytes': 4768}, indent=2) + '\n')
        require(sha(checkpoint) == expected_b250 and
                sha(reference) == expected_reference,
                'prior inputs changed after producer')
        resumed = run('consumer.run', ['timeout', '--foreground', '--signal=TERM',
            '--kill-after=5s', '900s', out / 'runtime', *args, 'resume'], env)
        require(not (out / 'consumer.run.stderr').read_bytes(),
                'fresh FILE consumer diagnostics')
        validate_resumed(resumed, reference.read_bytes())
        require((sha(wire), sha(values)) == producer_pins and
                sha(checkpoint) == expected_b250 and
                sha(reference) == expected_reference,
                'persisted FILE pair or prior inputs changed after consumer')
        program = 0
    except subprocess.CalledProcessError as error:
        program = error.returncode
    except Exception as error:
        (out / 'driver-error.txt').write_text(repr(error) + '\n')
    finally:
        try:
            inventory_depfiles(out)
            after = run('source.after', ['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'])
            require(before is not None and before == after,
                    'source revision changed')
            require(not run('source.after.status',
                            ['git', 'status', '--porcelain', '--untracked-files=all']),
                    'source tree changed after gate')
        except Exception as error:
            custody = 74
            (out / 'custody-error.txt').write_text(repr(error) + '\n')
        (out / 'driver-program.exit').write_text(str(program) + '\n')
        (out / 'driver-custody.exit').write_text(str(custody) + '\n')
        hashes(out, out / 'SHA256SUMS')
        (out / 'SHA256SUMS.sha256').write_text(
            hashlib.sha256((out / 'SHA256SUMS').read_bytes()).hexdigest() +
            '  SHA256SUMS\n')
    return custody or program


if __name__ == '__main__':
    if len(sys.argv) != 11:
        raise SystemExit('driver OUT PRIOR OUTER_SEAL COMBINED_SEAL B250_SHA REFERENCE_SHA RUNNER CC CXX COMPILER_SOURCE')
    raise SystemExit(gate(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
                          sys.argv[4], sys.argv[5], sys.argv[6],
                          Path(sys.argv[7]), Path(sys.argv[8]),
                          Path(sys.argv[9]), Path(sys.argv[10])))
