#!/usr/bin/env python3
"""Explicit A2/PA4 same-process paired timing and separate-process RSS evidence."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

if __package__:
    from . import attention_benchmark as base
else:
    import attention_benchmark as base
codec, common = base.codec, base.common
require, keys, ROOT = base.require, base.keys, base.ROOT
DEFINITION = 'eshkol-attention-comparison-definition'
REPORT = 'eshkol-attention-comparison-report'
PROVIDERS = ['a2.cpu-f32.serial', 'pa4.cpu-f32.row-scratch.candidate']
BASELINE_COMMIT = 'ee24ef461ef7e313c36c6c6f97ba4f3d8155bac7'
BASELINE_SHA256 = 'fdb4ca172ca7aa282e787ff42b2998592c8b77a30f67d35680f7e38b2f550b02'
IMAGE = 'sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99'
SOURCES = sorted(set(base.SOURCES + ['benchmarks/attention_compare_v1.c',
    'tests/b0/attention_compare.py', 'include/eshkol_transformer/pa4_attention_candidate.h']))
FLAGS = base.FLAGS
CONTRACT = {
    'id': 'a2-pa4-comparison-v1', 'providers': PROVIDERS,
    'device': 'cpu', 'dtype': 'f32', 'shapes': base.SHAPES,
    'operations': ['forward', 'backward'], 'scopes': ['dispatch', 'invoke'],
    'input_recipe': base.CONTRACT['input_recipe'], 'deterministic': True,
    'measurement': {'clock': 'clock_monotonic', 'timing': 'same-process-paired-batches',
        'invoke': 'prevalidated-immutable-call', 'pair_order': 'AB-BA-alternating',
        'rss': 'separate-selected-provider-process-linux-wait4-ru-maxrss-kib',
        'completion': 'linux-pidfd-poll', 'analysis': 'descriptive-ratios-min-lower-median-max-v1'},
    'warmup': 100, 'iterations': 10000, 'pairs': 12, 'rss_pairs': 4, 'timeout_ms': 30000}


def validate_definition(value):
    keys(value, set(CONTRACT), 'definition')
    adjustable = {'warmup', 'iterations', 'pairs', 'rss_pairs', 'timeout_ms'}
    for name in set(CONTRACT) - adjustable:
        require(codec.canonical_json(value[name]) == codec.canonical_json(CONTRACT[name]),
                'unsupported definition ' + name)
    for name, lower, upper in [('warmup', 0, 100000), ('iterations', 1, 1000000),
            ('pairs', 2, 100), ('rss_pairs', 2, 100), ('timeout_ms', 1, 3600000)]:
        codec._integer(value[name], name, lower, upper)
    require(value['pairs'] % 2 == value['rss_pairs'] % 2 == 0, 'unbalanced pairs')
    return value


def encode_definition(value):
    return codec._encode(DEFINITION, value, validate_definition)


def load_definition(path):
    return codec._load(path, DEFINITION, validate_definition)


def ratio(a, b):
    return format((Decimal(a)/Decimal(b)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN), 'f')


def interval(values):
    values = sorted(values)
    return {'min': values[0], 'median': values[(len(values)-1)//2], 'max': values[-1]}


def summarize(samples):
    pairs = [samples[i:i+2] for i in range(0,len(samples),2)]
    by_pair = [{s['provider']: s for s in pair} for pair in pairs]
    ratios = [Decimal(p[PROVIDERS[1]]['elapsed_ns'])/Decimal(p[PROVIDERS[0]]['elapsed_ns']) for p in by_pair]
    formatted = lambda xs: {k: format(v.quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN),'f') for k,v in interval(xs).items()}
    return {'elapsed_ns': {p: interval([s['elapsed_ns'] for s in samples if s['provider'] == p]) for p in PROVIDERS},
            'candidate_over_baseline': formatted(ratios),
            'ab_candidate_over_baseline': formatted(ratios[::2]),
            'ba_candidate_over_baseline': formatted(ratios[1::2]),
            'total_candidate_over_baseline': ratio(sum(s['elapsed_ns'] for s in samples if s['provider'] == PROVIDERS[1]),
                sum(s['elapsed_ns'] for s in samples if s['provider'] == PROVIDERS[0]))}


def rss_summary(samples):
    return {p:interval([s['process_peak_rss_kib'] for s in samples if s['provider'] == p]) for p in PROVIDERS}


def aggregate(cases):
    # Equal dispatch counts for every row; totals describe this bounded shape mix.
    result = []
    for operation in CONTRACT['operations']:
        for scope in CONTRACT['scopes']:
            samples = [s for c in cases if c['operation'] == operation and c['scope'] == scope for s in c['samples']]
            totals = {p:sum(s['elapsed_ns'] for s in samples if s['provider'] == p) for p in PROVIDERS}
            result.append({'operation':operation,'scope':scope,'total_elapsed_ns':totals,
                'candidate_over_baseline':ratio(totals[PROVIDERS[1]],totals[PROVIDERS[0]])})
    return result


def witness(bits, shape, operation):
    _, digest = base.native_output(json.dumps({'version':1, 'provider':PROVIDERS[0],
        'dispatch_elapsed_ns':1, 'output_bits':bits}).encode(), shape, operation)
    return digest


def validate_samples(samples, count, shape, operation, expected_bits, rss=False):
    require(type(samples) is list and len(samples) == count*2, 'sample count mismatch')
    digest = witness(expected_bits, shape, operation)
    for index, sample in enumerate(samples):
        names = {'pair','slot','provider','elapsed_ns','output_sha256'}
        if rss:
            names |= {'process_elapsed_ns','process_peak_rss_kib'}
        keys(sample, names, 'sample')
        for name in ['pair','slot']:
            codec._integer(sample[name], name, 0)
        pair, slot = divmod(index, 2)
        require(sample['pair'] == pair and sample['slot'] == slot and
                sample['provider'] == PROVIDERS[(pair % 2)^slot], 'pair/provider order mismatch')
        require(sample['output_sha256'] == digest, 'output witness mismatch')
        for name in ['elapsed_ns'] + (['process_elapsed_ns','process_peak_rss_kib'] if rss else []):
            codec._integer(sample[name], name, 1)
        if rss:
            require(sample['process_elapsed_ns'] >= sample['elapsed_ns'], 'batch exceeds process time')
    return digest


def native_output(raw, shape, operation, pairs, selected=None):
    try:
        value = json.loads(raw, object_pairs_hook=codec._pairs, parse_float=codec._forbid_float,
                           parse_constant=codec._forbid_constant)
    except (ValueError, UnicodeError) as error:
        raise codec.BenchmarkFormatError('malformed native output') from error
    keys(value, {'version','samples'}, 'native output')
    require(type(value['version']) is int and value['version'] == 1, 'native version')
    require(type(value['samples']) is list and len(value['samples']) == (pairs*2 if selected is None else 1), 'native sample count')
    result, expected_bits = [], None
    for index, sample in enumerate(value['samples']):
        keys(sample, {'pair','slot','provider','elapsed_ns','output_bits'}, 'native sample')
        pair, slot = divmod(index,2) if selected is None else (0,0)
        p = (pair % 2)^slot if selected is None else selected
        require(type(sample['pair']) is int and type(sample['slot']) is int and
                sample['pair'] == pair and sample['slot'] == slot and sample['provider'] == PROVIDERS[p], 'native order/provider')
        elapsed = codec._integer(sample['elapsed_ns'],'elapsed_ns',1)
        bits = sample['output_bits']
        digest = witness(bits, shape, operation)
        if expected_bits is None:
            expected_bits = bits
        require(bits == expected_bits, 'native provider/repetition bit parity mismatch')
        result.append({'pair':pair, 'slot':slot, 'provider':PROVIDERS[p], 'elapsed_ns':elapsed, 'output_sha256':digest})
    return result, expected_bits


def support_status(environment, compiler, image):
    release = environment['os_release']
    return ('supported-native-component' if image == IMAGE and
            re.search(r'^ID=ubuntu$', release, re.M) and
            re.search(r'^VERSION_ID="22\.04"$', release, re.M) and
            re.search(r'clang version 21\.1\.8(?:\s|$)', compiler['version']) and
            environment['architecture'] == 'x86_64' else 'compatibility-only')


def validate_report(value):
    keys(value, {'stable','volatile'}, 'report')
    stable, volatile = value['stable'], value['volatile']
    keys(stable, {'definition','definition_sha256','project','sources','compiler','binaries','baseline','container_image'}, 'stable')
    definition = validate_definition(stable['definition'])
    require(stable['definition_sha256'] == hashlib.sha256(encode_definition(definition)).hexdigest(), 'definition digest')
    require(stable['baseline'] == {'commit':BASELINE_COMMIT,'source_sha256':BASELINE_SHA256}, 'baseline identity')
    require(stable['container_image'] in [None, IMAGE], 'unrecognized image provenance')
    keys(stable['project'], {'commit','dirty'}, 'project')
    require(re.fullmatch('[0-9a-f]{40}',codec._string(stable['project']['commit'],'commit')) is not None and type(stable['project']['dirty']) is bool, 'project identity')
    keys(stable['sources'], set(SOURCES), 'sources')
    keys(stable['binaries'], {'target','helper','baseline_object','candidate_object'}, 'binaries')
    for digest in [*stable['sources'].values(), *stable['binaries'].values()]:
        codec._sha256(digest,'digest')
    keys(stable['compiler'], {'path','sha256','version','flags'}, 'compiler')
    for name in ['path','version']:
        codec._string(stable['compiler'][name],name,nonempty=True)
    codec._sha256(stable['compiler']['sha256'],'compiler digest')
    require(stable['compiler']['flags'] == FLAGS,'compiler flags')
    keys(volatile, {'observed_at_utc','support_status','environment','cases','aggregate'}, 'volatile')
    timestamp = codec._string(volatile['observed_at_utc'],'timestamp')
    require(codec._UTC_TIMESTAMP.fullmatch(timestamp) is not None,'timestamp format')
    try:
        datetime.fromisoformat(timestamp[:-1]+'+00:00')
    except ValueError as error:
        raise codec.BenchmarkFormatError('timestamp invalid') from error
    env = volatile['environment']
    keys(env, {'architecture','kernel','cpu_model','logical_cpus','os_release','cpu_affinity'}, 'environment')
    for name in ['architecture','kernel','cpu_model','os_release']:
        codec._string(env[name],name,nonempty=True)
    codec._integer(env['logical_cpus'],'logical_cpus',1)
    affinity = codec._array(env['cpu_affinity'],'cpu_affinity')
    require(bool(affinity),'empty affinity')
    for cpu in affinity:
        codec._integer(cpu,'cpu',0)
    require(affinity == sorted(set(affinity)),'invalid affinity')
    require(volatile['support_status'] == support_status(env,stable['compiler'],stable['container_image']), 'support label mismatch')
    cases = codec._array(volatile['cases'],'cases')
    identities = [(s,op,scope) for s in base.SHAPES for op in CONTRACT['operations'] for scope in CONTRACT['scopes']]
    require(len(cases) == len(identities),'case count')
    witnesses = {}
    for case,(shape,op,scope) in zip(cases,identities):
        keys(case, {'shape','operation','scope','output_bits','output_sha256','samples','summary','paired_process','rss_samples','rss_summary'},'case')
        require(codec.canonical_json(case['shape']) == codec.canonical_json(shape) and case['operation'] == op and case['scope'] == scope,'case order')
        digest = validate_samples(case['samples'],definition['pairs'],shape,op,case['output_bits'])
        require(case['output_sha256'] == digest,'case digest')
        require(witnesses.setdefault((tuple(shape),op),digest) == digest,'scope parity mismatch')
        validate_samples(case['rss_samples'],definition['rss_pairs'],shape,op,case['output_bits'],True)
        require(codec.canonical_json(case['rss_summary']) == codec.canonical_json(rss_summary(case['rss_samples'])),'RSS summary mismatch')
        require(codec.canonical_json(case['summary']) == codec.canonical_json(summarize(case['samples'])),'summary mismatch')
        keys(case['paired_process'], {'elapsed_ns','peak_rss_kib'},'paired process')
        for name in case['paired_process']:
            codec._integer(case['paired_process'][name],name,1)
        require(case['paired_process']['elapsed_ns'] >= sum(s['elapsed_ns'] for s in case['samples']),'timing exceeds process')
    require(codec.canonical_json(volatile['aggregate']) == codec.canonical_json(aggregate(cases)),'aggregate mismatch')
    return value


def encode_report(value):
    return codec._encode(REPORT,value,validate_report)


def load_report(path):
    return codec._load(path,REPORT,validate_report)


def build(compiler, directory):
    path = shutil.which(compiler)
    require(path is not None,'compiler unavailable')
    path = Path(path).resolve()
    identity = {'path':str(path),'sha256':common.sha256_file(path),'version':subprocess.check_output([str(path),'--version'],text=True).strip(),'flags':FLAGS}
    directory.mkdir(parents=True,exist_ok=True)
    frozen = subprocess.check_output(['git','show',BASELINE_COMMIT+':native/a2_attention_provider.c'],cwd=ROOT)
    require(hashlib.sha256(frozen).hexdigest() == BASELINE_SHA256,'frozen baseline source mismatch')
    baseline = directory/'baseline'
    baseline.mkdir(exist_ok=True)
    source = baseline/'a2_attention_provider.c'
    source.write_bytes(frozen)
    objects = [directory/'baseline.o',directory/'candidate.o']
    for src,obj,extra in [(source,objects[0],[]),(ROOT/'native/a2_attention_provider.c',objects[1],['-DET_PA4_ATTENTION_CANDIDATE'])]:
        subprocess.run([str(path),*FLAGS,*extra,'-I',str(ROOT/'include'),'-c',str(src),'-o',str(obj)],check=True)
    target,helper = directory/'attention-compare-v1',directory/'measure-linux-v1'
    subprocess.run([str(path),*FLAGS,'-I',str(ROOT/'include'),str(ROOT/'benchmarks/attention_compare_v1.c'),str(ROOT/'native/kernel_abi.c'),*[str(p) for p in objects],'-lm','-o',str(target)],check=True)
    subprocess.run([str(path),*FLAGS,str(ROOT/'benchmarks/measure_linux.c'),'-o',str(helper)],check=True)
    return target,helper,identity,dict(zip(['target','helper','baseline_object','candidate_object'],[common.sha256_file(p) for p in [target,helper,*objects]]))


def measure(helper,target,row,op,scope,definition,directory,selected=None):
    out,err = directory/'out',directory/'err'
    mode = 'paired' if selected is None else ['a2','pa4'][selected]
    argv = [str(helper),str(definition['timeout_ms']),str(out),str(err),'--',str(target),str(row),op,scope,
            str(definition['warmup']),str(definition['iterations']),str(definition['pairs']),mode]
    result = subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=definition['timeout_ms']/1000+5)
    require(result.returncode == 0 and not result.stderr,'helper failed: '+result.stderr)
    metrics = common.parse_metrics(result.stdout)
    require(not metrics['timed_out'] and metrics['exit_code'] == metrics['term_signal'] == 0 and err.read_bytes() == b'','native execution failed')
    samples,bits = native_output(out.read_bytes(),base.SHAPES[row],op,definition['pairs'],selected)
    return samples,bits,{'elapsed_ns':metrics['elapsed_ns'],'peak_rss_kib':metrics['peak_rss_kib']}


def run(definition_path,report_path,compiler,directory,image):
    require(platform.system() == 'Linux','Linux required')
    require(image in [None,IMAGE],'unknown container image')
    definition = load_definition(definition_path)
    sources = {p:common.sha256_file(ROOT/p) for p in SOURCES}
    target,helper,cc,binaries = build(compiler,directory)
    cases = []
    with tempfile.TemporaryDirectory(prefix='attention-compare.',dir=directory) as temp:
        for row,shape in enumerate(base.SHAPES):
            for op in CONTRACT['operations']:
                for scope in CONTRACT['scopes']:
                    samples,bits,process = measure(helper,target,row,op,scope,definition,Path(temp))
                    rss = []
                    for pair in range(definition['rss_pairs']):
                        for slot in range(2):
                            p = (pair % 2)^slot
                            measured,other_bits,metrics = measure(helper,target,row,op,scope,definition,Path(temp),p)
                            require(bits == other_bits,'separate-process parity mismatch')
                            rss.append({**measured[0],'pair':pair,'slot':slot,'process_elapsed_ns':metrics['elapsed_ns'],'process_peak_rss_kib':metrics['peak_rss_kib']})
                    cases.append({'shape':shape,'operation':op,'scope':scope,'output_bits':bits,'output_sha256':witness(bits,shape,op),
                        'samples':samples,'summary':summarize(samples),'paired_process':process,'rss_samples':rss,'rss_summary':rss_summary(rss)})
    require(sources == {p:common.sha256_file(ROOT/p) for p in SOURCES},'sources changed')
    require(cc['sha256'] == common.sha256_file(Path(cc['path'])),'compiler changed')
    require(binaries == dict(zip(binaries,[common.sha256_file(p) for p in [target,helper,directory/'baseline.o',directory/'candidate.o']])),'binaries changed')
    env = {'architecture':platform.machine(),'kernel':platform.release(),'cpu_model':common.cpu_model(),
        'logical_cpus':os.cpu_count(),'os_release':Path('/etc/os-release').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0))}
    report = {'stable':{'definition':definition,'definition_sha256':hashlib.sha256(encode_definition(definition)).hexdigest(),
        'project':{'commit':common.git_output('rev-parse','HEAD'),'dirty':bool(common.git_output('status','--porcelain','--untracked-files=all'))},
        'sources':sources,'compiler':cc,'binaries':binaries,'baseline':{'commit':BASELINE_COMMIT,'source_sha256':BASELINE_SHA256},'container_image':image},
        'volatile':{'observed_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'),
        'support_status':support_status(env,cc,image),'environment':env,'cases':cases,'aggregate':aggregate(cases)}}
    common.write_atomic(report_path,encode_report(report))
    print(f'attention comparison: PASS (40 cases; {report["volatile"]["support_status"]}); report: {report_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    running = commands.add_parser('run')
    running.add_argument('definition',type=Path)
    running.add_argument('report',type=Path)
    running.add_argument('--cc',default='clang')
    running.add_argument('--build-dir',type=Path,default=ROOT/'build/benchmarks/attention-compare-v1')
    running.add_argument('--container-image')
    verifying = commands.add_parser('verify')
    verifying.add_argument('definition',type=Path)
    verifying.add_argument('report',type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'run':
            run(args.definition.resolve(),args.report.resolve(),args.cc,args.build_dir.resolve(),args.container_image)
        else:
            require(load_definition(args.definition) == load_report(args.report)['stable']['definition'],'definition mismatch')
            print('attention comparison report: PASS')
    except (codec.BenchmarkFormatError,common.BenchmarkRunError,OSError,subprocess.SubprocessError) as error:
        print(f'error: {error}',file=os.sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
