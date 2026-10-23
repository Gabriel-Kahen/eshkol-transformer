"""Prepare exact sequential commands; deliberately never execute a compiler.

Root dispatches the native unit first, and the separate tiny-AOT gate only after
independent review and heavy-lock custody. Fresh output only, normal glibc only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
from tests.cpuv1_resource.generate_scheme import generate,ROOT
from tests.cpu4_v1_aggregate.closure import load_roster


def prepare(out,cc,cxx,eshkol_source,eshkol_build):
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh evidence output required')
    out.mkdir(mode=0o700);(out/'objects').mkdir();(out/'cache').mkdir()
    generation=generate(out/'generated')
    roster=load_roster()
    overlay=json.loads((ROOT/'tests/cpuv1_resource/native_overlay.json').read_text())
    selected=roster['translation_units']+overlay['additional_translation_units']
    include=sum((['-I',str(ROOT/d)] for d in roster['native_include_dirs']),[])
    include+=['-I',str(Path(eshkol_source)/'inc'),'-I',str(Path(eshkol_source)/'lib/core')]
    defs=['-D'+s for s in overlay['producer_defines']]
    common=roster['cflags']+['-O2']+include+defs+['-D'+s for s in roster['common_defines']]
    commands=[];objects=[];inputs=set()
    for unit in selected+[
        {'source':'native/cpuv1_resource_events.c','defines':[]},
        {'source':'native/cpuv1_resource_collect.c','defines':[]}]:
        source=unit['source'];obj=out/'objects'/(source.replace('/','_')+'.o')
        local=['-D'+s for s in unit['defines']+overlay['per_tu_defines'].get(source,[])]
        commands.append([cc]+common+local+['-MMD','-MF',str(obj.with_suffix('.d')),'-c',str(ROOT/source),'-o',str(obj)])
        objects.append(str(obj));inputs.add(source)
    cppflags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-fPIC','-fvisibility=hidden']+include+defs
    cpp=out/'objects/resource_runtime.o'
    commands.append([cxx]+cppflags+['-MMD','-MF',str(cpp.with_suffix('.d')),'-c',str(ROOT/'native/cpuv1_resource_runtime.cpp'),'-o',str(cpp)])
    objects.append(str(cpp));inputs.add('native/cpuv1_resource_runtime.cpp')
    archive=out/'libcpuv1_resource.a'
    commands.append(['ar','rcsD',str(archive)]+objects)
    unit=out/'producer-unit'
    native=commands+[[cxx]+cppflags+['-DET_F32_TENSOR_TESTING',str(ROOT/'tests/cpuv1_resource/producer_unit.cpp'),
        str(archive),str(Path(eshkol_build)/'libeshkol-runtime.a'),'-ldl','-lpthread','-lm','-o',str(unit)],
        ['timeout','--foreground','--signal=TERM','--kill-after=5s','60s',str(unit)]]
    aot=['env','-u','ESHKOL_PATH','-u','ESHKOL_JIT_CACHE_DIR','ESHKOL_JIT_CACHE=0',
         'XDG_CACHE_HOME='+str(out/'cache'),'ESHKOL_CXX_COMPILER='+cxx,'ESHKOL_LIB_DIR='+str(ROOT/'lib'),
         'timeout','--foreground','--signal=TERM','--kill-after=5s','1200s',str(Path(eshkol_build)/'eshkol-run'),
         '--dump-ir','--strict-types','--optimize','0','--no-stdlib']
    for directory in roster['source_include_dirs']:aot+=['-I',str(ROOT/directory)]
    aot+=['-L',str(out),'--lib','cpuv1_resource',str(out/'generated/tiny.esk'),'-o',str(out/'tiny')]
    # Tiny execution is a distinct gate AFTER emitted-IR allocation-free proof.
    run=['timeout','--foreground','--signal=TERM','--kill-after=5s','120s',str(out/'tiny')]
    plans={'native-unit':native,'tiny-compile':[aot],'tiny-run-after-ir-review':[run,run]}
    for name,plan in plans.items():
        (out/(name+'.commands.json')).write_text(json.dumps(plan,indent=2)+'\n')
        script='#!/usr/bin/env bash\nset -euo pipefail\n'
        for i,cmd in enumerate(plan):script+=shlex.join(cmd)+' >'+shlex.quote(str(out/f'{name}-{i}.stdout'))+' 2>'+shlex.quote(str(out/f'{name}-{i}.stderr'))+'\n'
        (out/(name+'.sh')).write_text(script)
    inputs|={p.relative_to(ROOT).as_posix() for p in (ROOT/'tests/cpuv1_resource').glob('*') if p.is_file()}
    inputs|={p.relative_to(ROOT).as_posix() for p in (ROOT/'native').glob('cpuv1_resource*') if p.is_file()}
    # Depfiles must later supplement this lexical source receipt.
    receipt={'schema':'cpuv1-stage-a-prepared-recipes-v1','execution':'NOT RUN',
        'native_units':len(selected),'generation':generation,
        'tool_arguments':dict(cc=cc,cxx=cxx,eshkol_source=eshkol_source,eshkol_build=eshkol_build),
        'inputs':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(inputs)}}
    (out/'recipes.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output')
    for name in ['cc','cxx','eshkol-source','eshkol-build']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();prepare(args.output,args.cc,args.cxx,args.eshkol_source,args.eshkol_build)
