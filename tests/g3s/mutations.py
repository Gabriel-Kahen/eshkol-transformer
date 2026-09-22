"""Compile isolated production mutations and require numerical-test rejection.

Every mutant must compile cleanly. A compiler/linker failure is not a killed
mutation. The production checkout and canonical archive are never changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


def replace(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after)


def mutations(source):
    plans = [
        ('binary64_sums', 'float sum=0.0f;', 'double sum=0.0;'),
        ('descending_id_ties', 'id<s->order[j-1]', 'id>s->order[j-1]'),
        ('skip_topk_renormalization', 's->b[i]=s->a[s->order[i]]/sum;', 's->b[i]=s->a[s->order[i]];'),
        ('top_p_before_renormalization', 'cumulative=cumulative+s->b[i];', 'cumulative=cumulative+s->a[s->order[i]];'),
        ('p_one_keep_all', 'if (cumulative>=p)', 'if (p<1.0f && cumulative>=p)'),
        ('strict_top_p', 'if (cumulative>=p)', 'if (cumulative>p)'),
        ('epsilon_top_p', 'if (cumulative>=p)', 'if (cumulative+0x1p-23f>=p)'),
        ('nonstrict_cdf', 'if (cdf>r)', 'if (cdf>=r)'),
        ('last_zero_tail', 'if (s->b[i]>0.0f) last=', 'last='),
        ('wrong_domain', 'UINT64_C(0x4733434154454731)', 'UINT64_C(0x4733434154454730)'),
        ('wrong_lane', '(lanes[0]>>8)', '(lanes[1]>>8)'),
        ('low24_mapping', '(lanes[0]>>8)', '(lanes[0]&0xffffffu)'),
        ('wrong_uniform_scale', '*0x1p-24f', '*0x1p-32f'),
        ('nine_philox_rounds', 'round < 10u', 'round < 9u'),
        ('counter_lane_swap', 'uint32_t c0 = (uint32_t)counter_low;', 'uint32_t c0 = (uint32_t)counter_high;'),
        ('key_lane_swap', 'uint32_t k0 = (uint32_t)key;', 'uint32_t k0 = (uint32_t)(key >> 32);'),
        ('double_counter_advance', 'successor=low+UINT64_C(1)', 'successor=low+UINT64_C(2)'),
        ('missing_counter_carry', 'high+(uint64_t)(successor==0)', 'high+UINT64_C(0)*(uint64_t)(successor==0)'),
        ('greedy_last_tie', 'if (logits[i]>logits[(size_t)*token])', 'if (logits[i]>=logits[(size_t)*token])'),
        ('repair_underflow', 's->a[i]=expf(scaled);', 's->a[i]=expf(scaled); if (s->a[i]==0.0f) s->a[i]=FLT_MIN;'),
        ('singleton_no_draw', 'const uint64_t low=decode_signed_word(state[2]);', 'if (scratch.count==1) { *token=(int64_t)scratch.order[0]; return 1; }\n  const uint64_t low=decode_signed_word(state[2]);'),
    ]
    for name,before,after in plans:
        yield name, replace(source,before,after)
    logit_sort = replace(source, 's->a[id]>s->a[s->order[j-1]]', 'logits[id]>logits[s->order[j-1]]')
    logit_sort = replace(logit_sort, 's->a[id]==s->a[s->order[j-1]]', 'logits[id]==logits[s->order[j-1]]')
    yield 'sort_logits_instead_of_rounded_probability', logit_sort
    yield 'reverse_id_softmax_sum', replace(source,
        '    sum=sum+s->a[i];\n    if (!isfinite(sum)) return 0;\n  }\n  for (size_t i=0;i<256;++i) {\n    s->a[i]=s->a[i]/sum;',
        '  }\n  for (size_t i=256;i>0;--i) sum=sum+s->a[i-1];\n  for (size_t i=0;i<256;++i) {\n    s->a[i]=s->a[i]/sum;')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cc',required=True)
    parser.add_argument('--header-dir',type=Path,required=True)
    parser.add_argument('--k1',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    source=(root/'native/g3s_sampling_provider.c').read_text()
    with tempfile.TemporaryDirectory(prefix='g3s-mutations-') as temp:
        directory=Path(temp)
        flags=[args.cc,'-std=c11','-O2','-Wall','-Wextra','-Werror','-Wpedantic',
               '-ffp-contract=off','-fexcess-precision=standard','-fno-fast-math',
               '-fstack-protector-all','-I'+str(root/'include'),'-I'+str(args.header_dir.resolve())]
        count=0
        for name,mutated in [('baseline',source),*mutations(source)]:
            path=directory/(name+'.c')
            executable=directory/name
            path.write_text(mutated)
            command=flags+['-DG3S_PROVIDER_SOURCE="'+str(path)+'"',str(root/'tests/g3s/numerical.c'),
                           str(args.k1.resolve()),'-lm','-o',str(executable)]
            built=subprocess.run(command,capture_output=True,text=True,timeout=60)
            assert built.returncode==0, (name,'mutation must compile',built.stderr)
            run=subprocess.run([str(executable)],capture_output=True,text=True,timeout=10)
            if name=='baseline':
                assert run.returncode==0 and 'PASS' in run.stdout, ('baseline',run.stdout,run.stderr)
                continue
            assert run.returncode!=0 and 'numerical:' in run.stderr, (name,'SURVIVED',run.stdout,run.stderr)
            count+=1
            print('G3-S mutation rejected: '+name)
        print('G3-S production numerical mutations PASS: %d/%d independently compiled mutants rejected'%(count,count))


if __name__=='__main__':
    main()
