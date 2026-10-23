"""Source branch/profile regression only; no C preprocessor or compiler run."""
from pathlib import Path
import ast,json,re,unittest
from tests.cpu4_v1_aggregate.closure import load_roster
ROOT=Path(__file__).resolve().parents[2]
OWNER='src/eshkol_transformer/g3c4_model_owner.c'
OPTIONAL={'ET_G3C4_LOGITS_RESERVATION_PRIVATE':
             ('et_g3c4_logits_internal','ET_G3C4_LOGITS_KIND'),
          'ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE':('et_g3c4_manual_frame_internal',)}


def source_branches(text,flags):
    """Select #if source text for tests, without macro expansion or C parsing.

    This small Boolean selector is only a guard-placement control. Actual C
    preprocessing, type availability and compiled layout remain NOT RUN here.
    """
    def condition(expression):
        expression=re.sub(r'defined\((\w+)\)',lambda m:str(m[1] in flags),expression)
        expression=expression.replace('&&',' and ').replace('||',' or ')
        expression=re.sub(r'!(?!=)',' not ',expression).strip()
        tree=ast.parse(expression,mode='eval')
        assert all(isinstance(n,(ast.Expression,ast.BoolOp,ast.And,ast.Or,
                                 ast.UnaryOp,ast.Not,ast.Constant)) for n in ast.walk(tree))
        return eval(compile(tree,'<source-guard>','eval'),{'__builtins__':{}})
    stack=[];active=True;output=[]
    for line in text.replace('\\\n',' ').splitlines():
        match=re.match(r'\s*#\s*(ifdef|ifndef|if|else|endif)\b\s*(.*)',line)
        if match:
            directive,value=match.groups()
            if directive in ('ifdef','ifndef','if'):
                chosen=(value in flags) if directive=='ifdef' else (value not in flags) if directive=='ifndef' else condition(value)
                stack.append((active,chosen));active=active and chosen
            elif directive=='else':
                parent,chosen=stack[-1];active=parent and not chosen
            else:active=stack.pop()[0]
        elif active:output.append(line)
    assert not stack
    return '\n'.join(output)


class Profile(unittest.TestCase):
    def setUp(self):
        roster=load_roster();unit=next(u for u in roster['translation_units'] if u['source']==OWNER)
        overlay=json.loads((ROOT/'tests/cpuv1_resource/native_overlay.json').read_text())
        self.flags=set(roster['common_defines']+unit['defines']+overlay['producer_defines']+
                       overlay['per_tu_defines'].get(OWNER,[]))
        self.census=(ROOT/'native/cpuv1_resource_c4.inc').read_text()

    def assert_no_unavailable_references(self,text):
        selected=source_branches(text,self.flags)
        for feature,names in OPTIONAL.items():
            self.assertNotIn(feature,self.flags)
            for name in names:self.assertNotIn(name,selected)
        return selected

    def test_frozen_profile_has_no_unconditional_type_or_enum_reference(self):
        selected=self.assert_no_unavailable_references(self.census)
        self.assertIn('ET_R_C4_NOT_APPLICABLE_KINDS',selected)
        self.assertIn('default:et_cpuv1_test_resource_fail_v1(packet,ET_R_COVERAGE);return;',selected)
        for kind in [0,1,2,3,4,5,6,7,9,10,11]:
            self.assertIn('resource_layout_v1(packet,ET_R_C4,'+str(kind)+',',selected)
        # Positive optional-source branch keeps the REAL owning type/enum;
        # this is a guard test, not permission to widen the execution profile.
        for feature,names in OPTIONAL.items():
            optional=source_branches(self.census,self.flags|{feature})
            for name in names:self.assertIn(name,optional)
        for name in ['et_g3c4_logits_internal','et_g3c4_manual_frame_internal','ET_G3C4_LOGITS_KIND']:
            with self.assertRaises(AssertionError):
                self.assert_no_unavailable_references(self.census+'\n'+name+'\n')

    def test_absence_is_explicit_and_conserved_history_still_rejects(self):
        selected=source_branches(self.census,self.flags)
        for kind in [8,12]:
            self.assertIn('ET_R_C4_COMPILED_KINDS]&=~(UINT64_C(1)<<'+str(kind)+')',selected)
            self.assertIn('ET_R_C4_NOT_APPLICABLE_KINDS]|=UINT64_C(1)<<'+str(kind),selected)
        events=(ROOT/'native/cpuv1_resource_events.c').read_text()
        for field in ['created','freed','allocated_requested','freed_requested',
                      'allocated_usable','freed_usable','failed','bases','requested',
                      'usable','live','pending','prepared','busy','dead',
                      'control_size','control_alignment']:
            self.assertIn('r->'+field,events.split('if ((c4_na&',1)[1].split('for (unsigned d=',1)[0])
        self.assertIn('if ((c4_compiled&',events)
        self.assertIn('c4_compiled|c4_na',events)
        unit=(ROOT/'tests/cpuv1_resource/producer_unit.cpp').read_text()
        self.assertIn('balanced absent-kind event history rejects profile',unit)
        self.assertIn('et_cpuv1_test_resource_mutation_v1(4)',unit)
        # Balanced created/freed and requested/usable totals cannot make an
        # unavailable kind a valid member merely because current bases are0.
        row={'created':1,'freed':1,'allocated_requested':16,'freed_requested':16,
             'allocated_usable':16,'freed_usable':16,'bases':0,'requested':0,'usable':0}
        self.assertEqual(row['created']-row['freed'],row['bases'])
        self.assertEqual(row['allocated_requested']-row['freed_requested'],row['requested'])
        self.assertEqual(row['allocated_usable']-row['freed_usable'],row['usable'])
        self.assertTrue(any(row.values()))

if __name__=='__main__':unittest.main()
