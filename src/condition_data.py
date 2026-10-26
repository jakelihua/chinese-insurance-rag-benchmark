"""Traceable, provisional waiting-period rules and controlled paired cases."""
import argparse
import hashlib
import itertools
import json
import unicodedata
from pathlib import Path

import fitz
import numpy as np

from experiment import write_json

POLICIES = {
    'pa_p0781': dict(title='平安关爱相伴团体医疗保险（B款）', insurer='平安', page=4,
                    start='2.2等待期', end='2.3保险责任', days=[90], accident_exempt=True,
                    approval_exempt=False, benefit='合同定义的特定疾病或轻症疾病医疗',
                    scope='仅研究该条约定的特定或轻症疾病；事故原因按条款意外定义。'),
    'pa_h865': dict(title='平安员工及家属住院医疗保险', insurer='平安', page=3,
                   start='等待期除另有约定外', end='如果在等待期后发生保险事故', days=[30],
                   accident_exempt=True, approval_exempt=True, benefit='一般住院医疗',
                   scope='仅研究一般住院医疗；没有其他特别约定。'),
    'cpic_86017': dict(title='太保互联网C款门急诊医疗保险', insurer='太平洋', page=4,
                      start='2.3等待期', end='2.4免赔额', days=[7,15,25],
                      accident_exempt=False, approval_exempt=True, benefit='互联网药品费用医疗',
                      scope='仅研究2.5.2互联网药品费用责任，不混用2.5.3意外门急诊规则。'),
    'cpic_82859': dict(title='太保互联网个人一年期住院E款医疗保险', insurer='太平洋', page=3,
                      start='2.3等待期', end='2.4免赔额', days=[30], accident_exempt=True,
                      approval_exempt=True, benefit='一般住院医疗',
                      scope='仅研究等待期排除，不代表满足全部赔付条件。'),
    'tk_group_2018': dict(title='泰康悦享环球（尊享版）高端医疗保险', insurer='泰康', page=4,
                         start='2.5等待期', end='2.6预授权', days=[30],
                         accident_exempt=True, approval_exempt=False, benefit='非精神心理疾病的一般住院医疗',
                         scope='首次或非连续投保、非精神心理疾病的一般住院；不研究按月计的其他责任。'),
}
TASK_CONTEXT = ('仅判断给定版本条款中的等待期排除，不判断实际是否赔付。假定其他承保和责任条件均满足，'
                '没有未披露的特别约定。疾病起始、首次确诊、治疗和医疗费用都发生在所述同一保单日，'
                '此前无相关症状、诊疗或费用。保单第1日为生效首日。')


def compact(s):
    return ''.join(unicodedata.normalize('NFKC', s).split())


def source_anchor(pdf, policy):
    page = pdf[policy['page'] - 1]
    chars, boxes = [], []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                for char in span['chars']:
                    for c in unicodedata.normalize('NFKC', char['c']):
                        if not c.isspace():
                            chars.append(c)
                            boxes.append(char['bbox'])
    normalized = ''.join(chars)
    start = normalized.index(compact(policy['start']))
    end = normalized.index(compact(policy['end']), start + len(compact(policy['start'])))
    text = normalized[start:end]
    rectangle = fitz.Rect(boxes[start])
    for box in boxes[start + 1:end]:
        rectangle |= fitz.Rect(box)
    anchor = {'pdf_page_1based': policy['page'], 'normalized_start': start, 'normalized_end': end,
              'normalization': 'NFKC + remove whitespace, rawdict character order',
              'normalized_span_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'bbox_pdf_points': list(rectangle), 'page_size_pdf_points': list(page.rect)}
    return anchor, text


def decision(frame):
    policy = POLICIES[frame['policy_id']]
    if frame['approved_waiver'] and policy['approval_exempt']:
        return 'WAIT_NOT_EXCLUDED'
    if frame['cause'] == 'accident' and policy['accident_exempt']:
        return 'WAIT_NOT_EXCLUDED'
    return 'WAIT_EXCLUDED' if frame['policy_day'] <= frame['plan_wait_days'] else 'WAIT_NOT_EXCLUDED'


def domains(frame):
    policy = POLICIES[frame['policy_id']]
    result = {'policy_day': [5,10,20,45,100], 'plan_wait_days': policy['days']}
    result['cause'] = ['illness', 'accident'] if policy['accident_exempt'] else ['illness']
    result['approved_waiver'] = [False,True] if policy['approval_exempt'] and frame['enrollment']=='new' else [False]
    return result


def pivotal_change(frame):
    original = decision(frame)
    # Select only an actual outcome-changing condition; preserve counterexample witnesses.
    for field in ['approved_waiver','plan_wait_days','cause','policy_day']:
        for value in domains(frame)[field]:
            changed = dict(frame, **{field:value})
            if decision(changed) != original:
                return field, changed
    raise ValueError('No pivotal condition in seed frame')


def question(frame, missing=None, surface=False):
    policy = POLICIES[frame['policy_id']]
    words = [f"按《{policy['title']}》这份条款，", '我是首次投保，' if frame['enrollment']=='first' else '我是重新投保，']
    if missing != 'policy_day':
        words.append(f"保单生效第{frame['policy_day']}天，")
    else:
        words.append('具体生效到第几天还不清楚，')
    if missing != 'cause':
        words.append('因意外伤害发生约定的医疗事项，' if frame['cause']=='accident' else '因非意外疾病发生约定的医疗事项，')
    else:
        words.append('是否由意外造成还不清楚，')
    words.append(f"问的是{policy['benefit']}责任，")
    if len(policy['days']) > 1:
        words.append('计划约定的等待期天数不清楚，' if missing=='plan_wait_days' else f"计划约定等待期为{frame['plan_wait_days']}天，")
    if frame['enrollment']=='new':
        if missing=='approved_waiver':
            words.append('是否已获保险公司批准免等待期还不清楚，')
        elif frame['approved_waiver']:
            words.append('已获保险公司明确批准免等待期，')
        else:
            words.append('未获保险公司批准免等待期，')
    words.append('是否会因等待期被排除？')
    text = ''.join(words)
    if surface:
        text = text.replace('我是首次投保，','头一次买这份保险，').replace('保单生效第','从保单生效算起第')
        text = text.replace('是否会因等待期被排除？','单看等待期这一条，会被挡住吗？')
    return text


def seed_frames():
    frames = []
    for policy_id, policy in POLICIES.items():
        default = dict(policy_id=policy_id, enrollment='first', cause='illness',
                       policy_day=20, plan_wait_days=policy['days'][-1], approved_waiver=False)
        if policy_id=='cpic_86017':
            changes = [dict(policy_day=10,plan_wait_days=7),dict(policy_day=10,plan_wait_days=25),
                       dict(policy_day=20,plan_wait_days=15),dict(policy_day=10,plan_wait_days=25,enrollment='new',approved_waiver=True)]
        else:
            changes = [dict(policy_day=20),dict(policy_day=45),dict(policy_day=20,cause='accident'),
                       dict(policy_day=20,enrollment='new',approved_waiver=True) if policy['approval_exempt'] else dict(policy_day=100)]
        frames += [dict(default,**change) for change in changes]
    return frames


def build(root, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'source_manifest.json').read_text())
    sources, evidence, policies = [], {}, []
    by_id = {row['id']: row for row in manifest if row['status']=='downloaded'}
    for policy_id, policy in POLICIES.items():
        row = by_id[policy_id]
        path = root/'raw'/(policy_id+'.pdf')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with fitz.open(path) as pdf:
            anchor, text = source_anchor(pdf,policy)
            (root/'previews'/(policy_id+'_anchor.png')).parent.mkdir(exist_ok=True)
            rect=fitz.Rect(anchor['bbox_pdf_points'])+(-3,-3,3,3)
            pdf[policy['page']-1].get_pixmap(matrix=fitz.Matrix(1.8,1.8),clip=rect).save(root/'previews'/(policy_id+'_anchor.png'))
        sources.append({**row,'title':policy['title'],'document_type':'formal_policy_clause',
                        'license_status':'publicly accessible; no redistribution license verified',
                        'review_status':'assistant extracted + page visually inspected; independent review pending',
                        'anchor':anchor})
        evidence[policy_id]=text
        policies.append({'policy_id':policy_id,'title':policy['title'],'insurer':policy['insurer'],
                         'waiting_days_options':policy['days'],'accident_exemption_in_studied_scope':policy['accident_exempt'],
                         'approved_new_contract_waiver':policy['approval_exempt'],'benefit_scope':policy['benefit'],
                         'scope_note':policy['scope'],'evidence':anchor,
                         'reference_status':'provisional assistant-curated rule; not expert gold'})
    cases, jobs = [], []
    frames=seed_frames()
    for i, frame in enumerate(frames):
        group=f'condition-{i+1:03d}'
        field, changed=pivotal_change(frame)
        label=decision(frame)
        for variant, f, missing in [('complete',frame,None),('surface',frame,None),
                                    ('missing_critical',frame,field),('counterfactual',changed,None)]:
            partial = {k:v for k,v in f.items() if k!=missing}
            witnesses = []
            if missing:
                witnesses=[{'completion':frame,'label':decision(frame)}, {'completion':changed,'label':decision(changed)}]
                assert witnesses[0]['label']!=witnesses[1]['label']
            cases.append({'id':group+'-'+variant,'group_id':group,'variant':variant,
                          'source_id':frame['policy_id'],'task_context':TASK_CONTEXT,
                          'question':question(f,missing,surface=variant=='surface'),
                          'visible_frame':partial,'missing_field':missing,
                          'reference_label':'CLARIFY' if missing else decision(f),
                          'counterexample_witnesses':witnesses,
                          'reference_status':'rule-derived provisional; independent human annotation pending',
                          'question_origin':'controlled synthetic, NOT real user log'})
        required=['policy_id','enrollment','policy_day','cause']
        if len(POLICIES[frame['policy_id']]['days'])>1:required.append('plan_wait_days')
        if frame['enrollment']=='new':required.append('approved_waiver')
        jobs.append({'seed_id':group,'frame':frame,'seed_question':question(frame),
                     'required_fields':required,'source_evidence':evidence[frame['policy_id']]})
    write_json(out/'sources.json',sources)
    write_json(out/'policy_rules.json',policies)
    with (out/'controlled_cases.jsonl').open('w',encoding='utf-8') as f:
        for row in cases:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    # Original evidence remains private and is not redistributed with outputs.
    write_json(root/'generation_inputs.json',{'task_context':TASK_CONTEXT,'jobs':jobs})
    write_json(root/'evidence_private.json',evidence)
    write_json(out/'data_manifest.json',{'date':'2026-10-26','formal_pdfs':len(sources),
        'source_groups':len(set(p['insurer'] for p in POLICIES.values())),
        'seed_groups':len(frames),'case_count':len(cases),'controlled_variants':4,
        'label_status':'provisional rule reference, NOT independently annotated',
        'all_cases_synthetic':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generation_inputs_sha256':hashlib.sha256((root/'generation_inputs.json').read_bytes()).hexdigest(),
        'source_failures':[r for r in manifest if r['status']!='downloaded'],
        'excluded_auxiliary_documents':[r['id'] for r in manifest if r['status']=='downloaded' and r['id'] not in POLICIES]})
    print('Built',len(sources),'formal sources,',len(frames),'groups,',len(cases),'controlled cases.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();build(args.root,args.out)
