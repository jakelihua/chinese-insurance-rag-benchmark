"""Out-of-sample official insurance clause evaluation, title-masked clause retrieval,
and equal-call-budget baseline experiments.

1. Evaluates the frozen development-set parser and reachability verifier on four
   held-out official Chinese insurance policy PDFs (pa_2011_36, pa_p0254, pa_p0321,
   cpic_93896; 16 seeds, 64 controlled cases with double-completion witnesses).
2. Compares generation and refinement baselines under matched LLM call budgets:
   seed_passthrough (0 calls), ordinary_style (1 call), condition_prompt (1 call),
   structured_json_rewrite (1 call), generic_self_refine (2 calls without symbolic
   parser feedback), and slot_guard_repair (<=2 calls with symbolic parser feedback).
3. Computes template divergence (1 - kappa_2(q', q_seed)) and title-masked clause
   retrieval across lexical (bm25_char2) and dense (bge_small_zh) retrievers.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None
import numpy as np
from scipy.stats import binomtest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import experiment as base
from analyze_v3 import bigram_containment, extract_fields_surface_normalized
from condition_data import POLICIES as DEV_POLICIES, TASK_CONTEXT, compact, source_anchor
from condition_guard import chinese_number
from cross_domain_study import (
    ALL_CROSS_FIELDS,
    CROSS_CLAUSES,
    build_seed_frames as build_cross_seeds,
    cross_decision,
    enumerate_cross_decisions,
    evaluate_cross_outcome,
    extract_cross_slots_v2,
    render_question as render_cross_question,
    should_trigger_slot_guard_repair,
)


def strip_book_title_spoilers(text):
    """Step 1 of controlled synthesis: strip textbook 《》 book-title spoilers and clause-number headers."""
    cleaned = re.sub(r'[《》]', '', text)
    cleaned = re.sub(
        r'(?:请问)?根据(?:条款)?第[一二三四五六七八九十0-9]+条(?:第[一二三四五六七八九十0-9（）()]+(?:款|项))?[，,]?',
        '',
        cleaned,
    )
    return cleaned.strip()


def clean_cross_question(raw, strip_book_titles=False):
    q = ' '.join(raw.strip().splitlines())
    q = re.sub(r'^(?:QUESTION|Question|改写后问题|修正后问题|用户提问|用户问题|提问|问题|改写问句)\s*[:：]\s*', '', q.strip())
    q = q.strip('“”"\' ')
    if strip_book_titles:
        q = strip_book_title_spoilers(q)
    return q


HELDOUT_POLICIES = {
    'pa_2011_36': dict(
        title='平安住院安心医疗保险',
        colloquial_name='平安住院安心医疗险',
        aliases=['平安住院安心医疗险', '平安住院安心险', '住院安心医疗保险', '住院安心医疗险'],
        insurer='平安',
        page=1,
        start='（一）一般住院医疗津贴',
        end='被保险人自本合同生效90日后因初次患癌症，经医院确诊必须住院治疗的，本公司按其实际住院日数乘',
        days=[30, 90],
        accident_exempt=True,
        approval_exempt=True,
        benefit='一般住院医疗津贴（30天等待期）与癌症住院医疗津贴（90天等待期）分项责任',
        benefits={
            'general_hospital': dict(
                label='一般住院医疗津贴',
                wait_days=30,
                accident_exempt=True,
                approval_exempt=True,
            ),
            'cancer_hospital': dict(
                label='癌症住院医疗津贴',
                wait_days=90,
                accident_exempt=False,
                approval_exempt=True,
            ),
        },
        scope='第四条分设两项责任：（一）一般住院医疗津贴等待期30天（意外伤害事故住院无等待期）；（二）癌症住院医疗津贴等待期90天（限初次患癌症确诊住院，无意外豁免）；连续投保或另有约定按约定执行。',
        url='https://resources.pingan.com/app_upload/file/annuity/2011052415580436.pdf',
    ),
    'pa_p0254': dict(
        title='平安住院医疗保险',
        colloquial_name='平安住院医疗险',
        aliases=['平安住院医疗险'],
        insurer='平安',
        page=1,
        start='第四条保险责任',
        end='（一）住院医疗社保内医疗费用保险金',
        days=[90],
        accident_exempt=True,
        approval_exempt=True,
        benefit='住院医疗社保内/外医疗费用保险金',
        scope='第四条住院医疗90天等待期；意外治疗无等待期，除另有约定外按条款执行。',
        url='https://resources.pingan.com/app_upload/file/paylx/20150422P0254.pdf',
    ),
    'pa_p0321': dict(
        title='平安肺癌疾病保险',
        colloquial_name='平安肺癌疾病险',
        aliases=['平安肺癌疾病险', '平安肺癌险', '肺癌疾病保险'],
        insurer='平安',
        page=1,
        start='第四条保险责任',
        end='第五条责任免除',
        days=[30],
        accident_exempt=False,
        approval_exempt=True,
        benefit='初次发生原发性肺癌疾病保险金',
        scope='第四条原发性肺癌（不包括原位癌）30天等待期（疾病险不设意外豁免）；连续投保或另有约定按约定执行。',
        url='https://resources.pingan.com/app_upload/file/annuity/20150309P0321.pdf',
    ),
    'cpic_93896': dict(
        title='附加（广西地区）学生幼儿住院医疗保险（2025版）',
        colloquial_name='太平洋广西学生幼儿住院医疗险',
        aliases=['附加广西地区学生幼儿住院医疗保险', '太平洋广西学生幼儿住院医疗险', '太保广西学生幼儿住院险', '广西学生幼儿住院医疗保险'],
        insurer='太平洋',
        page=1,
        start='第四条保险责任',
        end='针对以下两种情况',
        days=[30],
        accident_exempt=True,
        approval_exempt=False,
        benefit='学生幼儿住院医疗费用保险金',
        scope='第四条第一款住院医疗费用首次投保30天等待期（意外伤害住院无等待期；第1页写明续保不受30日限制，第3页第六条另规定本产品为不保证续保，本集4个种子均限定在首次投保场景）。',
        url='https://www.cpic.com.cn/upload/resources/file/2026/03/31/93896.pdf',
    ),
}

ALL_INSURANCE_FIELDS = [
    'policy_id', 'benefit_id', 'enrollment', 'cause', 'policy_day', 'plan_wait_days', 'approved_waiver'
]


def insurance_decision(frame, policies):
    policy = policies[frame['policy_id']]
    if 'benefits' in policy and frame.get('benefit_id') in policy['benefits']:
        b_rule = policy['benefits'][frame['benefit_id']]
        acc_exempt = b_rule['accident_exempt']
        app_exempt = b_rule['approval_exempt']
        wait_days = frame.get('plan_wait_days') if frame.get('plan_wait_days') is not None else b_rule['wait_days']
    else:
        acc_exempt = policy['accident_exempt']
        app_exempt = policy['approval_exempt']
        wait_days = frame['plan_wait_days']
    if frame['approved_waiver'] and app_exempt:
        return 'WAIT_NOT_EXCLUDED'
    if frame['cause'] == 'accident' and acc_exempt:
        return 'WAIT_NOT_EXCLUDED'
    return 'WAIT_EXCLUDED' if frame['policy_day'] <= wait_days else 'WAIT_NOT_EXCLUDED'


def extract_insurance_fields_frozen(text, policies=HELDOUT_POLICIES):
    """Unified surface-normalized slot extractor shared with analyze_v3.py."""
    return extract_fields_surface_normalized(text, policies=policies)


def enumerate_insurance_decisions_frozen(observed, policies, grant_source_policy=False, seed_frame=None):
    eff = dict(observed)
    if grant_source_policy and eff.get('policy_id') is None and seed_frame is not None:
        eff['policy_id'] = seed_frame['policy_id']
    pid = eff.get('policy_id')
    if pid not in policies:
        return []
    policy = policies[pid]
    decisions = set()
    enrollments = [eff['enrollment']] if eff.get('enrollment') is not None else ['first', 'new']
    days = (
        [eff['policy_day']]
        if eff.get('policy_day') is not None
        else [5, 7, 10, 15, 20, 25, 30, 45, 60, 90, 120]
    )
    if 'benefits' in policy:
        if eff.get('benefit_id') in policy['benefits']:
            b_ids = [eff['benefit_id']]
        else:
            b_ids = list(policy['benefits'].keys())
        for b_id in b_ids:
            b_rule = policy['benefits'][b_id]
            causes = [eff['cause']] if eff.get('cause') is not None else (
                ['illness', 'accident'] if b_rule['accident_exempt'] else ['illness']
            )
            w = b_rule['wait_days']
            for enr in enrollments:
                w_obs = eff.get('approved_waiver')
                if isinstance(w_obs, bool):
                    waivers = [w_obs]
                elif b_rule['approval_exempt'] and enr == 'new':
                    waivers = [False, True]
                else:
                    waivers = [False]
                for c in causes:
                    for d in days:
                        for aw in waivers:
                            cand = {
                                'policy_id': pid,
                                'benefit_id': b_id,
                                'enrollment': enr,
                                'cause': c,
                                'policy_day': d,
                                'plan_wait_days': w,
                                'approved_waiver': aw,
                            }
                            decisions.add(insurance_decision(cand, policies))
        return sorted(decisions)

    causes = [eff['cause']] if eff.get('cause') is not None else (
        ['illness', 'accident'] if policy['accident_exempt'] else ['illness']
    )
    if len(policy['days']) == 1:
        waits = policy['days']
    else:
        waits = (
            [eff['plan_wait_days']]
            if (eff.get('plan_wait_days') is not None and eff.get('plan_wait_days') in policy['days'])
            else policy['days']
        )
    for enr in enrollments:
        w_obs = eff.get('approved_waiver')
        if isinstance(w_obs, bool):
            waivers = [w_obs]
        elif policy['approval_exempt'] and enr == 'new':
            waivers = [False, True]
        else:
            waivers = [False]
        for c in causes:
            for d in days:
                for w in waits:
                    for aw in waivers:
                        cand = {
                            'policy_id': pid,
                            'enrollment': enr,
                            'cause': c,
                            'policy_day': d,
                            'plan_wait_days': w,
                            'approved_waiver': aw,
                        }
                        decisions.add(insurance_decision(cand, policies))
    return sorted(decisions)


def evaluate_insurance_outcome_frozen(observed, seed_frame, required_fields, policies, grant_source_policy=False):
    eff = dict(observed)
    if grant_source_policy and eff.get('policy_id') is None:
        eff['policy_id'] = seed_frame['policy_id']
    pid = seed_frame.get('policy_id')
    policy = policies.get(pid, {})
    eff_req = list(required_fields)
    if 'benefits' in policy and 'benefit_id' in seed_frame and 'benefit_id' not in eff_req:
        eff_req.insert(1, 'benefit_id')

    mutated = [
        f for f in ALL_INSURANCE_FIELDS
        if f in seed_frame and eff.get(f) is not None and eff.get(f) != seed_frame.get(f)
    ]
    if 'benefits' in policy and eff.get('benefit_id') in policy['benefits'] and eff.get('plan_wait_days') is not None:
        expected_w = policy['benefits'][eff['benefit_id']]['wait_days']
        if eff['plan_wait_days'] != expected_w and 'plan_wait_days' not in mutated:
            mutated.append('plan_wait_days')

    missing = [f for f in eff_req if eff.get(f) is None]
    ref_dec = insurance_decision(seed_frame, policies)
    reachable = enumerate_insurance_decisions_frozen(
        eff, policies, grant_source_policy=grant_source_policy, seed_frame=seed_frame
    )
    if mutated:
        return 'CONDITION_MUTATED', missing, mutated
    if len(reachable) == 1:
        if reachable[0] == ref_dec:
            return 'PRESERVED_DECISION', missing, mutated
        return 'CONDITION_MUTATED', missing, (mutated or list(missing))
    return 'DEGRADED_TO_CLARIFY', missing, mutated


def render_heldout_question(frame, policies, missing=None, surface=False):
    policy = policies[frame['policy_id']]
    pid = frame['policy_id']
    prod_name = policy.get('colloquial_name', policy['title']) if surface else policy['title']
    words = [
        f"我买的{prod_name}，",
        '这是首次投保，' if frame['enrollment'] == 'first' else '这是重新投保，',
    ]
    if missing != 'policy_day':
        words.append(f"保单生效第{frame['policy_day']}天，")
    else:
        words.append('具体是保单生效后第几天还不清楚，')

    if pid == 'pa_p0321':
        if missing != 'cause':
            words.append('因非意外疾病初次确诊原发性肺癌，' if frame['cause'] == 'illness' else '因意外伤害导致肺部病变，')
        else:
            words.append('具体确诊原因还不清楚，')
    elif pid == 'pa_2011_36' and frame.get('benefit_id') == 'cancer_hospital' and missing not in ('benefit_id', 'plan_wait_days'):
        if missing != 'cause':
            words.append('因非意外疾病（初次确诊癌症）住院治疗，' if frame['cause'] == 'illness' else '因意外伤害住院治疗，')
        else:
            words.append('这次住院到底是因疾病还是因意外伤害导致的还不清楚，')
    else:
        if missing != 'cause':
            words.append('因意外伤害住院治疗，' if frame['cause'] == 'accident' else '因非意外疾病住院治疗，')
        else:
            words.append('这次住院到底是因疾病还是因意外伤害导致的还不清楚，')

    if 'benefits' in policy:
        if missing in ('benefit_id', 'plan_wait_days'):
            words.append('想申请住院津贴，但目前还不清楚申请的是一般住院医疗津贴还是癌症住院医疗津贴（对应的等待期天数也不确定），')
        else:
            b_label = policy['benefits'][frame['benefit_id']]['label']
            words.append(f"想申请{b_label}（约定等待期为{frame['plan_wait_days']}天），")
    else:
        words.append(f"想申请{policy['benefit']}（条款等待期为{frame['plan_wait_days']}天），")

    if frame['enrollment'] == 'new':
        if missing == 'approved_waiver':
            words.append('是否已获保险公司批准免等待期还不清楚，')
        elif frame['approved_waiver']:
            words.append('并且已获保险公司明确批准免等待期，')
        else:
            words.append('但未获保险公司批准免等待期，')
    words.append('这种情况会因为等待期限制而不能理赔吗？')
    text = ''.join(words)
    if surface:
        text = text.replace('这是首次投保，', '头一次买这份保险，').replace('保单生效第', '从保单生效算起第')
        text = text.replace('这种情况会因为等待期限制而不能理赔吗？', '单看等待期这一条，会被卡住不能报销吗？')
    return strip_book_title_spoilers(text)


def build_atomic_clause_rubrics(frame, policy, anchor, pdf_file):
    """Decompose the reference evaluation into contract-clause evidence (bound to PDF filename,
    page number, and bounding-box coordinates) versus synthetic scenario parameters (from the seed frame)."""
    pg = anchor['pdf_page_1based']
    bbox = [round(float(x), 2) for x in anchor['bbox_pdf_points']]
    rubrics = [
        {
            'slot': 'policy_and_clause_rule',
            'source_type': 'pdf_contract_clause',
            'criterion': (
                f"合同条款规则：{policy['title']}（简称：{policy.get('colloquial_name', policy['title'])}），"
                f"约定等待期阈值{frame['plan_wait_days']}天"
            ),
            'pdf_file': pdf_file,
            'pdf_page_1based': pg,
            'bbox_pdf_points': bbox,
        },
        {
            'slot': 'scenario_case_parameters',
            'source_type': 'synthetic_seed_frame',
            'criterion': (
                f"合成案情参数（源自受控种子 frame，非 PDF 原文）："
                f"{'首次投保' if frame['enrollment'] == 'first' else '重新投保'}，"
                f"保单生效第{frame['policy_day']}天，"
                f"{'意外伤害' if frame['cause'] == 'accident' else '非意外疾病'}"
            ),
            'evaluated_against_pdf_anchor': f"{pdf_file}#page={pg}",
        },
    ]
    if 'benefits' in policy and frame.get('benefit_id') in policy['benefits']:
        b_label = policy['benefits'][frame['benefit_id']]['label']
        rubrics.insert(1, {
            'slot': 'benefit_id',
            'source_type': 'pdf_contract_clause',
            'criterion': f"合同分项责任条款：{b_label}（等待期{frame['plan_wait_days']}天）",
            'pdf_file': pdf_file,
            'pdf_page_1based': pg,
            'bbox_pdf_points': bbox,
        })
    if frame['enrollment'] == 'new':
        rubrics.append({
            'slot': 'approved_waiver',
            'source_type': 'synthetic_seed_frame',
            'criterion': (
                f"合成案情参数（续保免等待期批复状态）："
                f"{'已获保险公司批准免等待期' if frame.get('approved_waiver') else '未获批准免等待期'}"
            ),
            'evaluated_against_pdf_anchor': f"{pdf_file}#page={pg}",
        })
    return rubrics


def build_heldout_seeds_and_cases(raw_heldout_dir, out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[1] / 'data_v3'
    sources = []
    evidence = {}
    anchors = {}
    cached_sources = {}
    if (out_dir / 'heldout_clauses_sources.json').exists():
        for s in json.loads((out_dir / 'heldout_clauses_sources.json').read_text(encoding='utf-8')):
            cached_sources[s['id']] = s
    cached_ev = {}
    if (out_dir / 'dev_seed_inputs.json').exists():
        cached_ev = json.loads((out_dir / 'dev_seed_inputs.json').read_text(encoding='utf-8')).get('heldout_evidence', {})
    for pid, pol in HELDOUT_POLICIES.items():
        pdf_path = raw_heldout_dir / f'{pid}.pdf'
        if pdf_path.exists() and fitz is not None:
            sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            with fitz.open(pdf_path) as pdf:
                anchor, text = source_anchor(pdf, pol)
            anchors[pid] = anchor
            sources.append({
                'id': pid,
                'title': pol['title'],
                'colloquial_name': pol.get('colloquial_name', pol['title']),
                'aliases': pol.get('aliases', []),
                'insurer': pol['insurer'],
                'url': pol['url'],
                'sha256': sha,
                'bytes': pdf_path.stat().st_size,
                'document_type': 'heldout_formal_policy_clause',
                'license_status': 'publicly accessible; no redistribution license verified',
                'anchor': anchor,
            })
            evidence[pid] = text
        else:
            s_entry = cached_sources[pid]
            anchors[pid] = s_entry['anchor']
            sources.append(s_entry)
            evidence[pid] = cached_ev[pid]

    seed_configs = [
        # pa_2011_36: Clause 4(1) general_hospital (30d, accident_exempt=True) vs Clause 4(2) cancer_hospital (90d, accident_exempt=False)
        dict(policy_id='pa_2011_36', benefit_id='general_hospital', enrollment='first', cause='illness', policy_day=45, plan_wait_days=30, approved_waiver=False, pivotal='benefit_id', alt_val='cancer_hospital', alt_extra={'plan_wait_days': 90}),
        dict(policy_id='pa_2011_36', benefit_id='cancer_hospital', enrollment='first', cause='illness', policy_day=45, plan_wait_days=90, approved_waiver=False, pivotal='policy_day', alt_val=100),
        dict(policy_id='pa_2011_36', benefit_id='general_hospital', enrollment='first', cause='accident', policy_day=15, plan_wait_days=30, approved_waiver=False, pivotal='cause', alt_val='illness'),
        dict(policy_id='pa_2011_36', benefit_id='general_hospital', enrollment='new', cause='illness', policy_day=20, plan_wait_days=30, approved_waiver=True, pivotal='approved_waiver', alt_val=False),
        # pa_p0254: [90], accident_exempt=True, approval_exempt=True
        dict(policy_id='pa_p0254', enrollment='first', cause='illness', policy_day=45, plan_wait_days=90, approved_waiver=False, pivotal='policy_day', alt_val=100),
        dict(policy_id='pa_p0254', enrollment='first', cause='illness', policy_day=100, plan_wait_days=90, approved_waiver=False, pivotal='policy_day', alt_val=45),
        dict(policy_id='pa_p0254', enrollment='first', cause='accident', policy_day=20, plan_wait_days=90, approved_waiver=False, pivotal='cause', alt_val='illness'),
        dict(policy_id='pa_p0254', enrollment='new', cause='illness', policy_day=20, plan_wait_days=90, approved_waiver=True, pivotal='approved_waiver', alt_val=False),
        # pa_p0321: [30], accident_exempt=False (lung cancer disease insurance), approval_exempt=True
        dict(policy_id='pa_p0321', enrollment='first', cause='illness', policy_day=15, plan_wait_days=30, approved_waiver=False, pivotal='policy_day', alt_val=45),
        dict(policy_id='pa_p0321', enrollment='first', cause='illness', policy_day=45, plan_wait_days=30, approved_waiver=False, pivotal='policy_day', alt_val=15),
        dict(policy_id='pa_p0321', enrollment='new', cause='illness', policy_day=15, plan_wait_days=30, approved_waiver=True, pivotal='approved_waiver', alt_val=False),
        dict(policy_id='pa_p0321', enrollment='new', cause='illness', policy_day=15, plan_wait_days=30, approved_waiver=False, pivotal='approved_waiver', alt_val=True),
        # cpic_93896: [30], accident_exempt=True, approval_exempt=False (scoped to first-time enrollment)
        dict(policy_id='cpic_93896', enrollment='first', cause='illness', policy_day=15, plan_wait_days=30, approved_waiver=False, pivotal='policy_day', alt_val=45),
        dict(policy_id='cpic_93896', enrollment='first', cause='illness', policy_day=45, plan_wait_days=30, approved_waiver=False, pivotal='policy_day', alt_val=15),
        dict(policy_id='cpic_93896', enrollment='first', cause='accident', policy_day=15, plan_wait_days=30, approved_waiver=False, pivotal='cause', alt_val='illness'),
        dict(policy_id='cpic_93896', enrollment='first', cause='illness', policy_day=20, plan_wait_days=30, approved_waiver=False, pivotal='cause', alt_val='accident'),
    ]

    seeds = []
    cases = []
    for idx, cfg in enumerate(seed_configs):
        sid = f'heldout-{idx + 1:03d}'
        piv = cfg['pivotal']
        alt = cfg['alt_val']
        frame = {k: cfg[k] for k in ALL_INSURANCE_FIELDS if k in cfg}
        changed = dict(frame, **{piv: alt}, **cfg.get('alt_extra', {}))
        ref_dec = insurance_decision(frame, HELDOUT_POLICIES)
        alt_dec = insurance_decision(changed, HELDOUT_POLICIES)
        assert ref_dec != alt_dec, f'Expected opposite decisions on {sid}'

        req = ['policy_id', 'enrollment', 'policy_day', 'cause']
        if 'benefits' in HELDOUT_POLICIES[frame['policy_id']]:
            req.insert(1, 'benefit_id')
            req.append('plan_wait_days')
        elif len(HELDOUT_POLICIES[frame['policy_id']]['days']) > 1:
            req.append('plan_wait_days')
        if frame['enrollment'] == 'new':
            req.append('approved_waiver')

        seed_q = render_heldout_question(frame, HELDOUT_POLICIES)
        seeds.append({
            'seed_id': sid,
            'frame': frame,
            'required_fields': req,
            'pivotal_field': piv,
            'seed_question': seed_q,
            'reference_decision': ref_dec,
            'source_evidence': evidence[frame['policy_id']],
        })

        for variant, f_use, miss in [
            ('complete', frame, None),
            ('surface', frame, None),
            ('missing_critical', frame, piv),
            ('counterfactual', changed, None),
        ]:
            witnesses = []
            if miss:
                witnesses = [
                    {'completion': frame, 'label': ref_dec},
                    {'completion': changed, 'label': alt_dec},
                ]
            cases.append({
                'id': f'{sid}-{variant}',
                'group_id': sid,
                'variant': variant,
                'source_id': frame['policy_id'],
                'task_context': TASK_CONTEXT,
                'question': render_heldout_question(f_use, HELDOUT_POLICIES, missing=miss, surface=(variant == 'surface')),
                'missing_field': miss,
                'reference_label': 'CLARIFY' if miss else insurance_decision(f_use, HELDOUT_POLICIES),
                'counterexample_witnesses': witnesses,
                'atomic_rubrics': build_atomic_clause_rubrics(
                    f_use,
                    HELDOUT_POLICIES[frame['policy_id']],
                    anchors[frame['policy_id']],
                    f"{frame['policy_id']}.pdf",
                ),
            })
    return sources, evidence, seeds, cases


def extract_question_from_structured_output(raw_text):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in reversed(lines):
        for prefix in ['QUESTION:', 'Question:', '问句：', '用户提问：', '提问：', '改写问句：']:
            if line.startswith(prefix):
                cand = clean_cross_question(line[len(prefix):])
                if cand:
                    return cand
    for line in reversed(lines):
        if ('？' in line or '?' in line) and not line.startswith('{') and not line.startswith('}'):
            return clean_cross_question(line)
    return clean_cross_question(raw_text)


def mask_titles_and_insurers(text, titles):
    masked = text
    for t in sorted(titles, key=len, reverse=True):
        masked = masked.replace(f'《{t}》', '该条款')
        masked = masked.replace(t, '该条款')
    masked = re.sub(r'《[^》]{2,40}》', '该条款', masked)
    for ins in ['中国太平洋财产保险股份有限公司', '平安养老保险股份有限公司', '中国太平洋', '太平洋保险', '平安保险', '泰康人寿', '太平洋', '太保', '平安', '泰康']:
        masked = masked.replace(ins, '保险公司')
    return masked


def evaluate_dense_and_lexical_clause_retrieval(
    clause_docs, query_list, gold_indices, bge_tokenizer, bge_model, device, bge_cache=None
):
    bm25_res = base.evaluate_retrievers(clause_docs, {'q': (query_list, [{g} for g in gold_indices])})['q']
    idx_bm25 = base.SYSTEMS.index('bm25_char2')
    bm25_mrr = float(bm25_res[:, idx_bm25, 0].mean())
    bm25_top1 = float((bm25_res[:, idx_bm25, 0] == 1.0).mean())

    q_prefixed = ['为这个句子生成表示以用于检索相关文章：' + q for q in query_list]
    if bge_model is not None and bge_tokenizer is not None:
        with torch.inference_mode():
            d_enc = bge_tokenizer(clause_docs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            d_emb = torch.nn.functional.normalize(bge_model(**d_enc).last_hidden_state[:, 0], p=2, dim=1).cpu().numpy().astype(np.float64)
            q_enc = bge_tokenizer(q_prefixed, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            q_emb = torch.nn.functional.normalize(bge_model(**q_enc).last_hidden_state[:, 0], p=2, dim=1).cpu().numpy().astype(np.float64)
    elif bge_cache is not None:
        d_emb = np.stack([bge_cache[d] for d in clause_docs], axis=0)
        q_emb = np.stack([bge_cache[q] for q in q_prefixed], axis=0)
    else:
        raise FileNotFoundError('Neither local bge model nor bge_embedding_cache.npz is available.')
    sim = np.einsum('qd,cd->qc', q_emb, d_emb, optimize=True)
    bge_mrrs = []
    bge_top1s = []
    for i, g in enumerate(gold_indices):
        order = np.argsort(-sim[i], kind='stable')
        rank = int(np.where(order == g)[0][0]) + 1
        bge_mrrs.append(1.0 / rank if rank <= 10 else 0.0)
        bge_top1s.append(1.0 if rank == 1 else 0.0)
    return {
        'bm25_char2_top1': round(bm25_top1, 4),
        'bm25_char2_mrr10': round(bm25_mrr, 4),
        'bge_small_zh_top1': round(float(np.mean(bge_top1s)), 4),
        'bge_small_zh_mrr10': round(float(np.mean(bge_mrrs)), 4),
    }


def generate_llm_response_with_meta(tokenizer, model, device, system_prompt, user_prompt, max_new_tokens=120):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors='pt').to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs['input_ids'].shape[1]:]
    gen_token_count = int(gen_ids.shape[0])
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if device == 'mps':
        torch.mps.empty_cache()
    return raw, gen_token_count, bool(gen_token_count >= max_new_tokens)


def generate_llm_response(tokenizer, model, device, system_prompt, user_prompt, max_new_tokens=120):
    raw, _, _ = generate_llm_response_with_meta(tokenizer, model, device, system_prompt, user_prompt, max_new_tokens=max_new_tokens)
    return raw


def paired_mcnemar(rows, left, right, field='outcome', target='PRESERVED_DECISION'):
    left_map = {r['seed_id']: (r[field] == target) for r in rows if r['method'] == left}
    right_map = {r['seed_id']: (r[field] == target) for r in rows if r['method'] == right}
    wins = [sid for sid in left_map if left_map[sid] and not right_map[sid]]
    losses = [sid for sid in left_map if right_map[sid] and not left_map[sid]]
    discordant = len(wins) + len(losses)
    p_val = float(binomtest(len(wins), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        'left': left,
        'right': right,
        'n': len(left_map),
        'left_success': int(sum(left_map.values())),
        'right_success': int(sum(right_map.values())),
        'left_only_wins': wins,
        'right_only_wins': losses,
        'exact_mcnemar_two_sided_p': round(p_val, 6),
    }


def summarize_method_records(records, clause_docs, clause_ids, titles, bge_tokenizer, bge_model, device, bge_cache=None):
    n = len(records)
    qs = [r['question'] for r in records]
    golds = [clause_ids.index(r['clause_id']) for r in records]
    full_ret = evaluate_dense_and_lexical_clause_retrieval(
        clause_docs, qs, golds, bge_tokenizer, bge_model, device, bge_cache=bge_cache
    )
    masked_docs = [mask_titles_and_insurers(d, titles) for d in clause_docs]
    masked_qs = [mask_titles_and_insurers(q, titles) for q in qs]
    masked_ret = evaluate_dense_and_lexical_clause_retrieval(
        masked_docs, masked_qs, golds, bge_tokenizer, bge_model, device, bge_cache=bge_cache
    )

    novelties = [1.0 - bigram_containment(r['question'], r['seed_question']) for r in records]
    clause_copies = [bigram_containment(r['question'], clause_docs[clause_ids.index(r['clause_id'])]) for r in records]
    lengths = [len(r['question']) for r in records]

    return {
        'seed_count': n,
        'total_llm_calls': int(sum(r['llm_calls'] for r in records)),
        'format_valid': sum(1 for r in records if r['format_valid']),
        'strict_single_qmark': sum(1 for r in records if r['strict_single_qmark']),
        'all_slots_preserved': sum(1 for r in records if r['all_slots_preserved']),
        'non_pivotal_slot_omitted_decision_preserved': sum(
            1 for r in records if r['outcome'] == 'PRESERVED_DECISION' and not r['all_slots_preserved']
        ),
        'preserved_decision': sum(1 for r in records if r['outcome'] == 'PRESERVED_DECISION'),
        'degraded_to_clarify': sum(1 for r in records if r['outcome'] == 'DEGRADED_TO_CLARIFY'),
        'condition_mutated': sum(1 for r in records if r['outcome'] == 'CONDITION_MUTATED'),
        'granted_source_policy_all_slots_preserved': sum(
            1 for r in records if r.get('granted_source_policy_all_slots_preserved', r['all_slots_preserved'])
        ),
        'granted_source_policy_preserved_decision': sum(
            1 for r in records if r.get('granted_source_policy_outcome', r['outcome']) == 'PRESERVED_DECISION'
        ),
        'granted_source_policy_degraded_to_clarify': sum(
            1 for r in records if r.get('granted_source_policy_outcome', r['outcome']) == 'DEGRADED_TO_CLARIFY'
        ),
        'granted_source_policy_condition_mutated': sum(
            1 for r in records if r.get('granted_source_policy_outcome', r['outcome']) == 'CONDITION_MUTATED'
        ),
        'mean_template_novelty_1_minus_kappa2': round(float(np.mean(novelties)), 4),
        'mean_clause_copy_kappa2': round(float(np.mean(clause_copies)), 4),
        'mean_char_length': round(float(np.mean(lengths)), 2),
        'standard_retrieval': full_ret,
        'title_masked_retrieval': masked_ret,
    }


def main():
    proj_root = Path(__file__).resolve().parents[1]
    work_root = proj_root.parents[1] / 'work'
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=work_root / 'models' / 'qwen')
    p.add_argument('--bge-model', type=Path, default=work_root / 'models' / 'bge')
    p.add_argument('--raw-heldout', type=Path, default=work_root / 'condition_study' / 'raw_heldout')
    p.add_argument('--out-dir', type=Path, default=proj_root / 'data_v3')
    p.add_argument('--score-only', action='store_true')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        for req_file in [
            args.out_dir / 'heldout_generations.jsonl',
            args.out_dir / 'cross_domain_generations.jsonl',
            args.out_dir / 'cross_domain_equal_budget_generations.jsonl',
        ]:
            if not req_file.exists():
                raise FileNotFoundError(f'--score-only requires existing generation file: {req_file}')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    bge_tokenizer = None
    bge_model = None
    bge_cache = None
    if args.bge_model.exists():
        bge_tokenizer = AutoTokenizer.from_pretrained(str(args.bge_model), local_files_only=True)
        bge_model = AutoModel.from_pretrained(str(args.bge_model), local_files_only=True, use_safetensors=True).to(device).eval()
    elif (args.out_dir / 'bge_embedding_cache.npz').exists():
        cache_npz = np.load(args.out_dir / 'bge_embedding_cache.npz', allow_pickle=True)
        bge_cache = {str(t): emb for t, emb in zip(cache_npz['texts'], cache_npz['embeddings'])}
    else:
        raise FileNotFoundError(f'BGE model not found at {args.bge_model} and no cache at {args.out_dir / "bge_embedding_cache.npz"}')

    tokenizer = None
    model = None
    if not args.score_only:
        print(f'Loading Qwen2.5-1.5B-Instruct on {device} (float32, eager)...', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model),
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation='eager',
        ).to(device).eval()

    # 1. Held-out official insurance policies benchmark (4 unseen PDFs, 16 seeds)
    sources, evidence, held_seeds, held_cases = build_heldout_seeds_and_cases(args.raw_heldout, out_dir=args.out_dir)
    base.write_json(args.out_dir / 'heldout_clauses_sources.json', sources)
    with (args.out_dir / 'heldout_controlled_cases.jsonl').open('w', encoding='utf-8') as f:
        for c in held_cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    held_gen_path = args.out_dir / 'heldout_generations.jsonl'
    if args.score_only and held_gen_path.exists():
        held_raw_rows = [json.loads(line) for line in held_gen_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        held_raw_rows = []
        sys_ord = '你是一个保险咨询用户提问改写助手。把原问题改写成简短自然的真实投保人口语化单句提问，不要带《》书名号，只输出改写后的一个问句。'
        sys_cond = (
            '你是一个严谨的保险条款咨询问题改写助手。请用一线真实投保人日常咨询理赔的口语化口吻（不要使用《》书名号或死板条款号，像真实客户问客服“能不能报/卡不卡等待期”一样）把原问题改写成单句提问，但必须完整保留：'
            '（1）完整险种名称（不加《》书名号）；（2）首次投保还是重新投保；（3）保单生效第几天；（4）疾病还是意外伤害原因；'
            '（5）原问题明确申请的具体责任名称及对应的约定等待期天数（严禁把单项责任改写成二选一疑问）；'
            '（6）是否已获或未获保险公司批准免等待期。绝对不能遗漏或篡改任何数字与条件，只输出一个问句。'
        )
        sys_json = (
            '你是一个结构化条件守恒改写助手。请先在第一行输出JSON提取原问题的关键槽位：'
            '{"policy": "...", "benefit": "...", "enrollment": "...", "day": ..., "cause": "...", "wait_days": ..., "waiver": "..."}，'
            '然后在第二行以"QUESTION: "开头输出不带《》书名号、完整保留上述所有槽位、具体责任名称、险种全名和精确天数的自然口语化单句用户提问。'
        )
        sys_self = (
            '你是一个保险评测问题自检与修正助手。请对照【原问题】与【条款原文】，检查【初稿改写】是否遗漏了险种全名（不加《》书名号）、'
            '具体申请的单项责任名称、首次/重新投保、生效第几天、疾病/意外原因、等待期天数或批准免等待期状态，或者是否改错了数字。'
            '请直接输出修正后完整保留全部条件的自然口语化单句用户提问（不要带《》书名号，只输出一个问句）。'
        )
        sys_repair = (
            '你是一个保险条件槽位修复助手。自动验证器检测到你的初稿存在槽位缺失或篡改。'
            '请严格根据【原问题】和【缺失/错误槽位提示】，用自然的投保人咨询口吻（不带《》书名号）补齐所有缺失信息并修正数字，只输出修复后的一个完整单句提问。'
        )

        for s in held_seeds:
            sid = s['seed_id']
            frame = s['frame']
            req = s['required_fields']
            seed_q = s['seed_question']
            ev = s['source_evidence']
            pol = HELDOUT_POLICIES[frame['policy_id']]
            pol_title = pol['title']
            b_label = pol['benefits'][frame['benefit_id']]['label'] if 'benefits' in pol else pol['benefit']

            # 0. seed_passthrough (0 calls)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'seed_passthrough',
                'question': seed_q, 'seed_question': seed_q, 'llm_calls': 0, 'repaired': False,
                'raw_response': seed_q, 'generated_tokens': 0, 'max_new_tokens': 0, 'truncated_at_max_tokens': False,
            })

            # 1. ordinary_style (1 call)
            u_ord = f'原问题：{seed_q}\n请改写成简短自然的口语化提问（只输出一句话）：'
            raw_ord, tok_ord, trunc_ord = generate_llm_response_with_meta(tokenizer, model, device, sys_ord, u_ord, max_new_tokens=96)
            q_ord = clean_cross_question(raw_ord)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'ordinary_style',
                'question': q_ord, 'seed_question': seed_q, 'llm_calls': 1, 'repaired': False,
                'raw_response': raw_ord, 'generated_tokens': tok_ord, 'max_new_tokens': 96, 'truncated_at_max_tokens': trunc_ord,
            })

            # 2. condition_prompt (1 call)
            u_cond = (
                f'条款原文：{pol_title} {ev}\n原问题：{seed_q}\n'
                f'请改写为完整保留“{pol_title}”险种名称（不加《》书名号）、所申请的“{b_label}”单项责任及所有天数和条件的自然口语化单句提问：'
            )
            raw_cond, tok_cond, trunc_cond = generate_llm_response_with_meta(tokenizer, model, device, sys_cond, u_cond, max_new_tokens=110)
            q_cond = clean_cross_question(raw_cond, strip_book_titles=True)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'condition_prompt',
                'question': q_cond, 'seed_question': seed_q, 'llm_calls': 1, 'repaired': False,
                'raw_response': raw_cond, 'generated_tokens': tok_cond, 'max_new_tokens': 110, 'truncated_at_max_tokens': trunc_cond,
            })

            # 3. structured_json_rewrite (1 call)
            u_json = (
                f'条款原文：{pol_title} {ev}\n原问题：{seed_q}\n'
                f'请先输出JSON槽位，再在第二行以QUESTION: 开头输出完整保留“{pol_title}”（不加《》书名号）、所申请的“{b_label}”及全部条件的单句提问：'
            )
            raw_json, tok_json, trunc_json = generate_llm_response_with_meta(tokenizer, model, device, sys_json, u_json, max_new_tokens=160)
            q_json = strip_book_title_spoilers(extract_question_from_structured_output(raw_json))
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'structured_json_rewrite',
                'question': q_json, 'seed_question': seed_q, 'llm_calls': 1, 'repaired': False,
                'raw_response': raw_json, 'generated_tokens': tok_json, 'max_new_tokens': 160, 'truncated_at_max_tokens': trunc_json,
            })

            # 4. generic_self_refine_all (2 calls: refine condition_prompt without parser feedback)
            u_self = (
                f'条款原文：{pol_title} {ev}\n原问题：{seed_q}\n初稿改写：{q_cond}\n'
                f'请自检初稿是否遗漏“{pol_title}”险种名、单项责任“{b_label}”或任何条件/天数，直接输出修正后的单句提问（不要带《》书名号）：'
            )
            raw_self, tok_self, trunc_self = generate_llm_response_with_meta(tokenizer, model, device, sys_self, u_self, max_new_tokens=110)
            q_self = clean_cross_question(raw_self, strip_book_titles=True)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'generic_self_refine_all',
                'question': q_self, 'seed_question': seed_q, 'llm_calls': 2, 'repaired': True,
                'raw_response': raw_self, 'generated_tokens': tok_self, 'max_new_tokens': 110, 'truncated_at_max_tokens': trunc_self,
            })

            # 5. slot_guard_repair (<=2 calls: triggered by unified parser)
            obs_cond = extract_insurance_fields_frozen(q_cond, HELDOUT_POLICIES)
            out_cond, miss_cond, mut_cond = evaluate_insurance_outcome_frozen(obs_cond, frame, req, HELDOUT_POLICIES)
            if should_trigger_slot_guard_repair(miss_cond, mut_cond, out_cond):
                hints = []
                if 'policy_id' in miss_cond or 'policy_id' in mut_cond:
                    hints.append(f'必须完整包含险种名称“{pol_title}”（不加《》书名号）')
                if 'benefit_id' in miss_cond or 'benefit_id' in mut_cond:
                    hints.append(f'必须明确写出仅申请“{b_label}”（不要同时写两项责任或反问属于哪项责任）')
                if 'enrollment' in miss_cond or 'enrollment' in mut_cond:
                    hints.append('必须明确写出“首次投保”' if frame['enrollment'] == 'first' else '必须明确写出“重新投保”')
                if 'policy_day' in miss_cond or 'policy_day' in mut_cond:
                    hints.append(f"必须准确写出“保单生效第{frame['policy_day']}天”（不能改成其他天数）")
                if 'cause' in miss_cond or 'cause' in mut_cond:
                    hints.append('必须明确写出“因意外伤害”' if frame['cause'] == 'accident' else '必须明确写出“因非意外疾病”')
                if 'plan_wait_days' in miss_cond or 'plan_wait_days' in mut_cond:
                    hints.append(f"必须明确写出“约定等待期为{frame['plan_wait_days']}天”")
                if 'approved_waiver' in miss_cond or 'approved_waiver' in mut_cond:
                    hints.append('必须明确写出“已获保险公司明确批准免等待期”' if frame['approved_waiver'] else '必须明确写出“未获保险公司批准免等待期”')
                u_rep = (
                    f'原问题：{seed_q}\n初稿改写：{q_cond}\n'
                    f'自动槽位检查发现问题：{"；".join(hints)}。\n'
                    f'请输出修复后的单句用户提问（必须完整包含上述所有条件，不要带《》书名号，只输出一个问句）：'
                )
                raw_rep, tok_rep, trunc_rep = generate_llm_response_with_meta(tokenizer, model, device, sys_repair, u_rep, max_new_tokens=110)
                q_rep = clean_cross_question(raw_rep, strip_book_titles=True)
                rep_calls = 2
                rep_flag = True
            else:
                raw_rep, tok_rep, trunc_rep = raw_cond, tok_cond, trunc_cond
                q_rep = q_cond
                rep_calls = 1
                rep_flag = False

            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'slot_guard_repair',
                'question': q_rep, 'seed_question': seed_q, 'llm_calls': rep_calls, 'repaired': rep_flag,
                'raw_response': raw_rep, 'generated_tokens': tok_rep, 'max_new_tokens': 110, 'truncated_at_max_tokens': trunc_rep,
            })
            # Also record budget-matched generic_self_refine (ablation sharing the symbolic repair trigger schedule)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'generic_self_refine_budget_matched',
                'question': q_self if rep_flag else q_cond,
                'seed_question': seed_q, 'llm_calls': 2 if rep_flag else 1, 'repaired': rep_flag,
                'raw_response': raw_self if rep_flag else raw_cond,
                'generated_tokens': tok_self if rep_flag else tok_cond,
                'max_new_tokens': 110,
                'truncated_at_max_tokens': trunc_self if rep_flag else trunc_cond,
            })

    # Score all held-out rows with the unified insurance verifier
    held_seeds_map = {s['seed_id']: s for s in held_seeds}
    scored_held_rows = []
    for r in held_raw_rows:
        s = held_seeds_map[r['seed_id']]
        frame = s['frame']
        req = s['required_fields']
        seed_q = s['seed_question']
        q = seed_q if r['method'] == 'seed_passthrough' else strip_book_title_spoilers(r['question'])
        raw_resp = seed_q if r['method'] == 'seed_passthrough' else r['raw_response']
        obs = extract_insurance_fields_frozen(q, HELDOUT_POLICIES)
        outcome, miss, mut = evaluate_insurance_outcome_frozen(obs, frame, req, HELDOUT_POLICIES, grant_source_policy=False)
        outcome_g, miss_g, mut_g = evaluate_insurance_outcome_frozen(obs, frame, req, HELDOUT_POLICIES, grant_source_policy=True)
        qmarks = q.count('？') + q.count('?')
        fmt_valid = bool(len(q) >= 6 and ('？' in q or '?' in q) and '\n' not in q)
        scored_held_rows.append({
            **r,
            'question': q,
            'seed_question': seed_q,
            'raw_response': raw_resp,
            'observed_slots': {k: v for k, v in obs.items() if v is not None},
            'outcome': outcome,
            'missing_slots': miss,
            'mutated_slots': mut,
            'all_slots_preserved': bool(not miss and not mut),
            'granted_source_policy_outcome': outcome_g,
            'granted_source_policy_missing_slots': miss_g,
            'granted_source_policy_mutated_slots': mut_g,
            'granted_source_policy_all_slots_preserved': bool(not miss_g and not mut_g),
            'format_valid': fmt_valid,
            'strict_single_qmark': bool(fmt_valid and qmarks == 1),
        })

    with held_gen_path.open('w', encoding='utf-8') as f:
        for r in scored_held_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    held_clause_ids = list(HELDOUT_POLICIES.keys())
    held_clause_docs = [f"《{HELDOUT_POLICIES[pid]['title']}》 {evidence[pid]}" for pid in held_clause_ids]
    held_titles = [
        t
        for pid in held_clause_ids
        for t in [HELDOUT_POLICIES[pid]['title']] + HELDOUT_POLICIES[pid].get('aliases', [])
    ]

    # Combined 9-clause official insurance pool (5 dev + 4 held-out)
    dev_ev_path = work_root / 'condition_study' / 'evidence_private.json'
    if dev_ev_path.exists():
        dev_evidence = json.loads(dev_ev_path.read_text(encoding='utf-8'))
    else:
        dev_evidence = json.loads((args.out_dir / 'dev_seed_inputs.json').read_text(encoding='utf-8'))['evidence']
    combined_9_ids = list(DEV_POLICIES.keys()) + held_clause_ids
    combined_9_docs = (
        [f"《{DEV_POLICIES[pid]['title']}》 {dev_evidence[pid]}" for pid in DEV_POLICIES]
        + held_clause_docs
    )
    combined_9_titles = [DEV_POLICIES[pid]['title'] for pid in DEV_POLICIES] + held_titles

    held_methods_summary = {}
    for m in [
        'seed_passthrough', 'ordinary_style', 'condition_prompt',
        'structured_json_rewrite', 'generic_self_refine_budget_matched',
        'generic_self_refine_all', 'slot_guard_repair',
    ]:
        sub = [r for r in scored_held_rows if r['method'] == m]
        summary_4 = summarize_method_records(
            sub, held_clause_docs, held_clause_ids, held_titles, bge_tokenizer, bge_model, device, bge_cache=bge_cache
        )
        summary_9 = summarize_method_records(
            sub, combined_9_docs, combined_9_ids, combined_9_titles, bge_tokenizer, bge_model, device, bge_cache=bge_cache
        )
        summary_4['combined_9clause_standard_retrieval'] = summary_9['standard_retrieval']
        summary_4['combined_9clause_title_masked_retrieval'] = summary_9['title_masked_retrieval']
        held_methods_summary[m] = summary_4

    # 2. Equal-call-budget baselines on the 16-seed multi-threshold rule benchmark
    cross_seeds = build_cross_seeds()
    cross_seeds_map = {s['seed_id']: s for s in cross_seeds}
    existing_cross_rows = [
        json.loads(line)
        for line in (args.out_dir / 'cross_domain_generations.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    existing_by_key = {(r['seed_id'], r['method']): r for r in existing_cross_rows}

    cross_ext_path = args.out_dir / 'cross_domain_equal_budget_generations.jsonl'
    if args.score_only and cross_ext_path.exists():
        cross_ext_rows = [json.loads(line) for line in cross_ext_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        cross_ext_rows = []
        sys_cross_json = (
            '你是一个严谨的医疗与劳动法规评测问题生成助手。请先在第一行输出JSON槽位：'
            '{"clause": "...", "status": "...", "condition": "...", "elapsed_day": ..., "threshold_days": ..., "waiver": "..."}，'
            '然后在第二行以"QUESTION: "开头输出完整保留《规程/指引全名》、实际经过天数、规定阈值天数及豁免状态的单句用户提问。'
            '严禁把用户实际经过的天数改写成规定阈值天数。'
        )
        sys_cross_self = (
            '你是一个严谨的医疗与劳动法规评测问题自检助手。请对照【原问题】与【规则原文】，检查【初稿改写】是否：'
            '（1）遗漏了《规程/指引全名》；（2）把用户实际经过的天数（如7天、45天、120天）错误改写成了规则中的阈值天数（如14天、30天、90天）；'
            '（3）遗漏了资格/出血/责任条件或特批/延期豁免状态。请直接输出修正后的单句用户提问（只输出一个问句）。'
        )
        for s in cross_seeds:
            sid = s['seed_id']
            cid = s['clause_id']
            clause_info = CROSS_CLAUSES[cid]
            dom = clause_info['domain']
            seed_q = render_cross_question(s, surface=False)
            ord_row = existing_by_key[(sid, 'ordinary_style')]
            cond_row = existing_by_key[(sid, 'condition_prompt')]
            rep_row = existing_by_key[(sid, 'slot_guard_repair')]
            rep_triggered = bool(rep_row.get('repaired', False))

            # 0. seed_passthrough (0 calls)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'seed_passthrough', 'question': seed_q, 'seed_question': seed_q,
                'llm_calls': 0, 'repaired': False,
                'raw_response': seed_q, 'generated_tokens': 0, 'max_new_tokens': 0, 'truncated_at_max_tokens': False,
            })
            # 1. ordinary_style (1 call, from saved run)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'ordinary_style', 'question': ord_row['question'], 'seed_question': seed_q,
                'llm_calls': 1, 'repaired': False,
                'raw_response': ord_row['question'], 'generated_tokens': None, 'max_new_tokens': 96, 'truncated_at_max_tokens': False,
            })
            # 2. condition_prompt (1 call, from saved run)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'condition_prompt', 'question': cond_row['question'], 'seed_question': seed_q,
                'llm_calls': 1, 'repaired': False,
                'raw_response': cond_row['question'], 'generated_tokens': None, 'max_new_tokens': 110, 'truncated_at_max_tokens': False,
            })
            # 3. structured_json_rewrite (1 call)
            u_cjson = (
                f"规则原文：{clause_info['clause_text']}\n原问题：{seed_q}\n"
                f"请先在第一行输出JSON槽位，再在第二行以QUESTION: 开头输出完整保留《{clause_info['title']}》及所有天数/阈值的单句提问："
            )
            raw_cjson, tok_cjson, trunc_cjson = generate_llm_response_with_meta(
                tokenizer, model, device, sys_cross_json, u_cjson, max_new_tokens=160
            )
            q_cjson = extract_question_from_structured_output(raw_cjson)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'structured_json_rewrite', 'question': q_cjson, 'seed_question': seed_q,
                'llm_calls': 1, 'repaired': False,
                'raw_response': raw_cjson, 'generated_tokens': tok_cjson, 'max_new_tokens': 160, 'truncated_at_max_tokens': trunc_cjson,
            })
            # 4. generic_self_refine (2nd call refining condition_prompt without parser feedback)
            u_cself = (
                f"规则原文：{clause_info['clause_text']}\n原问题：{seed_q}\n初稿改写：{cond_row['question']}\n"
                f"请自检初稿是否遗漏《{clause_info['title']}》或把实际天数改成了阈值天数，直接输出修正后的单句提问："
            )
            raw_cself, tok_cself, trunc_cself = generate_llm_response_with_meta(
                tokenizer, model, device, sys_cross_self, u_cself, max_new_tokens=110
            )
            q_cself = clean_cross_question(raw_cself)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'generic_self_refine_budget_matched',
                'question': q_cself if rep_triggered else cond_row['question'],
                'seed_question': seed_q,
                'llm_calls': 2 if rep_triggered else 1,
                'repaired': rep_triggered,
                'raw_response': raw_cself if rep_triggered else cond_row['question'],
                'generated_tokens': tok_cself if rep_triggered else None,
                'max_new_tokens': 110,
                'truncated_at_max_tokens': trunc_cself if rep_triggered else False,
            })
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'generic_self_refine_all',
                'question': q_cself,
                'seed_question': seed_q,
                'llm_calls': 2,
                'repaired': True,
                'raw_response': raw_cself,
                'generated_tokens': tok_cself,
                'max_new_tokens': 110,
                'truncated_at_max_tokens': trunc_cself,
            })
            # 5. slot_guard_repair (<=2 calls, from saved run)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'slot_guard_repair', 'question': rep_row['question'], 'seed_question': seed_q,
                'llm_calls': 2 if rep_triggered else 1, 'repaired': rep_triggered,
                'raw_response': rep_row['question'], 'generated_tokens': None, 'max_new_tokens': 110, 'truncated_at_max_tokens': False,
            })

    scored_cross_ext = []
    for r in cross_ext_rows:
        s = cross_seeds_map[r['seed_id']]
        q = r['question']
        obs = extract_cross_slots_v2(q)
        outcome, miss, mut = evaluate_cross_outcome(obs, s, grant_source_clause=False)
        outcome_g, miss_g, mut_g = evaluate_cross_outcome(obs, s, grant_source_clause=True)
        qmarks = q.count('？') + q.count('?')
        fmt_valid = bool(len(q) >= 6 and ('？' in q or '?' in q) and '\n' not in q)
        scored_cross_ext.append({
            **r,
            'observed_slots': obs,
            'outcome': outcome,
            'missing_slots': miss,
            'mutated_slots': mut,
            'all_slots_preserved': bool(not miss and not mut),
            'granted_source_policy_outcome': outcome_g,
            'granted_source_policy_missing_slots': miss_g,
            'granted_source_policy_mutated_slots': mut_g,
            'granted_source_policy_all_slots_preserved': bool(not miss_g and not mut_g),
            'format_valid': fmt_valid,
            'strict_single_qmark': bool(fmt_valid and qmarks == 1),
        })

    with cross_ext_path.open('w', encoding='utf-8') as f:
        for r in scored_cross_ext:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    cross_clause_ids = list(CROSS_CLAUSES.keys())
    cross_clause_docs = [CROSS_CLAUSES[cid]['clause_text'] for cid in cross_clause_ids]
    cross_titles = [CROSS_CLAUSES[cid]['title'] for cid in cross_clause_ids]
    for cid in cross_clause_ids:
        cross_titles.extend(CROSS_CLAUSES[cid]['aliases'])

    cross_methods_summary = {}
    for m in [
        'seed_passthrough', 'ordinary_style', 'condition_prompt',
        'structured_json_rewrite', 'generic_self_refine_budget_matched',
        'generic_self_refine_all', 'slot_guard_repair',
    ]:
        sub = [r for r in scored_cross_ext if r['method'] == m]
        cross_methods_summary[m] = summarize_method_records(
            sub, cross_clause_docs, cross_clause_ids, cross_titles, bge_tokenizer, bge_model, device, bge_cache=bge_cache
        )

    cross_json_rows = [r for r in scored_cross_ext if r['method'] == 'structured_json_rewrite']
    held_json_rows = [r for r in scored_held_rows if r['method'] == 'structured_json_rewrite']

    out_summary = {
        'date': '2026-10-27',
        'study_description': (
            'Equal-call-budget baseline comparison (ordinary_style, condition_prompt, structured_json_rewrite, '
            'generic_self_refine, slot_guard_repair, and seed_passthrough upper bound), template-divergence '
            'metrics, title-masked retrieval evaluation, and out-of-sample evaluation on 4 held-out official '
            'insurance policy PDFs using the unified surface-normalized slot parser and decision reachability verifier.'
        ),
        'budget_matching_protocol_note': (
            'generic_self_refine_budget_matched is a trigger-matched ablation that applies a generic self-refinement '
            'prompt on the exact subset of seeds flagged by the symbolic slot parser (26 logical LLM calls on the 16 '
            'multi-threshold seeds), whereas generic_self_refine_all applies the second pass unconditionally across '
            'all 16 seeds (32 LLM calls). Prompt token lengths and wall-clock latencies are not matched across prompts.'
        ),
        'structured_json_truncation_diagnostics': {
            'cross_domain_multi_threshold': {
                'n': len(cross_json_rows),
                'max_new_tokens': 160,
                'truncated_at_max_tokens_count': sum(1 for r in cross_json_rows if r.get('truncated_at_max_tokens')),
                'stopped_after_json_without_question_line': sum(
                    1 for r in cross_json_rows
                    if not r.get('truncated_at_max_tokens') and 'QUESTION' not in (r.get('raw_response') or '') and not r['format_valid']
                ),
                'format_valid_count': sum(1 for r in cross_json_rows if r['format_valid']),
            },
            'heldout_insurance': {
                'n': len(held_json_rows),
                'max_new_tokens': 160,
                'truncated_at_max_tokens_count': sum(1 for r in held_json_rows if r.get('truncated_at_max_tokens')),
                'format_valid_count': sum(1 for r in held_json_rows if r['format_valid']),
            },
        },
        'paired_mcnemar_tests': {
            'cross_guard_vs_generic_budget_matched': paired_mcnemar(
                scored_cross_ext, 'slot_guard_repair', 'generic_self_refine_budget_matched'
            ),
            'cross_guard_vs_condition_prompt': paired_mcnemar(
                scored_cross_ext, 'slot_guard_repair', 'condition_prompt'
            ),
            'cross_guard_vs_generic_budget_matched_granted_clause_sensitivity': paired_mcnemar(
                scored_cross_ext,
                'slot_guard_repair',
                'generic_self_refine_budget_matched',
                field='granted_source_policy_outcome',
                target='PRESERVED_DECISION',
            ),
            'cross_guard_vs_condition_prompt_granted_clause_sensitivity': paired_mcnemar(
                scored_cross_ext,
                'slot_guard_repair',
                'condition_prompt',
                field='granted_source_policy_outcome',
                target='PRESERVED_DECISION',
            ),
            'heldout_guard_vs_condition_prompt': paired_mcnemar(
                scored_held_rows, 'slot_guard_repair', 'condition_prompt'
            ),
            'heldout_guard_vs_structured_json': paired_mcnemar(
                scored_held_rows, 'slot_guard_repair', 'structured_json_rewrite'
            ),
        },
        'heldout_official_insurance_confirmation': {
            'official_pdf_count': len(HELDOUT_POLICIES),
            'seed_count': len(held_seeds),
            'controlled_case_count': len(held_cases),
            'verifier_status': 'unified_surface_normalized_parser_with_benefit_specific_rules',
            'methods': held_methods_summary,
        },
        'cross_domain_equal_budget_comparison': {
            'clause_count': len(CROSS_CLAUSES),
            'seed_count': len(cross_seeds),
            'controlled_case_count': 64,
            'methods': cross_methods_summary,
        },
    }
    base.write_json(args.out_dir / 'equal_budget_and_heldout_summary.json', out_summary)
    print(json.dumps(out_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
