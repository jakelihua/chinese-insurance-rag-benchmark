"""Equal-Call-Budget Baselines, Held-Out Official Clause Confirmation,
Naturalness / Template-Divergence Metrics, and Title-Masked Retrieval Study.

Addresses the Research Landscape & Competitive Audit (2026-10-26):
1. Evaluates a frozen verifier (zero regex edits) on 4 newly collected, unseen
   official Chinese insurance policy PDFs (pa_2011_36, pa_p0254, pa_p0321, cpic_93896;
   16 held-out seeds, 64 controlled cases with double-completion witnesses).
2. Compares 6 generation regimes under explicit call budgets on both the Held-Out
   Official Insurance Confirmation Set (16 seeds) and the Multi-Threshold
   Rule-Compiler Stress Test (16 seeds):
   - seed_passthrough (0 LLM calls; information-preservation upper bound)
   - ordinary_style (1 LLM call; unconstrained colloquial rewrite)
   - condition_prompt (1 LLM call; open-loop slot-constrained prompt)
   - structured_json_rewrite (1 LLM call; schema-first JSON extraction + question)
   - generic_self_refine (2 LLM calls; equal-budget self-check without rule parser,
     evaluated at both matched 2-call budget and full 32-call budget)
   - slot_guard_repair (<=2 LLM calls; 1-step symbolic verifier-guided repair)
3. Measures naturalness / template divergence (1 - kappa_2(q', q_seed)) and
   Title-Masked Clause Retrieval (stripping policy titles and insurer names)
   across both lexical (bm25_char2) and dense (bge_small_zh) retrievers.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import fitz
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import experiment as base
from analyze_v3 import bigram_containment
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


def clean_cross_question(raw):
    q = ' '.join(raw.strip().splitlines())
    q = re.sub(r'^(?:QUESTION|Question|改写后问题|修正后问题|用户提问|用户问题|提问|问题|改写问句)\s*[:：]\s*', '', q.strip())
    return q.strip('“”"\' ')

HELDOUT_POLICIES = {
    'pa_2011_36': dict(
        title='平安住院安心医疗保险',
        insurer='平安',
        page=1,
        start='（一）一般住院医疗津贴',
        end='被保险人自本合同生效90日后因初次患癌症，经医院确诊必须住院治疗的，本公司按其实际住院日数乘',
        days=[30, 90],
        accident_exempt=True,
        approval_exempt=True,
        benefit='一般住院医疗津贴（30天等待期）或癌症住院医疗津贴（90天等待期）',
        scope='第四条（一）一般住院30天等待期及（二）癌症住院90天等待期；连续投保或另有约定按约定执行。',
        url='https://resources.pingan.com/app_upload/file/annuity/2011052415580436.pdf',
    ),
    'pa_p0254': dict(
        title='平安住院医疗保险',
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
        insurer='平安',
        page=1,
        start='第四条保险责任',
        end='第五条责任免除',
        days=[30],
        accident_exempt=False,
        approval_exempt=True,
        benefit='初次发生原发性肺癌疾病保险金',
        scope='第四条原发性肺癌30天等待期（疾病险不设意外豁免）；连续投保或另有约定按约定执行。',
        url='https://resources.pingan.com/app_upload/file/annuity/20150309P0321.pdf',
    ),
    'cpic_93896': dict(
        title='附加（广西地区）学生幼儿住院医疗保险（2025版）',
        insurer='太平洋',
        page=1,
        start='第四条保险责任',
        end='针对以下两种情况',
        days=[30],
        accident_exempt=True,
        approval_exempt=False,
        benefit='学生幼儿住院医疗费用保险金',
        scope='第四条第一款住院医疗费用30天等待期；意外伤害无等待期，不保证续保重新投保无特别豁免条款。',
        url='https://www.cpic.com.cn/upload/resources/file/2026/03/31/93896.pdf',
    ),
}

ALL_INSURANCE_FIELDS = ['policy_id', 'enrollment', 'cause', 'policy_day', 'plan_wait_days', 'approved_waiver']


def insurance_decision(frame, policies):
    policy = policies[frame['policy_id']]
    if frame['approved_waiver'] and policy['approval_exempt']:
        return 'WAIT_NOT_EXCLUDED'
    if frame['cause'] == 'accident' and policy['accident_exempt']:
        return 'WAIT_NOT_EXCLUDED'
    return 'WAIT_EXCLUDED' if frame['policy_day'] <= frame['plan_wait_days'] else 'WAIT_NOT_EXCLUDED'


def extract_insurance_fields_frozen(text, policies):
    """Exact frozen regex patterns from condition_guard.py + analyze_v3.py,
    parameterized only by the policy dictionary (zero regex edits).
    """
    norm = compact(text)
    fields = {}
    matches = [sid for sid, p in policies.items() if compact(p['title']) in norm]
    fields['policy_id'] = matches[0] if len(matches) == 1 else None
    first = bool(re.search(r'(?:首次|第一次|头一次)(?:投保|购买|买)', norm))
    new = bool(re.search(r'重新投保|再次投保|非连续投保', norm))
    fields['enrollment'] = 'first' if first and not new else 'new' if new and not first else None
    illness = bool(re.search(r'非意外|不是意外|并非意外|因疾病|因病|生病|患病|得病', norm))
    accident = '意外' in re.sub(r'非意外|不是意外|并非意外', '', norm)
    fields['cause'] = 'illness' if illness and not accident else 'accident' if accident and not illness else None
    num = r'([0-9零〇一二两三四五六七八九十百]+)'
    day = re.findall(r'生效(?:算起|后|起)?(?:的)?(?:第)?' + num + r'(?:天|日)', norm)
    fields['policy_day'] = chinese_number(day[0]) if len(day) == 1 else None
    if fields['policy_day'] is None:
        day2 = re.findall(r'生效(?:算起|后|起|是|为|已经|已)?(?:的)?(?:第)?' + num + r'(?:天|日)', norm)
        if len(day2) == 1:
            fields['policy_day'] = chinese_number(day2[0])
    wait = re.findall(r'(?:计划.{0,8}等待期(?:为|是)?|等待期为)' + num + r'(?:天|日)', norm)
    if not wait:
        wait = re.findall(num + r'(?:天|日)(?:的)?等待期', norm)
    fields['plan_wait_days'] = chinese_number(wait[0]) if len(wait) == 1 else None
    if fields['plan_wait_days'] is None:
        wait2 = re.findall(r'(?:计划|责任|约定).{0,10}等待期(?:为|是)?' + num + r'(?:天|日)', norm)
        if not wait2:
            wait2 = re.findall(r'等待期(?:为|是)' + num + r'(?:天|日)', norm)
        if len(wait2) == 1:
            fields['plan_wait_days'] = chinese_number(wait2[0])

    if re.search(
        r'(?:是否|有没有).{0,16}(?:批准|同意).{0,8}免(?:去|除)?等待期.{0,8}(?:还不清楚|尚不清楚|不清楚|不确定)',
        norm,
    ):
        fields['approved_waiver'] = 'unknown'
    elif re.search(
        r'(?:未获|尚未获|没有获得|没获得|未曾获|并未获|未经|(?:未|没有)(?:获|获得|得到|批准|同意)).{0,12}(?:批准|同意)?.{0,8}免(?:去|除)?等待期',
        norm,
    ):
        fields['approved_waiver'] = False
    elif re.search(r'(?:已获|已经|已).{0,15}(?:批准|同意).{0,8}免(?:去|除)?等待期', norm):
        fields['approved_waiver'] = True
    else:
        fields['approved_waiver'] = None
    return fields


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
    causes = [eff['cause']] if eff.get('cause') is not None else (
        ['illness', 'accident'] if policy['accident_exempt'] else ['illness']
    )
    days = (
        [eff['policy_day']]
        if eff.get('policy_day') is not None
        else [5, 7, 10, 15, 20, 25, 30, 45, 60, 90, 120]
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
        if w_obs == 'unknown':
            waivers = [False, True]
        elif isinstance(w_obs, bool):
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
    mutated = [
        f for f in ALL_INSURANCE_FIELDS
        if eff.get(f) is not None and eff.get(f) != 'unknown' and eff.get(f) != seed_frame.get(f)
    ]
    missing = [f for f in required_fields if eff.get(f) is None or eff.get(f) == 'unknown']
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
    words = [
        f"按《{policy['title']}》这份条款，",
        '我是首次投保，' if frame['enrollment'] == 'first' else '我是重新投保，',
    ]
    if missing != 'policy_day':
        words.append(f"保单生效第{frame['policy_day']}天，")
    else:
        words.append('具体生效到第几天还不清楚，')
    if missing != 'cause':
        words.append('因意外伤害发生约定的医疗事项，' if frame['cause'] == 'accident' else '因非意外疾病发生约定的医疗事项，')
    else:
        words.append('究竟是因疾病还是因意外伤害发生约定的医疗事项还不清楚，')
    words.append(f"问的是{policy['benefit']}责任，")
    if len(policy['days']) > 1:
        words.append(
            '计划约定的等待期天数不清楚，'
            if missing == 'plan_wait_days'
            else f"计划约定等待期为{frame['plan_wait_days']}天，"
        )
    if frame['enrollment'] == 'new':
        if missing == 'approved_waiver':
            words.append('是否已获保险公司批准免等待期还不清楚，')
        elif frame['approved_waiver']:
            words.append('已获保险公司明确批准免等待期，')
        else:
            words.append('未获保险公司批准免等待期，')
    words.append('是否会因等待期被排除？')
    text = ''.join(words)
    if surface:
        text = text.replace('我是首次投保，', '头一次买这份保险，').replace('保单生效第', '从保单生效算起第')
        text = text.replace('是否会因等待期被排除？', '单看等待期这一条，会被挡住吗？')
    return text


def build_heldout_seeds_and_cases(raw_heldout_dir):
    sources = []
    evidence = {}
    for pid, pol in HELDOUT_POLICIES.items():
        pdf_path = raw_heldout_dir / f'{pid}.pdf'
        sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        with fitz.open(pdf_path) as pdf:
            anchor, text = source_anchor(pdf, pol)
        sources.append({
            'id': pid,
            'title': pol['title'],
            'insurer': pol['insurer'],
            'url': pol['url'],
            'sha256': sha,
            'bytes': pdf_path.stat().st_size,
            'document_type': 'heldout_formal_policy_clause',
            'license_status': 'publicly accessible; no redistribution license verified',
            'anchor': anchor,
        })
        evidence[pid] = text

    seed_configs = [
        # pa_2011_36: multi-wait [30, 90], accident_exempt=True, approval_exempt=True
        dict(policy_id='pa_2011_36', enrollment='first', cause='illness', policy_day=45, plan_wait_days=30, approved_waiver=False, pivotal='plan_wait_days', alt_val=90),
        dict(policy_id='pa_2011_36', enrollment='first', cause='illness', policy_day=45, plan_wait_days=90, approved_waiver=False, pivotal='plan_wait_days', alt_val=30),
        dict(policy_id='pa_2011_36', enrollment='first', cause='accident', policy_day=15, plan_wait_days=30, approved_waiver=False, pivotal='cause', alt_val='illness'),
        dict(policy_id='pa_2011_36', enrollment='new', cause='illness', policy_day=20, plan_wait_days=30, approved_waiver=True, pivotal='approved_waiver', alt_val=False),
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
        # cpic_93896: [30], accident_exempt=True, approval_exempt=False
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
        frame = {k: cfg[k] for k in ALL_INSURANCE_FIELDS}
        changed = dict(frame, **{piv: alt})
        ref_dec = insurance_decision(frame, HELDOUT_POLICIES)
        alt_dec = insurance_decision(changed, HELDOUT_POLICIES)
        assert ref_dec != alt_dec, f'Expected opposite decisions on {sid}'

        req = ['policy_id', 'enrollment', 'policy_day', 'cause']
        if len(HELDOUT_POLICIES[frame['policy_id']]['days']) > 1:
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


def evaluate_dense_and_lexical_clause_retrieval(clause_docs, query_list, gold_indices, bge_tokenizer, bge_model, device):
    bm25_res = base.evaluate_retrievers(clause_docs, {'q': (query_list, [{g} for g in gold_indices])})['q']
    idx_bm25 = base.SYSTEMS.index('bm25_char2')
    bm25_mrr = float(bm25_res[:, idx_bm25, 0].mean())
    bm25_top1 = float((bm25_res[:, idx_bm25, 0] == 1.0).mean())

    q_prefixed = ['为这个句子生成表示以用于检索相关文章：' + q for q in query_list]
    with torch.inference_mode():
        d_enc = bge_tokenizer(clause_docs, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        d_emb = torch.nn.functional.normalize(bge_model(**d_enc).last_hidden_state[:, 0], p=2, dim=1).cpu().numpy().astype(np.float64)
        q_enc = bge_tokenizer(q_prefixed, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        q_emb = torch.nn.functional.normalize(bge_model(**q_enc).last_hidden_state[:, 0], p=2, dim=1).cpu().numpy().astype(np.float64)
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


def generate_llm_response(tokenizer, model, device, system_prompt, user_prompt, max_new_tokens=120):
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
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if device == 'mps':
        torch.mps.empty_cache()
    return raw


def summarize_method_records(records, clause_docs, clause_ids, titles, bge_tokenizer, bge_model, device):
    n = len(records)
    qs = [r['question'] for r in records]
    golds = [clause_ids.index(r['clause_id']) for r in records]
    full_ret = evaluate_dense_and_lexical_clause_retrieval(clause_docs, qs, golds, bge_tokenizer, bge_model, device)
    masked_docs = [mask_titles_and_insurers(d, titles) for d in clause_docs]
    masked_qs = [mask_titles_and_insurers(q, titles) for q in qs]
    masked_ret = evaluate_dense_and_lexical_clause_retrieval(masked_docs, masked_qs, golds, bge_tokenizer, bge_model, device)

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
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    bge_tokenizer = AutoTokenizer.from_pretrained(str(args.bge_model), local_files_only=True)
    bge_model = AutoModel.from_pretrained(str(args.bge_model), local_files_only=True, use_safetensors=True).to(device).eval()

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

    # =========================================================================
    # 1. Held-Out Official Insurance Clauses Confirmation Benchmark (4 new PDFs, 16 seeds)
    # =========================================================================
    sources, evidence, held_seeds, held_cases = build_heldout_seeds_and_cases(args.raw_heldout)
    base.write_json(args.out_dir / 'heldout_clauses_sources.json', sources)
    with (args.out_dir / 'heldout_controlled_cases.jsonl').open('w', encoding='utf-8') as f:
        for c in held_cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    held_gen_path = args.out_dir / 'heldout_generations.jsonl'
    if args.score_only and held_gen_path.exists():
        held_raw_rows = [json.loads(line) for line in held_gen_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        held_raw_rows = []
        sys_ord = '你是一个保险咨询用户提问改写助手。把原问题改写成简短自然的口语化单句提问，只输出改写后的一个问句。'
        sys_cond = (
            '你是一个严谨的保险条款评测问题生成助手。把原问题改写成自然口语化的单句提问，但必须100%完整保留：'
            '（1）《条款全名》；（2）首次投保还是重新投保；（3）保单生效第几天；（4）疾病还是意外伤害原因；'
            '（5）给定的计划约定等待期天数；（6）是否已获或未获保险公司批准免等待期。绝对不能遗漏或篡改任何数字与条件，只输出一个问句。'
        )
        sys_json = (
            '你是一个结构化条件守恒改写助手。请先在第一行输出JSON提取原问题的关键槽位：'
            '{"policy": "...", "enrollment": "...", "day": ..., "cause": "...", "wait_days": ..., "waiver": "..."}，'
            '然后在第二行以"QUESTION: "开头输出完整保留上述所有槽位、条款全名和精确天数的单句用户提问。'
        )
        sys_self = (
            '你是一个保险评测问题自检与修正助手。请对照【原问题】与【条款原文】，检查【初稿改写】是否遗漏了《条款全名》、'
            '首次/重新投保、生效第几天、疾病/意外原因、等待期天数或批准免等待期状态，或者是否改错了数字。'
            '请直接输出修正后完整保留全部条件的单句用户提问（只输出一个问句）。'
        )
        sys_repair = (
            '你是一个保险条件槽位修复助手。自动验证器检测到你的初稿存在槽位缺失或篡改。'
            '请严格根据【原问题】和【缺失/错误槽位提示】，补齐所有缺失信息并修正数字，只输出修复后的一个完整单句提问。'
        )

        for s in held_seeds:
            sid = s['seed_id']
            frame = s['frame']
            req = s['required_fields']
            seed_q = s['seed_question']
            ev = s['source_evidence']
            pol_title = HELDOUT_POLICIES[frame['policy_id']]['title']

            # 0. seed_passthrough (0 calls)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'seed_passthrough',
                'question': seed_q, 'seed_question': seed_q, 'llm_calls': 0, 'repaired': False,
            })

            # 1. ordinary_style (1 call)
            u_ord = f'原问题：{seed_q}\n请改写成简短自然的口语化提问（只输出一句话）：'
            q_ord = clean_cross_question(generate_llm_response(tokenizer, model, device, sys_ord, u_ord, max_new_tokens=96))
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'ordinary_style',
                'question': q_ord, 'seed_question': seed_q, 'llm_calls': 1, 'repaired': False,
            })

            # 2. condition_prompt (1 call)
            u_cond = f'条款原文：《{pol_title}》{ev}\n原问题：{seed_q}\n请改写为完整保留《{pol_title}》全名及所有天数和条件的单句提问：'
            q_cond = clean_cross_question(generate_llm_response(tokenizer, model, device, sys_cond, u_cond, max_new_tokens=110))
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'condition_prompt',
                'question': q_cond, 'seed_question': seed_q, 'llm_calls': 1, 'repaired': False,
            })

            # 3. structured_json_rewrite (1 call)
            u_json = f'条款原文：《{pol_title}》{ev}\n原问题：{seed_q}\n请先输出JSON槽位，再在第二行以QUESTION: 开头输出完整保留《{pol_title}》及全部条件的单句提问：'
            raw_json = generate_llm_response(tokenizer, model, device, sys_json, u_json, max_new_tokens=160)
            q_json = extract_question_from_structured_output(raw_json)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'structured_json_rewrite',
                'question': q_json, 'seed_question': seed_q, 'llm_calls': 1, 'repaired': False,
            })

            # 4. generic_self_refine_all (2 calls: refine condition_prompt without parser feedback)
            u_self = f'条款原文：《{pol_title}》{ev}\n原问题：{seed_q}\n初稿改写：{q_cond}\n请自检初稿是否遗漏《{pol_title}》或任何条件/天数，直接输出修正后的单句提问：'
            q_self = clean_cross_question(generate_llm_response(tokenizer, model, device, sys_self, u_self, max_new_tokens=110))
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'generic_self_refine_all',
                'question': q_self, 'seed_question': seed_q, 'llm_calls': 2, 'repaired': True,
            })

            # 5. slot_guard_repair (<=2 calls: triggered by frozen parser)
            obs_cond = extract_insurance_fields_frozen(q_cond, HELDOUT_POLICIES)
            _, miss_cond, mut_cond = evaluate_insurance_outcome_frozen(obs_cond, frame, req, HELDOUT_POLICIES)
            if should_trigger_slot_guard_repair(miss_cond, mut_cond):
                hints = []
                if 'policy_id' in miss_cond or 'policy_id' in mut_cond:
                    hints.append(f'必须逐字包含条款全名《{pol_title}》')
                if 'enrollment' in miss_cond or 'enrollment' in mut_cond:
                    hints.append('必须明确写出“首次投保”' if frame['enrollment'] == 'first' else '必须明确写出“重新投保”')
                if 'policy_day' in miss_cond or 'policy_day' in mut_cond:
                    hints.append(f"必须准确写出“保单生效第{frame['policy_day']}天”（不能改成其他天数）")
                if 'cause' in miss_cond or 'cause' in mut_cond:
                    hints.append('必须明确写出“因意外伤害”' if frame['cause'] == 'accident' else '必须明确写出“因非意外疾病”')
                if 'plan_wait_days' in miss_cond or 'plan_wait_days' in mut_cond:
                    hints.append(f"必须明确写出“计划约定等待期为{frame['plan_wait_days']}天”")
                if 'approved_waiver' in miss_cond or 'approved_waiver' in mut_cond:
                    hints.append('必须明确写出“已获保险公司明确批准免等待期”' if frame['approved_waiver'] else '必须明确写出“未获保险公司批准免等待期”')
                u_rep = (
                    f'原问题：{seed_q}\n初稿改写：{q_cond}\n'
                    f'自动槽位检查发现问题：{"；".join(hints)}。\n'
                    f'请输出修复后的单句用户提问（必须完整包含上述所有条件，只输出一个问句）：'
                )
                q_rep = clean_cross_question(generate_llm_response(tokenizer, model, device, sys_repair, u_rep, max_new_tokens=110))
                rep_calls = 2
                rep_flag = True
            else:
                q_rep = q_cond
                rep_calls = 1
                rep_flag = False

            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'slot_guard_repair',
                'question': q_rep, 'seed_question': seed_q, 'llm_calls': rep_calls, 'repaired': rep_flag,
            })
            # Also record budget-matched generic_self_refine (refining only when slot_guard_repair triggered)
            held_raw_rows.append({
                'seed_id': sid, 'clause_id': frame['policy_id'], 'method': 'generic_self_refine_budget_matched',
                'question': q_self if rep_flag else q_cond,
                'seed_question': seed_q, 'llm_calls': 2 if rep_flag else 1, 'repaired': rep_flag,
            })

    # Score all held-out rows with the frozen insurance verifier
    held_seeds_map = {s['seed_id']: s for s in held_seeds}
    scored_held_rows = []
    for r in held_raw_rows:
        s = held_seeds_map[r['seed_id']]
        frame = s['frame']
        req = s['required_fields']
        q = r['question']
        obs = extract_insurance_fields_frozen(q, HELDOUT_POLICIES)
        outcome, miss, mut = evaluate_insurance_outcome_frozen(obs, frame, req, HELDOUT_POLICIES, grant_source_policy=False)
        outcome_g, miss_g, mut_g = evaluate_insurance_outcome_frozen(obs, frame, req, HELDOUT_POLICIES, grant_source_policy=True)
        qmarks = q.count('？') + q.count('?')
        fmt_valid = bool(len(q) >= 6 and ('？' in q or '?' in q) and '\n' not in q)
        scored_held_rows.append({
            **r,
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
    held_titles = [HELDOUT_POLICIES[pid]['title'] for pid in held_clause_ids]

    # Combined 9-clause official insurance pool (5 dev + 4 held-out)
    dev_evidence = json.loads((work_root / 'condition_study' / 'evidence_private.json').read_text(encoding='utf-8'))
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
        summary_4 = summarize_method_records(sub, held_clause_docs, held_clause_ids, held_titles, bge_tokenizer, bge_model, device)
        summary_9 = summarize_method_records(sub, combined_9_docs, combined_9_ids, combined_9_titles, bge_tokenizer, bge_model, device)
        summary_4['combined_9clause_standard_retrieval'] = summary_9['standard_retrieval']
        summary_4['combined_9clause_title_masked_retrieval'] = summary_9['title_masked_retrieval']
        held_methods_summary[m] = summary_4

    # =========================================================================
    # 2. Equal-Call-Budget Baselines on the 16-Seed Multi-Threshold Rule-Compiler Benchmark
    # =========================================================================
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
            })
            # 1. ordinary_style (1 call, from saved run)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'ordinary_style', 'question': ord_row['question'], 'seed_question': seed_q,
                'llm_calls': 1, 'repaired': False,
            })
            # 2. condition_prompt (1 call, from saved run)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'condition_prompt', 'question': cond_row['question'], 'seed_question': seed_q,
                'llm_calls': 1, 'repaired': False,
            })
            # 3. structured_json_rewrite (1 call)
            u_cjson = (
                f"规则原文：{clause_info['clause_text']}\n原问题：{seed_q}\n"
                f"请先在第一行输出JSON槽位，再在第二行以QUESTION: 开头输出完整保留《{clause_info['title']}》及所有天数/阈值的单句提问："
            )
            raw_cjson = generate_llm_response(tokenizer, model, device, sys_cross_json, u_cjson, max_new_tokens=160)
            q_cjson = extract_question_from_structured_output(raw_cjson)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'structured_json_rewrite', 'question': q_cjson, 'seed_question': seed_q,
                'llm_calls': 1, 'repaired': False,
            })
            # 4. generic_self_refine (2nd call refining condition_prompt without parser feedback)
            u_cself = (
                f"规则原文：{clause_info['clause_text']}\n原问题：{seed_q}\n初稿改写：{cond_row['question']}\n"
                f"请自检初稿是否遗漏《{clause_info['title']}》或把实际天数改成了阈值天数，直接输出修正后的单句提问："
            )
            q_cself = clean_cross_question(generate_llm_response(tokenizer, model, device, sys_cross_self, u_cself, max_new_tokens=110))
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'generic_self_refine_budget_matched',
                'question': q_cself if rep_triggered else cond_row['question'],
                'seed_question': seed_q,
                'llm_calls': 2 if rep_triggered else 1,
                'repaired': rep_triggered,
            })
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'generic_self_refine_all',
                'question': q_cself,
                'seed_question': seed_q,
                'llm_calls': 2,
                'repaired': True,
            })
            # 5. slot_guard_repair (<=2 calls, from saved run)
            cross_ext_rows.append({
                'seed_id': sid, 'domain': dom, 'clause_id': cid,
                'method': 'slot_guard_repair', 'question': rep_row['question'], 'seed_question': seed_q,
                'llm_calls': 2 if rep_triggered else 1, 'repaired': rep_triggered,
            })

    scored_cross_ext = []
    for r in cross_ext_rows:
        s = cross_seeds_map[r['seed_id']]
        q = r['question']
        obs = extract_cross_slots_v2(q)
        outcome, miss, mut = evaluate_cross_outcome(obs, s)
        qmarks = q.count('？') + q.count('?')
        fmt_valid = bool(len(q) >= 6 and ('？' in q or '?' in q) and '\n' not in q)
        scored_cross_ext.append({
            **r,
            'observed_slots': obs,
            'outcome': outcome,
            'missing_slots': miss,
            'mutated_slots': mut,
            'all_slots_preserved': bool(not miss and not mut),
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
            sub, cross_clause_docs, cross_clause_ids, cross_titles, bge_tokenizer, bge_model, device
        )

    out_summary = {
        'date': '2026-10-26',
        'study_description': (
            'Equal-call-budget baseline comparison (ordinary_style, condition_prompt, structured_json_rewrite, '
            'generic_self_refine, slot_guard_repair, and seed_passthrough upper bound), naturalness / template-divergence '
            'metrics, title-masked retrieval evaluation, and out-of-sample confirmation on 4 newly collected official '
            'insurance policy PDFs using the frozen Round 3 slot parser and decision verifier.'
        ),
        'heldout_official_insurance_confirmation': {
            'official_pdf_count': len(HELDOUT_POLICIES),
            'seed_count': len(held_seeds),
            'controlled_case_count': len(held_cases),
            'verifier_status': 'frozen_round3_regex_and_enumerator_zero_edits',
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
