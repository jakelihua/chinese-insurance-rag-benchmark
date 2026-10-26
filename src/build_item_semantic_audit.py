"""Build item-by-item automated verifier explanation and manual-inspection watchlist
records across the 5 development insurance policies (40 items), 4 multi-threshold rules
(112 items), and 4 extension official insurance policies (112 items).

Provenance & Scope Disclosure:
- Every record in `data_v3/item_by_item_semantic_audit.jsonl` is a deterministic
  automated verifier explanation (`automated_verifier_explanation`) paired with a
  rule-based manual-inspection watchlist flag (`watchlist_flag` and `watchlist_reason`).
- Contract rule evidence is anchored to physical PDF filenames, page numbers, and
  bounding-box coordinates (`source_type = 'pdf_contract_clause'`), while synthetic
  case facts (enrollment status, policy day, claim cause, waiver status) are explicitly
  marked as originating from the synthetic seed frame (`source_type = 'synthetic_seed_frame'`).
- These records do NOT constitute an independent human annotation or external LLM
  semantic accuracy benchmark, and no circular agreement rate with the verifier is claimed.
"""
import json
from collections import Counter
from pathlib import Path

from analyze_v3 import (
    evaluate_slot_completeness,
    extract_fields_surface_normalized,
)
from condition_data import POLICIES as DEV_POLICIES, decision as dev_decision
from cross_domain_study import (
    CROSS_CLAUSES,
    build_seed_frames as build_cross_seeds,
    cross_decision,
    evaluate_cross_outcome,
    extract_cross_slots_v2,
)
from equal_budget_and_heldout_study import (
    HELDOUT_POLICIES,
    build_atomic_clause_rubrics,
    build_heldout_seeds_and_cases,
    insurance_decision,
)


PROVENANCE = (
    'automated_verifier_explanation_and_manual_inspection_watchlist; '
    'NOT_external_semantic_validation_or_human_annotation'
)

SLOT_ZH = {
    'policy_id': '险种名称',
    'benefit_id': '具体申领的分项责任（一般住院津贴或癌症住院津贴）',
    'enrollment': '首投还是续保状态',
    'policy_day': '出险时的保单生效天数',
    'cause': '出险原因（疾病还是意外）',
    'plan_wait_days': '计划或责任对应的等待期天数',
    'approved_waiver': '保险公司是否批复免除等待期',
    'clause_id': '适用规程或法规名称',
    'status': '临床分型/用工状态',
    'condition': '关键准入或排除条件',
    'elapsed_day': '当事人实际经过天数',
    'threshold_days': '规程法定阈值天数',
}


def format_slot_list(slots):
    return '、'.join(SLOT_ZH.get(s, s) for s in slots)


def build_insurance_verifier_explanation(
    seed_id, pol_title, benefit_label, frame, obs, outcome, missing, mutated, q_text, method, pdf_file, page_no, bbox
):
    ref_dec = '等待期内免责（WAIT_EXCLUDED）' if (
        frame.get('policy_day', 0) <= frame.get('plan_wait_days', 0)
        and not (frame.get('cause') == 'accident' and frame.get('accident_exempt', True))
        and not (frame.get('approved_waiver') and frame.get('approval_exempt', True))
    ) else '不受等待期免责限制（WAIT_NOT_EXCLUDED）'

    coord_tag = f'{pdf_file} 第{page_no}页条款坐标 [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]'
    enr_zh = '首次投保' if frame['enrollment'] == 'first' else '重新投保'
    cause_zh = '意外伤害' if frame['cause'] == 'accident' else '非意外疾病'

    if not q_text or ('？' not in q_text and '?' not in q_text):
        return (
            f'自动验证器解释（条款依据：{coord_tag}）：该记录在两段式 JSON 改写中未生成完整的问号结尾问句，'
            f'格式校验未通过（种子参考结论为 {ref_dec}）。',
            'format_incomplete',
        )

    if outcome == 'PRESERVED_DECISION':
        if not missing and not mutated:
            waiver_note = ''
            if frame['enrollment'] == 'new':
                waiver_note = '、已获批免等待期' if frame.get('approved_waiver') else '、未获批免等待期'
            return (
                f'自动验证器解释（条款依据：{coord_tag}）：解析器从问句中提取到完整案情要素（{pol_title}、{enr_zh}、'
                f'保单生效第{frame["policy_day"]}天、{cause_zh}、{benefit_label}及{frame["plan_wait_days"]}天等待期{waiver_note}），'
                f'无槽位冲突，可达结论唯一且等于参考结论 {ref_dec}。',
                'all_slots_matched',
            )
        miss_zh = format_slot_list(missing)
        return (
            f'自动验证器解释（条款依据：{coord_tag}）：问句未显式写出“{miss_zh}”，但在已提取条件'
            f'（{pol_title}、保单生效第{frame["policy_day"]}天、等待期阈值{frame["plan_wait_days"]}天）下，'
            f'枚举缺失槽位的所有合法取值均导出同一结论 {ref_dec}，故判定为结论可确定（PRESERVED_DECISION）。',
            'non_pivotal_slot_omitted_but_decision_preserved',
        )

    if outcome in ('CONDITION_MUTATED', 'COUNTERFACTUAL_FLIP'):
        mut_details = []
        for m in mutated:
            exp_v = frame.get(m)
            got_v = obs.get(m)
            mut_details.append(f'{SLOT_ZH.get(m, m)}由种子值“{exp_v}”变为提取值“{got_v}”')
        mut_str = '；'.join(mut_details) if mut_details else '提取槽位与种子案情冲突'
        return (
            f'自动验证器解释（条款依据：{coord_tag}）：解析器检出槽位数值或状态冲突（{mut_str}），'
            f'判定为条件篡改（CONDITION_MUTATED）。',
            'slot_value_mutated',
        )

    miss_zh = format_slot_list(missing) if missing else '关键条款或天数要素'
    if 'policy_id' in missing and len(missing) == 1:
        return (
            f'自动验证器解释（条款依据：{coord_tag}）：问句保留了出险天数（第{frame["policy_day"]}天）与原因，'
            f'但未包含可识别的产品名称（缺失 {miss_zh}）；独立评测下因跨保单等待期规则不同而判定为需追问（DEGRADED_TO_CLARIFY），'
            f'给定源保单上下文后结论可唯一确定。',
            'missing_product_identifier_only',
        )
    return (
        f'自动验证器解释（条款依据：{coord_tag}）：问句缺失“{miss_zh}”，枚举未观测槽位的合法补全空间同时包含'
        f'等待期内免责与不免责两种可达结论，判定为需追问澄清（DEGRADED_TO_CLARIFY）。',
        'missing_pivotal_slot',
    )


def build_cross_verifier_explanation(seed, obs, outcome, missing, mutated, q_text, method):
    title = CROSS_CLAUSES[seed['clause_id']]['title']
    ref_dec = '符合准入/支持条件（RULE_ELIGIBLE）' if cross_decision(seed) == 'RULE_ELIGIBLE' else '受阈值或禁忌限制排除（RULE_EXCLUDED）'

    if method == 'structured_json_rewrite' and ('？' not in q_text and '?' not in q_text):
        return (
            f'自动验证器解释（合成规则：{title}）：模型仅输出第一行 JSON 对象即停止生成，未输出第二行 QUESTION 自然问句，'
            f'两段式格式校验判为无效（format_valid=False）。',
            'json_only_early_stop',
        )

    if outcome == 'PRESERVED_DECISION':
        if not missing and not mutated:
            return (
                f'自动验证器解释（合成规则：{title}）：解析器提取到全部必需槽位（实际经过第{seed["elapsed_day"]}天、'
                f'规则阈值{seed["threshold_days"]}天及前置状态/资格条件），无槽位篡改，唯一可达结论为 {ref_dec}。',
                'all_slots_matched',
            )
        miss_zh = format_slot_list(missing)
        return (
            f'自动验证器解释（合成规则：{title}）：问句未显式写出“{miss_zh}”，但已提取条件在阈值等价区间枚举下'
            f'所有合法补全均导出唯一结论 {ref_dec}，判定为结论保持（PRESERVED_DECISION）。',
            'non_pivotal_slot_omitted_but_decision_preserved',
        )

    if outcome == 'CONDITION_MUTATED':
        if 'elapsed_day' in mutated:
            return (
                f'自动验证器解释（合成规则：{title}）：解析器检出经过天数槽位冲突——种子设定为第{seed["elapsed_day"]}天，'
                f'而问句中被提取为“{obs.get("elapsed_day")}”天（与规则阈值混淆），判定为条件篡改（CONDITION_MUTATED）。',
                'threshold_overwrite_mutation',
            )
        mut_zh = format_slot_list(mutated)
        return (
            f'自动验证器解释（合成规则：{title}）：解析器检出槽位冲突（{mut_zh}与种子设定不一致），'
            f'判定为条件篡改（CONDITION_MUTATED）。',
            'slot_value_mutated',
        )

    miss_zh = format_slot_list(missing) if missing else '关键时间或资格要素'
    return (
        f'自动验证器解释（合成规则：{title}）：问句缺失“{miss_zh}”，在多阈值规则下不同合法补全导出相反结论，'
        f'判定为需追问澄清（DEGRADED_TO_CLARIFY）。',
        'missing_pivotal_slot',
    )


def build_watchlist_annotation(record):
    """Identify verifiable format or surface-cue cases for future human inspection.
    Does not override verifier outcomes or fabricate text not present in `record['question']`."""
    q = record['question']
    raw = record.get('raw_response', '')
    method = record['method']
    missing = record['missing_slots']
    mutated = record['mutated_slots']

    # 1. Two-line JSON format failures or JSON-header / question-line split discrepancies
    if method == 'structured_json_rewrite':
        if '？' not in q and '?' not in q:
            if record.get('truncated_at_max_tokens'):
                return True, 'json_truncated_at_token_cap', (
                    '两段式 JSON 改写触及 160-token 生成上限导致末尾问句被截断，未形成完整问句；列为格式截断待核查样本。'
                )
            return True, 'json_stopped_after_json_line', (
                '两段式 JSON 改写仅输出第一行 JSON 字段即停止生成（未触及 160-token 上限），未输出 QUESTION 问句行；列为格式早停待核查样本。'
            )
        if 'policy_id' in missing and raw and any(
            pol['title'] in raw for pol in HELDOUT_POLICIES.values()
        ):
            return True, 'json_header_has_policy_but_question_omits_it', (
                '原始输出第一行 JSON 中包含险种名称，但第二行 QUESTION 问句中省略了险种名称，导致仅解析问句行时缺失 policy_id。'
            )

    # 2. Repaired non-compete queries that state domain-specific non-compete keywords without the formal rule title
    if (
        record['dataset_split'] == 'multi_threshold_rules_4clauses'
        and record['clause_id'] == 'law_noncompete'
        and 'clause_id' in missing
        and not mutated
        and ('竞业限制' in q)
    ):
        assert '竞业限制' in q
        return True, 'domain_phrase_without_formal_clause_title', (
            '问句中包含“竞业限制经济补偿/解除竞业限制协议”等业务短语，但未写出《劳动合同法第二十四条竞业限制经济补偿与单方解除合规指引》'
            '完整规程标题，因此在多规则池独立评测下因缺失 clause_id 被保守归入 DEGRADED_TO_CLARIFY。'
        )

    return False, None, None


def main():
    proj_root = Path(__file__).resolve().parents[1]
    work_root = proj_root.parents[1] / 'work'
    data_dir = proj_root / 'data_v3'

    dev_sources = {
        s['id']: s
        for s in json.loads((data_dir / 'sources.json').read_text(encoding='utf-8'))
        if 'anchor' in s
    }
    held_sources = {
        s['id']: s
        for s in json.loads((data_dir / 'heldout_clauses_sources.json').read_text(encoding='utf-8'))
        if 'anchor' in s
    }

    records = []

    # 1. Development insurance set (40 generated questions across 20 seeds x 2 methods)
    dev_inputs = json.loads((work_root / 'condition_study' / 'generation_inputs.json').read_text(encoding='utf-8'))
    dev_seeds = {r['seed_id']: r for r in dev_inputs['jobs']}
    dev_rows = [
        json.loads(line)
        for line in (data_dir / 'generated_questions.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    for r in dev_rows:
        seed = dev_seeds[r['seed_id']]
        frame = dict(seed['frame'])
        pid = frame['policy_id']
        pol = DEV_POLICIES[pid]
        anchor = dev_sources[pid]['anchor']
        frame['accident_exempt'] = pol['accident_exempt']
        frame['approval_exempt'] = pol['approval_exempt']
        req = seed['required_fields']
        obs = extract_fields_surface_normalized(r['question'], policies=DEV_POLICIES)
        _, out_s, mut_s, miss_s = evaluate_slot_completeness(obs, seed['frame'], req, grant_source_policy=False)
        _, out_g, mut_g, miss_g = evaluate_slot_completeness(obs, seed['frame'], req, grant_source_policy=True)
        explanation, diag_cat = build_insurance_verifier_explanation(
            r['seed_id'], pol['title'], pol['benefit'], frame, obs, out_s, miss_s, mut_s,
            r['question'], r['method'], f'{pid}.pdf', anchor['pdf_page_1based'], anchor['bbox_pdf_points']
        )
        rec = {
            'audit_id': f"dev-{r['seed_id']}-{r['method']}",
            'dataset_split': 'dev_insurance_5clauses',
            'seed_id': r['seed_id'],
            'clause_id': pid,
            'method': r['method'],
            'question': r['question'],
            'reference_decision': dev_decision(seed['frame']),
            'standalone_outcome': out_s,
            'granted_source_policy_outcome': out_g,
            'missing_slots': miss_s,
            'mutated_slots': mut_s,
            'diagnostic_category': diag_cat,
            'atomic_rubrics': build_atomic_clause_rubrics(frame, pol, anchor, f'{pid}.pdf'),
            'automated_verifier_explanation': explanation,
            'record_provenance': PROVENANCE,
        }
        w_flag, w_cat, w_reason = build_watchlist_annotation(rec)
        rec['watchlist_flag'] = w_flag
        if w_flag:
            rec['watchlist_category'] = w_cat
            rec['watchlist_reason'] = w_reason
        records.append(rec)

    # 2. Multi-threshold rule benchmark (112 equal-budget rows across 16 seeds x 7 methods)
    cross_seeds = {s['seed_id']: s for s in build_cross_seeds()}
    cross_rows = [
        json.loads(line)
        for line in (data_dir / 'cross_domain_equal_budget_generations.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    for r in cross_rows:
        s = cross_seeds[r['seed_id']]
        cid = r['clause_id']
        c_meta = CROSS_CLAUSES[cid]
        obs = extract_cross_slots_v2(r['question'])
        out_c, miss_c, mut_c = evaluate_cross_outcome(obs, s)
        explanation, diag_cat = build_cross_verifier_explanation(
            s, obs, out_c, miss_c, mut_c, r['question'], r['method']
        )
        rec = {
            'audit_id': f"cross-{r['seed_id']}-{r['method']}",
            'dataset_split': 'multi_threshold_rules_4clauses',
            'seed_id': r['seed_id'],
            'clause_id': cid,
            'method': r['method'],
            'question': r['question'],
            'raw_response': r.get('raw_response', ''),
            'truncated_at_max_tokens': bool(r.get('truncated_at_max_tokens')),
            'reference_decision': cross_decision(s),
            'standalone_outcome': out_c,
            'granted_source_policy_outcome': out_c,
            'missing_slots': miss_c,
            'mutated_slots': mut_c,
            'diagnostic_category': diag_cat,
            'atomic_rubrics': [
                {'slot': 'clause_id', 'criterion': f"合成规程规则：{c_meta['title']}（阈值{s['threshold_days']}天）", 'source_type': 'synthetic_composite_rule'},
                {'slot': 'scenario_case_parameters', 'criterion': f"合成案情参数（源自受控种子 frame）：实际经过第{s['elapsed_day']}天，status={s['status']}, condition={s['condition']}, waiver={s['approved_waiver']}", 'source_type': 'synthetic_seed_frame'},
            ],
            'automated_verifier_explanation': explanation,
            'record_provenance': PROVENANCE,
        }
        w_flag, w_cat, w_reason = build_watchlist_annotation(rec)
        rec.pop('raw_response', None)
        rec['watchlist_flag'] = w_flag
        if w_flag:
            rec['watchlist_category'] = w_cat
            rec['watchlist_reason'] = w_reason
        records.append(rec)

    # 3. Extension official insurance benchmark (112 rows across 16 seeds x 7 methods)
    _, _, held_seeds_list, _ = build_heldout_seeds_and_cases(work_root / 'condition_study' / 'raw_heldout')
    held_seeds = {s['seed_id']: s for s in held_seeds_list}
    held_rows = [
        json.loads(line)
        for line in (data_dir / 'heldout_generations.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    for r in held_rows:
        s = held_seeds[r['seed_id']]
        frame = dict(s['frame'])
        pid = frame['policy_id']
        pol = HELDOUT_POLICIES[pid]
        anchor = held_sources[pid]['anchor']
        if 'benefits' in pol and frame.get('benefit_id') in pol['benefits']:
            b_rule = pol['benefits'][frame['benefit_id']]
            b_label = b_rule['label']
            frame['accident_exempt'] = b_rule['accident_exempt']
            frame['approval_exempt'] = b_rule['approval_exempt']
        else:
            b_label = pol['benefit']
            frame['accident_exempt'] = pol['accident_exempt']
            frame['approval_exempt'] = pol['approval_exempt']
        obs = r['observed_slots']
        out_s = r['outcome']
        out_g = r['granted_source_policy_outcome']
        miss_s = r['missing_slots']
        mut_s = r['mutated_slots']
        explanation, diag_cat = build_insurance_verifier_explanation(
            r['seed_id'], pol['title'], b_label, frame, obs, out_s, miss_s, mut_s,
            r['question'], r['method'], f'{pid}.pdf', anchor['pdf_page_1based'], anchor['bbox_pdf_points']
        )
        rec = {
            'audit_id': f"heldout-{r['seed_id']}-{r['method']}",
            'dataset_split': 'heldout_insurance_4clauses',
            'seed_id': r['seed_id'],
            'clause_id': pid,
            'method': r['method'],
            'question': r['question'],
            'raw_response': r.get('raw_response', ''),
            'truncated_at_max_tokens': bool(r.get('truncated_at_max_tokens') or r.get('hit_token_cap')),
            'reference_decision': insurance_decision(s['frame'], HELDOUT_POLICIES),
            'standalone_outcome': out_s,
            'granted_source_policy_outcome': out_g,
            'missing_slots': miss_s,
            'mutated_slots': mut_s,
            'diagnostic_category': diag_cat,
            'atomic_rubrics': build_atomic_clause_rubrics(frame, pol, anchor, f'{pid}.pdf'),
            'automated_verifier_explanation': explanation,
            'record_provenance': PROVENANCE,
        }
        w_flag, w_cat, w_reason = build_watchlist_annotation(rec)
        rec.pop('raw_response', None)
        rec['watchlist_flag'] = w_flag
        if w_flag:
            rec['watchlist_category'] = w_cat
            rec['watchlist_reason'] = w_reason
        records.append(rec)

    jsonl_path = data_dir / 'item_by_item_semantic_audit.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    by_split = {}
    for split in ['dev_insurance_5clauses', 'multi_threshold_rules_4clauses', 'heldout_insurance_4clauses']:
        sub = [r for r in records if r['dataset_split'] == split]
        by_method = {}
        for m in sorted({r['method'] for r in sub}):
            m_sub = [r for r in sub if r['method'] == m]
            by_method[m] = {
                'count': len(m_sub),
                'standalone_outcomes': dict(Counter(r['standalone_outcome'] for r in m_sub)),
                'granted_source_policy_outcomes': dict(Counter(r['granted_source_policy_outcome'] for r in m_sub)),
                'diagnostic_categories': dict(Counter(r['diagnostic_category'] for r in m_sub)),
                'watchlist_count': sum(1 for r in m_sub if r['watchlist_flag']),
            }
        by_split[split] = {
            'total_records': len(sub),
            'watchlist_count': sum(1 for r in sub if r['watchlist_flag']),
            'methods': by_method,
        }

    watchlist_items = [
        {
            'audit_id': r['audit_id'],
            'dataset_split': r['dataset_split'],
            'method': r['method'],
            'question': r['question'],
            'standalone_outcome': r['standalone_outcome'],
            'missing_slots': r['missing_slots'],
            'mutated_slots': r['mutated_slots'],
            'watchlist_category': r.get('watchlist_category', ''),
            'watchlist_reason': r.get('watchlist_reason', ''),
        }
        for r in records
        if r['watchlist_flag']
    ]

    summary = {
        'generated_date': '2026-10-27',
        'total_records': len(records),
        'watchlist_count': len(watchlist_items),
        'record_provenance': PROVENANCE,
        'methodology_note': (
            'Deterministic item-by-item automated verifier explanations and a rule-based manual-inspection '
            'watchlist across the 5 development insurance policies (40 items), 4 multi-threshold rules '
            '(112 items), and 4 extension official insurance policies (112 items). Contract rules are linked '
            'to physical PDF page and bounding-box coordinates (`source_type = pdf_contract_clause`), while '
            'synthetic scenario facts are marked as `source_type = synthetic_seed_frame`. Because these explanations '
            'are produced directly from the deterministic parser and reachability verifier rather than an independent '
            'human or external judge, no circular agreement rate or false-positive/false-negative rate is reported.'
        ),
        'splits': by_split,
        'manual_inspection_watchlist': watchlist_items,
    }
    (data_dir / 'item_by_item_semantic_audit_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(f'Wrote {len(records)} automated verifier explanation records to {jsonl_path} (watchlist items: {len(watchlist_items)})')


if __name__ == '__main__':
    main()
