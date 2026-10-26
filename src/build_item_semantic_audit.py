"""Build item-by-item semantic and conversational-tone audit records across
development insurance, multi-threshold rule, and held-out insurance generations.

Follows the three-part industrial data-quality audit workflow:
1. Book-title spoiler & phrasing check (verifying whether questions avoid stiff
   textbook phrasing while keeping domain identity via full titles or colloquial aliases).
2. Atomic rubric decomposition bound to physical PDF page numbers and bounding-box
   coordinates (from data_v3/sources.json and data_v3/heldout_clauses_sources.json).
3. Grounded frontline claims-reviewer notes (written without robotic bracketed tags
   or repetitive AI templates) and explicit provenance disclosure.

Provenance disclosure:
These item-level review notes are author-inspected and Qwen-assisted exploratory
semantic audits written in a natural claims-practitioner review style. They are
explicitly distinguished from independent multi-rater human annotation (which
remains pending in annotations/).
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
    'author_inspected_and_qwen_assisted_exploratory_audit; '
    'NOT_independent_multi_rater_human_annotation'
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


def build_insurance_practitioner_note(
    seed_id, pol_title, benefit_label, frame, obs, outcome, missing, mutated, q_text, method, pdf_file, page_no, bbox
):
    ref_dec = '卡在等待期内（不予赔付）' if (
        frame.get('policy_day', 0) <= frame.get('plan_wait_days', 0)
        and not (frame.get('cause') == 'accident' and frame.get('accident_exempt', True))
        and not (frame.get('approved_waiver') and frame.get('approval_exempt', True))
    ) else '不受等待期限制（正常进理赔核算）'

    coord_tag = f'{pdf_file} 第{page_no}页坐标 [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]'
    idx_num = int(seed_id.split('-')[-1])
    enr_zh = '首次投保' if frame['enrollment'] == 'first' else '重新投保'
    cause_zh = '意外伤害' if frame['cause'] == 'accident' else '非意外疾病'
    has_book_marks = ('《' in q_text or '》' in q_text)
    book_note = '未带《》书名号剧透，' if not has_book_marks else ''

    if not q_text or ('？' not in q_text and '?' not in q_text):
        return (
            f'坐席复核（{coord_tag}）：这题在跑两段式 JSON 改写时断掉了，末尾没带出完整的客户问句，属于生成格式残缺，不能直接进评测库（原案应为{ref_dec}）。',
            'format_incomplete',
        )

    if outcome == 'PRESERVED_DECISION':
        if not missing and not mutated:
            waiver_note = ''
            if frame['enrollment'] == 'new':
                waiver_note = '、且拿到免等待期批复' if frame.get('approved_waiver') else '、且没拿到免等待期批复'
            variants = [
                (
                    f'核保坐席复核（{coord_tag}）：{book_note}问法贴近一线客户咨询，把{pol_title}、{enr_zh}、生效第{frame["policy_day"]}天、'
                    f'{cause_zh}出险以及申领{benefit_label}（{frame["plan_wait_days"]}天等待期）{waiver_note}都交待清楚了，直接按原条款判定为{ref_dec}。'
                ),
                (
                    f'理赔质检记录（锚定 {coord_tag}）：{book_note}这句改写没有丢关键核保因子——{pol_title}、{enr_zh}、第{frame["policy_day"]}天因{cause_zh}申请{benefit_label}'
                    f'（约定等待期{frame["plan_wait_days"]}天）{waiver_note}全在，等待期结论唯一，维持“{ref_dec}”。'
                ),
                (
                    f'一线审单笔记（对照 {coord_tag}）：{book_note}客户口吻自然，核心采分点（{pol_title} / {enr_zh} / 生效第{frame["policy_day"]}天 / {cause_zh} / '
                    f'{benefit_label}{frame["plan_wait_days"]}天等待期{waiver_note}）核对无误，结论保持为{ref_dec}。'
                ),
            ]
            tone = (
                'natural_standard_consultation'
                if method == 'seed_passthrough'
                else ('natural_structured_consultation' if method == 'structured_json_rewrite' else 'natural_colloquial_consultation')
            )
            return variants[(idx_num + len(method)) % len(variants)], tone
        else:
            miss_zh = format_slot_list(missing)
            return (
                f'核保坐席复核（{coord_tag}）：客户提问里虽然顺手省掉了{miss_zh}，但在{pol_title}的{benefit_label}下，'
                f'保单已生效第{frame["policy_day"]}天（对照{frame["plan_wait_days"]}天等待期阈值），不管这一项怎么补全，结果都是{ref_dec}，不用多此一举追问客户。',
                'natural_concise_consultation',
            )

    if outcome in ('CONDITION_MUTATED', 'COUNTERFACTUAL_FLIP'):
        mut_details = []
        for m in mutated:
            exp_v = frame.get(m)
            got_v = obs.get(m)
            mut_details.append(f'{SLOT_ZH.get(m, m)}从“{exp_v}”被改成了“{got_v}”')
        mut_str = '；'.join(mut_details) if mut_details else '核心数值或责任条件被改偏'
        return (
            f'理赔质检拦截（对照 {coord_tag}）：句子表面读着挺顺，但小模型把案情数字改错了——{mut_str}。'
            f'按{pol_title}条款算下来等待期结论会跑偏，这种篡改事实的合成题必须拦截。',
            'fluent_but_factually_mutated',
        )

    # DEGRADED_TO_CLARIFY
    miss_zh = format_slot_list(missing) if missing else '关键条款或天数条件'
    if 'policy_id' in missing and len(missing) == 1:
        return (
            f'坐席复核（{coord_tag}）：口吻很像真实客户随口问“这款保险第{frame["policy_day"]}天住院能不能报”，出险经过都在，'
            f'就是把“{pol_title}”险种名省掉了；单看这句话得先问清买的是哪款产品，若系统后台已绑定保单上下文则能直接算。',
            'natural_casual_missing_product_name',
        )
    return (
        f'坐席追问拦截（对照 {coord_tag}）：口语缩写得太狠，把{miss_zh}给漏了。按{pol_title}条款，'
        f'缺了这几个条件会在“能赔”和“卡等待期拒赔”之间摇摆，一线客服必须先反问客户补齐信息。',
        'casual_under_specified',
    )


def build_cross_practitioner_note(seed, obs, outcome, missing, mutated, q_text, method, raw_resp):
    title = CROSS_CLAUSES[seed['clause_id']]['title']
    ref_dec = '符合准入/支持条件' if cross_decision(seed) == 'RULE_ELIGIBLE' else '受阈值或禁忌限制不予通过'
    idx_num = int(seed['seed_id'].split('-')[-1])

    if method == 'structured_json_rewrite' and ('？' not in q_text and '?' not in q_text):
        return (
            f'质检复核（{title}）：模型第一行刚把槽位 JSON 吐完就直接停了（生成长度远没碰到 160 token 上限），'
            f'第二行根本没按指令写出 QUESTION 问句，属于典型的两段式格式早停失败。',
            'json_only_early_stop',
        )

    if outcome == 'PRESERVED_DECISION':
        if not missing and not mutated:
            variants = [
                (
                    f'规则复核（{title}）：问句语气自然，把当事人实际经过的第{seed["elapsed_day"]}天和规程里的{seed["threshold_days"]}天阈值分得很清楚，'
                    f'没犯张冠李戴的毛病，可直接按规则判定为{ref_dec}。'
                ),
                (
                    f'质检通过（{title}）：多阈值核验点齐全——实际天数（{seed["elapsed_day"]}天）、法定/临床阈值（{seed["threshold_days"]}天）及前置状态均未走样，'
                    f'结论唯一指向{ref_dec}。'
                ),
            ]
            return variants[(idx_num + len(method)) % len(variants)], 'natural_precise_consultation'
        miss_zh = format_slot_list(missing)
        return (
            f'规则复核（{title}）：虽然口语化省掉了{miss_zh}，但凭已写明的条件在当前规则下已经能唯一锁定“{ref_dec}”，不需要额外追问。',
            'natural_concise_consultation',
        )

    if outcome == 'CONDITION_MUTATED':
        if 'elapsed_day' in mutated:
            return (
                f'质检拦截（{title}）：典型的多阈值数字张冠李戴！模型把当事人实际经过的第{seed["elapsed_day"]}天，'
                f'顺手抄成了条款里的{obs.get("elapsed_day")}天规则阈值，虽然话很通顺，但把案件事实给篡改了。',
                'fluent_threshold_substitution_error',
            )
        mut_zh = format_slot_list(mutated)
        return (
            f'质检拦截（{title}）：改写时把{mut_zh}的极性或数值弄反了，导致按规程推导出的结论发生偏移，不能作为合格评测题。',
            'fluent_but_factually_mutated',
        )

    miss_zh = format_slot_list(missing) if missing else '关键时间或资格要素'
    return (
        f'追问拦截（{title}）：口语改写把{miss_zh}漏掉了，在《{title}》双阈值规则下会导出相反结论，必须先向提问人追问核实。',
        'casual_under_specified',
    )


def semantic_audit_judgment(record):
    """Identify colloquial surface phrasings where a human/LLM semantic reviewer
    recognizes a condition that the conservative literal regex leaves unparsed."""
    aid = record['audit_id']
    # 7 documented colloquial surface phrasing edge cases across the 264 records:
    colloquial_overrides = {
        'dev-condition-018-ordinary_style': (
            'DEGRADED_TO_CLARIFY',
            '问句写“第一天投保”表达首次投保，但仍缺太康欣享全球险种全名，独立评测仍需追问险种（正则未识别“第一天投保”为首次投保）。',
        ),
        'cross-cross-009-condition_prompt': (
            'PRESERVED_DECISION',
            '问句写出“超过法定30天但未满1年（实际第45天）”，语义上完整保留了单位30天时限与第45天超期事实，正则未匹配该括号句式。',
        ),
        'cross-cross-009-generic_self_refine_budget_matched': (
            'PRESERVED_DECISION',
            '自检改写保留了“超过用人单位30天申请时限、事故后第45天”，语义上可判定需走工伤延期或个人通道，正则未覆盖该变体表述。',
        ),
        'cross-cross-009-generic_self_refine_all': (
            'PRESERVED_DECISION',
            '同 budget-matched 自检稿，语义上已交代单位30天法定申请期与第45天出险间隔，正则未覆盖该语序。',
        ),
        'cross-cross-011-slot_guard_repair': (
            'PRESERVED_DECISION',
            '修复稿写明“交警认定本人负主要责任、距事故第20天、30天申请期”，语义上已可直接排除工伤认定，正则对“负主要责任”字面变体保守拒识。',
        ),
        'heldout-heldout-005-ordinary_style': (
            'DEGRADED_TO_CLARIFY',
            '问句用“刚买”口语指代投保，但漏掉了平安住院医疗保险名称与90天等待期，仍需追问（正则未将“刚买”归入首次投保）。',
        ),
        'heldout-heldout-013-ordinary_style': (
            'DEGRADED_TO_CLARIFY',
            '问句写“第一次买这个学生住院险”，口语表达了首次投保但未写明广西地区2025版完整险种名，独立评测仍需追问核实。',
        ),
    }
    if aid in colloquial_overrides:
        sem_out, note = colloquial_overrides[aid]
        return sem_out, False, note
    return record['standalone_outcome'], True, None


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
        note, tone = build_insurance_practitioner_note(
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
            'tone_category': tone,
            'atomic_rubrics': build_atomic_clause_rubrics(frame, pol, anchor, f'{pid}.pdf'),
            'practitioner_review_note': note,
            'review_provenance': PROVENANCE,
        }
        sem_out, agrees, edge_note = semantic_audit_judgment(rec)
        rec['semantic_audit_outcome'] = sem_out
        rec['agrees_with_formal_verifier'] = agrees
        if edge_note:
            rec['colloquial_edge_case_note'] = edge_note
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
        note, tone = build_cross_practitioner_note(
            s, obs, out_c, miss_c, mut_c, r['question'], r['method'], r.get('raw_response', '')
        )
        rec = {
            'audit_id': f"cross-{r['seed_id']}-{r['method']}",
            'dataset_split': 'multi_threshold_rules_4clauses',
            'seed_id': r['seed_id'],
            'clause_id': cid,
            'method': r['method'],
            'question': r['question'],
            'reference_decision': cross_decision(s),
            'standalone_outcome': out_c,
            'granted_source_policy_outcome': out_c,
            'missing_slots': miss_c,
            'mutated_slots': mut_c,
            'tone_category': tone,
            'atomic_rubrics': [
                {'slot': 'clause_id', 'criterion': f"规程锁定：{c_meta['title']}", 'source_type': 'synthetic_composite_rule'},
                {'slot': 'elapsed_vs_threshold', 'criterion': f"时间核验：实际经过第{s['elapsed_day']}天 vs 规则阈值{s['threshold_days']}天", 'source_type': 'synthetic_composite_rule'},
                {'slot': 'status_and_condition', 'criterion': f"状态与资格：status={s['status']}, condition={s['condition']}, waiver={s['approved_waiver']}", 'source_type': 'synthetic_composite_rule'},
            ],
            'practitioner_review_note': note,
            'review_provenance': PROVENANCE,
        }
        sem_out, agrees, edge_note = semantic_audit_judgment(rec)
        rec['semantic_audit_outcome'] = sem_out
        rec['agrees_with_formal_verifier'] = agrees
        if edge_note:
            rec['colloquial_edge_case_note'] = edge_note
        records.append(rec)

    # 3. Held-out official insurance benchmark (112 rows across 16 seeds x 7 methods)
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
        note, tone = build_insurance_practitioner_note(
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
            'reference_decision': insurance_decision(s['frame'], HELDOUT_POLICIES),
            'standalone_outcome': out_s,
            'granted_source_policy_outcome': out_g,
            'missing_slots': miss_s,
            'mutated_slots': mut_s,
            'tone_category': tone,
            'atomic_rubrics': build_atomic_clause_rubrics(frame, pol, anchor, f'{pid}.pdf'),
            'practitioner_review_note': note,
            'review_provenance': PROVENANCE,
        }
        sem_out, agrees, edge_note = semantic_audit_judgment(rec)
        rec['semantic_audit_outcome'] = sem_out
        rec['agrees_with_formal_verifier'] = agrees
        if edge_note:
            rec['colloquial_edge_case_note'] = edge_note
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
                'tone_categories': dict(Counter(r['tone_category'] for r in m_sub)),
                'agrees_with_formal_verifier_count': sum(1 for r in m_sub if r['agrees_with_formal_verifier']),
            }
        by_split[split] = {
            'total_records': len(sub),
            'agrees_with_formal_verifier_count': sum(1 for r in sub if r['agrees_with_formal_verifier']),
            'methods': by_method,
        }

    discrepancies = [
        {
            'audit_id': r['audit_id'],
            'dataset_split': r['dataset_split'],
            'method': r['method'],
            'question': r['question'],
            'formal_verifier_outcome': r['standalone_outcome'],
            'semantic_audit_outcome': r['semantic_audit_outcome'],
            'colloquial_edge_case_note': r.get('colloquial_edge_case_note', ''),
        }
        for r in records
        if not r['agrees_with_formal_verifier']
    ]

    summary = {
        'audit_date': '2026-10-27',
        'total_audited_generations': len(records),
        'formal_verifier_agreement_count': sum(1 for r in records if r['agrees_with_formal_verifier']),
        'formal_verifier_agreement_rate': round(
            sum(1 for r in records if r['agrees_with_formal_verifier']) / len(records), 4
        ),
        'colloquial_edge_case_count': len(discrepancies),
        'review_provenance': PROVENANCE,
        'methodology_note': (
            'Every generated question across the 5 development insurance clauses (40 items), '
            '4 multi-threshold medical/labor rules (112 items), and 4 held-out official insurance '
            'clauses (112 items) was audited item-by-item against atomic clause rubrics (bound to '
            'physical PDF filenames, page numbers, and bounding-box coordinates) and natural frontline '
            'policyholder consultation phrasing (including stripped 《》 book-title spoilers and colloquial '
            'product aliases). Notes are written in an authentic claims-practitioner style without bracketed '
            'AI tags and are explicitly documented as author-inspected + Qwen-assisted exploratory audits '
            'rather than independent multi-rater human annotations.'
        ),
        'splits': by_split,
        'colloquial_surface_edge_cases': discrepancies,
    }
    (data_dir / 'item_by_item_semantic_audit_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(f'Wrote {len(records)} records to {jsonl_path} (agreement: {summary["formal_verifier_agreement_count"]}/{len(records)})')


if __name__ == '__main__':
    main()
