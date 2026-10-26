"""Round 3 & Cross-Round Analytical Diagnostics (CPU-only, deterministic).

Computes:
1. Surface-normalized vs strict v1 slot-guard diagnostics and rule-decision
   corruption/collision rates on the 40 paired LLM generations in data_v3.
2. Clause retrieval discrimination (top-1 target policy identification and MRR)
   comparing complete, surface, missing_critical, counterfactual, ordinary_style,
   and condition_prompt queries over the 5 official clauses (standalone and
   embedded in the 3,746-document corpus).
3. Support-audited candidate-set abstention vs forced selection across the
   Round 2 validation_v2 per-query metric matrices (budgets 25/50/100/200).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import experiment as base
from condition_data import POLICIES, compact, decision
from condition_guard import chinese_number, extract_fields


def extract_fields_surface_normalized(text):
    """Surface-tolerant slot extractor used ONLY to diagnose v1 regex false negatives."""
    fields = extract_fields(text)
    norm = compact(text)
    num = r'([0-9零〇一二两三四五六七八九十百]+)'
    if fields['policy_day'] is None:
        day = re.findall(r'生效(?:算起|后|起|是|为|已经|已)?(?:的)?(?:第)?' + num + r'(?:天|日)', norm)
        if len(day) == 1:
            fields['policy_day'] = chinese_number(day[0])
    if fields['plan_wait_days'] is None:
        wait = re.findall(r'(?:计划|责任|约定).{0,10}等待期(?:为|是)?' + num + r'(?:天|日)', norm)
        if not wait:
            wait = re.findall(r'等待期(?:为|是)' + num + r'(?:天|日)', norm)
        if len(wait) == 1:
            fields['plan_wait_days'] = chinese_number(wait[0])
    if fields['approved_waiver'] is None:
        if re.search(
            r'(?:是否|有没有).{0,16}(?:批准|同意).{0,8}免(?:去|除)?等待期.{0,8}(?:还不清楚|尚不清楚|不清楚|不确定)',
            norm,
        ):
            fields['approved_waiver'] = None
        elif re.search(
            r'(?:未获|尚未获|没有获得|没获得|未曾获|并未获|未经|(?:未|没有)(?:获|获得|得到|批准|同意)).{0,12}(?:批准|同意)?.{0,8}免(?:去|除)?等待期',
            norm,
        ):
            fields['approved_waiver'] = False
        elif re.search(r'(?:已获|已经|已).{0,15}(?:批准|同意).{0,8}免(?:去|除)?等待期', norm):
            fields['approved_waiver'] = True
    return fields


def enumerate_admissible_decisions(observed, seed_frame=None, required_fields=None, grant_source_policy=False):
    """Enumerate all formal rule decisions reachable by completing missing fields
    for a known policy contract (either extracted from the query or granted by source context).
    Does not depend on seed_frame or required_fields unless grant_source_policy=True supplies policy_id.
    Covers all equivalence regions (< w, == w, > w) across all policy waiting periods {7, 15, 25, 30, 90}.
    """
    eff = dict(observed)
    if grant_source_policy and eff.get('policy_id') is None and seed_frame is not None:
        eff['policy_id'] = seed_frame['policy_id']
    pid = eff.get('policy_id')
    if pid not in POLICIES:
        return []
    policy = POLICIES[pid]
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
        if eff.get('approved_waiver') is not None:
            waivers = [eff['approved_waiver']]
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
                        decisions.add(decision(cand))
    return sorted(decisions)


def evaluate_rule_outcome_under_observed(observed, seed_frame, required_fields, grant_source_policy=False):
    """Evaluate whether a rewritten question preserves the rule-derived waiting-period outcome
    by enumerating all admissible completions of any missing required fields.

    Returns one of:
    - 'PRESERVED_DECISION': no required fields are mutated, and every admissible completion of any
      missing fields yields the exact same decision(frame) as the seed frame (including when a
      non-pivotal slot is omitted).
    - 'COUNTERFACTUAL_FLIP': resolved fields contain a mutated value that flips decision(frame).
    - 'CONDITION_MUTATED': resolved fields contain a mutated value without flipping decision(frame).
    - 'DEGRADED_TO_CLARIFY': one or more missing required fields admit disagreeing completions
      ('WAIT_EXCLUDED' vs 'WAIT_NOT_EXCLUDED'), making the question genuinely ambiguous.
    """
    eff = dict(observed)
    if grant_source_policy and eff.get('policy_id') is None:
        eff['policy_id'] = seed_frame['policy_id']
    mutated = [f for f in required_fields if eff.get(f) is not None and eff.get(f) != seed_frame[f]]
    missing = [f for f in required_fields if eff.get(f) is None]
    ref_dec = decision(seed_frame)
    if mutated:
        candidate = dict(seed_frame)
        for f in mutated:
            candidate[f] = eff[f]
        flipped = (candidate.get('policy_id') in POLICIES and decision(candidate) != ref_dec)
        return ('COUNTERFACTUAL_FLIP' if flipped else 'CONDITION_MUTATED'), mutated, missing
    if missing:
        reachable = enumerate_admissible_decisions(
            eff, seed_frame, required_fields, grant_source_policy=grant_source_policy
        )
        if reachable == [ref_dec]:
            return 'PRESERVED_DECISION', mutated, missing
        return 'DEGRADED_TO_CLARIFY', mutated, missing
    return 'PRESERVED_DECISION', mutated, missing


def evaluate_slot_completeness(observed, seed_frame, required_fields, grant_source_policy=False):
    """Separate strict slot-level completeness from formal rule decision determinacy."""
    outcome, mutated, missing = evaluate_rule_outcome_under_observed(
        observed, seed_frame, required_fields, grant_source_policy=grant_source_policy
    )
    if not mutated and not missing:
        return 'ALL_SLOTS_PRESERVED', outcome, mutated, missing
    if not mutated and missing and outcome == 'PRESERVED_DECISION':
        return 'NON_PIVOTAL_SLOT_OMITTED_DECISION_PRESERVED', outcome, mutated, missing
    return outcome, outcome, mutated, missing


def analyze_v3_generations(root, data_dir):
    inputs = json.loads((root / 'generation_inputs.json').read_text(encoding='utf-8'))
    seeds = {r['seed_id']: r for r in inputs['jobs']}
    gen_rows = [json.loads(s) for s in (data_dir / 'generated_questions.jsonl').read_text(encoding='utf-8').splitlines()]

    fields_order = ['policy_id', 'enrollment', 'policy_day', 'cause', 'plan_wait_days', 'approved_waiver']
    required_denominators = {
        f: sum(f in s['required_fields'] for s in seeds.values()) for f in fields_order
    }
    report = {
        'required_field_denominators': required_denominators,
        'methods': {},
        'collisions': [],
        'per_seed_comparison': [],
    }
    for method in ['ordinary_style', 'condition_prompt']:
        rows = [r for r in gen_rows if r['method'] == method]
        strict_pass = 0
        norm_pass = 0
        strict_single_qmark = 0
        strict_field_flags = {f: 0 for f in fields_order}
        norm_field_flags = {f: 0 for f in fields_order}
        standalone_outcomes = {'PRESERVED_DECISION': 0, 'DEGRADED_TO_CLARIFY': 0, 'CONDITION_MUTATED': 0, 'COUNTERFACTUAL_FLIP': 0}
        granted_policy_outcomes = {'PRESERVED_DECISION': 0, 'DEGRADED_TO_CLARIFY': 0, 'CONDITION_MUTATED': 0, 'COUNTERFACTUAL_FLIP': 0}
        standalone_slot_outcomes = {
            'ALL_SLOTS_PRESERVED': 0,
            'NON_PIVOTAL_SLOT_OMITTED_DECISION_PRESERVED': 0,
            'DEGRADED_TO_CLARIFY': 0,
            'CONDITION_MUTATED': 0,
            'COUNTERFACTUAL_FLIP': 0,
        }
        granted_slot_outcomes = {
            'ALL_SLOTS_PRESERVED': 0,
            'NON_PIVOTAL_SLOT_OMITTED_DECISION_PRESERVED': 0,
            'DEGRADED_TO_CLARIFY': 0,
            'CONDITION_MUTATED': 0,
            'COUNTERFACTUAL_FLIP': 0,
        }

        by_question_text = {}
        for r in rows:
            seed = seeds[r['seed_id']]
            frame = seed['frame']
            req = seed['required_fields']
            qmark_count = r['question'].count('？') + r['question'].count('?')
            if r['format_valid'] and qmark_count == 1:
                strict_single_qmark += 1
            for f in r.get('mismatched_or_unresolved_fields', []):
                if f in strict_field_flags:
                    strict_field_flags[f] += 1
            if r['format_valid'] and r['proxy_all_required_fields_preserved']:
                strict_pass += 1
            norm_obs = extract_fields_surface_normalized(r['question'])
            norm_fail = [f for f in req if norm_obs.get(f) != frame[f]]
            for f in norm_fail:
                norm_field_flags[f] += 1
            if r['format_valid'] and not norm_fail:
                norm_pass += 1

            slot_s, out_standalone, _, _ = evaluate_slot_completeness(norm_obs, frame, req, grant_source_policy=False)
            slot_g, out_granted, _, _ = evaluate_slot_completeness(norm_obs, frame, req, grant_source_policy=True)
            standalone_outcomes[out_standalone] += 1
            granted_policy_outcomes[out_granted] += 1
            standalone_slot_outcomes[slot_s] += 1
            granted_slot_outcomes[slot_g] += 1
            by_question_text.setdefault(r['question'], []).append((r['seed_id'], decision(frame), frame))

        for q_text, items in by_question_text.items():
            if len(items) > 1:
                labels = sorted({lbl for _, lbl, _ in items})
                report['collisions'].append({
                    'method': method,
                    'question': q_text,
                    'colliding_seeds': [sid for sid, _, _ in items],
                    'reference_labels': labels,
                    'opposite_label_collision': len(labels) > 1,
                })

        report['methods'][method] = {
            'attempts': len(rows),
            'format_valid': sum(r['format_valid'] for r in rows),
            'strict_single_question_mark_valid': strict_single_qmark,
            'strict_v1_proxy_pass': strict_pass,
            'strict_v1_proxy_flagged': len(rows) - strict_pass,
            'strict_v1_field_flags': strict_field_flags,
            'surface_normalized_proxy_pass': norm_pass,
            'surface_normalized_proxy_flagged': len(rows) - norm_pass,
            'v1_false_negative_on_surface_phrasing': norm_pass - strict_pass,
            'surface_normalized_field_flags': norm_field_flags,
            'standalone_slot_completeness_outcomes': standalone_slot_outcomes,
            'source_policy_granted_slot_completeness_outcomes': granted_slot_outcomes,
            'standalone_decision_outcomes': standalone_outcomes,
            'source_policy_granted_decision_outcomes': granted_policy_outcomes,
        }

    # Clause retrieval discrimination over the 5 official clauses and combined 3751-doc corpus
    evidence = json.loads((root / 'evidence_private.json').read_text(encoding='utf-8'))
    policy_ids = list(POLICIES.keys())
    clause_docs = [ f"《{POLICIES[pid]['title']}》 {evidence[pid]}" for pid in policy_ids ]

    cases = [json.loads(s) for s in (data_dir / 'controlled_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    query_groups = {}
    for variant in ['complete', 'surface', 'missing_critical', 'counterfactual']:
        sub = [c for c in cases if c['variant'] == variant]
        query_groups['controlled_' + variant] = ([c['question'] for c in sub], [{policy_ids.index(c['source_id'])} for c in sub])
    for method in ['ordinary_style', 'condition_prompt']:
        sub = [r for r in gen_rows if r['method'] == method]
        query_groups['generated_' + method] = ([r['question'] for r in sub], [{policy_ids.index(r['source_id'])} for r in sub])

    clause_metrics = base.evaluate_retrievers(clause_docs, query_groups)
    retrieval_table = {}
    for qname, arr in clause_metrics.items():
        retrieval_table[qname] = {}
        for s_idx, s_name in enumerate(base.SYSTEMS):
            mrr = float(arr[:, s_idx, 0].mean())
            top1 = float((arr[:, s_idx, 0] == 1.0).mean())
            retrieval_table[qname][s_name] = {'mrr10': round(mrr, 4), 'top1_accuracy': round(top1, 4)}
    report['clause_retrieval_5docs'] = retrieval_table
    return report


def bigram_containment(query, doc):
    q_s = re.sub(r'\s+', '', query)
    d_s = re.sub(r'\s+', '', doc)
    if len(q_s) < 2:
        return 0.0
    q_bg = {q_s[i:i + 2] for i in range(len(q_s) - 1)}
    d_bg = {d_s[i:i + 2] for i in range(max(0, len(d_s) - 1))}
    return len(q_bg & d_bg) / len(q_bg)


def analyze_v2_abstention_and_ppi(v2_dir):
    npz = np.load(v2_dir / 'per_query_metrics.npz')
    manifest = json.loads((v2_dir / 'manifest.json').read_text(encoding='utf-8'))
    trials = json.loads((v2_dir / 'trials.json').read_text(encoding='utf-8'))
    systems = manifest['systems']
    real_cal = npz['real_calibration'][:, :, 0]
    real_test = npz['real_test'][:, :, 0]
    truth = real_test.mean(axis=0)
    best_idx = int(np.argmax(truth))

    # Group existing trials by (budget, repeat, generation)
    trial_map = {}
    for t in trials:
        trial_map[(t['budget'], t['repeat'], t['generation'], t['method'])] = t

    demos = [79, 513, 623]
    eval_inputs_path = Path(manifest['all_generation_rows_preserved_in_work_directory']).parent / 'evaluation_inputs.json'
    cal_ov2 = None
    syn_ov2_by_mode = {}
    if eval_inputs_path.exists():
        eval_in = json.loads(eval_inputs_path.read_text(encoding='utf-8'))
        demos = eval_in['demonstration_indices_within_calibration']
        docs = eval_in['documents']
        cal_ov2 = np.array([max(bigram_containment(r['question'], docs[g]) for g in r['gold']) for r in eval_in['calibration']])
        common = manifest['matched_common_sources']
        common_set = set(common)
        gen_path = Path(manifest['all_generation_rows_preserved_in_work_directory']) / 'generations.jsonl'
        if gen_path.exists():
            gen_rows = [json.loads(line) for line in gen_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            for m_name in ['zero_shot', 'few_shot_3']:
                sub = [r for r in gen_rows if r['mode'] == m_name and r['doc_id'] in common_set]
                sub.sort(key=lambda r: common.index(r['doc_id']))
                syn_ov2_by_mode[m_name] = np.array([bigram_containment(r['question'], docs[r['doc_id']]) for r in sub])
        kw_queries, _ = base.synthetic_queries(docs)
        syn_ov2_by_mode['keywords_matched'] = np.array([bigram_containment(kw_queries[d], docs[d]) for d in common])
    pool = sorted(set(range(len(real_cal))) - set(demos))

    summary_rows = []
    for budget in [25, 50, 100, 200]:
        for mode in ['keywords_matched', 'zero_shot', 'few_shot_3']:
            syn_scores = npz[mode][:, :, 0]
            syn_mean = syn_scores.mean(axis=0)
            syn_ov2 = syn_ov2_by_mode.get(mode)

            forced_real_wrong = []
            forced_mix_wrong = []
            ppi_rectified_wrong = []
            ppi_rectified_regret = []
            overlap_gated_wrong = []
            overlap_gated_regret = []
            audit_wrong_confident = []
            audit_abstain_rate = []
            audit_set_coverage = []
            audit_mean_set_size = []

            for repeat in range(5):
                seed = 20261026 + budget * 100 + repeat
                rng = np.random.default_rng(seed)
                selected = demos + rng.choice(pool, budget - len(demos), replace=False).tolist()
                cal_sub = real_cal[selected]
                cal_mean = cal_sub.mean(axis=0)

                # 1. Forced real_only and mix_real_density_0.5 from existing trials
                t_real = trial_map[(budget, repeat, mode, 'real_only')]
                t_mix = trial_map[(budget, repeat, mode, 'mix_real_density_0.5')]
                forced_real_wrong.append(float(not t_real['optimal']))
                forced_mix_wrong.append(float(not t_mix['optimal']))

                # 2. Power-tuned / variance-weighted linear control variate (unpaired PPI-style shrinkage)
                cal_var = cal_sub.var(axis=0, ddof=1) / len(selected)
                syn_var = syn_scores.var(axis=0, ddof=1) / len(syn_scores)
                bias_sq = np.square(syn_mean - cal_mean)
                w_mse = cal_var / np.maximum(cal_var + syn_var + bias_sq, 1e-12)
                shrunk = (1.0 - w_mse) * cal_mean + w_mse * syn_mean
                chosen_shrunk = int(np.argmax(shrunk))
                ppi_rectified_wrong.append(float(chosen_shrunk != best_idx))
                ppi_rectified_regret.append(float(truth.max() - truth[chosen_shrunk]))

                # 2b. Relational Source-Overlap Gating (calibrating on query-to-source bigram containment)
                if cal_ov2 is not None and syn_ov2 is not None:
                    tau_cal = float(np.median(cal_ov2[selected]))
                    mask_ov = syn_ov2 <= tau_cal
                    if mask_ov.sum() < 5:
                        mask_ov = syn_ov2 <= float(np.quantile(syn_ov2, 0.20))
                    gated_mean = syn_scores[mask_ov].mean(axis=0)
                    chosen_gated = int(np.argmax(gated_mean))
                    overlap_gated_wrong.append(float(chosen_gated != best_idx))
                    overlap_gated_regret.append(float(truth.max() - truth[chosen_gated]))

                # 3. Support-Audited Candidate-Set / Abstention Rule:
                boot_rng = np.random.default_rng(seed + 99999)
                boot_idx = boot_rng.integers(0, len(selected), size=(200, len(selected)))
                boot_means = cal_sub[boot_idx].mean(axis=1)  # (200, 8)
                cal_winner = int(np.argmax(cal_mean))
                diffs = boot_means - boot_means[:, [cal_winner]]
                upper_95 = np.quantile(diffs, 0.95, axis=0)
                candidate_set = [s for s in range(len(systems)) if upper_95[s] >= 0.0]

                abstained = len(candidate_set) > 1
                confident_wrong = (not abstained) and (candidate_set[0] != best_idx)
                covered = best_idx in candidate_set

                audit_abstain_rate.append(float(abstained))
                audit_wrong_confident.append(float(confident_wrong))
                audit_set_coverage.append(float(covered))
                audit_mean_set_size.append(float(len(candidate_set)))

            row_out = {
                'budget': budget,
                'generation': mode,
                'real_only_wrong_selection_rate': float(np.mean(forced_real_wrong)),
                'mix_0.5_wrong_selection_rate': float(np.mean(forced_mix_wrong)),
                'mse_shrunk_wrong_selection_rate': float(np.mean(ppi_rectified_wrong)),
                'mse_shrunk_mean_regret': float(np.mean(ppi_rectified_regret)),
                'candidate_set_confident_wrong_rate': float(np.mean(audit_wrong_confident)),
                'candidate_set_abstention_rate': float(np.mean(audit_abstain_rate)),
                'candidate_set_optimal_coverage': float(np.mean(audit_set_coverage)),
                'candidate_set_mean_size': float(np.mean(audit_mean_set_size)),
            }
            if overlap_gated_wrong:
                row_out['source_overlap_gated_wrong_selection_rate'] = float(np.mean(overlap_gated_wrong))
                row_out['source_overlap_gated_mean_regret'] = float(np.mean(overlap_gated_regret))
            summary_rows.append(row_out)
    return {
        'date': '2026-10-26',
        'note': 'Exploratory retrospective evaluation on validation_v2 per_query_metrics.npz; not an independent confirmation set.',
        'rows': summary_rows,
    }


def rescore_multi_scale_summary(root, data_dir, v3_diag):
    """Deterministically rescore saved Qwen-Plus and Qwen-Max generation logs alongside local 1.5B."""
    inputs = json.loads((root / 'generation_inputs.json').read_text(encoding='utf-8'))
    seeds = {r['seed_id']: r for r in inputs['jobs']}
    fields_order = ['policy_id', 'enrollment', 'policy_day', 'cause', 'plan_wait_days', 'approved_waiver']
    denoms = v3_diag['required_field_denominators']

    multi = {
        'required_field_denominators': denoms,
        'qwen2.5-1.5b-instruct': v3_diag['methods'],
    }
    for model_key, subdir in [
        ('qwen-plus', 'generation_qwen_plus'),
        ('qwen-max', 'generation_qwen_max'),
    ]:
        gpath = root / subdir / 'generations.jsonl'
        if not gpath.exists():
            continue
        rows_all = [json.loads(line) for line in gpath.read_text(encoding='utf-8').splitlines() if line.strip()]
        model_summary = {}
        for method in ['ordinary_style', 'condition_prompt']:
            rows = [r for r in rows_all if r['method'] == method]
            strict_v1 = 0
            surface_v2 = 0
            strict_single_qmark = 0
            granted_all_slots = 0
            standalone_dec = 0
            granted_dec = 0
            standalone_non_pivotal = 0
            granted_non_pivotal = 0
            flips = 0
            mutations = 0
            slot_mismatches = {f: 0 for f in fields_order}
            for r in rows:
                seed = seeds[r['seed_id']]
                frame = seed['frame']
                req = seed['required_fields']
                qmark_count = r['question'].count('？') + r['question'].count('?')
                if r['format_valid'] and qmark_count == 1:
                    strict_single_qmark += 1
                if r['format_valid'] and r.get('proxy_all_required_fields_preserved'):
                    strict_v1 += 1
                norm_obs = extract_fields_surface_normalized(r['question'])
                norm_fail = [f for f in req if norm_obs.get(f) != frame[f]]
                for f in norm_fail:
                    slot_mismatches[f] += 1
                if r['format_valid'] and not norm_fail:
                    surface_v2 += 1
                slot_s, out_s, _, _ = evaluate_slot_completeness(norm_obs, frame, req, grant_source_policy=False)
                slot_g, out_g, _, _ = evaluate_slot_completeness(norm_obs, frame, req, grant_source_policy=True)
                if slot_g == 'ALL_SLOTS_PRESERVED':
                    granted_all_slots += 1
                if out_s == 'PRESERVED_DECISION':
                    standalone_dec += 1
                if out_g == 'PRESERVED_DECISION':
                    granted_dec += 1
                if slot_s == 'NON_PIVOTAL_SLOT_OMITTED_DECISION_PRESERVED':
                    standalone_non_pivotal += 1
                if slot_g == 'NON_PIVOTAL_SLOT_OMITTED_DECISION_PRESERVED':
                    granted_non_pivotal += 1
                if out_g == 'COUNTERFACTUAL_FLIP':
                    flips += 1
                if out_g in ('COUNTERFACTUAL_FLIP', 'CONDITION_MUTATED'):
                    mutations += 1
            model_summary[method] = {
                'attempts': len(rows),
                'format_valid': sum(r['format_valid'] for r in rows),
                'strict_single_question_mark_valid': strict_single_qmark,
                'strict_v1_preserved': strict_v1,
                'surface_v2_preserved': surface_v2,
                'granted_policy_all_slots_preserved': granted_all_slots,
                'standalone_decision_preserved': standalone_dec,
                'standalone_non_pivotal_slot_omitted_decision_preserved': standalone_non_pivotal,
                'granted_policy_decision_preserved': granted_dec,
                'granted_policy_non_pivotal_slot_omitted_decision_preserved': granted_non_pivotal,
                'counterfactual_flips': flips,
                'condition_mutations': mutations,
                'slot_mismatch_counts_v2': slot_mismatches,
            }
        multi[model_key] = model_summary
    base.write_json(data_dir / 'multi_scale_condition_summary.json', multi)
    return multi


def main():
    proj_root = Path(__file__).resolve().parents[1]
    work_root = proj_root.parents[1] / 'work'
    p = argparse.ArgumentParser()
    p.add_argument('--work-condition', type=Path, default=work_root / 'condition_study')
    p.add_argument('--data-v3', type=Path, default=proj_root / 'data_v3')
    p.add_argument('--results-v2', type=Path, default=proj_root / 'results' / 'validation_v2')
    args = p.parse_args()

    v3_diag = analyze_v3_generations(args.work_condition, args.data_v3)
    base.write_json(args.data_v3 / 'condition_drift_diagnostics.json', v3_diag)
    multi = rescore_multi_scale_summary(args.work_condition, args.data_v3, v3_diag)

    v2_abst = analyze_v2_abstention_and_ppi(args.results_v2)
    base.write_json(args.results_v2 / 'abstention_and_shrinkage_diagnostics.json', v2_abst)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        fields = ['policy_id', 'enrollment', 'policy_day', 'cause', 'plan_wait_days', 'approved_waiver']
        denoms = [v3_diag['required_field_denominators'][f] for f in fields]
        x = np.arange(len(fields))
        ord_norm_sat = [
            denoms[i] - v3_diag['methods']['ordinary_style']['surface_normalized_field_flags'][f]
            for i, f in enumerate(fields)
        ]
        cnd_strict_sat = [
            denoms[i] - v3_diag['methods']['condition_prompt']['strict_v1_field_flags'][f]
            for i, f in enumerate(fields)
        ]
        cnd_norm_sat = [
            denoms[i] - v3_diag['methods']['condition_prompt']['surface_normalized_field_flags'][f]
            for i, f in enumerate(fields)
        ]
        ord_pct = [100.0 * s / d for s, d in zip(ord_norm_sat, denoms)]
        cnd_strict_pct = [100.0 * s / d for s, d in zip(cnd_strict_sat, denoms)]
        cnd_norm_pct = [100.0 * s / d for s, d in zip(cnd_norm_sat, denoms)]
        axes[0].bar(x - 0.25, ord_pct, 0.24, label='ordinary_style (surface norm)')
        axes[0].bar(x, cnd_strict_pct, 0.24, label='condition_prompt (v1 strict)')
        axes[0].bar(x + 0.25, cnd_norm_pct, 0.24, label='condition_prompt (surface norm)')
        axes[0].set_xticks(
            x,
            [f"{f.replace('_', chr(10))}\n(n={d})" for f, d in zip(fields, denoms)],
            fontsize=7.5,
        )
        axes[0].set_ylim(0, 118)
        axes[0].set_ylabel('Required slot satisfaction rate (%)')
        axes[0].set_title('Per-slot preservation rate (1.5B, true denominator n)')
        axes[0].legend(fontsize=7.5, loc='lower right')

        cats = ['Qwen2.5-1.5B\n(Local Edge)', 'Qwen-Plus\n(Cloud Mid-Tier)', 'Qwen-Max\n(Flagship)']
        cx = np.arange(len(cats))
        ord_slot = [
            v3_diag['methods']['ordinary_style']['surface_normalized_proxy_pass'],
            multi.get('qwen-plus', {}).get('ordinary_style', {}).get('surface_v2_preserved', 0),
            multi.get('qwen-max', {}).get('ordinary_style', {}).get('surface_v2_preserved', 0),
        ]
        cnd_slot = [
            v3_diag['methods']['condition_prompt']['surface_normalized_proxy_pass'],
            multi.get('qwen-plus', {}).get('condition_prompt', {}).get('surface_v2_preserved', 0),
            multi.get('qwen-max', {}).get('condition_prompt', {}).get('surface_v2_preserved', 0),
        ]
        cnd_dec = [
            v3_diag['methods']['condition_prompt']['standalone_decision_outcomes']['PRESERVED_DECISION'],
            multi.get('qwen-plus', {}).get('condition_prompt', {}).get('standalone_decision_preserved', 0),
            multi.get('qwen-max', {}).get('condition_prompt', {}).get('standalone_decision_preserved', 0),
        ]
        axes[1].bar(cx - 0.24, ord_slot, 0.23, label='ordinary_style (slot match)')
        axes[1].bar(cx, cnd_slot, 0.23, label='condition_prompt (slot match)')
        axes[1].bar(cx + 0.24, cnd_dec, 0.23, label='condition_prompt (decision determined)')
        axes[1].set_xticks(cx, cats, fontsize=8)
        axes[1].set_ylim(0, 25)
        axes[1].set_ylabel('Seeds preserved (out of 20)')
        axes[1].set_title('Cross-scale slot match vs. decision determinacy')
        axes[1].legend(fontsize=7.5, loc='upper left')
        fig.tight_layout()
        fig.savefig(args.data_v3 / 'condition_preservation_comparison.png', dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print('Wrote condition_drift_diagnostics.json, multi_scale_condition_summary.json, and abstention_and_shrinkage_diagnostics.json')


if __name__ == '__main__':
    main()
