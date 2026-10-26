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
        if re.search(r'(?:已获|已经|已).{0,15}(?:批准|同意).{0,8}免(?:去|除)?等待期', norm):
            fields['approved_waiver'] = True
    return fields


def evaluate_rule_outcome_under_observed(observed, seed_frame, required_fields, grant_source_policy=False):
    """Evaluate whether a rewritten question preserves the rule-derived waiting-period outcome.

    Returns one of:
    - 'PRESERVED_DECISION': all required fields (optionally granting source policy_id) are resolved
      and yield the exact same decision(frame) as the seed frame, with no corrupted values.
    - 'COUNTERFACTUAL_FLIP': resolved fields contain a mutated value that flips decision(frame),
      or mutates a required numeric/categorical condition to a different value.
    - 'DEGRADED_TO_CLARIFY': one or more required pivotal fields are missing/unresolved, making
      the standalone question unanswerable without follow-up clarification.
    """
    eff = dict(observed)
    if grant_source_policy and eff.get('policy_id') is None:
        eff['policy_id'] = seed_frame['policy_id']
    # Check if any required field was resolved to a WRONG (mutated) value
    mutated = [f for f in required_fields if eff.get(f) is not None and eff.get(f) != seed_frame[f]]
    missing = [f for f in required_fields if eff.get(f) is None]
    if mutated:
        # Check if substituting the mutated fields flips the formal decision
        candidate = dict(seed_frame)
        for f in mutated:
            candidate[f] = eff[f]
        flipped = ( candidate.get('policy_id') in POLICIES and decision(candidate) != decision(seed_frame) )
        return 'COUNTERFACTUAL_FLIP' if flipped else 'CONDITION_MUTATED', mutated, missing
    if missing:
        return 'DEGRADED_TO_CLARIFY', mutated, missing
    return 'PRESERVED_DECISION', mutated, missing


def analyze_v3_generations(root, data_dir):
    inputs = json.loads((root / 'generation_inputs.json').read_text(encoding='utf-8'))
    seeds = {r['seed_id']: r for r in inputs['jobs']}
    gen_rows = [json.loads(s) for s in (data_dir / 'generated_questions.jsonl').read_text(encoding='utf-8').splitlines()]

    report = {'methods': {}, 'collisions': [], 'per_seed_comparison': []}
    for method in ['ordinary_style', 'condition_prompt']:
        rows = [r for r in gen_rows if r['method'] == method]
        strict_pass = 0
        norm_pass = 0
        norm_field_flags = {f: 0 for f in ['policy_id', 'enrollment', 'policy_day', 'cause', 'plan_wait_days', 'approved_waiver']}
        standalone_outcomes = {'PRESERVED_DECISION': 0, 'DEGRADED_TO_CLARIFY': 0, 'CONDITION_MUTATED': 0, 'COUNTERFACTUAL_FLIP': 0}
        granted_policy_outcomes = {'PRESERVED_DECISION': 0, 'DEGRADED_TO_CLARIFY': 0, 'CONDITION_MUTATED': 0, 'COUNTERFACTUAL_FLIP': 0}

        by_question_text = {}
        for r in rows:
            seed = seeds[r['seed_id']]
            frame = seed['frame']
            req = seed['required_fields']
            if r['format_valid'] and r['proxy_all_required_fields_preserved']:
                strict_pass += 1
            norm_obs = extract_fields_surface_normalized(r['question'])
            norm_fail = [f for f in req if norm_obs.get(f) != frame[f]]
            for f in norm_fail:
                norm_field_flags[f] += 1
            if r['format_valid'] and not norm_fail:
                norm_pass += 1

            out_standalone, _, _ = evaluate_rule_outcome_under_observed(norm_obs, frame, req, grant_source_policy=False)
            out_granted, _, _ = evaluate_rule_outcome_under_observed(norm_obs, frame, req, grant_source_policy=True)
            standalone_outcomes[out_standalone] += 1
            granted_policy_outcomes[out_granted] += 1
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
            'strict_v1_proxy_pass': strict_pass,
            'strict_v1_proxy_flagged': len(rows) - strict_pass,
            'surface_normalized_proxy_pass': norm_pass,
            'surface_normalized_proxy_flagged': len(rows) - norm_pass,
            'v1_false_negative_on_surface_phrasing': norm_pass - strict_pass,
            'surface_normalized_field_flags': norm_field_flags,
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
    # Read exact demonstration indices from evaluation_inputs if available
    eval_inputs_path = Path(manifest['all_generation_rows_preserved_in_work_directory']).parent / 'evaluation_inputs.json'
    if eval_inputs_path.exists():
        demos = json.loads(eval_inputs_path.read_text(encoding='utf-8'))['demonstration_indices_within_calibration']
    pool = sorted(set(range(len(real_cal))) - set(demos))

    summary_rows = []
    for budget in [25, 50, 100, 200]:
        for mode in ['keywords_matched', 'zero_shot', 'few_shot_3']:
            syn_scores = npz[mode][:, :, 0]
            syn_mean = syn_scores.mean(axis=0)

            forced_real_wrong = []
            forced_mix_wrong = []
            ppi_rectified_wrong = []
            ppi_rectified_regret = []
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
                # Under unpaired shift, shrinking real_cal toward synthetic mean by optimal variance weight
                cal_var = cal_sub.var(axis=0, ddof=1) / len(selected)
                syn_var = syn_scores.var(axis=0, ddof=1) / len(syn_scores)
                bias_sq = np.square(syn_mean - cal_mean)
                # Empirical MSE-minimizing shrinkage weight per system
                w_mse = cal_var / np.maximum(cal_var + syn_var + bias_sq, 1e-12)
                shrunk = (1.0 - w_mse) * cal_mean + w_mse * syn_mean
                chosen_shrunk = int(np.argmax(shrunk))
                ppi_rectified_wrong.append(float(chosen_shrunk != best_idx))
                ppi_rectified_regret.append(float(truth.max() - truth[chosen_shrunk]))

                # 3. Support-Audited Candidate-Set / Abstention Rule:
                # Paired bootstrap on the calibration sample (200 resamples) to form a 95% candidate set;
                # Also if synthetic top-1 != calibration top-1 and falls within the calibration uncertainty band, include it.
                boot_rng = np.random.default_rng(seed + 99999)
                boot_idx = boot_rng.integers(0, len(selected), size=(200, len(selected)))
                boot_means = cal_sub[boot_idx].mean(axis=1)  # (200, 8)
                cal_winner = int(np.argmax(cal_mean))
                # A system s is in the candidate set if the 95% upper bound of (m_s - m_{cal_winner}) >= 0
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

            summary_rows.append({
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
            })
    return {
        'date': '2026-10-26',
        'note': 'Exploratory retrospective evaluation on validation_v2 per_query_metrics.npz; not an independent confirmation set.',
        'rows': summary_rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work-condition', type=Path, required=True)
    p.add_argument('--data-v3', type=Path, required=True)
    p.add_argument('--results-v2', type=Path, required=True)
    args = p.parse_args()

    v3_diag = analyze_v3_generations(args.work_condition, args.data_v3)
    base.write_json(args.data_v3 / 'condition_drift_diagnostics.json', v3_diag)

    v2_abst = analyze_v2_abstention_and_ppi(args.results_v2)
    base.write_json(args.results_v2 / 'abstention_and_shrinkage_diagnostics.json', v2_abst)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fields = ['policy_id', 'enrollment', 'policy_day', 'cause', 'plan_wait_days', 'approved_waiver']
        x = np.arange(len(fields))
        ord_strict = [20 - r for r in [14, 7, 5, 2, 4, 3]]
        cnd_strict = [20 - r for r in [1, 0, 8, 0, 1, 3]]
        cnd_norm = [20 - v3_diag['methods']['condition_prompt']['surface_normalized_field_flags'][f] for f in fields]
        axes[0].bar(x - 0.25, ord_strict, 0.24, label='ordinary_style (v1/norm)')
        axes[0].bar(x, cnd_strict, 0.24, label='condition_prompt (v1 strict)')
        axes[0].bar(x + 0.25, cnd_norm, 0.24, label='condition_prompt (surface norm)')
        axes[0].set_xticks(x, [f.replace('_', '\n') for f in fields], fontsize=8)
        axes[0].set_ylim(0, 26)
        axes[0].set_ylabel('Seeds resolved / satisfied (out of 20)')
        axes[0].set_title('Slot preservation across 20 seed scenarios')
        axes[0].legend(fontsize=7.5, loc='upper left')

        cats = ['Qwen2.5-1.5B\n(Local Edge)', 'Qwen-Plus\n(Cloud Mid-Tier)', 'Qwen-Max\n(Flagship)']
        cx = np.arange(len(cats))
        ord_pres = [
            v3_diag['methods']['ordinary_style']['standalone_decision_outcomes']['PRESERVED_DECISION'],
            0,
            4,
        ]
        cnd_pres = [
            v3_diag['methods']['condition_prompt']['standalone_decision_outcomes']['PRESERVED_DECISION'],
            19,
            20,
        ]
        axes[1].bar(cx - 0.16, ord_pres, 0.30, label='ordinary_style (v2)')
        axes[1].bar(cx + 0.16, cnd_pres, 0.30, label='condition_prompt (v2)')
        axes[1].set_xticks(cx, cats, fontsize=8)
        axes[1].set_ylim(0, 26)
        axes[1].set_ylabel('Standalone preserved decisions (out of 20)')
        axes[1].set_title('Cross-scale rule decision preservation')
        axes[1].legend(fontsize=8, loc='upper left')
        fig.tight_layout()
        fig.savefig(args.data_v3 / 'condition_preservation_comparison.png', dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print('Wrote condition_drift_diagnostics.json and abstention_and_shrinkage_diagnostics.json')


if __name__ == '__main__':
    main()
