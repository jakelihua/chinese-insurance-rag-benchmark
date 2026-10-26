"""Unblind and score two independent human annotation JSON files (reviewer_A & reviewer_B).

Computes raw agreement, Cohen's kappa (for the 80 scenario action items and 40
generation fidelity items), compares against the provisional rule reference, and
outputs a disagreement adjudication sheet.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from experiment import write_json


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    records = data.get('records', [])
    by_id = {}
    for r in records:
        if r.get('id') and r.get('label'):
            by_id[r['id']] = {'label': r['label'].strip(), 'reason': r.get('reason', '').strip()}
    return data.get('reviewer', Path(path).stem), by_id


def compute_task_metrics(keys, map_a, map_b, mapping, reference_field=None):
    common = [k for k in sorted(keys) if k in map_a and k in map_b]
    if not common:
        return {'completed_common': 0, 'expected': len(keys), 'status': 'incomplete'}
    labels_a = [map_a[k]['label'] for k in common]
    labels_b = [map_b[k]['label'] for k in common]
    raw_agree = float(np.mean([a == b for a, b in zip(labels_a, labels_b)]))
    kappa = float(cohen_kappa_score(labels_a, labels_b)) if len(set(labels_a) | set(labels_b)) > 1 else 1.0
    disagreements = []
    for k in common:
        if map_a[k]['label'] != map_b[k]['label']:
            disagreements.append({
                'blind_id': k,
                'meta': mapping[k],
                'reviewer_A': map_a[k],
                'reviewer_B': map_b[k],
            })
    result = {
        'expected': len(keys),
        'completed_common': len(common),
        'raw_agreement': round(raw_agree, 4),
        'cohen_kappa': round(kappa, 4),
        'disagreement_count': len(disagreements),
        'disagreements': disagreements,
    }
    if reference_field:
        ref_agree_a = float(np.mean([map_a[k]['label'] == mapping[k][reference_field] for k in common]))
        ref_agree_b = float(np.mean([map_b[k]['label'] == mapping[k][reference_field] for k in common]))
        result['agreement_with_provisional_rule_A'] = round(ref_agree_a, 4)
        result['agreement_with_provisional_rule_B'] = round(ref_agree_b, 4)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reviewer-a', type=Path, required=True)
    p.add_argument('--reviewer-b', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding='utf-8'))
    rev_a, map_a = load_annotations(args.reviewer_a)
    rev_b, map_b = load_annotations(args.reviewer_b)

    action_keys = [k for k in mapping if k.startswith('A')]
    fidelity_keys = [k for k in mapping if k.startswith('F')]

    action_res = compute_task_metrics(action_keys, map_a, map_b, mapping, reference_field='reference_label')
    fidelity_res = compute_task_metrics(fidelity_keys, map_a, map_b, mapping)

    # Unblind fidelity counts by method where both reviewers agree
    method_fidelity = {'ordinary_style': {'agreed_preserved': 0, 'agreed_not_preserved': 0, 'disagreed_or_uncertain': 0, 'total': 0},
                       'condition_prompt': {'agreed_preserved': 0, 'agreed_not_preserved': 0, 'disagreed_or_uncertain': 0, 'total': 0}}
    for k in fidelity_keys:
        if k in map_a and k in map_b:
            m = mapping[k]['method']
            method_fidelity[m]['total'] += 1
            la, lb = map_a[k]['label'], map_b[k]['label']
            if la == lb == 'PRESERVED':
                method_fidelity[m]['agreed_preserved'] += 1
            elif la == lb == 'NOT_PRESERVED':
                method_fidelity[m]['agreed_not_preserved'] += 1
            else:
                method_fidelity[m]['disagreed_or_uncertain'] += 1

    summary = {
        'reviewer_a_id': rev_a,
        'reviewer_b_id': rev_b,
        'action_task_80': action_res,
        'fidelity_task_40': fidelity_res,
        'unblinded_fidelity_by_method': method_fidelity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
