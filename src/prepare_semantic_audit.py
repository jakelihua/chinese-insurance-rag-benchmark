"""Deterministic author-side audit sample; never alters retrieval qrels or scores."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from validate_v2 import validate_logs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--generations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    inputs = json.loads(args.inputs.read_text())
    records = [json.loads(s) for s in args.generations.read_text().splitlines()]
    modes, common = validate_logs(records, inputs['selected_documents'])
    if len(common) < 20:
        raise ValueError('Not enough paired valid sources for frozen audit sample')
    selected = sorted(np.random.default_rng(20261028).choice(common, 20, replace=False).tolist())
    args.out.mkdir(parents=True, exist_ok=True)
    sample = []
    for d in selected:
        for mode in ['zero_shot', 'few_shot_3']:
            sample.append({'doc_id': d, 'mode': mode, 'question': modes[mode][d]['question'],
                           'source_answer': inputs['documents'][d],
                           'support_label': '', 'review_note': ''})
    (args.out / 'semantic_audit_sample.json').write_text(
        json.dumps({'seed':20261028,'purpose':'AI-assisted author sanity review, not expert gold labels',
                    'sample':sample},ensure_ascii=False,indent=2))
    with (args.out / 'semantic_audit_template.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['doc_id','mode','support_label','review_note'])
        writer.writeheader()
        writer.writerows({k:row[k] for k in writer.fieldnames} for row in sample)
    print('Prepared 40 paired questions for author-side semantic audit; not used to filter scores.')


if __name__ == '__main__':
    main()
