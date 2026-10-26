"""Freeze generation-only inputs separately from real evaluation questions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import experiment as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items, documents, cal, test, audit = base.prepare(args.data, 20261025)
    rng = np.random.default_rng(20261026)
    demo_indices = sorted(rng.choice(len(cal), 3, replace=False).tolist())
    demo_item_indices = [cal[i] for i in demo_indices]
    excluded_documents = {d for i in demo_item_indices for d in items[i]['gold']}
    allowed = sorted(set(range(len(documents))) - excluded_documents)
    selected = sorted(rng.choice(allowed, 256, replace=False).tolist())
    generation = {
        'seed': 20261026,
        'demonstration_questions': [items[i]['question'] for i in demo_item_indices],
        'sources': [{'doc_id': i, 'answer': documents[i]} for i in selected],
    }
    evaluation = {
        'seed': 20261025, 'audit': audit, 'documents': documents,
        'calibration': [{'question': items[i]['question'], 'gold': items[i]['gold']} for i in cal],
        'test': [{'question': items[i]['question'], 'gold': items[i]['gold']} for i in test],
        'demonstration_indices_within_calibration': demo_indices,
        'selected_documents': selected,
    }
    assert not set(demo_item_indices).intersection(test)
    assert not excluded_documents.intersection(selected)
    base.write_json(args.out / 'generation_inputs.json', generation)
    base.write_json(args.out / 'evaluation_inputs.json', evaluation)
    base.write_json(args.out / 'preparation_manifest.json', {
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'generation_inputs_sha256': hashlib.sha256((args.out / 'generation_inputs.json').read_bytes()).hexdigest(),
        'evaluation_inputs_sha256': hashlib.sha256((args.out / 'evaluation_inputs.json').read_bytes()).hexdigest(),
        'selected_source_count': len(selected), 'demo_count': len(demo_item_indices),
        'demo_test_overlap': 0, 'demo_source_selected_overlap': 0,
    })
    print('Frozen 256 sources and 3 calibration demonstrations; generation file has no test questions.')


if __name__ == '__main__':
    main()
