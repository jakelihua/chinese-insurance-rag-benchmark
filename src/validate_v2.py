"""Compare LLM-generated query sets with real questions using 8 fixed systems."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import CountVectorizer

import experiment as base
from embed_v2 import Encoder

SYSTEMS = base.SYSTEMS + ['bge_small_zh', 'rrf_bm25_bge']


def order_metrics(order, gold):
    ranks = [i + 1 for i, doc in enumerate(order[:10]) if int(doc) in gold]
    rank = min(ranks) if ranks else None
    return [1 / rank if rank else 0.0, float(rank is not None and rank <= 5)]


def rrf_scores(lexical_scores, dense_scores, k=60, depth=100):
    if lexical_scores.shape != dense_scores.shape:
        raise ValueError('Fusion requires the same ordered candidate corpus')
    result = np.zeros(len(lexical_scores), dtype=float)
    for scores, require_positive in [(lexical_scores, True), (dense_scores, False)]:
        order = np.argsort(-scores, kind='stable')[:depth]
        if require_positive:
            order = order[scores[order] > 0]
        result[order] += 1 / (k + np.arange(1, len(order) + 1))
    return result


def compare(estimate, truth):
    chosen = int(np.argmax(estimate))
    tau = float(kendalltau(estimate, truth).statistic)
    return {'selected': SYSTEMS[chosen], 'regret': float(truth.max() - truth[chosen]),
            'tau_b': tau if np.isfinite(tau) else None,
            'optimal': bool(np.isclose(truth[chosen], truth.max(), atol=1e-12, rtol=0))}


def validate_logs(records, selected):
    by_mode = {'zero_shot': {}, 'few_shot_3': {}}
    for row in records:
        if row['mode'] not in by_mode or row['doc_id'] not in selected:
            raise ValueError('Unexpected mode/source in generation logs')
        if row['doc_id'] in by_mode[row['mode']]:
            raise ValueError('Duplicate generation key')
        by_mode[row['mode']][row['doc_id']] = row
    if any(set(rows) != set(selected) for rows in by_mode.values()):
        raise ValueError('Incomplete generation logs: do not silently drop missing jobs')
    common = sorted(set.intersection(*[
        {d for d, row in rows.items() if row['format_valid']} for rows in by_mode.values()]))
    return by_mode, common


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--generations', type=Path, required=True)
    p.add_argument('--embeddings', type=Path, required=True)
    p.add_argument('--bge-model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    inputs = json.loads(args.inputs.read_text())
    generation_manifest = json.loads((args.generations / 'manifest.json').read_text())
    embedding_manifest = json.loads((args.embeddings / 'manifest.json').read_text())
    expected_hash = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    if embedding_manifest['evaluation_inputs_sha256'] != expected_hash:
        raise ValueError('Embedding cache does not match evaluation inputs')
    preparation = json.loads((args.inputs.parent / 'preparation_manifest.json').read_text())
    if preparation['generation_inputs_sha256'] != generation_manifest['input_sha256']:
        raise ValueError('Generation inputs changed after freezing')
    records = [json.loads(line) for line in (args.generations / 'generations.jsonl').read_text().splitlines()]
    by_mode, common = validate_logs(records, inputs['selected_documents'])
    quality = {'attempted_per_mode': len(inputs['selected_documents']), 'common_valid_sources': len(common),
               'valid_counts': {m: sum(r['format_valid'] for r in rows.values()) for m, rows in by_mode.items()},
               'note': 'Format checks only; semantic validity and insurance correctness are not verified'}
    if len(common) < 32:
        base.write_json(args.out / 'generation_quality.json', quality)
        raise ValueError('Too few common valid sources; report generation failure before comparing systems')
    keywords, _ = base.synthetic_queries(inputs['documents'])
    datasets = {
        'real_calibration': ([r['question'] for r in inputs['calibration']], [set(r['gold']) for r in inputs['calibration']]),
        'real_test': ([r['question'] for r in inputs['test']], [set(r['gold']) for r in inputs['test']]),
        'keywords_matched': ([keywords[d] for d in common], [{d} for d in common]),
    }
    for mode in by_mode:
        datasets[mode] = ([by_mode[mode][d]['question'] for d in common], [{d} for d in common])
    lexical_results = base.evaluate_retrievers(inputs['documents'], datasets)
    cache = np.load(args.embeddings / 'embeddings.npz')
    embeddings = {name: cache[name] for name in ['documents', 'real_calibration', 'real_test']}
    encoder = Encoder(args.bge_model)
    truncation = {}
    for mode in ['keywords_matched', 'zero_shot', 'few_shot_3']:
        embeddings[mode], truncation[mode] = encoder.encode(datasets[mode][0], query=True)
    for name, array in embeddings.items():
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
            raise ValueError('Invalid cached embedding: ' + name)
    document_matrix = embeddings['documents'].astype(np.float64)
    # Run lexical and dense scores over exactly the same candidate order.
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    bm25_docs = base.bm25_documents(vectorizer.fit_transform(inputs['documents']))
    metrics = {}
    for name, (queries, gold) in datasets.items():
        print('Dense and fixed RRF:', name, flush=True)
        qx = vectorizer.transform(queries).astype(float)
        qx.data[:] = 1
        extra = np.zeros((len(queries), 2, 2))
        for start in range(0, len(queries), 128):
            end = min(start + 128, len(queries))
            dense = embeddings[name][start:end].astype(np.float64) @ document_matrix.T
            if not np.isfinite(dense).all():
                raise ValueError('Nonfinite similarity scores')
            lexical = (qx[start:end] @ bm25_docs.T).toarray()
            for j in range(end - start):
                i = start + j
                extra[i, 0] = order_metrics(np.argsort(-dense[j], kind='stable'), gold[i])
                fused = rrf_scores(lexical[j], dense[j])
                extra[i, 1] = order_metrics(np.argsort(-fused, kind='stable'), gold[i])
        metrics[name] = np.concatenate([lexical_results[name], extra], axis=1)
    # Native scores must match the first-round lexical experiment exactly.
    reference = np.load(Path(__file__).resolve().parents[1] / 'results/per_query_metrics.npz')
    for name in ['real_calibration', 'real_test']:
        if not np.array_equal(metrics[name][:, :6], reference[name]):
            raise ValueError('First-round lexical result mismatch; inspect data/index drift')
    np.savez_compressed(args.out / 'per_query_metrics.npz', **metrics)
    demos = inputs['demonstration_indices_within_calibration']
    pool = sorted(set(range(len(inputs['calibration']))) - set(demos))
    frozen_estimates = []
    for budget in [25, 50, 100, 200]:
        for repeat in range(5):
            seed = 20261026 + budget * 100 + repeat
            rng = np.random.default_rng(seed)
            selected = demos + rng.choice(pool, budget - len(demos), replace=False).tolist()
            cal_queries = [datasets['real_calibration'][0][i] for i in selected]
            cal_scores = metrics['real_calibration'][selected, :, 0]
            for mode in ['keywords_matched', 'zero_shot', 'few_shot_3']:
                print('Selection', budget, repeat, mode, flush=True)
                estimates, diagnostic = base.calibration_estimates(
                    cal_queries, cal_scores, datasets[mode][0], metrics[mode][:, :, 0], budget, seed)
                frozen_estimates.append((budget, repeat, mode, estimates, diagnostic))
    truth = metrics['real_test'][:, :, 0].mean(axis=0)
    trials = []
    for budget, repeat, mode, estimates, diagnostic in frozen_estimates:
        for method, estimated in estimates.items():
            trials.append({'budget': budget, 'repeat': repeat, 'generation': mode, 'method': method,
                           **compare(estimated, truth), **diagnostic,
                           'estimated_scores': estimated.tolist()})
    selection = []
    for budget in [25, 50, 100, 200]:
        for mode in ['keywords_matched', 'zero_shot', 'few_shot_3']:
            for method in frozen_estimates[0][3]:
                rows = [r for r in trials if r['budget'] == budget and r['generation'] == mode and r['method'] == method]
                taus = [r['tau_b'] for r in rows if r['tau_b'] is not None]
                selection.append({'budget': budget, 'generation': mode, 'method': method,
                                  'mean_regret': float(np.mean([r['regret'] for r in rows])),
                                  'mean_tau_b': float(np.mean(taus)) if taus else None,
                                  'optimal_fraction': float(np.mean([r['optimal'] for r in rows])),
                                  'domain_auc': float(np.mean([r['domain_auc'] for r in rows])),
                                  'density_ess': float(np.mean([r['density_ess'] for r in rows]))})
    table = []
    for j, system in enumerate(SYSTEMS):
        row = {'system': system}
        for name, values in metrics.items():
            row[name + '_mrr10'] = float(values[:, j, 0].mean())
            row[name + '_hit5'] = float(values[:, j, 1].mean())
        table.append(row)
    # Paired, pointwise descriptive intervals; not simultaneous significance claims.
    rng = np.random.default_rng(20261027)
    real = metrics['real_test'][:, :, 0]
    boot = np.stack([real[rng.integers(len(real), size=len(real))].mean(axis=0) for _ in range(1000)])
    confidence = []
    reference_index = SYSTEMS.index('bm25_char2')
    for j, system in enumerate(SYSTEMS):
        delta = boot[:, j] - boot[:, reference_index]
        confidence.append({'system': system, 'mrr10': float(truth[j]),
                           'pointwise_95_percentile_interval': np.quantile(boot[:, j], [0.025, 0.975]).tolist(),
                           'delta_vs_bm25_char2': float(truth[j] - truth[reference_index]),
                           'paired_pointwise_delta_interval': np.quantile(delta, [0.025, 0.975]).tolist()})
    for filename, value in [('generation_quality.json', quality), ('retrieval_summary.json', table),
                            ('calibration_summary.json', selection), ('trials.json', trials),
                            ('bootstrap_intervals.json', confidence)]:
        base.write_json(args.out / filename, value)
    base.write_json(args.out / 'manifest.json', {
        'evaluation_inputs_sha256': expected_hash, 'preparation': preparation,
        'generation': generation_manifest, 'embedding': embedding_manifest,
        'generation_log_sha256': hashlib.sha256((args.generations / 'generations.jsonl').read_bytes()).hexdigest(),
        'systems': SYSTEMS, 'matched_common_sources': common, 'synthetic_embedding_truncation': truncation,
        'rrf': {'k': 60, 'depth': 100}, 'elapsed_seconds': time.time() - started,
        'cosine_scoring_dtype': 'float64; finite values and embedding unit norms checked',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'status': 'exploratory, previously examined test set, single source, weak labels',
        'all_generation_rows_preserved_in_work_directory': str(args.generations.resolve()),
    })
    print('Completed v2; exact lexical reproduction check passed.', flush=True)


if __name__ == '__main__':
    main()
