"""Exploratory answer retrieval, with isolated calibration and final evaluation."""
import argparse
import csv
import hashlib
import json
import platform
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import sparse
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


SYSTEMS = ['tfidf_char1', 'tfidf_char2', 'tfidf_char1_3',
           'tfidf_char3_5', 'bm25_char1', 'bm25_char2']
PII = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|'
                 r'(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)0\d{2,3}[- ]?\d{7,8}(?!\d)')


def normalize(text):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', unicodedata.normalize('NFKC', text).lower())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def prepare(path, seed):
    with open(path, encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    required = {'title', 'question', 'reply', 'is_best'}
    if not raw or not required.issubset(raw[0]):
        raise ValueError('Missing dataset fields or empty CSV')
    grouped, excluded = {}, defaultdict(int)
    for row in raw:
        if row['is_best'].strip() != '1':
            excluded['not_best'] += 1
            continue
        title, detail, answer = [row[k].strip() for k in ['title', 'question', 'reply']]
        if detail.lower() == 'nan':
            detail = ''
        question = title if not detail or normalize(detail) in normalize(title) else title + ' ' + detail
        if not (4 <= len(question) <= 500 and 15 <= len(answer) <= 3000):
            excluded['length'] += 1
            continue
        if PII.search(question + '\n' + answer):
            excluded['pii_pattern'] += 1
            continue
        key = normalize(title)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = {'title': title, 'question': question, 'answers': set()}
        grouped[key]['answers'].add(answer)
    items = list(grouped.values())
    doc_by_key = {}
    for item in items:
        for answer in sorted(item['answers']):
            doc_by_key.setdefault(normalize(answer), answer)
    doc_keys = sorted(doc_by_key)
    doc_ids = {key: i for i, key in enumerate(doc_keys)}
    documents = [doc_by_key[key] for key in doc_keys]
    for item in items:
        item['gold'] = sorted({doc_ids[normalize(a)] for a in item.pop('answers')})
    if len(items) < 500:
        raise ValueError('Too few usable question groups for this protocol')
    uf, owner = UnionFind(len(items)), {}
    for i, item in enumerate(items):
        for doc in item['gold']:
            if doc in owner:
                uf.union(i, owner[doc])
            owner[doc] = i
    title_x = TfidfVectorizer(analyzer='char', ngram_range=(2, 4)).fit_transform(
        [normalize(item['title']) for item in items])
    similarity = (title_x @ title_x.T).tocoo()
    for a, b, value in zip(similarity.row, similarity.col, similarity.data):
        if a < b and value >= 0.90:
            uf.union(int(a), int(b))
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[uf.find(i)].append(i)
    rng = np.random.default_rng(seed)
    roots = rng.permutation(sorted(groups))
    cal = sorted(i for root in roots[:max(1, len(roots) // 5)] for i in groups[root])
    test = sorted(set(range(len(items))) - set(cal))
    if len(cal) < 200 or not test:
        raise ValueError('Insufficient calibration/test support')
    cal_docs = {d for i in cal for d in items[i]['gold']}
    test_docs = {d for i in test for d in items[i]['gold']}
    assert not cal_docs.intersection(test_docs)
    max_cross_similarity = float((title_x[cal] @ title_x[test].T).max())
    return items, documents, cal, test, {
        'raw_rows': len(raw), 'excluded': dict(excluded), 'question_groups': len(items),
        'documents': len(documents), 'connected_groups': len(groups),
        'calibration_pool': len(cal), 'test_questions': len(test),
        'max_cross_split_title_cosine': max_cross_similarity,
        'calibration_answer_overlap_test': 0,
        'provenance': 'Public mirror; upstream reports Baidu Zhidao; original posts not verified',
        'labels': 'Observed best-answer links, NOT verified insurance truth or exhaustive relevance',
    }


def synthetic_queries(documents):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=30000)
    x = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    queries = []
    for i, document in enumerate(documents):
        row = x.getrow(i)
        ordered = sorted(zip(row.indices, row.data), key=lambda v: (-v[1], terms[v[0]]))
        selected = []
        for index, _ in ordered:
            term = terms[index]
            if re.fullmatch(r'[\u4e00-\u9fff]{2,3}', term) and not any(term in s or s in term for s in selected):
                selected.append(term)
            if len(selected) == 6:
                break
        queries.append('请问' + ' '.join(selected) + '如何处理？')
    # Deliberately source-copying diagnostic, not a valid real-user surrogate.
    prefixes = [re.split(r'[。！？\n]', d)[0][:60] or d[:60] for d in documents]
    return queries, prefixes


def bm25_documents(counts, k1=1.2, b=0.75):
    counts = counts.tocsr().astype(np.float64)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    mean_length = max(float(lengths.mean()), 1e-12)
    dfs = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((counts.shape[0] - dfs + 0.5) / (dfs + 0.5))
    coo = counts.tocoo()
    values = idf[coo.col] * coo.data * (k1 + 1) / (
        coo.data + k1 * (1 - b + b * lengths[coo.row] / mean_length))
    return sparse.csr_matrix((values, (coo.row, coo.col)), shape=counts.shape)


def retrieval_metrics(scores, gold):
    # Positive scores only: empty/OOV queries do not receive arbitrary tie credit.
    order = np.argsort(-scores, kind='stable')[:10]
    ranks = [i + 1 for i, d in enumerate(order) if scores[d] > 0 and int(d) in gold]
    rank = min(ranks) if ranks else None
    return [1.0 / rank if rank else 0.0, float(rank is not None and rank <= 5)]


def evaluate_retrievers(documents, query_sets):
    all_queries, spans = [], {}
    for name, (queries, gold) in query_sets.items():
        spans[name] = (len(all_queries), len(all_queries) + len(queries), gold)
        all_queries.extend(queries)
    results = {name: np.zeros((len(q), len(SYSTEMS), 2)) for name, (q, _) in query_sets.items()}
    for system_index, system in enumerate(SYSTEMS):
        print('Retrieval:', system, flush=True)
        if system.startswith('tfidf'):
            lo, hi = {'tfidf_char1': (1, 1), 'tfidf_char2': (2, 2),
                      'tfidf_char1_3': (1, 3), 'tfidf_char3_5': (3, 5)}[system]
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(lo, hi), dtype=np.float64)
            docs = vectorizer.fit_transform(documents)
            queries_x = vectorizer.transform(all_queries)
        else:
            ngram = 1 if system == 'bm25_char1' else 2
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(ngram, ngram))
            docs = bm25_documents(vectorizer.fit_transform(documents))
            queries_x = vectorizer.transform(all_queries).astype(np.float64)
            queries_x.data[:] = 1.0
        for name, (start, end, gold) in spans.items():
            for offset in range(start, end, 128):
                scores = (queries_x[offset:min(end, offset + 128)] @ docs.T).toarray()
                for j, row in enumerate(scores):
                    i = offset - start + j
                    results[name][i, system_index] = retrieval_metrics(row, gold[i])
    return results


def ess(weights):
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum())


def length_weights(real_queries, synthetic):
    edges = np.array([0, 10, 20, 30, 50, 80, 120, np.inf])
    ri = np.digitize([len(q) for q in real_queries], edges[1:-1])
    si = np.digitize([len(q) for q in synthetic], edges[1:-1])
    # Half-count smoothing; clipping is explicit bias/variance regularization.
    p = np.bincount(ri, minlength=7) + 0.5
    q = np.bincount(si, minlength=7) + 0.5
    ratio = (p / p.sum()) / (q / q.sum())
    return np.clip(ratio[si], 0.1, 10.0)


def density_weights(real_queries, synthetic, seed):
    texts = list(real_queries) + list(synthetic)
    y = np.r_[np.ones(len(real_queries)), np.zeros(len(synthetic))]
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=4000)
    x = vectorizer.fit_transform(texts)
    predictions = np.zeros(len(texts))
    for train, valid in StratifiedKFold(2, shuffle=True, random_state=seed).split(x, y):
        classifier = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=500)
        classifier.fit(x[train], y[train])
        predictions[valid] = classifier.predict_proba(x[valid])[:, 1]
    prob = np.clip(predictions[len(real_queries):], 1e-4, 1 - 1e-4)
    # Balanced classes approximate equal priors, so odds approximate a density ratio.
    return np.clip(prob / (1 - prob), 0.1, 10.0), float(roc_auc_score(y, predictions))


def calibration_estimates(cal_queries, cal_scores, synthetic, synthetic_scores, budget, seed):
    """No test queries, test scores, or test relevance labels enter this function."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(cal_queries), budget, replace=False)
    qs = [cal_queries[i] for i in indices]
    real = cal_scores[indices].mean(axis=0)
    uniform = synthetic_scores.mean(axis=0)
    sw = length_weights(qs, synthetic)
    dw, auc = density_weights(qs, synthetic, seed)
    weighted = np.average(synthetic_scores, weights=dw, axis=0)
    sample = rng.choice(len(synthetic), min(budget, len(synthetic)), replace=False)
    return {
        'synthetic_uniform': uniform,
        'synthetic_budget_matched': synthetic_scores[sample].mean(axis=0),
        'synthetic_length_weighted': np.average(synthetic_scores, weights=sw, axis=0),
        'synthetic_density_weighted': weighted,
        'real_only': real,
        'mix_real_density_0.5': 0.5 * real + 0.5 * weighted,
    }, {'density_ess': ess(dw), 'length_ess': ess(sw), 'domain_auc': auc}


def compare(estimated, truth):
    selected = int(np.argmax(estimated))
    tau = float(kendalltau(estimated, truth).statistic)
    return {'selected': SYSTEMS[selected], 'regret': float(truth.max() - truth[selected]),
            'tau_b': tau if np.isfinite(tau) else None,
            'optimal': bool(np.isclose(truth[selected], truth.max(), atol=1e-12, rtol=0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    items, documents, cal, test, audit = prepare(args.data, args.seed)
    synthetic, prefixes = synthetic_queries(documents)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    query_sets = {
        'real_calibration': ([items[i]['question'] for i in cal], [set(items[i]['gold']) for i in cal]),
        'real_test': ([items[i]['question'] for i in test], [set(items[i]['gold']) for i in test]),
        'synthetic_keywords': (synthetic, [{i} for i in range(len(documents))]),
        'source_prefix_diagnostic': (prefixes, [{i} for i in range(len(documents))]),
    }
    metrics = evaluate_retrievers(documents, query_sets)
    np.savez_compressed(args.out / 'per_query_metrics.npz', **metrics)
    split_record = {
        'calibration': [hashlib.sha256(normalize(items[i]['title']).encode()).hexdigest() for i in cal],
        'test': [hashlib.sha256(normalize(items[i]['title']).encode()).hexdigest() for i in test],
    }
    write_json(args.out / 'split_hashes.json', split_record)
    estimates = []
    for budget in [25, 50, 100, 200]:
        for repeat in range(args.repeats):
            run_seed = args.seed + budget * 100 + repeat
            print('Calibration budget/repeat:', budget, repeat, flush=True)
            predicted, diagnostics = calibration_estimates(
                query_sets['real_calibration'][0], metrics['real_calibration'][:, :, 0],
                synthetic, metrics['synthetic_keywords'][:, :, 0], budget, run_seed)
            estimates.append((budget, repeat, predicted, diagnostics))
    # Read held-out outcomes only after all model-selection estimates are frozen.
    truth = metrics['real_test'][:, :, 0].mean(axis=0)
    rows = []
    for budget, repeat, predicted, diagnostics in estimates:
        for method, estimate in predicted.items():
            rows.append({'budget': budget, 'repeat': repeat, 'method': method,
                         **compare(estimate, truth), **diagnostics,
                         'estimated_scores': estimate.tolist()})
    summary = []
    for budget in [25, 50, 100, 200]:
        for method in estimates[0][2]:
            rr = [r for r in rows if r['budget'] == budget and r['method'] == method]
            taus = [r['tau_b'] for r in rr if r['tau_b'] is not None]
            summary.append({'budget': budget, 'method': method,
                            'mean_regret': float(np.mean([r['regret'] for r in rr])),
                            'mean_tau_b': float(np.mean(taus)) if taus else None,
                            'optimal_selection_fraction': float(np.mean([r['optimal'] for r in rr])),
                            'mean_density_ess': float(np.mean([r['density_ess'] for r in rr])),
                            'mean_domain_auc': float(np.mean([r['domain_auc'] for r in rr]))})
    write_json(args.out / 'audit.json', audit)
    write_json(args.out / 'trials.json', rows)
    write_json(args.out / 'calibration_summary.json', summary)
    table = []
    for j, system in enumerate(SYSTEMS):
        entry = {'system': system}
        for name, values in metrics.items():
            entry[name + '_mrr10'] = float(values[:, j, 0].mean())
            entry[name + '_hit5'] = float(values[:, j, 1].mean())
        table.append(entry)
    write_json(args.out / 'retrieval_summary.json', table)
    manifest = {'seed': args.seed, 'repeats': args.repeats, 'python': platform.python_version(),
                'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'sklearn': sklearn.__version__, 'elapsed_seconds': time.time() - started,
                'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_url': 'https://raw.githubusercontent.com/meowcao/InsuranceModel/main/data/csv/baoxianzhidao_filter.csv',
                'synthetic_method': 'deterministic extractive proxy, NOT LLM generation',
                'status': 'exploratory, weak relevance labels, single source, lexical retrievers only'}
    write_json(args.out / 'manifest.json', manifest)
    print('Completed:', manifest['elapsed_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
