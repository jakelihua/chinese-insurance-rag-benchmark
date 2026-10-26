"""Reproducible Full-Document MaxP Ablation & Round 2 Source-Confounding / Gating Sensitivity Diagnostics.

Computes:
1. Full-Document MaxP (512-token window, 256-token stride) over the 509 documents exceeding 512 tokens
   (1,205 overflow chunks, 0 remaining truncated documents) on real_test (N=2,994).
2. Source-sampling confounding analysis: comparing the full real_test set (N=2,994) against the exact
   source-matched real_test subsets for Qwen2.5-1.5B (n=209 queries across 207 docs) and Qwen-Plus
   (n=153 queries across 151 docs), with 10,000-resample paired bootstrap 95% confidence intervals
   and within-source query-style lexical overlap (kappa_2) shifts.
3. Quantile sensitivity sweep (q in {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00})
   for relational source-overlap gating, plus matched-size random subsampling and query-length
   gating baselines.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer

import experiment as base
from embed_v2 import INSTRUCTION
from validate_v2 import SYSTEMS, order_metrics, rrf_scores


def bigram_containment(query, doc):
    q_s = re.sub(r'\s+', '', query)
    d_s = re.sub(r'\s+', '', doc)
    if len(q_s) < 2:
        return 0.0
    q_bg = {q_s[i:i + 2] for i in range(len(q_s) - 1)}
    d_bg = {d_s[i:i + 2] for i in range(max(0, len(d_s) - 1))}
    return len(q_bg & d_bg) / len(q_bg)


def paired_bootstrap_ci(diff_array, n_boot=10000, seed=20261026):
    rng = np.random.default_rng(seed)
    n = len(diff_array)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff_array[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff_array.mean()), [float(lo), float(hi)]


def compute_strata_and_sweep(ov2, bge_mrr, rrf_mrr, lengths, seed=20261026):
    strata = {}
    for s_name, mask in [
        ('low_lt_0.25', ov2 < 0.25),
        ('mid_0.25_to_0.50', (ov2 >= 0.25) & (ov2 < 0.50)),
        ('high_ge_0.50', ov2 >= 0.50),
    ]:
        if mask.sum() > 0:
            strata[s_name] = {
                'count': int(mask.sum()),
                'share': float(mask.mean()),
                'bge_mrr': float(bge_mrr[mask].mean()),
                'rrf_mrr': float(rrf_mrr[mask].mean()),
                'delta_bge_minus_rrf': float(bge_mrr[mask].mean() - rrf_mrr[mask].mean()),
            }

    q20 = float(np.quantile(ov2, 0.20))
    m20 = ov2 <= q20
    k20 = int(m20.sum())

    # Quantile sweep
    quantile_sweep = {}
    for q in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
        thr = float(np.quantile(ov2, q))
        m_q = ov2 <= thr
        b_m = float(bge_mrr[m_q].mean())
        r_m = float(rrf_mrr[m_q].mean())
        quantile_sweep[f'q_{int(round(q * 100)):02d}'] = {
            'quantile': q,
            'threshold': thr,
            'count': int(m_q.sum()),
            'bge_mrr': b_m,
            'rrf_mrr': r_m,
            'delta_bge_minus_rrf': b_m - r_m,
            'bge_selected': bool(b_m > r_m),
        }

    # Matched-size random subsampling baseline (1,000 draws of size k20)
    rng = np.random.default_rng(seed)
    rand_bge_wins = 0
    rand_deltas = []
    for _ in range(1000):
        sub_idx = rng.choice(len(ov2), size=k20, replace=False)
        d_val = float(bge_mrr[sub_idx].mean() - rrf_mrr[sub_idx].mean())
        rand_deltas.append(d_val)
        if d_val > 0:
            rand_bge_wins += 1

    # Query-length gating baselines (shortest k20 vs longest k20 queries)
    order_len = np.argsort(lengths, kind='stable')
    short_idx = order_len[:k20]
    long_idx = order_len[-k20:]
    short_delta = float(bge_mrr[short_idx].mean() - rrf_mrr[short_idx].mean())
    long_delta = float(bge_mrr[long_idx].mean() - rrf_mrr[long_idx].mean())

    return {
        'count': int(len(ov2)),
        'mean_kappa2': float(ov2.mean()),
        'median_kappa2': float(np.median(ov2)),
        'strata': strata,
        'cold_start_q20': {
            'threshold': q20,
            'count': k20,
            'bge_mrr': float(bge_mrr[m20].mean()),
            'rrf_mrr': float(rrf_mrr[m20].mean()),
            'delta_bge_minus_rrf': float(bge_mrr[m20].mean() - rrf_mrr[m20].mean()),
            'optimal_selected': bool(bge_mrr[m20].mean() > rrf_mrr[m20].mean()),
        },
        'quantile_sensitivity_sweep': quantile_sweep,
        'matched_size_baselines_at_q20': {
            'subset_size_k': k20,
            'random_subsample_1000_bge_win_rate': float(rand_bge_wins / 1000.0),
            'random_subsample_mean_delta_bge_minus_rrf': float(np.mean(rand_deltas)),
            'shortest_k_queries_delta_bge_minus_rrf': short_delta,
            'shortest_k_queries_bge_selected': bool(short_delta > 0),
            'longest_k_queries_delta_bge_minus_rrf': long_delta,
            'longest_k_queries_bge_selected': bool(long_delta > 0),
        },
    }


def run_maxp_ablation(eval_in, emb_npz, bge_model_path):
    docs = eval_in['documents']
    test_queries = [r['question'] for r in eval_in['test']]
    test_gold = [set(r['gold']) for r in eval_in['test']]

    doc_emb_512 = emb_npz['documents'].astype(np.float64)
    q_emb_test = emb_npz['real_test'].astype(np.float64)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(str(bge_model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(bge_model_path), local_files_only=True, use_safetensors=True).to(device).eval()

    overflow_doc_ids = []
    overflow_texts = []
    truncated_count = 0
    for d_idx, text in enumerate(docs):
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(ids) > 512:
            truncated_count += 1
            content_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            # First 510 content tokens are already covered by doc_emb_512; encode sliding windows with stride 256
            for start in range(256, len(content_ids), 256):
                window_ids = content_ids[start:start + 510]
                if not window_ids:
                    break
                chunk_text = tokenizer.decode(window_ids, skip_special_tokens=True)
                overflow_doc_ids.append(d_idx)
                overflow_texts.append(chunk_text)
                if start + 510 >= len(content_ids):
                    break

    chunk_vecs = []
    batch_size = 32
    for i in range(0, len(overflow_texts), batch_size):
        batch = overflow_texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.inference_mode():
            cls = model(**encoded).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        chunk_vecs.append(cls.cpu().numpy().astype(np.float64))
    overflow_matrix = np.concatenate(chunk_vecs, axis=0)
    overflow_doc_ids_arr = np.array(overflow_doc_ids, dtype=np.int64)

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    bm25_docs = base.bm25_documents(vectorizer.fit_transform(docs))
    qx = vectorizer.transform(test_queries).astype(float)
    qx.data[:] = 1

    trunc_bge, trunc_rrf = [], []
    maxp_bge, maxp_rrf = [], []

    for start in range(0, len(test_queries), 128):
        end = min(start + 128, len(test_queries))
        q_batch = q_emb_test[start:end]
        dense_512 = q_batch @ doc_emb_512.T
        dense_maxp = dense_512.copy()
        chunk_sim = q_batch @ overflow_matrix.T
        for j in range(end - start):
            np.maximum.at(dense_maxp[j], overflow_doc_ids_arr, chunk_sim[j])
        lexical = (qx[start:end] @ bm25_docs.T).toarray()
        for j in range(end - start):
            g = test_gold[start + j]
            trunc_bge.append(order_metrics(np.argsort(-dense_512[j], kind='stable'), g))
            fused_512 = rrf_scores(lexical[j], dense_512[j])
            trunc_rrf.append(order_metrics(np.argsort(-fused_512, kind='stable'), g))

            maxp_bge.append(order_metrics(np.argsort(-dense_maxp[j], kind='stable'), g))
            fused_maxp = rrf_scores(lexical[j], dense_maxp[j])
            maxp_rrf.append(order_metrics(np.argsort(-fused_maxp, kind='stable'), g))

    trunc_bge = np.array(trunc_bge)
    trunc_rrf = np.array(trunc_rrf)
    maxp_bge = np.array(maxp_bge)
    maxp_rrf = np.array(maxp_rrf)

    _, ci_512 = paired_bootstrap_ci(trunc_bge[:, 0] - trunc_rrf[:, 0])
    _, ci_maxp = paired_bootstrap_ci(maxp_bge[:, 0] - maxp_rrf[:, 0])

    return {
        'corpus_size': len(docs),
        'truncated_docs_at_512': truncated_count,
        'sliding_window_max_tokens': 512,
        'sliding_window_stride_tokens': 256,
        'overflow_tail_chunks_encoded': int(len(overflow_texts)),
        'remaining_truncated_docs': 0,
        'real_test_queries': len(test_queries),
        '512_trunc': {
            'bge_small_zh': {
                'mrr_at_10': float(trunc_bge[:, 0].mean()),
                'hit_at_5': float(trunc_bge[:, 1].mean()),
            },
            'rrf_bm25_bge': {
                'mrr_at_10': float(trunc_rrf[:, 0].mean()),
                'hit_at_5': float(trunc_rrf[:, 1].mean()),
            },
            'delta_bge_minus_rrf_mrr': float(trunc_bge[:, 0].mean() - trunc_rrf[:, 0].mean()),
            'paired_bootstrap_95_ci': ci_512,
        },
        'full_maxp': {
            'bge_small_zh': {
                'mrr_at_10': float(maxp_bge[:, 0].mean()),
                'hit_at_5': float(maxp_bge[:, 1].mean()),
            },
            'rrf_bm25_bge': {
                'mrr_at_10': float(maxp_rrf[:, 0].mean()),
                'hit_at_5': float(maxp_rrf[:, 1].mean()),
            },
            'delta_bge_minus_rrf_mrr': float(maxp_bge[:, 0].mean() - maxp_rrf[:, 0].mean()),
            'paired_bootstrap_95_ci': ci_maxp,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work-v2', type=Path, required=True)
    p.add_argument('--results-v2', type=Path, required=True)
    p.add_argument('--results-qwen-plus', type=Path, required=True)
    p.add_argument('--bge-model', type=Path, required=True)
    args = p.parse_args()

    eval_in = json.loads((args.work_v2 / 'evaluation_inputs.json').read_text(encoding='utf-8'))
    docs = eval_in['documents']
    emb_npz = np.load(args.work_v2 / 'embedding_cache' / 'embeddings.npz')

    v2_npz = np.load(args.results_v2 / 'per_query_metrics.npz')
    v2_manifest = json.loads((args.results_v2 / 'manifest.json').read_text(encoding='utf-8'))
    qp_npz = np.load(args.results_qwen_plus / 'per_query_metrics.npz')
    qp_manifest = json.loads((args.results_qwen_plus / 'manifest.json').read_text(encoding='utf-8'))

    idx_bm25 = SYSTEMS.index('bm25_char2')
    idx_bge = SYSTEMS.index('bge_small_zh')
    idx_rrf = SYSTEMS.index('rrf_bm25_bge')

    # Real test & calibration overlap and lengths
    test_ov2 = np.array([max(bigram_containment(r['question'], docs[g]) for g in r['gold']) for r in eval_in['test']])
    test_len = np.array([len(r['question']) for r in eval_in['test']])
    cal_ov2 = np.array([max(bigram_containment(r['question'], docs[g]) for g in r['gold']) for r in eval_in['calibration']])
    cal_len = np.array([len(r['question']) for r in eval_in['calibration']])

    real_test_mrr = v2_npz['real_test'][:, :, 0]
    real_cal_mrr = v2_npz['real_calibration'][:, :, 0]

    out = {
        'date': '2026-10-26',
        'note': (
            'Exploratory post-hoc diagnostic analysis separating source-sampling confounding from within-source '
            'query-style overlap inflation, with full quantile sensitivity sweeps and Full-MaxP ablation.'
        ),
        'real_test_strata': compute_strata_and_sweep(
            test_ov2, real_test_mrr[:, idx_bge], real_test_mrr[:, idx_rrf], test_len
        ),
        'real_calibration_strata': compute_strata_and_sweep(
            cal_ov2, real_cal_mrr[:, idx_bge], real_cal_mrr[:, idx_rrf], cal_len
        ),
    }

    # Synthetic strata & quantile sweeps for 1.5B and Qwen-Plus
    for label, manifest, npz_data in [
        ('qwen_1.5b', v2_manifest, v2_npz),
        ('qwen_plus', qp_manifest, qp_npz),
    ]:
        common = manifest['matched_common_sources']
        common_set = set(common)
        gen_dir = Path(manifest['all_generation_rows_preserved_in_work_directory'])
        gen_rows = [json.loads(line) for line in (gen_dir / 'generations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        out[label] = {}
        for mode in ['zero_shot', 'few_shot_3']:
            sub = [r for r in gen_rows if r['mode'] == mode and r['doc_id'] in common_set]
            sub.sort(key=lambda r: common.index(r['doc_id']))
            ov2 = np.array([bigram_containment(r['question'], docs[r['doc_id']]) for r in sub])
            lengths = np.array([len(r['question']) for r in sub])
            mrr = npz_data[mode][:, :, 0]
            mode_info = compute_strata_and_sweep(ov2, mrr[:, idx_bge], mrr[:, idx_rrf], lengths)
            diff_rrf_bge = mrr[:, idx_rrf] - mrr[:, idx_bge]
            mean_diff, ci_diff = paired_bootstrap_ci(diff_rrf_bge)
            mode_info['overall_bge_mrr10'] = float(mrr[:, idx_bge].mean())
            mode_info['overall_rrf_mrr10'] = float(mrr[:, idx_rrf].mean())
            mode_info['overall_bm25_char2_mrr10'] = float(mrr[:, idx_bm25].mean())
            mode_info['delta_rrf_minus_bge'] = mean_diff
            mode_info['delta_rrf_minus_bge_95_ci'] = ci_diff
            out[label][mode] = mode_info

    # Source-sampling confounding analysis: Full Real Test (N=2994) vs Source-Matched Real Test Subsets
    full_diff, full_ci = paired_bootstrap_ci(real_test_mrr[:, idx_bge] - real_test_mrr[:, idx_rrf])
    confounding = {
        'full_real_test': {
            'query_count': int(len(real_test_mrr)),
            'unique_gold_docs': int(len({g for r in eval_in['test'] for g in r['gold']})),
            'mean_kappa2': float(test_ov2.mean()),
            'bm25_char2_mrr10': float(real_test_mrr[:, idx_bm25].mean()),
            'bge_small_zh_mrr10': float(real_test_mrr[:, idx_bge].mean()),
            'rrf_bm25_bge_mrr10': float(real_test_mrr[:, idx_rrf].mean()),
            'delta_bge_minus_rrf': full_diff,
            'paired_bootstrap_95_ci': full_ci,
        },
        'matched_subsets': {},
    }
    for label, manifest, npz_data in [
        ('qwen_1.5b_matched_sources', v2_manifest, v2_npz),
        ('qwen_plus_matched_sources', qp_manifest, qp_npz),
    ]:
        common = manifest['matched_common_sources']
        common_set = set(common)
        common_pos = {d: idx for idx, d in enumerate(common)}
        matched_q_indices = []
        matched_syn_pos = []
        for i, r in enumerate(eval_in['test']):
            hits = [g for g in r['gold'] if g in common_pos]
            if hits:
                matched_q_indices.append(i)
                matched_syn_pos.append(common_pos[hits[0]])
        matched_docs_covered = sorted({g for i in matched_q_indices for g in eval_in['test'][i]['gold'] if g in common_set})
        sub_real = real_test_mrr[matched_q_indices]
        sub_ov2 = test_ov2[matched_q_indices]
        diff_m, ci_m = paired_bootstrap_ci(sub_real[:, idx_bge] - sub_real[:, idx_rrf])

        # Align synthetic queries both by unique covered doc IDs (207/151) and 1-to-1 with matched real queries (209/153)
        syn_idx = [common.index(d) for d in matched_docs_covered]
        zs_sub = npz_data['zero_shot'][syn_idx, :, 0]
        fs_sub = npz_data['few_shot_3'][syn_idx, :, 0]
        zs_q_aligned = npz_data['zero_shot'][matched_syn_pos, :, 0]
        fs_q_aligned = npz_data['few_shot_3'][matched_syn_pos, :, 0]

        confounding['matched_subsets'][label] = {
            'common_valid_synthetic_sources': len(common),
            'matched_real_test_queries': len(matched_q_indices),
            'matched_unique_target_docs': len(matched_docs_covered),
            'real_on_matched_sources': {
                'mean_kappa2': float(sub_ov2.mean()),
                'bm25_char2_mrr10': float(sub_real[:, idx_bm25].mean()),
                'bge_small_zh_mrr10': float(sub_real[:, idx_bge].mean()),
                'rrf_bm25_bge_mrr10': float(sub_real[:, idx_rrf].mean()),
                'delta_bge_minus_rrf': diff_m,
                'paired_bootstrap_95_ci': ci_m,
            },
            'zero_shot_query_aligned_on_matched_sources': {
                'pair_count': len(matched_syn_pos),
                'bm25_char2_mrr10': float(zs_q_aligned[:, idx_bm25].mean()),
                'bge_small_zh_mrr10': float(zs_q_aligned[:, idx_bge].mean()),
                'rrf_bm25_bge_mrr10': float(zs_q_aligned[:, idx_rrf].mean()),
                'delta_bge_minus_rrf': float(zs_q_aligned[:, idx_bge].mean() - zs_q_aligned[:, idx_rrf].mean()),
            },
            'few_shot_3_query_aligned_on_matched_sources': {
                'pair_count': len(matched_syn_pos),
                'bm25_char2_mrr10': float(fs_q_aligned[:, idx_bm25].mean()),
                'bge_small_zh_mrr10': float(fs_q_aligned[:, idx_bge].mean()),
                'rrf_bm25_bge_mrr10': float(fs_q_aligned[:, idx_rrf].mean()),
                'delta_bge_minus_rrf': float(fs_q_aligned[:, idx_bge].mean() - fs_q_aligned[:, idx_rrf].mean()),
            },
            'zero_shot_on_same_target_docs': {
                'doc_count': len(syn_idx),
                'bm25_char2_mrr10': float(zs_sub[:, idx_bm25].mean()),
                'bge_small_zh_mrr10': float(zs_sub[:, idx_bge].mean()),
                'rrf_bm25_bge_mrr10': float(zs_sub[:, idx_rrf].mean()),
                'delta_bge_minus_rrf': float(zs_sub[:, idx_bge].mean() - zs_sub[:, idx_rrf].mean()),
            },
            'few_shot_3_on_same_target_docs': {
                'doc_count': len(syn_idx),
                'bm25_char2_mrr10': float(fs_sub[:, idx_bm25].mean()),
                'bge_small_zh_mrr10': float(fs_sub[:, idx_bge].mean()),
                'rrf_bm25_bge_mrr10': float(fs_sub[:, idx_rrf].mean()),
                'delta_bge_minus_rrf': float(fs_sub[:, idx_bge].mean() - fs_sub[:, idx_rrf].mean()),
            },
        }
    out['source_sampling_confounding_analysis'] = confounding

    # Full-Document MaxP ablation
    print('Running Full-Document MaxP sliding-window ablation...', flush=True)
    out['maxp_full_document_ablation'] = run_maxp_ablation(eval_in, emb_npz, args.bge_model)

    out_path = args.results_v2 / 'mitigation_and_maxp_diagnostics.json'
    base.write_json(out_path, out)
    print(f'Wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
