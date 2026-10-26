# Clause-Grounded Decision-Preserving Verification for Synthetic RAG Evaluation
### Eligibility-Condition Drift, Out-of-Sample Policy Confirmation, and Retriever Selection Diagnostics in Chinese Insurance

**Author:** Shunchi Wang (Independent Researcher, `jakelihua@gmail.com`)  
**Paper Artifacts:**
- **LaTeX Source & Overleaf Package:** [`paper/overleaf_package/main.tex`](paper/overleaf_package/main.tex) | [`paper/overleaf_submission.zip`](paper/overleaf_submission.zip)
- **English Manuscript (Markdown):** [`paper/draft_en.md`](paper/draft_en.md)
- **Chinese Manuscript:** [`paper/draft_zh.md`](paper/draft_zh.md)

---

## Overview

When benchmarking a vertical-domain retrieval-augmented generation (RAG) system, engineering teams often synthesize evaluation queries from source passages using instruction-tuned LLMs. Across community Q&A retrieval, official insurance policy wordings ($5$ development policies and $4$ held-out policies), and equal-call-budget baseline comparisons spanning three model tiers (`Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, and `Qwen-Max`), we show that high retrieval scores—even $100\%$ Top-1—do not imply that a synthetic query preserves its ground-truth business decision or clarification state. No LLM or embedding model weights are pre-trained or fine-tuned in this repository; all experiments evaluate inference-time generation, dense/lexical indexing, statistical calibration, and symbolic rule verification.

1. **Group-Disjoint Retrieval Benchmark ($3{,}746$ answers, $3{,}725$ connected groups, $2{,}994$ held-out real queries):**
   - Prevents duplicate-answer leakage via Union-Find connected-component splitting ($750$ calibration queries, $2{,}994$ held-out test queries, zero shared answers) on `baoxianzhidao_filter.csv` (originating via `meowcao/InsuranceModel` $\leftarrow$ `SophonPlus/ChineseNlpCorpus` from historical Baidu Zhidao community Q&A with weak best-answer labels and unverified open redistribution licensing).
   - Naive source-prefix copy proxies (`source_prefix_diagnostic`) invert the ranking between `bm25_char2` ($0.3817$ real MRR@10 vs. $0.9311$ prefix MRR@10) and `tfidf_char3_5` ($0.2727$ real MRR@10 vs. $0.9330$ prefix MRR@10), incurring $0.1090$ selection regret.
2. **Source-Matched Decomposition, Exploratory Source-Overlap Gating, and Candidate-Set Abstention:**
   - On the full held-out real user query set ($N=2{,}994$), pure dense retrieval (`bge_small_zh`, $\text{MRR@10}=0.5216, \text{Hit@5}=0.6453$) outperforms hybrid fusion (`rrf_bm25_bge`, $\text{MRR@10}=0.5030, \text{Hit@5}=0.6075$; paired $\Delta = +0.0187$, 10,000-resample 95% bootstrap CI $[+0.0077, +0.0296]$; zero-truncation sliding-window `Full-MaxP` over all $509$ long documents [$762$ overflow tail chunks across $1{,}271$ total windows] confirms $\text{MRR@10}=0.5165$ vs. $0.5007$, $\Delta = +0.0158$).
   - **Decomposing Source-Sampling Confounding vs. Within-Source Lexical Copy Inflation:** Both local `Qwen2.5-1.5B-Instruct` ($N=249$ common valid sources) and cloud `Qwen-Plus` ($N=179$ common valid sources) rank `rrf_bm25_bge` first under both `zero_shot` ($0.5647$ and $0.7371$) and `few_shot_3` ($0.4124$ and $0.6813$). Aligning real test queries with synthetic queries on the same target documents ($n=209$ queries / $207$ docs for `1.5B`; $n=153$ queries / $151$ docs for `Qwen-Plus`) shows that (a) small-subsample evaluation introduces source-sampling confounding where real RRF already slightly leads real BGE in point estimate ($0.4975$ vs. $0.4696$ and $0.4980$ vs. $0.4668$, with 95% CIs $[-0.0645, +0.0096]$ and $[-0.0687, +0.0073]$ spanning zero), and (b) while `zero_shot` synthesis increases query-to-source bigram containment $\kappa_2(q', d^*)$ ($0.3253 \rightarrow 0.4775$ on `1.5B` and $0.3167 \rightarrow 0.4638$ on `Qwen-Plus`) and boosts `bm25_char2` by $+0.0925$ ($95\%$ CI $[+0.0227, +0.1633]$) and $+0.2457$ ($[+0.1695, +0.3216]$) MRR@10 on those paired queries, the paired difference-in-differences in RRF-over-BGE advantage ($\Delta\Delta_{\text{RRF}-\text{BGE}} = +0.0123$, CI $[-0.0375, +0.0649]$ on `1.5B zero_shot`; $-0.0020$, CI $[-0.0479, +0.0438]$ on `Qwen-Plus zero_shot`) spans zero across all four settings, meaning within-source phrasing effects on the relative RRF$-$BGE gap cannot be statistically separated from source-subsampling variance at current sample sizes.
   - **Exploratory Relational Source-Overlap Gating (`source_overlap_gated`) vs. Retriever-Selection Candidate-Set Abstention:** While marginal query-feature density reweighting fails ($0\%$ optimal selection rate), an exploratory retrospective gate on query-to-source bigram containment $\kappa_2(q', d^*) \le \hat{\tau}_{\text{cal}}^{(B)}$ recovers `bge_small_zh` across all calibrated budgets $B \in \{25, 50, 100, 200\}$ and cold-start quantiles $q \in [0.10, 0.30]$ (flipping back to RRF once high-overlap queries enter at $q \ge 0.40\text{--}0.50$, whereas 1,000 matched-size random subsamples at $Q_{0.20}$ select `bge_small_zh` in only $5.9\%$ and $18.2\%$ of draws). Meanwhile, paired-bootstrap candidate-set abstention on `real_only` returns $\widehat{\mathcal{C}}_{0.95}$ when small calibration budgets cannot statistically separate top contenders, reducing confident wrong commitments at $B=25$ from $60\%\text{--}100\%$ to $0.0\%$ ($80\%$ abstention, mean set size $2.60$ of $8$, $100\%$ optimal coverage).
3. **Clause-Grounded Decision Verification Across $9$ Official Insurance Policies ($5$ Development + $4$ Held-Out):**
   - Across five development official Chinese insurance policy wordings (`pa_p0781`, `pa_h865`, `cpic_86017`, `cpic_82859`, `tk_group_2018`; $20$ seeds, $80$ controlled cases), every `missing_critical` (`CLARIFY`) benchmark case is certified by two counterexample completion witnesses yielding opposite rule outcomes (`WAIT_EXCLUDED` vs. `WAIT_NOT_EXCLUDED`). Enumerating complete threshold-partitioned equivalence regions of missing slots (`enumerate_admissible_decisions`, never reading ground-truth `seed_frame` metadata) shows that `ordinary_style` preserves standalone decision determinacy in $30\%$ (`1.5B`), $0\%$ (`Qwen-Plus`), and $45\%$ (`Qwen-Max`), whereas `condition_prompt` reaches $95\%$ (`1.5B`) and $100\%$ (`Qwen-Plus`/`Qwen-Max`).
   - **Out-of-Sample Confirmation on 4 Held-Out Official Insurance Policies & Title-Masked Retrieval:** On four held-out official insurance policy PDFs (`pa_2011_36`, `pa_p0254`, `pa_p0321`, `cpic_93896`; $16$ seeds, $64$ controlled cases), the frozen development-set slot parser and reachability verifier (zero regex edits) certifies $100\%$ ($64/64$) of controlled cases, showing that `ordinary_style` gets high template novelty ($1-\kappa_2=0.6410$) but $0/16$ ($0.0\%$) standalone decision determinacy ($5/16$ [$31.2\%$] when granted the source policy title), whereas `condition_prompt` and `slot_guard_repair` achieve $15/16$ ($93.8\%$) automatic frozen-parser decision determinacy ($16/16$ under human semantic audit) and $0/16$ ($0.0\%$) condition mutations. Masking product titles and insurer names drops 9-clause Top-1 retrieval from $75.0\%$ to $31.2\%\text{--}56.2\%$ (`bm25_char2`) and from $62.5\%\text{--}81.2\%$ to $25.0\%\text{--}56.2\%$ (`bge_small_zh`).
4. **Equal-Call-Budget Baseline Comparison on the Multi-Threshold Rule-Compiler Stress Test ($16$ seeds, $64$ cases, $7$ methods):**
   - On a synthetic multi-threshold rule-compiler stress test inspired by healthcare clinical pathways and labor-law rules (`med_egfr_tki`, `med_vte_prophylaxis`, `law_work_injury`, `law_noncompete`), all 6 non-JSON methods achieve $100\%$ ($16/16$) Top-1 retrieval on both `bm25_char2` and `bge_small_zh` (both standard and title-masked), yet `ordinary_style` preserves decision determinacy in only $6.2\%$ ($1/16$), and 1-call `structured_json_rewrite` fails single-line format validation ($0/16$ valid, $31.2\%$ decision preservation).
   - At the same $26$-call budget ($16$ initial $+$ $10$ second-pass calls), an LLM self-check without symbolic parser feedback (`generic_self_refine_budget_matched`, $26$ calls; unchanged at $32$ calls in `generic_self_refine_all`) achieves $37.5\%$ ($6/16$) all-slot match and $50.0\%$ ($8/16$) decision determinacy with $25.0\%$ ($4/16$) threshold mutations, whereas symbolic parser-guided `slot_guard_repair` ($26$ calls) increases all-slot match to $68.8\%$ ($11/16$, $+31.3$ pp), increases formal decision determinacy to $75.0\%$ ($12/16$, $+25.0$ pp), and reduces parser-detected threshold mutations to $0.0\%$ ($0/16$).

---

## Repository Structure

```text
.
├── data_v3/                        # Official insurance policy, held-out confirmation & multi-threshold rule benchmarks
│   ├── controlled_cases.jsonl      # 80 dev insurance controlled cases (complete, surface, missing_critical, counterfactual)
│   ├── generated_questions.jsonl   # 40 paired local LLM rewrites (ordinary_style vs. condition_prompt)
│   ├── condition_drift_diagnostics.json
│   ├── multi_scale_condition_summary.json  # 1.5B vs. Qwen-Plus vs. Qwen-Max summary (120 rewrites)
│   ├── heldout_clauses_sources.json        # 4 unseen official insurance policy PDFs (SHA256 & bbox anchors)
│   ├── heldout_controlled_cases.jsonl      # 64 held-out controlled cases with double-completion witnesses
│   ├── heldout_generations.jsonl           # 112 held-out generation records across 7 methods
│   ├── cross_domain_controlled_cases.jsonl # 64 Healthcare & Legal controlled cases with double witnesses
│   ├── cross_domain_generations.jsonl      # 48 local generations (ordinary, condition_prompt, slot_guard_repair)
│   ├── cross_domain_equal_budget_generations.jsonl # 112 equal-call-budget comparison records across 7 methods
│   ├── cross_domain_summary.json           # Cross-domain preservation, mutation & retrieval summary
│   ├── equal_budget_and_heldout_summary.json # Held-out confirmation, title-masked retrieval & equal-budget summary
│   ├── condition_preservation_comparison.png
│   └── independent_review/         # Blinded two-rater HTML evaluation suite (reviewer_A.html & reviewer_B.html)
├── results/                        # Community Q&A retrieval & calibration outputs
│   ├── retrieval_summary.json      # Lexical retrieval metrics
│   ├── validation_v2/              # 1.5B dense/hybrid evaluation, abstention, source_overlap_gated & Full-MaxP
│   └── validation_v2_qwen_plus/    # Qwen-Plus cross-scale evaluation & source_overlap_gated
├── src/                            # Reproducible evaluation, generation, and diagnostic scripts
│   ├── experiment.py               # Group-disjoint split & lexical retrieval
│   ├── generate_v2.py              # Local LLM query synthesis
│   ├── generate_v2_api.py          # Cloud API multi-scale query synthesis
│   ├── validate_v2.py              # Dense/hybrid evaluation & bootstrap intervals
│   ├── condition_data.py           # Deterministic policy rule engine & counterexamples
│   ├── condition_generate.py       # Local LLM paired rewriting
│   ├── condition_generate_api.py   # Cloud API multi-scale paired rewriting
│   ├── cross_domain_study.py       # Synthetic multi-threshold rule-compiler stress test & slot_guard_repair
│   ├── equal_budget_and_heldout_study.py # Held-out 4-policy confirmation, title-masked retrieval & equal-budget baselines
│   ├── maxp_ablation.py            # Source-sampling confounding, quantile sensitivity & Full-MaxP ablation
│   ├── analyze_v3.py               # Slot completeness vs. decision determinacy, source_overlap_gated & abstention
│   └── score_annotations.py        # Blinded human annotation unblinding & Cohen's kappa scorer
├── tests/                          # Unit test suite (28/28 deterministic tests)
└── paper/                          # English & Chinese manuscripts + Overleaf LaTeX package
```

---

## Reproducing the Experiments & Running Unit Tests

### 1. Run the Deterministic Unit Test Suite (28 Tests)
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Group-Disjoint Split & Lexical Proxy Evaluation
```bash
python3 src/experiment.py --data ../../work/baoxianzhidao_filter.csv --out results --seed 20261025 --repeats 5
python3 src/make_report.py
```

### 3. LLM Query Synthesis & Retriever Selection Analysis (`1.5B` & Cloud API)
```bash
# Local Qwen2.5-1.5B-Instruct generation & evaluation
python3 src/generate_v2.py --inputs ../../work/validation_v2/generation_inputs.json --model ../../work/models/qwen --out ../../work/validation_v2/generation_float32 --batch-size 8 --dtype float32 --attention eager
python3 src/validate_v2.py --inputs ../../work/validation_v2/evaluation_inputs.json --generations ../../work/validation_v2/generation_float32 --embeddings ../../work/validation_v2/embedding_cache --bge-model ../../work/models/bge --out results/validation_v2

# Cloud API (Qwen-Plus) cross-scale generation & evaluation
DASHSCOPE_API_KEY="your_api_key" python3 src/generate_v2_api.py --inputs ../../work/validation_v2/generation_inputs.json --model-name qwen-plus --out ../../work/validation_v2/generation_qwen_plus --workers 8
python3 src/validate_v2.py --inputs ../../work/validation_v2/evaluation_inputs.json --generations ../../work/validation_v2/generation_qwen_plus --embeddings ../../work/validation_v2/embedding_cache --bge-model ../../work/models/bge --out results/validation_v2_qwen_plus
```

### 4. Official Policy Condition Verification, Held-Out Confirmation & Equal-Call-Budget Study
```bash
# Build 80 controlled cases with single-field counterexamples
python3 src/condition_data.py --root ../../work/condition_study --out data_v3

# Cloud API multi-scale paired rewriting (qwen-plus / qwen-max)
DASHSCOPE_API_KEY="your_api_key" python3 src/condition_generate_api.py --inputs ../../work/condition_study/generation_inputs.json --model-name qwen-plus --out ../../work/condition_study/generation_qwen_plus
DASHSCOPE_API_KEY="your_api_key" python3 src/condition_generate_api.py --inputs ../../work/condition_study/generation_inputs.json --model-name qwen-max --out ../../work/condition_study/generation_qwen_max

# Run cross-round diagnostics, cross-domain stress test rescoring, held-out & equal-budget rescoring, and Full-MaxP ablation
python3 src/analyze_v3.py
python3 src/cross_domain_study.py --score-only
python3 src/equal_budget_and_heldout_study.py --score-only
python3 src/maxp_ablation.py
```
