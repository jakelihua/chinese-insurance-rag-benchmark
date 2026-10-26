# When Do Synthetic Queries Mislead Retriever Selection?
### Selection Reversal, Candidate-Set Abstention, and Eligibility-Condition Preservation in Chinese Insurance Retrieval

**Author:** Shunchi Wang (Independent Researcher, `jakelihua@gmail.com`)  
**Paper Artifacts:**
- 📄 **LaTeX Source & Overleaf Package:** [`paper/overleaf_package/main.tex`](paper/overleaf_package/main.tex) | [`paper/overleaf_submission.zip`](paper/overleaf_submission.zip)
- 🇬🇧 **English Manuscript (Markdown):** [`paper/draft_en.md`](paper/draft_en.md)
- 🇨🇳 **Chinese Manuscript (v0.3):** [`paper/draft_zh.md`](paper/draft_zh.md)

---

## Abstract & Key Findings

When cold-starting a vertical-domain retrieval-augmented generation (RAG) system, engineering teams routinely rely on rule-based proxies or LLM-generated synthetic queries to benchmark candidate retrievers before real user logs exist. Across a three-round diagnostic study on Chinese insurance retrieval and clause-level underwriting/claims eligibility verification spanning three model tiers (`Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, and `Qwen-Max`), we demonstrate:

1. **Round 1 — Group-Disjoint Retrieval Benchmark ($3{,}746$ answers, $3{,}725$ connected groups, $2{,}994$ held-out real queries):**
   - Prevents duplicate-answer leakage via Union-Find connected-component splitting ($750$ calibration queries, $2{,}994$ held-out test queries, zero shared answers).
   - Naive source-prefix copy proxies (`source_prefix_diagnostic`) invert the ranking between `bm25_char2` ($0.3817$ real MRR@10 vs. $0.9311$ prefix MRR@10) and `tfidf_char3_5` ($0.2727$ real MRR@10 vs. $0.9330$ prefix MRR@10), incurring $0.1090$ selection regret.
2. **Round 2 — Source-Matched Decomposition, Exploratory Source-Overlap Gating & Candidate-Set Abstention:**
   - On the full held-out real user query set ($N=2{,}994$), pure dense retrieval (`bge_small_zh`, $\text{MRR@10}=0.5216, \text{Hit@5}=0.6453$) significantly outperforms hybrid fusion (`rrf_bm25_bge`, $\text{MRR@10}=0.5030, \text{Hit@5}=0.6075$; paired $\Delta = +0.0187$, 10,000-resample 95% bootstrap CI $[+0.0077, +0.0296]$; zero-truncation sliding-window `Full-MaxP` over all $509$ long documents [$762$ overflow tail chunks across $1{,}271$ total windows] confirms $\text{MRR@10}=0.5165$ vs. $0.5007$, $\Delta = +0.0158$).
   - **Decomposing Source-Sampling Confounding vs. Within-Source Lexical Copy Inflation:** Both local `Qwen2.5-1.5B-Instruct` ($N=249$ common valid sources) and cloud `Qwen-Plus` ($N=179$ common valid sources) rank `rrf_bm25_bge` first under both `zero_shot` ($0.5647$ and $0.7371$) and `few_shot_3` ($0.4124$ and $0.6813$). Aligning real test queries with synthetic queries on the exact same target documents ($n=209$ queries / $207$ docs for `1.5B`; $n=153$ queries / $151$ docs for `Qwen-Plus`) shows that (a) small-subsample evaluation introduces **source-sampling confounding** where real RRF already slightly leads real BGE in point estimate ($0.4975$ vs. $0.4696$ and $0.4980$ vs. $0.4668$, with 95% CIs $[-0.0645, +0.0096]$ and $[-0.0687, +0.0073]$ spanning zero), and (b) while `zero_shot` synthesis significantly inflates query-to-source bigram containment $\kappa_2(q', d^*)$ ($0.3253 \rightarrow 0.4775$ on `1.5B` and $0.3167 \rightarrow 0.4638$ on `Qwen-Plus`) and boosts `bm25_char2` by $+0.0925$ ($95\%$ CI $[+0.0227, +0.1633]$) and $+0.2457$ ($[+0.1695, +0.3216]$) MRR@10 on those paired queries, the paired difference-in-differences in RRF-over-BGE advantage ($\Delta\Delta_{\text{RRF}-\text{BGE}} = +0.0123$, CI $[-0.0375, +0.0649]$ on `1.5B zero_shot`; $-0.0020$, CI $[-0.0479, +0.0438]$ on `Qwen-Plus zero_shot`) spans zero across all four settings, meaning within-source phrasing effects on the relative RRF$-$BGE gap cannot be statistically separated from source-subsampling variance at current sample sizes.
   - **Exploratory Relational Source-Overlap Gating (`source_overlap_gated`) vs. Retriever-Selection Candidate-Set Abstention:** While marginal query-feature density reweighting fails ($0\%$ optimal selection rate), an exploratory retrospective gate on query-to-source bigram containment $\kappa_2(q', d^*) \le \hat{\tau}_{\text{cal}}^{(B)}$ recovers `bge_small_zh` across all calibrated budgets $B \in \{25, 50, 100, 200\}$ and cold-start quantiles $q \in [0.10, 0.30]$ (flipping back to RRF once high-overlap queries enter at $q \ge 0.40\text{--}0.50$, whereas 1,000 matched-size random subsamples at $Q_{0.20}$ select `bge_small_zh` in only $5.9\%$ and $18.2\%$ of draws). Meanwhile, paired-bootstrap candidate-set abstention on `real_only` returns $\widehat{\mathcal{C}}_{0.95}$ when small calibration budgets cannot statistically separate top contenders—cutting confident wrong commitments at $B=25$ from $60\%\text{--}100\%$ down to **$0.0\%$** ($80\%$ abstention, mean set size $2.60$ of $8$, **$100\%$** optimal coverage).
3. **Round 3 — Clause-Grounded Eligibility-Condition Preservation, Slot Completeness vs. Decision Determinacy & Synthetic Multi-Threshold Rule-Compiler Stress Test ($80$ insurance cases + $64$ stress-test cases):**
   - Across five official Chinese insurance policy wordings (`pa_p0781`, `pa_h865`, `cpic_86017`, `cpic_82859`, `tk_group_2018`), every `missing_critical` (`CLARIFY`) benchmark case is certified by **two explicit counterexample completion witnesses** yielding opposite rule outcomes (`WAIT_EXCLUDED` vs. `WAIT_NOT_EXCLUDED`), and **$100\%$ ($120/120$)** of LLM rewrites pass permissive `format_valid` checks ($33/40$ on `1.5B` and $40/40$ on `Qwen-Plus`/`Qwen-Max` under strict single-`?`).
   - **Separating All-Slot Completeness from Reference-Independent Formal Decision Determinacy:** Enumerating complete threshold-partitioned equivalence regions of missing slots using only extracted question observations and public policy rules (`enumerate_admissible_decisions`, never reading ground-truth `seed_frame` metadata) separates non-pivotal slot omissions from genuine ambiguity. Moving from `1.5B` to the commercial API models `Qwen-Plus` and `Qwen-Max` (undisclosed parameter counts) reduces observed numeric value mutations on this 20-seed sample from $15\%$ ($3/20$) to $0\%$ ($0/20$), but `ordinary_style` preserves standalone all-slot match in only $30\%$ (`1.5B`), $0\%$ (`Qwen-Plus`), and $20\%$ (`Qwen-Max`) of seeds (and standalone decision determinacy in $30\%$, $0\%$, and $45\%$). Explicit slot constraints (`condition_prompt`) restore standalone all-slot match to **$95\%\text{--}100\%$** and standalone decision determinacy to **$95\%$ (`1.5B`) and $100\%$ (`Qwen-Plus`/`Qwen-Max`)** ($100\%$ granted-policy determinacy across all three models).
   - **Synthetic Multi-Threshold Rule-Compiler Stress Test & Closed-Loop `slot_guard_repair`:** On a researcher-constructed synthetic rule-compiler stress test inspired by healthcare clinical pathways and labor-law rules ($4$ synthetic clauses, $16$ seeds, $64$ controlled cases, $48$ method records from $42$ local `1.5B` calls), pairing `condition_prompt` with a 1-step deterministic **closed-loop slot-guard repair (`slot_guard_repair`)** lifts parser-verified all-slot match from $6.2\%$ (`ordinary_style`) and $37.5\%$ (`condition_prompt`) to **$68.8\%$ ($11/16$)**, raises reference-independent formal decision determinacy to **$75.0\%$ ($12/16$, including $100.0\%$ [$8/8$] on healthcare-inspired rules)**, and eliminates **$100\%$ of parser-detected threshold mutations ($31.2\% \rightarrow 0.0\%$)**.

---

## Repository Structure

```text
.
├── data_v3/                        # Round 3 insurance & cross-domain clause-grounded benchmarks
│   ├── controlled_cases.jsonl      # 80 insurance controlled cases (complete, surface, missing_critical, counterfactual)
│   ├── generated_questions.jsonl   # 40 paired local LLM rewrites (ordinary_style vs. condition_prompt)
│   ├── condition_drift_diagnostics.json
│   ├── multi_scale_condition_summary.json  # 1.5B vs. Qwen-Plus vs. Qwen-Max summary (120 rewrites)
│   ├── cross_domain_controlled_cases.jsonl # 64 Healthcare & Legal controlled cases with double witnesses
│   ├── cross_domain_generations.jsonl      # 48 local generations (ordinary, condition_prompt, slot_guard_repair)
│   ├── cross_domain_summary.json           # Cross-domain preservation, mutation & retrieval summary
│   ├── condition_preservation_comparison.png
│   └── independent_review/         # Blinded two-rater HTML evaluation suite (reviewer_A.html & reviewer_B.html)
├── results/                        # Round 1 & Round 2 evaluation outputs
│   ├── retrieval_summary.json      # Round 1 full-sample retrieval metrics
│   ├── validation_v2/              # Round 2 1.5B evaluation, abstention, source_overlap_gated & Full-MaxP
│   └── validation_v2_qwen_plus/    # Round 2 Qwen-Plus cross-scale evaluation & source_overlap_gated
├── src/                            # Reproducible evaluation, generation, and diagnostic scripts
│   ├── experiment.py               # Round 1 group-disjoint split & lexical retrieval
│   ├── generate_v2.py              # Round 2 local LLM query synthesis
│   ├── generate_v2_api.py          # Round 2 cloud API multi-scale query synthesis
│   ├── validate_v2.py              # Round 2 dense/hybrid evaluation & bootstrap intervals
│   ├── condition_data.py           # Round 3 deterministic policy rule engine & counterexamples
│   ├── condition_generate.py       # Round 3 local LLM paired rewriting
│   ├── condition_generate_api.py   # Round 3 cloud API multi-scale paired rewriting
│   ├── cross_domain_study.py       # Synthetic multi-threshold rule-compiler stress test & slot_guard_repair
│   ├── maxp_ablation.py            # Source-sampling confounding, quantile sensitivity & Full-MaxP ablation
│   ├── analyze_v3.py               # Slot completeness vs. decision determinacy, source_overlap_gated & abstention
│   └── score_annotations.py        # Blinded human annotation unblinding & Cohen's kappa scorer
├── tests/                          # Unit test suite (26/26 deterministic tests)
└── paper/                          # English & Chinese manuscripts + Overleaf LaTeX package
```

---

## Reproducing the Experiments & Running Unit Tests

### 1. Run the Deterministic Unit Test Suite (26 Tests)
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Round 1: Group-Disjoint Split & Heuristic Proxy Evaluation
```bash
python3 src/experiment.py --data ../../work/baoxianzhidao_filter.csv --out results --seed 20261025 --repeats 5
python3 src/make_report.py
```

### 3. Round 2: LLM Query Synthesis & Retriever Selection Reversal (`1.5B` & Cloud API)
```bash
# Local Qwen2.5-1.5B-Instruct generation & evaluation
python3 src/generate_v2.py --inputs ../../work/validation_v2/generation_inputs.json --model ../../work/models/qwen --out ../../work/validation_v2/generation_float32 --batch-size 8 --dtype float32 --attention eager
python3 src/validate_v2.py --inputs ../../work/validation_v2/evaluation_inputs.json --generations ../../work/validation_v2/generation_float32 --embeddings ../../work/validation_v2/embedding_cache --bge-model ../../work/models/bge --out results/validation_v2

# Cloud API (Qwen-Plus) cross-scale generation & evaluation
DASHSCOPE_API_KEY="your_api_key" python3 src/generate_v2_api.py --inputs ../../work/validation_v2/generation_inputs.json --model-name qwen-plus --out ../../work/validation_v2/generation_qwen_plus --workers 8
python3 src/validate_v2.py --inputs ../../work/validation_v2/evaluation_inputs.json --generations ../../work/validation_v2/generation_qwen_plus --embeddings ../../work/validation_v2/embedding_cache --bge-model ../../work/models/bge --out results/validation_v2_qwen_plus
```

### 4. Round 3: Clause-Grounded Eligibility-Condition Preservation (`1.5B`, `Qwen-Plus`, `Qwen-Max`)
```bash
# Build 80 controlled cases with single-field counterexamples
python3 src/condition_data.py --root ../../work/condition_study --out data_v3

# Cloud API multi-scale paired rewriting (qwen-plus / qwen-max)
DASHSCOPE_API_KEY="your_api_key" python3 src/condition_generate_api.py --inputs ../../work/condition_study/generation_inputs.json --model-name qwen-plus --out ../../work/condition_study/generation_qwen_plus
DASHSCOPE_API_KEY="your_api_key" python3 src/condition_generate_api.py --inputs ../../work/condition_study/generation_inputs.json --model-name qwen-max --out ../../work/condition_study/generation_qwen_max

# Run cross-round diagnostics, cross-domain stress test rescoring, and Full-MaxP ablation (works with default paths or explicit flags)
python3 src/analyze_v3.py
python3 src/cross_domain_study.py --score-only
python3 src/maxp_ablation.py
```
