# When Do Synthetic Queries Mislead Retriever Selection?
### Selection Reversal, Candidate-Set Abstention, and Eligibility-Condition Preservation in Chinese Insurance Retrieval

**Author:** Shunchi Wang (Independent Researcher)  
**Paper Artifacts:**
- 📄 **LaTeX Source & Overleaf Package:** [`paper/overleaf_package/main.tex`](paper/overleaf_package/main.tex) | [`paper/overleaf_submission.zip`](paper/overleaf_submission.zip)
- 🇬🇧 **English Manuscript (Markdown):** [`paper/draft_en.md`](paper/draft_en.md)
- 🇨🇳 **Chinese Manuscript (v0.3):** [`paper/draft_zh.md`](paper/draft_zh.md)

---

## Abstract & Key Findings

When cold-starting a vertical-domain retrieval-augmented generation (RAG) system, engineering teams routinely rely on rule-based proxies or LLM-generated synthetic queries to benchmark candidate retrievers before real user logs exist. Across a three-round diagnostic study on Chinese insurance retrieval and clause-level underwriting/claims eligibility verification spanning three model tiers (`Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, and `Qwen-Max`), we demonstrate:

1. **Round 1 — Group-Disjoint Retrieval Benchmark ($3{,}746$ answers, $3{,}725$ connected groups, $2{,}994$ held-out real queries):**
   - Prevents duplicate-answer leakage via Union-Find connected-component splitting.
   - Naive snippet-copy proxies invert the ranking between `bm25_char2` ($0.4907$ MRR@10) and `tfidf_char3_5` ($0.3817$ MRR@10), incurring $0.1090$ selection regret.
2. **Round 2 — Cross-Scale Retriever Selection Reversal & Candidate-Set Abstention:**
   - On held-out real user queries, pure dense retrieval (`bge_small_zh`, $\text{MRR@10}=0.5216$) significantly outperforms hybrid fusion (`rrf_bm25_bge`, $0.5030$; paired $\Delta = +0.0187$, 95% bootstrap CI $[+0.0080, +0.0300]$).
   - **Selection Reversal Persists Across Model Scales:** Both local `Qwen2.5-1.5B-Instruct` ($N=249$) and production cloud `Qwen-Plus` ($N=179$) rank `rrf_bm25_bge` first under both `zero_shot` ($0.5647$ and $0.7371$) and `few_shot_3` ($0.4124$ and $0.6813$) due to passage-conditioned lexical overlap inflation.
   - **Candidate-Set Abstention:** While surface-feature density reweighting fails ($0\%$ optimal selection rate), paired-bootstrap candidate-set abstention eliminates confident wrong commitments at small calibration budgets ($B=25$: from $60\text{--}100\%$ down to **$0.0\%$**) while preserving **$100\%$** coverage of the true oracle winner.
3. **Round 3 — Clause-Grounded Eligibility-Condition Preservation Across Model Scales ($80$ controlled cases, $120$ paired LLM rewrites):**
   - Across five official Chinese insurance policy wordings (`pa_p0781`, `pa_h865`, `cpic_86017`, `cpic_82859`, `tk_group_2018`), **$100\%$ ($120/120$)** of generated queries pass syntactic format checks (`format_valid`).
   - **Scale-Dependent Hallucination vs. Scale-Invariant Omission:** Scaling from `1.5B` to `Qwen-Plus` and flagship `Qwen-Max` eliminates numeric value mutations (`生效第20天 -> 第二天` drops from $15\% \rightarrow 0\%$), but **fails to prevent pragmatic slot omission** under unconstrained conversational prompting (`ordinary_style` preserves standalone rule decisions in only $30\%$ [`1.5B`], $0\%$ [`Qwen-Plus`], and $20\%$ [`Qwen-Max`] of cases).
   - **Universal Effectiveness of `condition_prompt`:** Explicit slot-constrained prompting restores standalone rule-decision preservation to **$95\%$ ($19/20$)** on `1.5B` and `Qwen-Plus`, and **$100\%$ ($20/20$)** on `Qwen-Max`, lifting top-1 clause retrieval to $95\%\text{--}100\%$.

---

## Repository Structure

```text
.
├── data_v3/                        # Round 3 clause-grounded benchmark & diagnostics
│   ├── controlled_cases.jsonl      # 80 controlled cases (complete, surface, missing_critical, counterfactual)
│   ├── generated_questions.jsonl   # 40 paired local LLM rewrites (ordinary_style vs. condition_prompt)
│   ├── condition_drift_diagnostics.json
│   ├── multi_scale_condition_summary.json  # 1.5B vs. Qwen-Plus vs. Qwen-Max summary (120 rewrites)
│   ├── condition_preservation_comparison.png
│   └── review/                     # Blinded two-rater HTML evaluation suite (reviewer_A.html & reviewer_B.html)
├── results/                        # Round 1 & Round 2 evaluation outputs
│   ├── retrieval_summary.json      # Round 1 full-sample retrieval metrics
│   ├── validation_v2/              # Round 2 Qwen2.5-1.5B-Instruct evaluation & abstention diagnostics
│   └── validation_v2_qwen_plus/    # Round 2 Qwen-Plus cross-scale evaluation metrics
├── src/                            # Reproducible evaluation, generation, and diagnostic scripts
│   ├── experiment.py               # Round 1 group-disjoint split & lexical retrieval
│   ├── generate_v2.py              # Round 2 local LLM query synthesis
│   ├── generate_v2_api.py          # Round 2 cloud API multi-scale query synthesis
│   ├── validate_v2.py              # Round 2 dense/hybrid evaluation & bootstrap intervals
│   ├── condition_data.py           # Round 3 deterministic policy rule engine & counterexamples
│   ├── condition_generate.py       # Round 3 local LLM paired rewriting
│   ├── condition_generate_api.py   # Round 3 cloud API multi-scale paired rewriting
│   ├── analyze_v3.py               # Synonym-aware slot parser, abstention & retrieval diagnostics
│   └── score_annotations.py        # Blinded human annotation unblinding & Cohen's kappa scorer
├── tests/                          # Unit test suite (21/21 deterministic tests)
└── paper/                          # English & Chinese manuscripts + Overleaf LaTeX package
```

---

## Reproducing the Experiments & Running Unit Tests

### 1. Run the Deterministic Unit Test Suite (21 Tests)
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

# Run cross-round diagnostics
python3 src/analyze_v3.py --work-condition ../../work/condition_study --data-v3 data_v3 --results-v2 results/validation_v2
```
