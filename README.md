# Do Synthetic Queries Preserve Insurance Eligibility Decisions?

**Author:** Shunchi Wang (Independent Researcher, `jakelihua@gmail.com`)  
**Paper & Thesis Artifacts:**
- **Chinese Master's Thesis (Full 7-Chapter Monograph):** [`paper/master_thesis_zh.md`](paper/master_thesis_zh.md)
- **LaTeX Source & Overleaf Package:** [`paper/overleaf_package/main.tex`](paper/overleaf_package/main.tex) | [`paper/overleaf_submission.zip`](paper/overleaf_submission.zip)
- **English Manuscript (Markdown):** [`paper/draft_en.md`](paper/draft_en.md)
- **Chinese Manuscript:** [`paper/draft_zh.md`](paper/draft_zh.md)

---

## Overview

Retrieval-augmented generation (RAG) benchmarks often synthesize test queries from source passages using instruction-tuned large language models. In rule-governed insurance eligibility and claims workflows, a rewritten query can retrieve the source clause at rank 1 while omitting or altering conditions that determine whether a claim is covered, excluded, or requires clarification. Across **26 official Chinese insurance policy wordings** ($5$ development policies, $4$ post-refinement extension policies, $11$ zero-alias confirmation policies, and $6$ untouched post-hardening holdout policies, totaling $328$ controlled cases) and a multi-threshold rule testbed spanning two distinct model families (`Qwen2.5-1.5B-Instruct`, `Qwen-Plus` [`response_model="qwen-plus"`], `Qwen-Max`, and `DeepSeek` [`deepseek-chat` API endpoint, returning `response_model="deepseek-flash"`]), we evaluate clause-grounded eligibility condition preservation, shared-rule-base dual-implementation consistency, fair structured baselines, equal-budget repair versus filtering tradeoffs, and strictly isolated downstream RAG system selection impact. No LLM or embedding model weights are pre-trained or fine-tuned in this repository; all experiments evaluate inference-time generation, dense/lexical indexing, statistical calibration, and symbolic rule verification.

1. **Clause-Grounded Decision Verification on $5$ Development Insurance Policies ($20$ seeds, $80$ controlled cases, $120$ multi-scale rewrites):**
   - Across five development official Chinese insurance policy wordings (`pa_p0781`, `pa_h865`, `cpic_86017`, `cpic_82859`, `tk_group_2018`), every `missing_critical` (`CLARIFY`) benchmark case is certified by two counterexample completion witnesses yielding opposite rule outcomes (`WAIT_EXCLUDED` vs. `WAIT_NOT_EXCLUDED`).
   - Enumerating complete threshold-partitioned equivalence regions of missing slots (`enumerate_admissible_decisions`, without reading ground-truth `seed_frame` metadata) shows that unconstrained conversational rewriting (`ordinary_style`) preserves standalone decision determinacy in only $30.0\%$ (`1.5B`), $0.0\%$ (`Qwen-Plus`), and $45.0\%$ (`Qwen-Max`), whereas explicit slot-preserving prompting (`condition_prompt`) reaches $95.0\%$ (`1.5B`) and $100.0\%$ (`Qwen-Plus`/`Qwen-Max`).
2. **Extension to Four Official Insurance Policies, Benefit-Specific Rules, and Title-Masked Retrieval ($4$ PDFs, $16$ seeds, $64$ controlled cases):**
   - On four additional official insurance policy PDFs (`pa_2011_36`, `pa_p0254`, `pa_p0321`, `cpic_93896`) with textbook book-title markers (`《》`) and clause-number headers stripped, we conduct a retrospective post-refinement evaluation after adding colloquial product aliases and benefit-level disambiguation (`benefit_id`) to the shared parser (`extract_fields_surface_normalized`), explicitly modeling benefit-specific waiting-period rules (distinguishing general inpatient allowance [Article 4(1), 30-day wait, accident & continuous renewal exempt] from cancer inpatient allowance [Article 4(2), 90-day wait, continuous renewal exempt, no accident exemption] in `pa_2011_36`, and primary lung cancer excluding carcinoma in situ in `pa_p0321`) and separating contract PDF clause anchors (`source_type='pdf_contract_clause'`) from synthetic seed scenario parameters (`source_type='synthetic_seed_frame'`) in `atomic_rubrics`.
   - The unified verifier certifies $100.0\%$ ($64/64$) of controlled cases. Conversational rewrites (`ordinary_style`) achieve high lexical novelty ($1-\kappa_2=0.5380$) but $0/16$ ($0.0\%$) standalone decision determinacy ($5/16$ [$31.2\%$] when granted the source policy ID), while single-pass `condition_prompt` already reaches $16/16$ ($100.0\%$), so `slot_guard_repair` triggers $0$ repair calls (exact two-sided McNemar $p=1.0$ vs. `condition_prompt`, $p=0.03125$ vs. `structured_json_rewrite` at $10/16$). Masking product titles and insurer names drops 9-clause Top-1 retrieval from $75.0\%$ to $31.2\%$ on `bge_small_zh` for title-preserving queries.
3. **Equal-Call-Budget Baseline Comparison on Multi-Threshold Composite Rules ($16$ seeds, $64$ cases, $7$ methods):**
   - On a 16-seed ($64$-case) synthetic multi-threshold rule testbed inspired by clinical pathways and labor statutes (`med_egfr_tki`, `med_vte_prophylaxis`, `law_work_injury`, `law_noncompete`), all 6 non-JSON methods achieve $100.0\%$ ($16/16$) Top-1 retrieval on both `bm25_char2` and `bge_small_zh` (both standard and title-masked), yet `ordinary_style` preserves standalone decision determinacy in only $6.2\%$ ($1/16$; $43.8\%$ [$7/16$] when the source `clause_id` is granted, while $9/16$ still fail on substantive conditions), and 1-call `structured_json_rewrite` fails two-line format validation ($0/16$ valid; $0/16$ hit the 160-token cap with `generated_tokens` $\in [45, 111]$, as the model stops immediately after closing the first-line JSON brace).
   - At an identical $26$-call budget ($16$ initial $+$ $10$ second-pass calls on the shared 10-seed trigger schedule), unguided LLM self-refinement (`generic_self_refine_budget_matched`, $26$ calls; unchanged at $32$ calls in `generic_self_refine_all`) achieves $37.5\%$ ($6/16$) all-slot match and $50.0\%$ ($8/16$) decision determinacy (both standalone and granted-clause) with $25.0\%$ ($4/16$) threshold mutations, whereas symbolic parser-guided `slot_guard_repair` ($26$ calls) reaches $68.8\%$ ($11/16$) standalone all-slot match, $75.0\%$ ($12/16$) standalone decision determinacy ($93.8\%$ [$15/16$] under granted-clause sensitivity once the 3 `law_noncompete` title-omission cases `cross-013`, `014`, `016` are granted `clause_id`), and $0.0\%$ ($0/16$) parser-detected mutations (primary standalone exact two-sided McNemar $p=0.125$ vs. `generic_self_refine_budget_matched` and $p=0.0625$ vs. `condition_prompt`; post-hoc granted-clause sensitivity $p=0.015625$ and $p=0.007812$).
4. **Deterministic Verifier Diagnostic Log, Watchlist & Appendix A Community-QA Retrieval Diagnostics:**
   - A deterministic 264-item verifier diagnostic log and manual-inspection watchlist (`data_v3/item_by_item_semantic_audit.jsonl`, explicitly marked as an automated parser explanation and 29-item surface-edge-case watchlist rather than external semantic validation or human annotation) documents per-slot extraction traces and edge cases.
   - Appendix A reports corpus-level retrieval ranking and calibration diagnostics on `baoxianzhidao_filter.csv` ($3{,}746$ answers, $2{,}994$ held-out real queries, `Full-MaxP` zero-truncation check, source-sampling confounding vs. lexical copying decomposition, negative propensity-reweighting results, and heuristic paired-bootstrap candidate-set abstention).
5. **Dual-Implementation Consistency Check, 40-Seed Confirmation Set, Fair Structured Baselines, 24-Seed Untouched Post-Hardening Holdout (`v3.1-untouched-holdout`), and Strictly Isolated Downstream Pipeline Selection Impact:**
   - **Three-Layer Audit & Constructive Diagnostic Suite (`N=120`)**: Layer 1 audits $20$ confirmation/dev/extension official insurance PDFs (`data_v3/clause_rule_mapping_audit.json`, author engineering compilation time estimate $11.72$ min/contract, total $234.5$ min), $6$ untouched post-hardening holdout Ping An PDFs (`data_v3/untouched_holdout_sources.json`), and $4$ multi-threshold rules. Layer 2 evaluates the verifier on a $120$-case constructive diagnostic suite across $8$ categories (`data_v3/gold_diagnostic_summary.json`, $60$ gold-positive and $60$ gold-negative): `false_accept_fp = 0` (**False Accept Rate $0.0\%$**, **Precision $100.0\%$**), `false_reject_fn = 0` on supported day syntax (Categories 1–7), and $15/15$ unsupported month/week/date expressions in Category 8 conservatively intercepted as `PARSE_UNCERTAIN` via a text-only hard gate without reading `seed_frame`.
   - **Shared-Rule-Base Dual-Implementation Consistency Check Across $400$ Confirmation Generations**: Evaluated across $11$ zero-alias official PDFs ($24$ insurance seeds) + $16$ multi-threshold seeds ($40$ seeds, $160$ cases) on `qwen-plus` (`response_model="qwen-plus"`) and `deepseek-chat` (`response_model="deepseek-flash"`). Using forward reachability enumeration (`evaluate_independent_text_semantics`) that computes reachable outcomes directly from extracted slots without shortcutting through `seed['reference_decision']`—distinguishing same-partition numeric drift (`FACT_MUTATED_SAME_DECISION`, e.g., Day 20 $\rightarrow$ Day 21 on a 30-day wait keeping `WAIT_EXCLUDED`) from threshold-crossing flips (`DECISION_CHANGED`, e.g., Day 20 $\rightarrow$ Day 45)—and explicitly documented as a **`shared_rule_base_dual_implementation_consistency_check`** rather than external human annotation:
     - **Pure Semantic Reachability Consistency (`semantic_reachability_consistency`, ignoring format truncation)**: **$\text{TP} = 296, \text{FP} = 0$ (Semantic False Accept Rate $0.0\%$, Precision $100.0\%$), $\text{TN} = 86, \text{FN} = 18$ ($\text{FRR} = 5.73\%$)**.
     - **Joint Format-and-Semantic Usability (`joint_format_and_semantic_usability`, requiring `format_valid = True`)**: **$\text{TP} = 290, \text{FP} = 6$ (Precision $97.97\%$, False Accept Rate $6.38\%$), $\text{TN} = 88, \text{FN} = 16$ (Recall $94.77\%$)**, where **all $6$ false accepts arise from 220-token cap truncations in `structured_json_rewrite` ($5$ on `Qwen-Plus` and $1$ on `DeepSeek`)**, while **`slot_guard_repair` ($n=80$), `condition_prompt` ($n=80$), and `generic_self_refine_budget_matched` ($n=80$) achieve $0$ false accepts (Precision $100.0\%$)**.
   - **Dual-Endpoint Confirmation ($N=40$ Seeds / $80$ Pairs), Fair Structured Baselines, and 24-Seed Untouched Holdout ($N=48$ Pairs)**:
     - On the 40-seed confirmation set under the joint `format_and_slot` endpoint, `slot_guard_repair` (**$78/80 = 97.50\%$**: $39/40$ on `Qwen-Plus`, $39/40$ on `DeepSeek`) outperforms `condition_prompt` (**$70/80 = 87.50\%$**, 8W/0L, McNemar **$p=0.0078125$** [Holm **$p=0.0234375$**], 40-seed clustered **$p=0.0078125$** [Holm **$p=0.0234375$**], 15-clause clustered $p=0.0625$ [$5$ nonzero clause clusters]), `generic_self_refine_budget_matched` (**$70/80 = 87.50\%$**, 8W/0L, McNemar **$p=0.0078125$**, seed-clustered **$p=0.015625$** [Holm **$p=0.03125$**], 15-clause clustered **$p=0.03125$**), and 220-token `structured_json_rewrite` (**$71/80 = 88.75\%$** [$33/40$ Qwen + $38/40$ DeepSeek], 7W/0L, McNemar **$p=0.015625$** [Holm **$p=0.0234375$**], seed-clustered **$p=0.03125$** [Holm **$p=0.03125$**], 15-clause clustered $p=0.0625$; on `slot_only`, $78/80$ vs. $77/80$, $p=1.0$).
     - Fair structured baselines (`data_v3/fair_structured_baseline_generations.jsonl`): raising the cap to `max_new_tokens=320` (`structured_json_untruncated_320`) reaches $80/80$ format validity and **$73/80$ ($91.25\%$)** usable queries (`slot_guard_repair` wins 5:0, McNemar $p=0.0625$); direct JSON schema rendering (`structured_json_schema_render`) reaches $80/80$ format validity and **$70/80$ ($87.50\%$)** usable queries (`slot_guard_repair` wins 8:0, McNemar $p=0.0078125$, seed-clustered $p=0.015625$).
     - On the **24-seed untouched post-hardening holdout (`v3.1-untouched-holdout`, $6$ new Ping An PDFs + $12$ unseen multi-threshold seeds = $48$ cross-model pairs in `data_v3/untouched_holdout_generations.jsonl`)**, **`slot_guard_repair` reaches $48/48$ ($100.0\%$)**, outperforming `condition_prompt` ($39/48 = 81.25\%$, McNemar **$p=0.00390625$**, 24-seed clustered **$p=0.0078125$**, 10-clause clustered **$p=0.03125$**), 220-token `structured_json_rewrite` ($37/48 = 77.08\%$, McNemar **$p=0.00097656$**), `structured_json_schema_render` ($39/48 = 81.25\%$, McNemar **$p=0.00390625$**), and `structured_json_untruncated_320` / `generic_self_refine_budget_matched` ($43/48 = 89.58\%$, $p=0.0625$).
   - **Strictly Isolated Retrieval + Symbolic Rule-Decision Pipeline Selection Impact (`data_v3/downstream_rag_selection_study.json`)**: All $5$ candidate pipelines execute through the isolated top-level function `predict(question, retrieved_clause, inference_config)` (which never receives `seed` metadata and is invariant to flipping `seed['reference_decision']`) with $100\%$ pure `BAAI/bge-small-zh-v1.5` dense vector cache coverage across all $80$ reference queries ($48$ insurance + $32$ multi-threshold, zero lexical fallback; `S1_dense_rule_clarify` [$97.50\%$] $>$ `S2_bm25_rule_clarify` [$93.75\%$] $>$ `S5` [$41.25\%$] $>$ `S3` [$35.00\%$] $>$ `S4` [$31.25\%$]). Unverified conversational rewrites (`ordinary_style`) across all $40$ seeds (`combined_40_seeds`) suffer $97.5\%\text{--}100.0\%$ label corruption and invert system rankings (Kendall's $\tau = -0.7746$ on `qwen-plus`, $\tau = -0.8367$ on `deepseek-chat`, selection regret $0.6250$), whereas both `verify_only_filter` and `slot_guard_repair` achieve **$0.0\%$ label corruption** and **Kendall's $\tau = 1.0000$, Spearman's $\rho = 1.0000$** ($0.0$ selection regret).

---

## Repository Structure

```text
.
├── data_v3/                        # Official insurance policy, held-out confirmation & multi-threshold rule benchmarks
│   ├── controlled_cases.jsonl      # 80 dev insurance controlled cases (complete, surface, missing_critical, counterfactual)
│   ├── generated_questions.jsonl   # 40 paired local LLM rewrites (ordinary_style vs. condition_prompt)
│   ├── condition_drift_diagnostics.json
│   ├── multi_scale_condition_summary.json  # 1.5B vs. Qwen-Plus vs. Qwen-Max summary (120 rewrites)
│   ├── heldout_clauses_sources.json        # 4 held-out official insurance policy PDFs (SHA256, aliases & bbox anchors)
│   ├── heldout_controlled_cases.jsonl      # 64 held-out controlled cases with atomic_rubrics & dual witnesses
│   ├── heldout_generations.jsonl           # 112 held-out generation records across 7 methods
│   ├── cross_domain_controlled_cases.jsonl # 64 multi-threshold controlled cases with dual witnesses
│   ├── cross_domain_generations.jsonl      # 48 local generations (ordinary, condition_prompt, slot_guard_repair)
│   ├── cross_domain_equal_budget_generations.jsonl # 112 equal-call-budget comparison records across 7 methods
│   ├── cross_domain_summary.json           # Multi-threshold preservation, mutation & retrieval summary
│   ├── equal_budget_and_heldout_summary.json # Held-out confirmation, title-masked retrieval & equal-budget summary
│   ├── item_by_item_semantic_audit.jsonl   # 264 deterministic verifier diagnostic & 29-item watchlist records
│   ├── item_by_item_semantic_audit_summary.json
│   ├── clause_rule_mapping_audit.json      # Layer 1 clause-to-rule audit across 20 official PDFs + 4 rules
│   ├── gold_diagnostic_suite.jsonl         # 120-case constructive gold diagnostic & adversarial suite
│   ├── gold_diagnostic_summary.json        # Verifier precision/recall & PARSE_UNCERTAIN breakdown
│   ├── frozen_confirmation_sources.json    # 11 strictly frozen confirmation official insurance PDFs
│   ├── frozen_confirmation_cases.jsonl     # 160 dual-witness controlled cases on 40 frozen confirmation seeds
│   ├── frozen_confirmation_generations_qwen_plus.jsonl # 320 method records on Qwen-Plus (8 methods x 40 seeds)
│   ├── frozen_confirmation_generations_deepseek.jsonl  # 320 method records on DeepSeek-V3 (8 methods x 40 seeds)
│   ├── frozen_confirmation_independent_audit_labels.json # Standalone SHA256-indexed dual-implementation consistency labels
│   ├── frozen_confirmation_three_layer_audit.jsonl     # 400 dual-implementation audited generations across 2 model families
│   ├── fair_structured_baseline_generations.jsonl      # Fair structured baselines (max_new_tokens=320 & JSON schema render)
│   ├── untouched_holdout_sources.json                  # 6 untouched post-hardening holdout Ping An policy PDFs
│   ├── untouched_holdout_generations.jsonl             # 336 method records on 24 untouched holdout seeds across 2 models
│   ├── downstream_rag_selection_study.json # Strictly isolated downstream RAG + Decision system ranking study
│   ├── frozen_confirmation_summary.json    # Full confirmation, fair structured, untouched holdout & Holm-Bonferroni summary
│   ├── condition_preservation_comparison.png
│   └── independent_review/         # Blinded two-rater HTML evaluation suite (reviewer_A.html & reviewer_B.html)
├── results/                        # Appendix A: Community Q&A retrieval & calibration outputs
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
│   ├── equal_budget_and_heldout_study.py # Extension 4-policy evaluation, title-masked retrieval & equal-budget baselines
│   ├── build_item_semantic_audit.py # Deterministic verifier diagnostic log & manual-inspection watchlist builder
│   ├── frozen_confirmation_and_rag_study.py # Confirmation, fair structured baselines, 24-seed untouched holdout & isolated RAG study
│   ├── maxp_ablation.py            # Source-sampling confounding, quantile sensitivity & Full-MaxP ablation
│   ├── analyze_v3.py               # Slot completeness vs. decision determinacy, source_overlap_gated & abstention
│   └── score_annotations.py        # Blinded human annotation unblinding & Cohen's kappa scorer
├── tests/                          # Deterministic unit test suite (34 tests)
├── requirements.txt                # Pinned Python dependencies for offline reproduction
└── paper/                          # Master's thesis, English & Chinese manuscripts + Overleaf LaTeX package
```

---

## Reproducing the Experiments & Running Unit Tests

### 0. Environment Setup
```bash
pip install -r requirements.txt
```

### 1. Run the Deterministic Unit Test Suite (34 Tests)
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Self-Contained Offline Rescoring & Verifier Diagnostics (Works in Clean Checkout Without `../../work/`)
All five primary verification and diagnostic commands below run deterministically in an isolated git checkout (including with `HF_HUB_OFFLINE=1`) using the tracked seed definitions, generation logs, and precomputed embedding/overlap caches in `data_v3/` and `results/validation_v2/`:
```bash
python3 src/analyze_v3.py
python3 src/cross_domain_study.py --score-only
python3 src/equal_budget_and_heldout_study.py --score-only
python3 src/build_item_semantic_audit.py
python3 src/frozen_confirmation_and_rag_study.py --score-only
```

### 3. Optional Full Regeneration from Raw Third-Party PDFs / Local Model Weights (Requires Local `../../work/`)
```bash
python3 src/condition_data.py --root ../../work/condition_study --out data_v3
python3 src/experiment.py --data ../../work/baoxianzhidao_filter.csv --out results --seed 20261025 --repeats 5
python3 src/validate_v2.py --inputs ../../work/validation_v2/evaluation_inputs.json --generations ../../work/validation_v2/generation_float32 --embeddings ../../work/validation_v2/embedding_cache --bge-model ../../work/models/bge --out results/validation_v2
python3 src/maxp_ablation.py
```
