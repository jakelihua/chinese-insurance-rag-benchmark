# Do Synthetic Queries Preserve Insurance Eligibility Decisions?

**Author:** Shunchi Wang (Independent Researcher, `jakelihua@gmail.com`)  
**Paper & Thesis Artifacts:**
- **Chinese Master's Thesis (Full 7-Chapter Monograph):** [`paper/master_thesis_zh.md`](paper/master_thesis_zh.md)
- **LaTeX Source & Overleaf Package:** [`paper/overleaf_package/main.tex`](paper/overleaf_package/main.tex) | [`paper/overleaf_submission.zip`](paper/overleaf_submission.zip)
- **English Manuscript (Markdown):** [`paper/draft_en.md`](paper/draft_en.md)
- **Chinese Manuscript:** [`paper/draft_zh.md`](paper/draft_zh.md)

---

## Overview

Retrieval-augmented generation (RAG) benchmarks often synthesize test queries from source passages using instruction-tuned large language models. In rule-governed insurance eligibility and claims workflows, a rewritten query can retrieve the source clause at rank 1 while omitting or altering conditions that determine whether a claim is covered, excluded, or requires clarification. Across **20 official Chinese insurance policy wordings** ($5$ development policies, $4$ post-refinement extension policies, and $11$ strictly frozen confirmation policies with $304$ dual-witness-certified controlled cases in total) and a 16-seed ($64$-case) multi-threshold rule testbed spanning two distinct model families (`Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, `Qwen-Max`, and `DeepSeek-V3` / `deepseek-chat`), we evaluate clause-grounded eligibility condition preservation, verifier precision/recall, equal-budget repair versus filtering tradeoffs, and downstream RAG system selection impact. No LLM or embedding model weights are pre-trained or fine-tuned in this repository; all experiments evaluate inference-time generation, dense/lexical indexing, statistical calibration, and symbolic rule verification.

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
5. **Decoupled Three-Layer Reference Audit, 40-Seed Confirmation Set (`v3.0-frozen` & `v3.1-hardened`), Cross-Model Generalization (`Qwen-Plus` & `DeepSeek-V3`), and Controlled Downstream Pipeline Selection Impact:**
   - **Three-Layer Reference Audit & Constructive Diagnostic Suite (`N=120`)**: Layer 1 audits all $20$ official insurance PDFs and $4$ multi-threshold rules (`data_v3/clause_rule_mapping_audit.json`, author engineering compilation time estimate $11.72$ min/contract, total $234.5$ min). Layer 2 evaluates the verifier on a $120$-case constructive diagnostic suite across $8$ categories (`data_v3/gold_diagnostic_summary.json`, $60$ gold-positive and $60$ gold-negative): `false_accept_fp = 0` (**False Accept Rate $0.0\%$**, **Precision $100.0\%$**), `false_reject_fn = 0` on supported day syntax (Categories 1–7), and $15/15$ unsupported month/week/date expressions in Category 8 conservatively intercepted as `PARSE_UNCERTAIN` via a text-only hard gate without reading `seed_frame`.
   - **Decoupled Layer-3 Audit Across $400$ Confirmation Generations**: Evaluated across $11$ new official PDFs (`aliases=[]`, $24$ insurance seeds) + $16$ multi-threshold seeds ($40$ seeds, $160$ cases) on `qwen-plus` and `deepseek-chat` (`DeepSeek-V3`). Using a standalone SHA256-indexed reference label table (`data_v3/frozen_confirmation_independent_audit_labels.json`) and a decoupled rule evaluator (`evaluate_independent_text_semantics`) that never reads `r['outcome']` (verified by counterfactual nonsense-text perturbation), across all $400$ generated queries (`data_v3/frozen_confirmation_three_layer_audit.jsonl`) the verifier achieves **$\text{TP} = 290, \text{FP} = 6$ (Precision $97.97\%$, False Accept Rate $6.38\%$), $\text{TN} = 88, \text{FN} = 16$ (Recall $94.77\%$)**, where **all $6$ false accepts arise from mid-sentence token-cap truncations in `structured_json_rewrite` on `Qwen-Plus` (`format_valid = False`)**, while **`slot_guard_repair` ($n=80$), `condition_prompt` ($n=80$), and `generic_self_refine_budget_matched` ($n=80$) achieve $0$ false accepts (Precision $100.0\%$)**.
   - **Repair (`slot_guard_repair`) vs. Filtering (`verify_only_filter`) & `structured_json_rewrite`**: On both `qwen-plus` ($47$ calls, $20{,}503$ prompt + $3{,}477$ completion = $23{,}980$ tokens) and `deepseek-chat` ($53$ calls, $19{,}820$ prompt + $3{,}250$ completion = $23{,}070$ tokens), `slot_guard_repair` reaches **$39/40$ ($97.5\%$)** decision preservation, **$40/40$ ($100\%$)** format validity, and **$0/40$** mutations. While `structured_json_rewrite` reaches $38/40$ and $39/40$ slot-level preservation ($p=1.0$ vs. `slot_guard_repair` on slots alone), its JSON header inflates completion tokens by $98.9\%\text{--}129.1\%$ and causes $7/40$ ($17.5\%$) mid-sentence truncations on `qwen-plus` (`format_valid = 33/40`, end-to-end valid & preserved yield $32/40 = 80.0\%$, pooled $70/80$ vs. $78/80$ for `slot_guard_repair`, McNemar $p=0.007812$, seed-clustered permutation $p=0.015625$). Pooled across both model families ($N=80$ paired seed-model instances) and verified under **seed-clustered ($N_{\text{seed}}=40$) and clause-clustered ($N_{\text{clause}}=15$) exact two-sided sign-flip permutation tests**, `slot_guard_repair` ($78/80 = 97.5\%$) significantly outperforms `condition_prompt` ($70/80 = 87.5\%$, McNemar **$p=0.007812$**, seed-clustered **$p=0.0078125$**, clause-clustered **$p=0.03125$**; `deepseek-chat` per-model $p=0.03125$, `qwen-plus` per-model $p=0.50$), `generic_self_refine_budget_matched` ($70/80 = 87.5\%$, McNemar **$p=0.007812$**, seed-clustered **$p=0.015625$**, clause-clustered **$p=0.03125$**), and `generic_self_refine_all` ($68/80 = 85.0\%$, McNemar **$p=0.001953$**, seed-clustered **$p=0.003906$**, clause-clustered **$p=0.03125$**), while avoiding the $7.5\%\text{--}17.5\%$ ($25.0\%$ multi-threshold) scenario loss of `verify_only_filter`.
   - **Controlled Retrieval + Symbolic Rule-Decision Pipeline Selection Impact (`data_v3/downstream_rag_selection_study.json`)**: Across $5$ controlled retrieval + symbolic rule-decision pipelines (`S1_dense_rule_clarify` [$97.50\%$ oracle accuracy on $80$ paired reference cases] $>$ `S2_bm25_rule_clarify` [$93.75\%$] $>$ `S5` [$41.25\%$] $>$ `S4` [$32.50\%$] $>$ `S3` [$31.25\%$], running real `bge_small_zh` / `bm25_char2` retrieval and text slot parsing on every query), unverified conversational rewrites (`ordinary_style`) across all $40$ seeds (`combined_40_seeds`) suffer $97.5\%\text{--}100.0\%$ label corruption and invert system rankings (Kendall's $\tau = -0.6708$ on `qwen-plus`, $\tau = -0.5270$ on `deepseek-chat`, selection regret $0.5625\text{--}0.6625$), whereas both `verify_only_filter` and `slot_guard_repair` achieve **$0.0\%$ label corruption** and **Kendall's $\tau = 1.0000$, Spearman's $\rho = 1.0000$** ($0.0$ selection regret).

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
│   ├── frozen_confirmation_independent_audit_labels.json # Standalone SHA256-indexed decoupled reference audit labels
│   ├── frozen_confirmation_three_layer_audit.jsonl     # 400 three-layer audited generations across 2 model families
│   ├── downstream_rag_selection_study.json # Downstream RAG + Decision system ranking & selection regret study
│   ├── frozen_confirmation_summary.json    # Full frozen confirmation, cost/latency & paired McNemar summary
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
│   ├── frozen_confirmation_and_rag_study.py # Strictly frozen confirmation (v3.0-frozen), 120-case gold suite, cross-model & downstream RAG study
│   ├── maxp_ablation.py            # Source-sampling confounding, quantile sensitivity & Full-MaxP ablation
│   ├── analyze_v3.py               # Slot completeness vs. decision determinacy, source_overlap_gated & abstention
│   └── score_annotations.py        # Blinded human annotation unblinding & Cohen's kappa scorer
├── tests/                          # Deterministic unit test suite (33 tests)
├── requirements.txt                # Pinned Python dependencies for offline reproduction
└── paper/                          # Master's thesis, English & Chinese manuscripts + Overleaf LaTeX package
```

---

## Reproducing the Experiments & Running Unit Tests

### 0. Environment Setup
```bash
pip install -r requirements.txt
```

### 1. Run the Deterministic Unit Test Suite (33 Tests)
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
