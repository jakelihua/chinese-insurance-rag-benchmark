# Do Synthetic Queries Preserve Insurance Eligibility Decisions?

**Author:** Shunchi Wang (Independent Researcher, `jakelihua@gmail.com`)  
**Paper Artifacts:**
- **LaTeX Source & Overleaf Package:** [`paper/overleaf_package/main.tex`](paper/overleaf_package/main.tex) | [`paper/overleaf_submission.zip`](paper/overleaf_submission.zip)
- **English Manuscript (Markdown):** [`paper/draft_en.md`](paper/draft_en.md)
- **Chinese Manuscript:** [`paper/draft_zh.md`](paper/draft_zh.md)

---

## Overview

Retrieval-augmented generation (RAG) benchmarks often synthesize test queries from source passages using instruction-tuned large language models. In rule-governed insurance eligibility and claims workflows, a rewritten query can retrieve the source clause at rank 1 while omitting or altering conditions that determine whether a claim is covered, excluded, or requires clarification. Across nine official Chinese insurance policy wordings ($5$ development policies and $4$ extension policies, $144$ controlled cases certified with dual completion witnesses) and a 16-seed ($64$-case) multi-threshold rule testbed spanning three model tiers (`Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, and `Qwen-Max`), we evaluate clause-grounded eligibility condition preservation and decision determinacy. No LLM or embedding model weights are pre-trained or fine-tuned in this repository; all experiments evaluate inference-time generation, dense/lexical indexing, statistical calibration, and symbolic rule verification.

1. **Clause-Grounded Decision Verification on $5$ Development Insurance Policies ($20$ seeds, $80$ controlled cases, $120$ multi-scale rewrites):**
   - Across five development official Chinese insurance policy wordings (`pa_p0781`, `pa_h865`, `cpic_86017`, `cpic_82859`, `tk_group_2018`), every `missing_critical` (`CLARIFY`) benchmark case is certified by two counterexample completion witnesses yielding opposite rule outcomes (`WAIT_EXCLUDED` vs. `WAIT_NOT_EXCLUDED`).
   - Enumerating complete threshold-partitioned equivalence regions of missing slots (`enumerate_admissible_decisions`, without reading ground-truth `seed_frame` metadata) shows that unconstrained conversational rewriting (`ordinary_style`) preserves standalone decision determinacy in only $30.0\%$ (`1.5B`), $0.0\%$ (`Qwen-Plus`), and $45.0\%$ (`Qwen-Max`), whereas explicit slot-preserving prompting (`condition_prompt`) reaches $95.0\%$ (`1.5B`) and $100.0\%$ (`Qwen-Plus`/`Qwen-Max`).
2. **Extension to Four Official Insurance Policies, Benefit-Specific Rules, and Title-Masked Retrieval ($4$ PDFs, $16$ seeds, $64$ controlled cases):**
   - On four additional official insurance policy PDFs (`pa_2011_36`, `pa_p0254`, `pa_p0321`, `cpic_93896`) with textbook book-title markers (`《》`) and clause-number headers stripped, we conduct a retrospective post-refinement evaluation after adding colloquial product aliases and benefit-level disambiguation (`benefit_id`) to the shared parser (`extract_fields_surface_normalized`), explicitly modeling benefit-specific waiting-period rules (distinguishing general inpatient allowance [Article 4(1), 30-day wait, accident & continuous renewal exempt] from cancer inpatient allowance [Article 4(2), 90-day wait, continuous renewal exempt, no accident exemption] in `pa_2011_36`, and primary lung cancer excluding carcinoma in situ in `pa_p0321`) and separating contract PDF clause anchors (`source_type='pdf_contract_clause'`) from synthetic seed scenario parameters (`source_type='synthetic_seed_frame'`) in `atomic_rubrics`.
   - The unified verifier certifies $100.0\%$ ($64/64$) of controlled cases. Conversational rewrites (`ordinary_style`) achieve high lexical novelty ($1-\kappa_2=0.5380$) but $0/16$ ($0.0\%$) standalone decision determinacy ($5/16$ [$31.2\%$] when granted the source policy ID), while single-pass `condition_prompt` already reaches $16/16$ ($100.0\%$), so `slot_guard_repair` triggers $0$ repair calls (exact two-sided McNemar $p=1.0$ vs. `condition_prompt`, $p=0.03125$ vs. `structured_json_rewrite` at $10/16$). Masking product titles and insurer names drops 9-clause Top-1 retrieval from $75.0\%$ to $31.2\%$ on `bge_small_zh` for title-preserving queries.
3. **Equal-Call-Budget Baseline Comparison on Multi-Threshold Composite Rules ($16$ seeds, $64$ cases, $7$ methods):**
   - On a 16-seed ($64$-case) synthetic multi-threshold rule testbed inspired by clinical pathways and labor statutes (`med_egfr_tki`, `med_vte_prophylaxis`, `law_work_injury`, `law_noncompete`), all 6 non-JSON methods achieve $100.0\%$ ($16/16$) Top-1 retrieval on both `bm25_char2` and `bge_small_zh` (both standard and title-masked), yet `ordinary_style` preserves decision determinacy in only $6.2\%$ ($1/16$), and 1-call `structured_json_rewrite` fails two-line format validation ($0/16$ valid; $0/16$ hit the 160-token cap with `generated_tokens` $\in [45, 111]$, as the model stops immediately after closing the first-line JSON brace).
   - At an identical $26$-call budget ($16$ initial $+$ $10$ second-pass calls on the shared 10-seed trigger schedule), unguided LLM self-refinement (`generic_self_refine_budget_matched`, $26$ calls; unchanged at $32$ calls in `generic_self_refine_all`) achieves $37.5\%$ ($6/16$) all-slot match and $50.0\%$ ($8/16$) decision determinacy with $25.0\%$ ($4/16$) threshold mutations, whereas symbolic parser-guided `slot_guard_repair` ($26$ calls) reaches $68.8\%$ ($11/16$) all-slot match, $75.0\%$ ($12/16$) decision determinacy, and $0.0\%$ ($0/16$) parser-detected mutations (exact two-sided McNemar $p=0.125$ vs. `generic_self_refine_budget_matched`, $p=0.0625$ vs. `condition_prompt`).
4. **Deterministic Verifier Diagnostic Log, Watchlist & Appendix A Community-QA Retrieval Diagnostics:**
   - A deterministic 264-item verifier diagnostic log and manual-inspection watchlist (`data_v3/item_by_item_semantic_audit.jsonl`, explicitly marked as an automated parser explanation and 29-item surface-edge-case watchlist rather than external semantic validation or human annotation) documents per-slot extraction traces and edge cases.
   - Appendix A reports corpus-level retrieval ranking and calibration diagnostics on `baoxianzhidao_filter.csv` ($3{,}746$ answers, $2{,}994$ held-out real queries, `Full-MaxP` zero-truncation check, source-sampling confounding vs. lexical copying decomposition, negative propensity-reweighting results, and paired-bootstrap candidate-set abstention).

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
│   ├── maxp_ablation.py            # Source-sampling confounding, quantile sensitivity & Full-MaxP ablation
│   ├── analyze_v3.py               # Slot completeness vs. decision determinacy, source_overlap_gated & abstention
│   └── score_annotations.py        # Blinded human annotation unblinding & Cohen's kappa scorer
├── tests/                          # Unit test suite (29/29 deterministic tests)
└── paper/                          # English & Chinese manuscripts + Overleaf LaTeX package
```

---

## Reproducing the Experiments & Running Unit Tests

### 1. Run the Deterministic Unit Test Suite (29 Tests)
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Official Policy Condition Verification, Held-Out Confirmation & Equal-Call-Budget Study
```bash
# Build 80 development controlled cases with single-field counterexamples
python3 src/condition_data.py --root ../../work/condition_study --out data_v3

# Run cross-round diagnostics, multi-threshold rescoring, held-out & equal-budget rescoring, and exploratory semantic audit
python3 src/analyze_v3.py
python3 src/cross_domain_study.py --score-only
python3 src/equal_budget_and_heldout_study.py --score-only
python3 src/build_item_semantic_audit.py
```

### 3. Appendix A: Group-Disjoint Community-QA Split, LLM Query Synthesis & Full-MaxP Diagnostics
```bash
python3 src/experiment.py --data ../../work/baoxianzhidao_filter.csv --out results --seed 20261025 --repeats 5
python3 src/validate_v2.py --inputs ../../work/validation_v2/evaluation_inputs.json --generations ../../work/validation_v2/generation_float32 --embeddings ../../work/validation_v2/embedding_cache --bge-model ../../work/models/bge --out results/validation_v2
python3 src/maxp_ablation.py
```
