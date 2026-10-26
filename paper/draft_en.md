# When Do Synthetic Queries Mislead Retriever Selection?
## Selection Reversal, Candidate-Set Abstention, and Eligibility-Condition Preservation in Chinese Insurance Retrieval

**Author:** Shunchi Wang (Independent Researcher)  
**Repository:** [https://github.com/jakelihua/chinese-insurance-rag-benchmark](https://github.com/jakelihua/chinese-insurance-rag-benchmark)

---

## Abstract

When cold-starting a domain-specific retrieval-augmented generation (RAG) system before user query logs exist, practitioners often synthesize queries from knowledge-base passages to rank candidate retrievers. This paper examines two practical questions in Chinese insurance retrieval: (1) when does ranking retrievers on synthetic queries reverse the winner observed on real user questions, and (2) to what extent do format-valid synthetic rewrites drop or alter underwriting and claims eligibility conditions across model scales?

We study these questions in three rounds. In **Round 1**, we deduplicate $3{,}746$ Chinese insurance answers and partition QA pairs into $3{,}725$ connected components via Union-Find so that no shared answer spans calibration ($750$ queries) and test ($2{,}994$ queries) splits. Evaluating six lexical retrievers, we show that a source-prefix diagnostic proxy selects character $3\text{--}5$-gram TF-IDF ($\text{MRR@10}=0.9330$) over character-bigram BM25 ($0.9311$), whereas real test queries favor bigram BM25 ($0.3817$ vs. $0.2727$, selection regret $0.1090$). In **Round 2**, we add dense (`bge_small_zh`) and hybrid (`rrf_bm25_bge`) retrievers and generate zero-shot and 3-shot synthetic queries from held-out answers using both a local 1.5B model (`Qwen2.5-1.5B-Instruct`, $N=249$ common valid sources) and a cloud production model (`Qwen-Plus`, $N=179$). On real test queries, pure dense retrieval ranks first ($\text{MRR@10}=0.5216$ vs. $0.5030$ for hybrid RRF; paired difference $+0.0187$, 95% bootstrap CI $[+0.0080, +0.0300]$). On synthetic queries from both models and both prompting modes, passage-level lexical reuse inflates BM25 scores and reverses the top-1 choice in favor of hybrid RRF (regret $0.0187$). Surface-feature importance weighting fails to fix this reversal ($0\%$ optimal selection across budgets $B \in \{25, 50, 100, 200\}$), while a paired-bootstrap **Candidate-Set Abstention** rule on small real calibration sets avoids confident wrong commitments at $B=25$ ($0\%$ confident wrong rate, $80\%$ abstention, $100\%$ optimal coverage, mean set size $2.60$ of $8$). In **Round 3**, we build $80$ controlled clause-grounded cases with single-field counterexamples from five official insurance policy wordings and evaluate $120$ paired rewrites across `Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, and `Qwen-Max`. All $120/120$ rewrites pass syntactic format checks, yet unconstrained conversational rewriting (`ordinary_style`) preserves standalone rule decisions in only $30\%$ (`1.5B`), $0\%$ (`Qwen-Plus`), and $20\%$ (`Qwen-Max`) of seeds. Scaling up model size eliminates numeric value mutations ($15\% \rightarrow 0\%$) but does not stop conversational omission of product names or enrollment status. Explicitly listing required condition slots (`condition_prompt`) raises standalone decision preservation to $95\%$ on `1.5B` and `Qwen-Plus`, and $100\%$ ($20/20$) on `Qwen-Max`.

---

## 1. Introduction

Selecting a retriever for a vertical-domain RAG system—for instance, choosing among character $n$-gram BM25, dense bi-encoders, or Reciprocal Rank Fusion (RRF)—often happens before real user traffic is available. A common workflow in both retrieval research and automated RAG evaluation frameworks is to generate synthetic questions from corpus passages using an instruction-tuned LLM, score candidate retrievers on the synthetic set, and deploy the top-ranked system.

While prior work has studied macro-level rank correlations between synthetic and real test collections, two specific failure modes matter when making concrete deployment decisions in insurance retrieval:
1. **Dense vs. Hybrid Selection Reversal**: Passage-conditioned query generation tends to reuse domain terms from the source passage. When synthetic queries have higher lexical overlap with target documents than colloquial user queries do, BM25 scores are artificially elevated, which in turn boosts hybrid RRF over pure dense retrieval even when dense retrieval performs better on real user questions.
2. **Loss of Underwriting and Claims Conditions**: In insurance clauses, coverage outcomes (`WAIT_EXCLUDED` vs. `WAIT_NOT_EXCLUDED`) depend on specific variables: product identity, first-time vs. renewal enrollment, policy effective day, sickness vs. accidental injury cause, plan-specific waiting period (e.g., 7 vs. 25 days), and insurer waiver approval. A rewrite that satisfies standard format filters (single line, question mark, valid length) can still drop or alter one of these variables, making the question undecidable (`CLARIFY`) or flipping its coverage outcome.

This paper reports a three-round empirical and methodological study on Chinese insurance retrieval, pairing diagnostic failure analysis with concrete, falsifiable mitigations. To avoid terminology ambiguity, we explicitly distinguish two levels of abstention studied in this work:
- **Retriever-Selection Candidate-Set Abstention (Round 2)**: At the benchmark evaluation level, returning a bootstrap confidence set of candidate retrievers $\widehat{\mathcal{C}}_{0.95}$ rather than committing to a single noisy winner when the real calibration budget $B$ cannot statistically separate top contenders.
- **Query-Level Clarification Abstention (`CLARIFY`, Round 3)**: At the individual question level, withholding a binary exclusion decision when a pivotal underwriting condition slot is omitted, formally certified via two opposite-outcome completion witnesses.

Our scope is intentionally bounded: we evaluate retrieval selection and clause condition conservation rather than end-to-end online claims adjudication, and we separate deterministic slot checks from human verification. Specifically:
- **Round 1 (Group-Disjoint Lexical Baseline)**: On $3{,}746$ deduplicated answers from a public Chinese insurance QA corpus partitioned into $3{,}725$ connected groups ($750$ calibration, $2{,}994$ held-out test queries), we compare six lexical retrievers against extractive keyword and prefix-copy proxies, quantifying how long-$n$-gram copy bias selects `tfidf_char3_5` over the true lexical winner `bm25_char2` (regret $0.1090$ MRR@10).
- **Round 2 (Cross-Scale Selection Reversal and Candidate-Set Abstention)**: Expanding to eight systems (six lexical, `bge_small_zh`, and `rrf_bm25_bge`), we evaluate zero-shot and 3-shot queries generated by local `Qwen2.5-1.5B-Instruct` ($N=249$ common valid sources) and cloud `Qwen-Plus` ($N=179$). Both models select `rrf_bm25_bge` under both prompting strategies, reversing the real-test winner `bge_small_zh` ($\text{MRR@10}=0.5216$ vs. $0.5030$). We show why unpaired surface-feature density weighting fails ($0\%$ optimal selection), and demonstrate that paired-bootstrap **candidate-set abstention** on small real calibration samples reduces confident wrong retriever commitments at $B=25$ from $60\%\text{--}100\%$ to $0\%$.
- **Round 3 (Double-Witness Counterexamples and Multi-Scale Condition Preservation)**: Using five official Chinese insurance policy clauses, we formalize a waiting-period rule compiler and construct $20$ seed scenarios ($80$ controlled variants where every `missing_critical` `CLARIFY` case carries two opposite-outcome completion witnesses) alongside $120$ paired LLM rewrites across `Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, and `Qwen-Max`. We separate scale-dependent numeric mutations ($15\%$ on `1.5B` vs. $0\%$ on `Qwen-Plus`/`Qwen-Max`) from scale-invariant pragmatic slot omission under ordinary conversational prompting, and show that explicit slot constraints (`condition_prompt`) raise standalone decision preservation from $0\%\text{--}30\%$ to $95\%\text{--}100\%$ and top-1 clause retrieval to $95\%\text{--}100\%$.

---

## 2. Related Work

- **Synthetic Queries in IR Evaluation**: Promptagator and InPars-v2 use LLM-generated queries to train retrievers when in-domain labels are scarce. Evaluation frameworks such as ARES and RAGAs also use synthetic queries to score RAG pipelines. Recent analyses by Rahmani et al. (2024) and Boyd et al. (2025) note that LLM-generated evaluation sets carry generator-specific biases. We focus on a concrete mechanism—lexical overlap inflation shifting the winner from pure dense retrieval to hybrid BM25+dense fusion—and test it across both 1.5B and commercial cloud generators.
- **Calibration and Abstention under Small Budgets**: Prediction-Powered Inference (PPI) and ARES combine synthetic predictions with small gold samples to construct confidence intervals on paired data. In cold-start retriever selection, however, synthetic queries are generated from documents without paired real user questions on the same test split. We evaluate unpaired shrinkage and surface-density weighting as negative baselines, and formulate a bootstrap candidate-set abstention rule that withholds point commitment when calibration uncertainty spans multiple systems.
- **Insurance QA Benchmarks**: InsuranceQA and its Chinese adaptations provide community question-answering pairs for answer selection. Because community answers contain duplicates and lack formal underwriting parameters, we augment our Round 1–2 FAQ experiments with a Round 3 benchmark built directly from five official Chinese insurance policy wordings with deterministic waiting-period rules.

---

## 3. Round 1: Group-Disjoint Split and Lexical Proxy Diagnostics

### 3.1 Leakage-Free Group Split
Public Chinese insurance FAQ corpora (`baoxianzhidao_filter.csv`, $8{,}362$ raw rows) contain repeated questions and shared boilerplate answers. Filtering for `is_best == '1'`, length bounds ($4 \le |q| \le 500$, $15 \le |d| \le 3{,}000$), and PII removal yields $3{,}746$ unique answers and $3{,}744$ valid QA groups. We build an undirected graph linking any pairs that share normalized questions, normalized answers, or character $2\text{--}4$-gram TF-IDF title cosine similarity $\ge 0.90$, yielding $3{,}725$ connected components via Union-Find. Assigning whole components to splits (`seed=20261025`) produces:
- **Real Calibration Pool**: $750$ queries across $745$ components (zero shared answers with test; max cross-split title cosine $0.8965$).
- **Real Test Set**: $2{,}994$ queries across $2{,}980$ components.

### 3.2 Lexical Systems and Round 1 Results
We index all $3{,}746$ answers under six character $n$-gram lexical retrievers: sublinear TF-IDF cosine similarity over character unigrams (`tfidf_char1`), bigrams (`tfidf_char2`), $1\text{--}3$-grams (`tfidf_char1_3`), and $3\text{--}5$-grams (`tfidf_char3_5`), alongside Okapi BM25 ($k_1=1.2, b=0.75$) over character unigrams (`bm25_char1`) and bigrams (`bm25_char2`).

| System | Real Test MRR@10 | Real Test Hit@5 | Keyword Proxy MRR@10 | Keyword Proxy Hit@5 | Prefix Copy MRR@10 | Prefix Copy Hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| `tfidf_char1` | 0.2296 | 0.3313 | 0.7647 | 0.9039 | 0.6692 | 0.7603 |
| `bm25_char1` | 0.2529 | 0.3427 | 0.6921 | 0.8489 | 0.8483 | 0.9063 |
| `tfidf_char3_5` | 0.2727 | 0.3557 | 0.4578 | 0.6602 | **0.9330** | **0.9613** |
| `tfidf_char2` | 0.3180 | 0.4218 | 0.8890 | 0.9821 | 0.8795 | 0.9354 |
| `tfidf_char1_3` | 0.3187 | 0.4285 | 0.8704 | 0.9720 | 0.9010 | 0.9474 |
| `bm25_char2` | **0.3817** | **0.4826** | **0.9335** | **0.9888** | 0.9311 | 0.9605 |

On held-out real queries, `bm25_char2` ranks first ($\text{MRR@10}=0.3817, \text{Hit@5}=0.4826$), whereas `tfidf_char3_5` ranks fourth ($\text{MRR@10}=0.2727, \text{Hit@5}=0.3557$). On `source_prefix_diagnostic`, contiguous substring copying pushes `tfidf_char3_5` to first place ($0.9330$ vs. $0.9311$ for `bm25_char2`), causing a true-test selection regret of $0.3817 - 0.2727 = 0.1090$. Across $5$ random draws per calibration budget $B \in \{25, 50, 100, 200\}$, `real_only` selects `bm25_char2` in $80\%$ ($4/5$) of draws at $B=25$ (mean regret $0.0126$, $\tau_b = 0.4133$) and $100\%$ ($5/5$) of draws at $B \ge 50$ (regret $0.0000$, $\tau_b = 0.6267, 0.8933, 0.9200$).

---

## 4. Round 2: LLM Synthetic Queries, Selection Reversal, and Abstention

### 4.1 Setup and Shared Valid Source Sets
We expand the candidate pool to eight systems by adding:
- `bge_small_zh`: Dense retrieval with `BAAI/bge-small-zh-v1.5` (512-dim L2-normalized CLS embeddings; 512-token limit, truncating $509$ of $3{,}746$ long answers, while $2$ of $256$ source answers exceed $512$ tokens during LLM prompt construction).
- `rrf_bm25_bge`: Reciprocal Rank Fusion ($k=60$, depth $100$) combining `bm25_char2` (requiring positive lexical score) and `bge_small_zh`.

From $256$ target answer passages disjoint from the 3 calibration style demonstrations (`indices=[79, 513, 623]`), we run two generators under `zero_shot` and `few_shot_3`:
1. `Qwen2.5-1.5B-Instruct` (local MPS, `float32/eager`, greedy decoding, `max_new_tokens=64`): $255/256$ (`zero_shot`) and $250/256$ (`few_shot_3`) pass format validation, leaving **$N=249$ common valid sources**. (An initial `float16+sdpa` run on MPS exhibited numerical underflow and is archived in `failed_float16_run.json`.)
2. `Qwen-Plus` (cloud API, `temperature=0`, `max_tokens=64`): $243/256$ (`zero_shot`) and $184/256$ (`few_shot_3`) pass the single-question filter, yielding **$N=179$ common valid sources**.

| Query Set / Generator | Metric | `bm25_char2` | `tfidf_char3_5` | `bge_small_zh` | `rrf_bm25_bge` | Selected System (of 8) |
|---|---|---:|---:|---:|---:|---|
| Held-Out Real Test ($N=2{,}994$) | MRR@10 / Hit@5 | 0.3817 / 0.4826 | 0.2727 / 0.3557 | **0.5216** / **0.6453** | 0.5030 / 0.6075 | **`bge_small_zh` (True Winner)** |
| Real Calibration Pool ($N=750$) | MRR@10 / Hit@5 | 0.3681 / 0.4800 | 0.2687 / 0.3587 | **0.5370** / **0.6533** | 0.5034 / 0.6333 | **`bge_small_zh`** |
| Matched Keyword Proxy ($N=249$) | MRR@10 / Hit@5 | **0.9359** / **0.9920** | 0.4132 / 0.6586 | 0.5165 / 0.6707 | 0.7639 / 0.8916 | `bm25_char2` (Regret $0.1399$) |
| `Qwen2.5-1.5B` `zero_shot` ($N=249$) | MRR@10 / Hit@5 | 0.5013 / 0.6265 | 0.4360 / 0.5542 | 0.5202 / 0.6787 | **0.5647** / **0.6948** | `rrf_bm25_bge` (Reversed, Regret $0.0187$) |
| `Qwen2.5-1.5B` `few_shot_3` ($N=249$) | MRR@10 / Hit@5 | 0.3225 / 0.4016 | 0.2776 / 0.3414 | 0.4024 / **0.4940** | **0.4124** / 0.4779 | `rrf_bm25_bge` (Reversed on MRR@10) |
| `Qwen-Plus` `zero_shot` ($N=179$) | MRR@10 / Hit@5 | 0.6528 / 0.7486 | 0.4553 / 0.5754 | 0.7005 / **0.8380** | **0.7371** / **0.8380** | `rrf_bm25_bge` (Reversed, Regret $0.0187$) |
| `Qwen-Plus` `few_shot_3` ($N=179$) | MRR@10 / Hit@5 | 0.5747 / 0.6927 | 0.3832 / 0.5307 | 0.6794 / 0.8324 | **0.6813** / **0.8547** | `rrf_bm25_bge` (Reversed, Regret $0.0187$) |

### 4.2 Mechanism of Selection Reversal and Candidate-Set Abstention
1. **Real Test Ordering**: On the $2{,}994$ held-out real questions, `bge_small_zh` achieves $\text{MRR@10}=0.5216$ ($\text{Hit@5}=0.6453$), beating `rrf_bm25_bge` ($0.5030$, $\text{Hit@5}=0.6075$) by $+0.0187$ (1,000-sample paired bootstrap 95% CI: $[+0.0080, +0.0300]$).
2. **Lexical Copy Inflation**: On real test queries (mean length $25.25$ chars; calibration $27.12$ chars), query-to-target bigram and trigram containment are $0.3243$ and $0.1808$. Under `zero_shot`, bigram and trigram containment jump to $0.4744$ and $0.3442$ on `Qwen2.5-1.5B` ($13.94$ chars), and $0.4640$ and $0.2850$ on `Qwen-Plus` ($22.83$ chars). Under `few_shot_3`, lengths are $20.63$ (`1.5B`, bigram $0.3180$, trigram $0.2199$) and $24.03$ (`Qwen-Plus`, bigram $0.4108$, trigram $0.2313$). On `1.5B`, mimicking 3 unrelated style examples introduces semantic drift in $6/20$ audited queries (`semantic_audit.json`), whereas on `Qwen-Plus` `few_shot_3` narrows the RRF-vs-BGE gap to $-0.0019$ ($0.6813$ vs. $0.6794$) but still reverses the point winner.
3. **Failure of Unpaired Density Reweighting vs. Candidate-Set Abstention**: A 2-fold cross-fit character $1\text{--}3$-gram logistic classifier separates real calibration from synthetic queries with $\text{ROC-AUC}=0.8120$ (`zero_shot`) and $0.7733$ (`few_shot_3`) at $B=25$ (rising to $0.8765$ and $0.9148$ at $B=200$), yet `synthetic_density_weighted` selects `bge_small_zh` in **$0\%$ of trials** across all budgets. Conversely, our 95% paired-bootstrap **Candidate-Set Abstention** rule ($200$ resamples over $\mathcal{Q}_{\text{cal}}^{(B)}$) eliminates confident wrong commitments at $B=25$:

| Selection Policy | Metric | $B=25$ | $B=50$ | $B=100$ | $B=200$ |
|---|---|---:|---:|---:|---:|
| `synthetic_density_weighted` | Wrong Selection $\downarrow$ | 100% | 100% | 100% | 100% |
| `mix_real_density_0.5` | Wrong Selection $\downarrow$ | 80% | 60% | 40% | 60% |
| `mse_shrunk` (Unpaired) | Wrong Selection $\downarrow$ | 60% | 40% | 20% | 0% |
| `real_only` (Point Commit) | Wrong Selection $\downarrow$ | 60% | 40% | 0% | 0% |
| **Candidate-Set Abstention** | **Confident Wrong** $\downarrow$ | **0%** | 20% | **0%** | **0%** |
| **Candidate-Set Abstention** | **Abstention Rate** | 80% | 60% | 40% | 60% |
| **Candidate-Set Abstention** | **Optimal Coverage** $\uparrow$ | **100%** | 80% | **100%** | **100%** |
| **Candidate-Set Abstention** | **Mean Set Size (of 8)** | 2.60 | 1.60 | 1.40 | 1.60 |

---

## 5. Round 3: Eligibility-Condition Preservation Across Model Scales

### 5.1 Traceable Clauses and Double-Witness Counterexamples
We collect five official Chinese insurance policy PDFs (`data_v3/sources.json`) with verified SHA256 hashes, page numbers, NFKC character offsets, and PDF bounding boxes:
- `pa_p0781`: *Ping An Caring Companion Group Medical Insurance (Plan B)* (90-day wait, p. 4, accident exempts, no waiver).
- `pa_h865`: *Ping An Employee & Dependent Inpatient Medical Insurance* (30-day wait, p. 3, accident or approved waiver exempts).
- `cpic_86017`: *CPIC Internet Outpatient Medical Insurance (Plan C)* (7/15/25-day plan wait on Sec. 2.5.2 online medication, p. 4, approved waiver exempts).
- `cpic_82859`: *CPIC Internet Personal 1-Year Inpatient Medical Insurance (Plan E)* (30-day wait, p. 3, accident or approved waiver exempts).
- `tk_group_2018`: *Taikang Joyful Global (Premier) High-End Medical Insurance* (30-day wait, p. 4, accident exempts, no waiver).

Each structured scenario frame $z \in \mathcal{Z}$ specifies up to six required slots: `policy_id` ($20/20$ seeds), `enrollment` ($20/20$), `policy_day` ($20/20$), `cause` ($20/20$), `plan_wait_days` ($4/20$ seeds on `cpic_86017`), and `approved_waiver` ($3/20$ renewal seeds). From $20$ seed frames, we generate $80$ controlled cases (`complete`, `surface`, `missing_critical`, `counterfactual`), where every `missing_critical` (`CLARIFY`) item carries two explicit counterexample completions $v_1, v_2$ such that $f(z_{-k}, v_1) \neq f(z_{-k}, v_2)$.

### 5.2 Multi-Scale Paired Rewriting (`1.5B`, `Qwen-Plus`, `Qwen-Max`)
We evaluate $120$ paired rewrites ($3$ model tiers $\times 20$ seeds $\times 2$ prompts: `ordinary_style` vs. `condition_prompt`):

| Diagnostic Metric | `1.5B` `ordinary` | `1.5B` `cond_prompt` | `Qwen-Plus` `ordinary` | `Qwen-Plus` `cond_prompt` | `Qwen-Max` `ordinary` | `Qwen-Max` `cond_prompt` |
|---|---:|---:|---:|---:|---:|---:|
| Syntactic `format_valid` Rate | 100% (20/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) | 100% (20/20) | **100% (20/20)** |
| Slot Missing/Mismatch: `policy_id` (/20) | 70% (14/20) | **5% (1/20)** | 100% (20/20) | **0% (0/20)** | 50% (10/20) | **0% (0/20)** |
| Slot Missing/Mismatch: `enrollment` (/20) | 35% (7/20) | **0% (0/20)** | 5% (1/20) | **0% (0/20)** | 30% (6/20) | **0% (0/20)** |
| Slot Missing/Mismatch: `policy_day` (/20) | 25% (5/20) | **0% (0/20)** | 35% (7/20) | **0% (0/20)** | 20% (4/20) | **0% (0/20)** |
| Slot Missing/Mismatch: `cause` (/20) | 10% (2/20) | **0% (0/20)** | 15% (3/20) | 5% (1/20) | **0% (0/20)** | **0% (0/20)** |
| Slot Missing/Mismatch: `plan_wait_days` (/4) | 100% (4/4) | **0% (0/4)** | 25% (1/4) | **0% (0/4)** | 25% (1/4) | **0% (0/4)** |
| Slot Missing/Mismatch: `approved_waiver` (/3) | 100% (3/3) | **0% (0/3)** | 33% (1/3) | **0% (0/3)** | **0% (0/3)** | **0% (0/3)** |
| Numeric Value Mutation Rate (`20d` $\to$ `2d`) | 15% (3/20) | **0% (0/20)** | **0% (0/20)** | **0% (0/20)** | **0% (0/20)** | **0% (0/20)** |
| Strict Regex Pass (`v1_strict_regex`) | 30% (6/20) | 50% (10/20) | 0% (0/20) | 95% (19/20) | 20% (4/20) | **100% (20/20)** |
| **Standalone Decision Preserved (v2)** | 30% (6/20) | **95% (19/20)** | 0% (0/20) | **95% (19/20)** | 20% (4/20) | **100% (20/20)** |
| **Decision Given Policy Context (v2)** | 50% (10/20) | **100% (20/20)** | 45% (9/20) | **95% (19/20)** | 50% (10/20) | **100% (20/20)** |
| 5-Clause Top-1 Accuracy (`bm25_char2`) | 50% (10/20) | **100% (20/20)** | 80% (16/20) | **100% (20/20)** | 90% (18/20) | **95% (19/20)** |
| 5-Clause MRR@10 (`bm25_char2`) | 0.6792 | **1.0000** | 0.8542 | **1.0000** | 0.9350 | **0.9750** |

### 5.3 Key Findings on Condition Drift
1. **Format Validation Does Not Catch Condition Loss**: All $120/120$ generated questions pass `format_valid`, yet under `ordinary_style`, $70\%$ (`1.5B`), $100\%$ (`Qwen-Plus`), and $80\%$ (`Qwen-Max`) fail to preserve standalone rule decidability.
2. **Scale-Dependent Value Mutation vs. Scale-Invariant Omission**: On `1.5B`, `ordinary_style` mutates `"保单生效第20天"` into `"保单生效后的第二天"` in $3/20$ seeds (`condition-013, 015, 016`) and collapses `condition-009` (`WAIT_NOT_EXCLUDED`, 7-day wait) and `condition-010` (`WAIT_EXCLUDED`, 25-day wait) into the exact same unanswerable string. Scaling to `Qwen-Plus` and `Qwen-Max` eliminates numeric mutations ($0\%$), but larger models still omit product names ($100\%$ on `Qwen-Plus`, $50\%$ on `Qwen-Max`) or rewrite first-time enrollment into ambiguous phrasing ($30\%$ on `Qwen-Max`) when instructed to sound colloquial.
3. **Universal Effectiveness of `condition_prompt`**: Explicitly constraining required slots restores standalone decision preservation to $95\%$ (`1.5B` and `Qwen-Plus`) and $100\%$ (`Qwen-Max`), and lifts 5-clause `bm25_char2` Top-1 accuracy to $95\%\text{--}100\%$.

---

## 6. Cross-Domain Generalizability and Limitations

- **Structural Generalizability Beyond Insurance**: Both failure mechanisms and our proposed mitigations apply directly to other rule-governed vertical domains. First, passage-conditioned lexical co-voting bias arises whenever formal corpus documents (e.g., statutory tax codes, clinical practice guidelines, or legal compliance manuals) are paired with colloquial user queries; paired-bootstrap candidate-set abstention requires only per-query metric matrices and transfers without domain modification. Second, insurance waiting-period clauses are structurally isomorphic to **clinical trial eligibility screening** (lab thresholds, washout days, prior-therapy waivers), **statutory legal/tax advisory** (holding periods, filing status exemptions), and **financial underwriting**. Any domain governed by deterministic multi-slot decision tables $f: \mathcal{Z} \to \mathcal{Y}$ can directly instantiate our double-witness criterion ($f(z_{-k}, v_1) \neq f(z_{-k}, v_2)$) and slot-enumerated prompting (`condition_prompt`).
- **Limitations**: First, our Round 1–2 FAQ corpus comes from a single community insurance dataset with weak historical answer labels, and `bge_small_zh` truncates $509/3{,}746$ answers exceeding $512$ tokens. Second, our multi-scale experiments evaluate three models within the Qwen family (`Qwen2.5-1.5B-Instruct`, `Qwen-Plus`, and `Qwen-Max`) and one dense encoder. Third, Round 3 covers five official medical and outpatient clauses ($80$ controlled cases, $120$ LLM rewrites), and independent two-rater human annotation (`reviewer_A.html` and `reviewer_B.html`, scored via `score_annotations.py`) remains pending.
