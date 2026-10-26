# Can a Few Authentic Queries Calibrate Synthetic Retrieval Evaluation? An Empirical Study on Selection Reversal and Business-Condition Conservation in Chinese Insurance QA

**Status: Exploratory Research Manuscript v0.3 (Updated 2026-10-26; Pre-Confirmation Draft).** Author list and affiliations to be confirmed by human contributors. Current experiments evaluate historical answer retrieval (Rounds 1–2) and controlled waiting-period clause condition conservation (Round 3); they do not claim end-to-end online RAG/Agent claims adjudication gains. Double-blind human annotation packs for Round 3 are prepared and awaiting independent completion.

---

## Abstract

Publicly available user question-answering (QA) datasets in Chinese commercial insurance often lack traceable contract provenance, verified answer accuracy, and complete product condition specifications. In enterprise settings where authentic user queries are scarce, teams routinely rely on synthetic queries generated from knowledge-base documents to select among candidate retrievers. In this paper, we conduct a three-round reproducible empirical investigation into when synthetic evaluation misleads retriever selection, whether small-budget covariate calibration can correct selection bias, and why unconstrained style rewriting corrupts business answerability:

1. **Round 1 (Lexical Retrieval Baseline & Extractive Diagnostics)**: Over 3,746 deduplicated insurance answers, 750 calibration queries, and 2,994 held-out authentic test queries (partitioned via 3,725 Union-Find connected components with zero answer overlap), we compare six character-level TF-IDF and BM25 retrievers. While an extractive keyword proxy and the held-out test set both select `bm25_char2` ($\text{MRR@10}=0.9335$ and $0.3817$, respectively), a source-prefix copying diagnostic erroneously selects `tfidf_char3_5` (incurring a true selection regret of $0.1090$). At real-query budgets $B \ge 50$, evaluating directly on the real calibration sample (`real_only`) selects the optimal lexical retriever in $100\%$ of trials, whereas character-length and logistic density-ratio reweighting provide no additional selection benefit.
2. **Round 2 (Local LLM Generation, Dense/Hybrid Selection Reversal & Support Audit)**: Sampling 256 source answers disjoint from calibration demonstrations, we generate 512 synthetic queries using local `Qwen2.5-1.5B-Instruct` (`float32/eager` on Apple MPS; 249 common format-valid sources) under zero-shot and 3-shot authentic-style guidance. Expanding the candidate pool to eight systems by adding dense retrieval (`bge-small-zh-v1.5`) and fixed Reciprocal Rank Fusion (`rrf_bm25_bge`, $k=60$, depth $100$) across 360 calibration trials reveals a **system selection reversal**: both zero-shot and 3-shot LLM synthetic benchmarks select hybrid retrieval `rrf_bm25_bge` ($\text{MRR@10}=0.5647$ and $0.4124$), whereas the held-out authentic test set favors pure dense retrieval `bge_small_zh` ($\text{MRR@10}=0.5216$ vs. $0.5030$; paired difference $+0.0187$, 1,000-sample paired bootstrap $95\%$ interval $[+0.0080, +0.0300]$). Surface character density-ratio reweighting fails to overcome this LLM co-voting bias ($0\%$ optimal selection across all budgets), and fixed $0.5$ mixing degrades `zero_shot` selection accuracy at $B=200$ from $100\%$ (`real_only`) to $40\%$. Conversely, a calibration paired-bootstrap **Candidate-Set Abstention** rule reduces the Confident Wrong Selection Rate at $B=25$ from $60\%\text{--}100\%$ to **$0.0\%$** ($80\%$ abstention rate, $100\%$ optimal coverage, mean set size $2.6$). Furthermore, an author-side semantic support audit on 40 paired queries ($20$ sources) shows that 3-shot style guidance introduces unsupported business condition mutations or intent shifts in $6/20$ ($30\%$) queries (versus $0/20$ in zero-shot).
3. **Round 3 (Traceable Policy Clauses, Counterexample Witnesses & Condition-Conserving Synthesis)**: To isolate and remedy condition drift, we collect five official insurance contract PDFs across three major Chinese insurance groups (Ping An, CPIC, Taikang) with verifiable SHA256 hashes, page numbers, character spans, and bounding-box anchors. Within a formally bounded waiting-period exclusion domain, we construct 20 seed scenarios $\times$ 4 controlled variants (`complete`, `surface`, `missing_critical`, and `counterfactual`, totaling 80 cases), requiring every `missing_critical` (`CLARIFY`) case to carry **two explicit counterexample completion witnesses** yielding opposite rule decisions (`WAIT_EXCLUDED` vs. `WAIT_NOT_EXCLUDED`). Paired rewriting experiments (40 generations) with local `Qwen2.5-1.5B-Instruct` comparing unconstrained colloquial rewriting (`ordinary_style`) against explicit business-condition preservation prompting (`condition_prompt`) demonstrate that: (a) $100\%$ ($40/40$) of generated queries pass surface format checks, proving format gates are blind to semantic corruption; (b) `ordinary_style` drops product identity in $14/20$ ($70\%$) cases, drops multi-plan waiting periods in $4/4$ ($100\%$) cases, mutates `"policy day 20"` into `"the second day"` in $3/20$ cases, and causes an **opposite-label collision**—rewriting two scenarios with opposite ground-truth labels (`WAIT_NOT_EXCLUDED` under a 7-day plan vs. `WAIT_EXCLUDED` under a 25-day plan on day 10) into the exact same character-for-character unanswerable string; and (c) `condition_prompt` raises standalone rule-decision preservation from $30\%$ ($6/20$) to **$95\%$ ($19/20$)** ($100\%$ when source policy identity is granted) and restores `bm25_char2` Top-1 target-clause retrieval accuracy from $50.0\%$ to **$100.0\%$**, while exposing a $9/20$ conservative false-negative rate of strict regex slot guards on synonymous surface phrasings.

---

## 1 Introduction

Offline evaluation of enterprise Retrieval-Augmented Generation (RAG) and knowledge QA systems must support not merely score reporting, but **system selection**—identifying which candidate retriever $s \in \mathcal{S}$ maximizes expected utility $\mu_s = \mathbb{E}_{q \sim P}[m(s, q)]$ under the target user query distribution $P$. In high-stakes vertical domains such as commercial insurance policy consultation, underwriting, and pre-claim triage, authentic user logs are frequently scarce or restricted by privacy regulations. Consequently, practitioners generate synthetic evaluation queries $q' \sim Q$ from corpus documents or historical answers.

In Chinese insurance QA, document-to-query synthesis encounters two compounding structural failure modes:

1. **Surface Lexical and Distributional Divergence ($Q \neq P$)**: Insurance clauses and curated customer-service answers employ standardized legal and actuarial terminology (e.g., *"upon expiration of the waiting period"*, *"first-time non-continuous enrollment"*), whereas lay users ask brief, colloquial, and context-elliptical questions (*"Can I claim if I got sick 20 days after buying?"*). When synthetic queries inherit lexical fragments from the source answer, lexical retrievers (TF-IDF, BM25) and lexical-dense hybrid fusion (RRF) receive inflated matching signals, distorting comparative system rankings.
2. **Pivotal Business-Condition Drift and Answerability Corruption ($Y(q') \neq Y(q)$)**: To make synthetic queries resemble natural user utterances, researchers often apply few-shot style prompting with authentic user examples or instruct LLMs to rewrite queries colloquially. However, insurance contract outcomes are governed by strict conjunctive business conditions: product identity and version (where waiting periods vary across $7$, $15$, $25$, $30$, or $90$ days), enrollment type (first-time vs. re-enrollment with or without an approved waiting-period waiver), policy elapsed day, event cause (accidental injury exemptions vs. illness exclusions), and selected plan tier. When stylistic rewriting omits a product name, drops a plan tier, or mutates a numeric/categorical slot, the source document's relevance and decision label are invalidated—turning a deterministically answerable scenario into an unanswerable query requiring clarification (`CLARIFY`) or silently flipping the ground-truth outcome (`WAIT_EXCLUDED` $\leftrightarrow$ `WAIT_NOT_EXCLUDED`).

Because prior work has already established general synthetic test collection frameworks [1–4] and documented rank divergence between synthetic and authentic RAG benchmarks [5, 6], we do not claim the existence of distribution shift itself as a novel finding. Instead, we investigate three narrower, falsifiable research questions:

- **RQ1 (Limits of Unpaired Covariate Calibration)**: Given a small budget $B \in \{25, 50, 100, 200\}$ of authentic user queries, does importance reweighting (via length histograms or out-of-fold character n-gram density ratios) or linear shrinkage consistently reduce retriever selection regret compared to simply evaluating on those $B$ authentic queries directly (`real_only`)?
- **RQ2 (Risk Control via Candidate-Set Abstention)**: When finite calibration budgets cannot statistically separate top contenders (`bge_small_zh` vs. `rrf_bm25_bge`) and synthetic preferences conflict with real calibration samples, can a paired-bootstrap candidate-set abstention mechanism eliminate confident wrong selections while maintaining high coverage of the true optimal system?
- **RQ3 (Business-Condition Conservation and Counterexample Verification)**: On traceable official insurance policy clauses, to what extent does unconstrained colloquial rewriting corrupt pivotal business conditions and flip waiting-period decisions? Can explicit condition-preserving prompts (`condition_prompt`) coupled with deterministic slot guards prevent label corruption and opposite-label collisions, and where do regex slot guards fail due to surface synonymy?

---

## 2 Related Work

**Synthetic Retrieval Evaluation and Rank Divergence.** Rahmani et al. [1] studied synthetic test collections for retrieval evaluation. RAGEval [3] uses scenario schemas to synthesize domain documents, questions, answers, and grounding evidence, while DataMorgana [4] controls question categories and user traffic distributions in enterprise QA. Recently, van Elburg et al. [5] and Kucia & Gawlik [6] compared synthetic and human benchmarks on RAG system rankings, showing that query distribution divergence alters retriever selection. Our Round 2 experiments corroborate this selection reversal in Chinese insurance retrieval (where LLM synthesis favors RRF hybrid retrieval while authentic queries favor BGE dense retrieval) and go further by auditing *why* 3-shot authentic-style prompting exacerbates semantic condition drift.

**Prediction-Powered Inference (PPI) and Importance Calibration.** ARES [2] combines synthetic judge training with limited human labels via Prediction-Powered Inference (PPI) [7], and Divekar [8] extends PPI to LLM-based ranking evaluation. Crucially, standard PPI rectifies *judge measurement error* on the *same* query distribution by pairing gold labels and model predictions on identical items. By contrast, when synthetic queries $q' \sim Q$ and authentic queries $q \sim P$ are unpaired, covariate density weighting $\hat{w}(q') = \hat{P}(\phi(q'))/\hat{Q}(\phi(q'))$ requires conditional invariance $\mathbb{E}_Q[m(s, q) \mid \phi(q)] = \mathbb{E}_P[m(s, q) \mid \phi(q)]$. Because document-derived synthetic queries exhibit intrinsic lexical coupling with their generator source document $d^*$, this conditional invariance breaks down—explaining our negative empirical findings on density-ratio reweighting.

**Chinese Insurance Benchmarks and Answerability.** InsQABench [9] provides Chinese insurance QA datasets, but relies heavily on LLM-generated queries and answers that cannot substitute for authentic user queries or verified contract rules. In Round 3, rather than relying on LLM self-judgments of answerability, we anchor every rule to character spans and bounding boxes in official insurer PDFs and enforce a formal **double-witness criterion** for `CLARIFY` cases.

---

## 3 Problem Formulation and Methodology

### 3.1 Retriever Selection, Reweighting, and Shrinkage Estimators

Let $\mathcal{D} = \{d_1, \dots, d_N\}$ denote the answer corpus and $\mathcal{S} = \{s_1, \dots, s_K\}$ the candidate retrievers. For query $q$ with relevant document set $G(q) \subseteq \mathcal{D}$, let $m(s, q) \in [0, 1]$ be the evaluation metric ($\text{MRR@10}$ as primary, $\text{Hit@5}$ as secondary; lexical retrievers require strictly positive matching scores $score(d) > 0$ to prevent zero-score tie inflation). The true target quality under authentic distribution $P$ is $\mu_s = \mathbb{E}_{q \sim P}[m(s, q)]$. Choosing $\hat{s} = \arg\max_{s \in \mathcal{S}} \hat{\mu}_s$ incurs selection regret:
$$\text{Regret}(\hat{s}) = \max_{s \in \mathcal{S}} \mu_s - \mu_{\hat{s}}$$
Given $M$ synthetic queries $\{q'_j\}_{j=1}^M \sim Q$ and a calibration budget of $B$ authentic queries $\mathcal{Q}_{\text{cal}}^{(B)} \sim P$ (strictly disjoint from the held-out test set $\mathcal{Q}_{\text{test}}$), we compare:
1. `synthetic_uniform`: unweighted synthetic mean $\hat{\mu}_s^{\text{syn}}$.
2. `synthetic_budget_matched`: unweighted mean of $\min(B, M)$ randomly sampled synthetic queries.
3. `synthetic_length_weighted`: importance weighting via smoothed character-length histogram ratios clipped to $[0.1, 10.0]$.
4. `synthetic_density_weighted`: 2-fold stratified cross-fit logistic regression (`C=1.0, class_weight='balanced'`) over 1–3 character n-gram TF-IDF features (4,000 max features), using out-of-fold odds $\hat{w}_{\text{dens}}(q') = \text{clip}(\hat{p}(q')/(1-\hat{p}(q')), 0.1, 10.0)$, accompanied by domain classifier $\text{ROC-AUC}$ and Kish Effective Sample Size $\text{ESS}(w) = (\sum w_j)^2 / \sum w_j^2$.
5. `real_only`: direct sample mean $\hat{\mu}_s^{\text{real}}$ on the $B$ authentic calibration queries.
6. `mix_real_density_0.5`: fixed convex combination $0.5 \hat{\mu}_s^{\text{real}} + 0.5 \hat{\mu}_s^{\text{dens}}$.
7. `mse_shrunk`: empirical MSE-minimizing shrinkage $\hat{\mu}_s^{\text{shrunk}} = (1 - \lambda_s^*)\hat{\mu}_s^{\text{real}} + \lambda_s^* \hat{\mu}_s^{\text{syn}}$ with $\lambda_s^* = \hat{\sigma}_{s,\text{real}}^2 / (\hat{\sigma}_{s,\text{real}}^2 + \hat{\sigma}_{s,\text{syn}}^2 + (\hat{\mu}_s^{\text{syn}} - \hat{\mu}_s^{\text{real}})^2)$.

### 3.2 Support-Audited Candidate-Set Abstention

To prevent forced single-system selection under high small-sample uncertainty, we construct a $95\%$ paired-bootstrap candidate set over $\mathcal{Q}_{\text{cal}}^{(B)}$. Let $\hat{s}_{\text{cal}} = \arg\max_{s \in \mathcal{S}} \hat{\mu}_s^{\text{real}}$. Across $R=200$ paired bootstrap resamples of $\mathcal{Q}_{\text{cal}}^{(B)}$, let $U_{0.95}(s)$ be the $95\text{th}$ percentile of the paired difference $\bar{m}^{(r)}(s) - \bar{m}^{(r)}(\hat{s}_{\text{cal}})$. The candidate set is:
$$\widehat{\mathcal{C}}_{0.95} = \{ s \in \mathcal{S} \mid U_{0.95}(s) \ge 0 \}$$
When $|\widehat{\mathcal{C}}_{0.95}| > 1$, the selector abstains from a single point decision and returns $\widehat{\mathcal{C}}_{0.95}$. We jointly report Confident Wrong Selection Rate, Abstention Rate, Optimal System Coverage $\Pr(s^* \in \widehat{\mathcal{C}}_{0.95})$, and Mean Set Size $\mathbb{E}[|\widehat{\mathcal{C}}_{0.95}|]$.

### 3.3 Formal Waiting-Period Rule World and Double-Witness Criterion

In Round 3, each scenario is defined by a structured business condition frame:
$$z = (\texttt{policy\_id}, \texttt{enrollment}, \texttt{policy\_day}, \texttt{cause}, \texttt{plan\_wait\_days}, \texttt{approved\_waiver}) \in \mathcal{Z}$$
A deterministic rule compiler $f: \mathcal{Z} \to \{\texttt{WAIT\_EXCLUDED}, \texttt{WAIT\_NOT\_EXCLUDED}\}$ evaluates whether the waiting-period clause alone excludes coverage under explicitly bounded assumptions (all other policy terms satisfied; illness onset, diagnosis, hospitalization/purchase, and expense occur on the same specified policy day $t \ge 1$).

When deleting field $k$ yields partial frame $z_{-k}$, we label $z_{-k}$ as `CLARIFY` **if and only if** there exist two valid domain values $v_1, v_2 \in \Omega_k(z)$ such that:
$$f(z_{-k}, k=v_1) \neq f(z_{-k}, k=v_2)$$
Both completions $((z_{-k}, v_1, f(z_{-k}, v_1)), (z_{-k}, v_2, f(z_{-k}, v_2)))$ are stored in `controlled_cases.jsonl` as verifiable **counterexample witnesses**.

---

## 4 Experimental Setup and Data Provenance

### 4.1 Round 1 & Round 2: Historical QA Partitioning and Local Model Setup

- **Corpus Cleaning & Union-Find Partitioning**: From 8,362 raw records in `baoxianzhidao_filter.csv` (SHA256 `f98991b5...`), we retain `is_best == '1'` entries, enforce length bounds ($4 \le |q| \le 500$, $15 \le |d| \le 3000$), and filter PII patterns, yielding 3,744 question groups and 3,746 unique answers. Using Union-Find, we merge questions sharing identical normalized titles, shared answers, or character 2–4-gram TF-IDF title cosine similarity $\ge 0.90$ into 3,725 connected components, splitting components (`seed=20261025`) into **750 calibration queries** and **2,994 held-out test queries** (zero shared answers across splits; maximum cross-split title cosine $0.8965$).
- **Local LLM Generation & Numerical Stability Audit**: In Round 2, 3 calibration questions (`indices=[79, 513, 623]`) serve as style demonstrations, and 256 source answers disjoint from those demonstrations are frozen in `generation_inputs.json`. Using local `Qwen/Qwen2.5-1.5B-Instruct` (revision `989aa798...`), we run greedy decoding (`do_sample=False`, `max_new_tokens=64`) for `zero_shot` and `few_shot_3`. An initial `float16 + sdpa` run on Apple MPS exhibited numerical underflow (archived in `failed_float16_run.json`); switching to `float32 + eager` resolved the issue, producing 255/256 valid `zero_shot` and 250/256 valid `few_shot_3` queries (**249 common valid sources**, evaluated without any retriever-based filtering).
- **Dense and Hybrid Systems**: We encode the 3,746 answers and all query sets with `BAAI/bge-small-zh-v1.5` (revision `7999e1d3...`, CLS pooling, L2-normalized, `float64` cosine scoring, max length 512 tokens, truncating 509 long answers) and fuse `bm25_char2` and `bge_small_zh` via fixed `RRF(k=60, depth=100)` (`rrf_bm25_bge`).

### 4.2 Round 3: Traceable Insurance Clauses and Paired Rewriting

We collect five official policy clause PDFs from three insurance groups, verifying SHA256 checksums, page numbers, NFKC compact character spans, and PDF bounding boxes (`data_v3/sources.json`):

| Policy ID | Official Clause Title (Chinese / English Gloss) | Insurer | Page | Studied Benefit Scope | Waiting Days | Accident Exempt | Approved Waiver |
|---|---|---|---:|---|---|---|---|
| `pa_p0781` | 平安关爱相伴团体医疗保险（B款） (*Ping An Group Medical Plan B*) | Ping An | p. 4 | Specified / mild critical illness medical | `[90]` | Yes | No |
| `pa_h865` | 平安员工及家属住院医疗保险 (*Ping An Employee & Family Inpatient*) | Ping An | p. 3 | General inpatient medical | `[30]` | Yes | Yes |
| `cpic_86017` | 太保互联网C款门急诊医疗保险 (*CPIC Online Outpatient Plan C*) | CPIC | p. 4 | Sec. 2.5.2 Online pharmacy medication | `[7, 15, 25]` | No | Yes |
| `cpic_82859` | 太保互联网个人一年期住院E款医疗保险 (*CPIC Online 1-Yr Inpatient Plan E*) | CPIC | p. 3 | General inpatient medical | `[30]` | Yes | Yes |
| `tk_group_2018` | 泰康悦享环球（尊享版）高端医疗保险 (*Taikang Global High-End Medical*) | Taikang | p. 4 | Non-psychiatric general inpatient | `[30]` | Yes | No |

From 20 complete seed frames (4 per policy), we deterministically generate 80 controlled cases (`complete`, `surface`, `missing_critical`, `counterfactual`) and run 40 paired local `Qwen2.5-1.5B-Instruct` (`float32/eager`, `batch_size=1`) rewrites comparing `ordinary_style` against `condition_prompt`.

---

## 5 Empirical Results and Mechanism Analysis

### 5.1 Round 1 & Round 2: Retrieval Selection Reversal and Calibration Limits

Table 1 reports $\text{MRR@10}$ and $\text{Hit@5}$ across all eight systems on the 2,994 held-out authentic test queries alongside Round 1 and Round 2 synthetic benchmarks:

| System | Held-Out Real Test MRR@10 | Held-Out Real Test Hit@5 | Paired $\Delta$ vs. `bm25_char2` (95% Bootstrap CI) | R1 Prefix Copy MRR@10 | R2 Matched Keyword MRR@10 | R2 Qwen Zero-Shot MRR@10 | R2 Qwen 3-Shot MRR@10 |
|---|---:|---:|---|---:|---:|---:|---:|
| `tfidf_char1` | 0.2296 | 0.3313 | $-0.1521 \; [-0.1660, -0.1400]$ | 0.6692 | 0.7564 | 0.2582 | 0.2120 |
| `tfidf_char2` | 0.3180 | 0.4218 | $-0.0637 \; [-0.0729, -0.0544]$ | 0.8795 | 0.8800 | 0.4176 | 0.2872 |
| `tfidf_char1_3` | 0.3187 | 0.4285 | $-0.0630 \; [-0.0715, -0.0540]$ | 0.9010 | 0.8642 | 0.4136 | 0.2960 |
| `tfidf_char3_5` | 0.2727 | 0.3557 | $-0.1090 \; [-0.1203, -0.0986]$ | **0.9330** | 0.4132 | 0.4360 | 0.2776 |
| `bm25_char1` | 0.2529 | 0.3427 | $-0.1288 \; [-0.1410, -0.1161]$ | 0.8483 | 0.6910 | 0.2702 | 0.1944 |
| `bm25_char2` | 0.3817 | 0.4826 | $\phantom{+}0.0000 \; [\phantom{+}0.0000, \phantom{+}0.0000]$ | 0.9311 | **0.9359** | 0.5013 | 0.3225 |
| `bge_small_zh` | **0.5216** | **0.6453** | $+0.1399 \; [+0.1250, +0.1551]$ | — | 0.5165 | 0.5202 | 0.4024 |
| `rrf_bm25_bge` | 0.5030 | 0.6075 | $+0.1212 \; [+0.1115, +0.1308]$ | — | 0.7639 | **0.5647** | **0.4124** |

**Key Findings in Rounds 1 & 2:**
1. **Retriever Selection Reversal**: On held-out authentic user queries, dense retrieval `bge_small_zh` achieves the highest performance ($0.5216$), outperforming `rrf_bm25_bge` ($0.5030$) by $+0.0187$ with a 1,000-sample paired bootstrap $95\%$ interval of $[+0.0080, +0.0300]$. Because authentic user questions diverge lexically from standardized answers, `bm25_char2` trails `bge_small_zh` by $0.1399$, so unweighted RRF fusion dilutes top-1 dense hits. Conversely, both `zero_shot` and `few_shot_3` LLM synthetic benchmarks select `rrf_bm25_bge` ($0.5647$ and $0.4124$), because LLM queries generated from source answers retain domain terms that boost `bm25_char2` ($0.5013$ on `zero_shot`), creating artificial lexical-dense co-voting synergy.
2. **Failure of Unpaired Density Reweighting vs. Effectiveness of Candidate-Set Abstention**: Across all budgets $B \in \{25, 50, 100, 200\}$, character density-ratio reweighting (`synthetic_density_weighted`) selects the true optimal system in $0\%$ of trials. At $B=25$, forced single-system selection (`real_only`, `mix_0.5`, `mse_shrunk`) makes a confident wrong choice in $60\%\text{--}80\%$ of trials. By contrast, our $95\%$ paired-bootstrap **Candidate-Set Abstention** rule reduces the Confident Wrong Selection Rate to **$0.0\%$** at $B=25$ ($80\%$ abstention rate, $100\%$ optimal coverage, mean set size $2.6$) and **$0.0\%$** at $B=100$ ($40\%$ abstention rate, $100\%$ optimal coverage, mean set size $1.4$).
3. **Why 3-Shot Style Guidance Degrades Source Support**: In our 40-query semantic audit (`seed=20261028`), `zero_shot` queries are fully supported by their source answer in $12/20$ ($60\%$) cases with $0/20$ unsupported mutations, whereas `few_shot_3` drops to $9/20$ ($45\%$) supported and introduces **$6/20$ ($30\%$) unsupported condition or intent mutations**—such as mutating *"equal traffic liability"* into *"other party's full liability"* (`doc_id=545`) or mutating *"medical expense advance conditions"* into *"eligibility to buy insurance after hospitalization"* (`doc_id=2213`).

---

### 5.2 Round 3: Business-Condition Conservation and Opposite-Label Collision

Table 2 compares `ordinary_style` and `condition_prompt` across the 20 complete seed scenarios (40 paired generations) on format validity, strict v1 regex slot checking (`condition_guard.py`), surface-normalized slot verification, rule-derived decision outcomes, and 5-clause `bm25_char2` retrieval:

| Rewriting Method | Format Valid | Strict v1 Slot Guard Pass | Surface-Normalized Slot Pass | Standalone Decision Preserved | Degraded to `CLARIFY` | Numeric Condition Mutated | Opposite-Label Collisions | Target Clause `bm25_char2` Top-1 (MRR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_style` | 20/20 (100%) | 6/20 (30%) | 6/20 (30%) | 6/20 (30%) | 11/20 (55%) | 3/20 (15%) | **1 pair (2 seeds)** | 50.0% (0.6792) |
| `condition_prompt` | 20/20 (100%) | 10/20 (50%) | **19/20 (95%)** | **19/20 (95%)** | 1/20 (5%) | **0/20 (0%)** | **0 pairs** | **100.0% (1.0000)** |

Field-level omission/mismatch counts (out of required occurrences: 20 for `policy_id`, `enrollment`, `policy_day`, `cause`; 4 for `plan_wait_days`; 3 for `approved_waiver`):
- `ordinary_style` (both v1 and surface-normalized): `policy_id` **14/20 (70%)**, `enrollment` **7/20 (35%)**, `policy_day` **5/20 (25%)**, `cause` **2/20 (10%)**, `plan_wait_days` **4/4 (100%)**, `approved_waiver` **3/3 (100%)**.
- `condition_prompt` (v1 strict $\to$ surface-normalized): `policy_id` $1 \to \mathbf{1}$, `enrollment` $0 \to \mathbf{0}$, `policy_day` $8 \to \mathbf{0}$, `cause` $0 \to \mathbf{0}$, `plan_wait_days` $1 \to \mathbf{0}$, `approved_waiver` $3 \to \mathbf{0}$.

**Representative Case Studies from Round 3 (`generated_questions.jsonl`):**

1. **Opposite-Label Collision (`condition-009` vs. `condition-010` on `cpic_86017`)**:
   - Under *CPIC Online Outpatient Plan C* (`cpic_86017`), online pharmacy medication carries a 7-, 15-, or 25-day waiting period depending on the purchased plan.
   - `condition-009` specifies `plan_wait_days=7, policy_day=10` $\implies$ **`WAIT_NOT_EXCLUDED`** (day 10 exceeds the 7-day wait).
   - `condition-010` specifies `plan_wait_days=25, policy_day=10` $\implies$ **`WAIT_EXCLUDED`** (day 10 falls within the 25-day wait).
   - Under `ordinary_style`, `Qwen2.5-1.5B-Instruct` drops both the product title and the plan waiting days on both seeds, outputting the **exact same Chinese string** for both opposite-outcome scenarios:
     > *"我是在网上看病花了钱，是第一次买这个保险，保单生效第10天了，因为不是意外病，想问一下这属于等待期吗？"*
     > (*"I spent money seeing a doctor online, it's my first time buying this insurance, the policy has been in effect for 10 days, and it's not an accidental illness—does this fall under the waiting period?"*)
   - By contrast, `condition_prompt` preserves *"计划中等待期是7天"* (*"waiting period in the plan is 7 days"*) in `condition-009` and *"这个责任的等待期是25天"* (*"waiting period for this coverage is 25 days"*) in `condition-010`, maintaining full answerability and label validity.
2. **Numeric Day Mutation and Waiver Deletion (`condition-013, 015, 016` on `cpic_82859`)**:
   - In three seeds where the policy elapsed day is `policy_day=20` (*"保单生效第20天"*), `ordinary_style` mutates the number into **`"保单生效后的第二天"` (*"the second day after policy effect"*, `policy_day=2`)**, and in `condition-016` simultaneously deletes the approved re-enrollment waiting-period waiver (`approved_waiver=True`), inverting the substantive rule outcome from `WAIT_NOT_EXCLUDED` to `WAIT_EXCLUDED`.
3. **Precision–Recall Trade-off of Deterministic Regex Slot Guards**:
   - While `condition_guard.py` (v1) has zero false positives on corrupted queries, it flags 9 valid `condition_prompt` outputs solely due to minor surface phrasing variations (e.g., *"保单生效是第20天"* with copula *"是"* instead of *"保单生效第20天"*, and *"明确同意免去/免除等待期"* instead of *"同意免等待期"*). This confirms that deterministic regex guards serve as conservative triage filters rather than semantic gold standards.

---

## 6 Limitations and Confirmation Gate

1. **Single-Source Historical QA and Model Scope**: Rounds 1–2 rely on a single public Baidu Zhidao mirror lacking original post URLs or timestamps, and evaluate one small local generator (`Qwen2.5-1.5B-Instruct`) and one dense retriever (`BGE-small-zh-v1.5`, with 512-token truncation on 509 long answers).
2. **Pending Independent Human Annotation**: In Round 3, all 80 controlled cases and 40 generated questions are packaged into method-blinded, un-prefilled review interfaces (`reviewer_A.html/csv` and `reviewer_B.html/csv`, 120 items each). Until two independent human annotators complete these files and compute inter-annotator agreement (Cohen's $\kappa$ via `score_annotations.py`), our rule compiler labels remain provisional reference labels rather than expert-verified gold truth.

---

## 7 Conclusion

Synthetic retrieval evaluation in vertical domains such as Chinese insurance QA cannot be validated by surface format checks or corrected by post-hoc character density reweighting alone. Introducing dense and hybrid retrieval reveals a systematic selection reversal (where LLM synthetic queries favor RRF hybrid fusion while held-out authentic queries favor BGE dense retrieval), and small-budget selection risk is more reliably managed via calibration bootstrap candidate-set abstention than covariate reweighting. Furthermore, unconstrained colloquial style guidance frequently strips product identities, mutates elapsed policy days, and even collapses opposite-label multi-plan scenarios into identical unanswerable questions. Anchoring synthesis to traceable contract clauses with explicit counterexample witnesses and business-condition conservation prompts substantially reduces condition drift ($95\%$ standalone decision preservation and $100\%$ target-clause top-1 retrieval), providing an auditable methodology for domain-specific RAG and Agent evaluation.

---

## References

1. H. A. Rahmani, N. Craswell, E. Yilmaz, B. Mitra, and D. Campos. Synthetic Test Collections for Retrieval Evaluation. In *Proceedings of SIGIR*, 2024. https://arxiv.org/abs/2405.07767
2. J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. In *Proceedings of NAACL*, 2024. https://aclanthology.org/2024.naacl-long.20/
3. K. Zhu, Y. Luo, D. Xu, et al. RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework. In *Proceedings of ACL*, 2025. https://aclanthology.org/2025.acl-long.418/
4. S. Filice, G. Horowitz, D. Carmel, et al. Generating Q&A Benchmarks for RAG Evaluation in Enterprise Settings. In *Proceedings of ACL (Industry Track)*, 2025. https://aclanthology.org/2025.acl-industry.33/
5. R. van Elburg, et al. Can we Evaluate RAGs with Synthetic Data? *arXiv preprint arXiv:2508.11758*, 2025. https://arxiv.org/abs/2508.11758
6. M. Kucia and P. Gawlik. Beyond Benchmark Scores: How Synthetic and Authentic Query Distributions Diverge in RAG Evaluation. *arXiv preprint arXiv:2609.14579*, 2026. https://arxiv.org/abs/2609.14579
7. A. N. Angelopoulos, S. Bates, C. Fannjiang, M. I. Jordan, and T. Zrnic. Prediction-Powered Inference. *Science*, 382(6671):669–674, 2023. https://arxiv.org/abs/2301.09633
8. A. Divekar. Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference. *arXiv preprint arXiv:2606.05308*, 2026. https://arxiv.org/abs/2606.05308
9. Y. Ding, et al. InsQABench: Benchmarking Chinese Insurance Domain Question Answering with Large Language Models. *arXiv preprint arXiv:2501.10943*, 2025. https://arxiv.org/abs/2501.10943
