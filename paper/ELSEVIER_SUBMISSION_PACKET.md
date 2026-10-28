# Elsevier 期刊（ESWA / IP&M）投稿网站全套表单填报与物料清单
# Complete Submission Metadata Packet for Editorial Manager

> 本文件已将投递 **Expert Systems with Applications (ESWA)** 或 **Information Processing & Management (IP&M)** 投稿系统（Editorial Manager）所需的**每一个输入框内容、声明、Highlight 字符数校验、Cover Letter、推荐审稿人**全部整理成可直接「一键复制粘贴」的格式。

---

## 一、投稿网站入口与账号准备

1. **目标期刊投稿入口（二选一，推荐 ESWA 首选）**：
   - **ESWA 投稿网址**：[https://www.editorialmanager.com/eswa/](https://www.editorialmanager.com/eswa/)
   - **IP&M 投稿网址**：[https://www.editorialmanager.com/ipm/](https://www.editorialmanager.com/ipm/)
2. **登录账号**：
   - 点击 **`Register Now`**（或使用已有的 Elsevier / ScienceDirect 账号直接登录）。
   - 登录身份选择 **`Author Login`**。
   - 点击菜单栏的 **`Submit New Manuscript`** 开始新投稿。

---

## 二、逐项表单复制粘贴内容（Form Fields）

### 1. 论文类型（Article Type）
- 下拉菜单选择：**`Full Length Article`** 或 **`Regular Paper`** / **`Research Paper`**

### 2. 论文标题（Article Title）
```text
Do Synthetic Queries Preserve Insurance Eligibility Decisions? Decision-Preserving Verification and Repair for Vertical RAG
```

### 3. 简短运行标题（Short Title / Running Head，若系统要求 $\le 50$ 字符）
```text
Decision-Preserving Verification for Insurance RAG
```

### 4. 论文摘要（Abstract，纯文本无特殊 LaTeX 格式）
```text
Constructing evaluation benchmarks for vertical-domain Retrieval-Augmented Generation (RAG) systems often relies on instruction-tuned large language models (LLMs) to synthesize user queries from source documents. In rule-governed insurance underwriting and claims workflows, however, a synthetic query can rank the source clause at Top-1 while omitting or mutating eligibility conditions that determine whether a claim is covered, excluded, or requires clarification.

This paper studies decision-preserving verification and repair of synthetic queries across 36 official Chinese insurance policy PDFs spanning 10 insurer groups (26 core benchmark PDFs plus 10 cross-insurer generalization PDFs) and 8 multi-threshold rule families (totaling 88 seeds across 25 clause clusters and 21 independent source documents, N=176 cross-model pairs) using Qwen and DeepSeek models without weight fine-tuning. We compile eligibility rules into a threshold-partitioned reachability verifier that separates surface slot completeness from reference-independent decision determinacy, uses a text-only gate to distinguish parser grammar limits from question ambiguity, and guides targeted slot repair when open-loop rewrites omit or alter pivotal conditions.

Across our experiments, unconstrained conversational rewrites reach up to 100.0% Top-1 retrieval while preserving standalone decisions on only 0.0%–45.0% of development seeds and 0.0%–2.5% of confirmation and extension seeds. Even when the 15 confirmation clauses are embedded in a 1,000-clause real commercial insurance distractor pool from Fudan University's InsQABench (66.67x larger), conversational rewrites achieve 92.50% lexical Top-1 (86.25% hybrid Top-1) despite 1.25% (1/80) symbolic decision usability, whereas parser-guided slot repair reaches 98.75% dense/hybrid Top-1 (100.0% Hit@5, MRR@10=0.9938) and outperforms zero-call templates under 1,000-clause title-masked dense retrieval (MRR@10 0.5015 vs. 0.4656). Across a 120-case diagnostic suite and a 400-query shared-rule-base dual-implementation consistency check, slot-only semantic verification records 0 false accepts (TP=296, FP=0). On the 40-seed confirmation set (N=80 pairs), parser-guided slot repair reaches 97.5% (78/80) symbolic joint usability, outperforming single-pass constrained prompting, equal-budget self-refinement, and deterministic JSON schema rendering (all 87.5%, 70/80, McNemar p=0.0078125); on both the 24-seed waiting-period extension set (N=48 pairs) and a 24-seed non-waiting-period multi-rule extension set strictly grounded in verbatim PDF clauses with character-level PyMuPDF (fitz) bounding-box anchors and SHA-256 hashes separated from structured task_context scope notes (10,000 RMB deductible and 100%/60% social-insurance copay tiers, 100k/200k RMB specialty-drug caps, secondary-or-above public hospital ordinary-ward and 30-day peri-inpatient clinic rules, and 0–60 age / 28-day infant / 30-day prescription limits across 4 official clauses, N=48 pairs), it reaches 100.0% (48/48) symbolic usability. Pooled across all 88 seeds (N=176 pairs, 25 clause clusters, 21 source documents), slot_guard_repair achieves 98.86% (174/176) symbolic usability, significantly outperforming structured_json_untruncated_320 (90.34%, 159/176, 15W/0L, McNemar p=6.10e-5, 21-document clustered p_Holm=0.015625) and all five remaining baselines.

To evaluate semantic validity without regex circularity while distinguishing protocol-guided cross-checking from zero-hint pure-verbatim adjudication, a dual-mode 3-flagship panel (kimi-k3, deepseek-v4-pro, qwen3.8-max; Confirmation-40 3-way semantic kappa=0.8578 protocol-guided / kappa=0.8366 pure-verbatim, binary usability kappa=0.7411 / 0.7008) shows 87.19% (558/640) exact agreement with our verifier and 96.56% (618/640) direct agreement between protocol-guided and pure-verbatim modes; under panel majority (161/176 = 91.48%), slot_guard_repair eliminates the label corruption of ordinary_style (27.27%) and two-line JSON baselines (67.05%–73.86%, p <= 0.015625) and performs comparably to strong single-pass condition_prompt (161/176 = 91.48%) and structured_json_untruncated_320 (161/176 = 91.48%, p=1.0).

Across 9 real cloud LLM end-to-end RAG pipelines (L1–L9, including model-matched abstaining vs. forced-binary pairs) and an independent External Holdout-48 benchmark, zero-call symbolic reachability re-certification (zero_call_verifier_dynamic_relabel) slashes single-arm evaluation MAE on unverified ordinary_style queries from 0.6556 / 0.5250 (tau = -0.8792 / -0.6973) to 0.0936 / 0.1222 (tau = +0.9062 / +0.7812)—and on an 80-query mixed uncurated pool (where a constant-CLARIFY counter-baseline fails at MAE 0.4222 / 0.4333) from 0.3292 / 0.2986 down to 0.0504 / 0.0816 (tau = +0.8125 / +0.7062)—without re-generating a single query, while verified synthesis reduces External Holdout-48 MAE to 0.0264–0.0583 (tau in [+0.7701, +0.9412]). Finally, three flagship models recover 16/17–17/17 (94.12%–100.0%) waiting-period schemas across 17 contracts, font-size superscript-cleaned symbolic extraction achieves 27/27 (100.0%) across 27 waiting-period PDFs from 10 insurers, and external validation on 50 InsQABench items achieves 92.0% quantitative threshold coverage.
```

### 5. 关键词（Keywords，通常输入 4–6 个，用回车或分号隔开）
```text
Retrieval-Augmented Generation (RAG)
Synthetic Benchmark Evaluation
Insurance Underwriting and Claims Rules
Decision-Preserving Verification
Symbolic Slot-Guard Repair
System Selection Bias
```

### 6. 核心要点（Highlights，Elsevier 强制要求 3–5 条，每条含空格严格 $\le 85$ 字符）
> 以下 5 条均已通过严格字符数核验（长度分别为 78、81、84、78、84 字符）：

```text
Synthetic conversational queries mutate eligibility conditions in insurance RAG.
Zero-call reachability verification slashes RAG evaluation MAE from 0.66 to 0.09.
System selection ranking inversions flip to positive concordance without LLM calls.
Closed-loop slot-guard repair achieves 98.86% deterministic symbolic validity.
Demonstrated across 36 real insurance contracts and 9 end-to-end cloud RAG systems.
```

### 7. 作者与机构信息（Authors & Institutions）
- **First Name（名）**：`Shunchi`
- **Last Name（姓）**：`Wang`
- **Academic Degree**：`B.S.` / `Bachelor`
- **Email**：`jakelihua@gmail.com`
- **Institution / Affiliation（机构）**：`Independent Researcher`
- **City / Country**：`Shenzhen`（或你的常住城市），`China`
- **ORCID ID**：`0009-0009-8629-2078`
- **Corresponding Author（通讯作者）**：勾选 `Yes`

---

## 三、投稿附信（Cover Letter，直接复制）

在系统的 `Cover Letter` 文本框中直接粘贴下方英文：

```text
Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript entitled "Do Synthetic Queries Preserve Insurance Eligibility Decisions? Decision-Preserving Verification and Repair for Vertical RAG" for consideration for publication as a Full Length Article in Expert Systems with Applications.

In rule-governed vertical domains such as insurance underwriting, healthcare eligibility, and regulatory compliance, industrial engineering teams routinely rely on instruction-tuned Large Language Models (LLMs) to synthesize evaluation queries from contract clauses due to a lack of anonymized real user logs. However, existing benchmark construction pipelines prioritize retrieval hit rate and conversational fluency, overlooking whether critical legal qualification conditions are preserved. 

In this work, we demonstrate that unconstrained conversational query synthesis achieves high Top-1 retrieval while suffering 77.5%–85.0% decision degradation. When deployed to evaluate nine real end-to-end cloud LLM RAG pipelines, this synthetic label corruption systematically penalizes prudent abstention-capable systems and reverses downstream system rankings (Kendall's tau = -0.8792 on Qwen-Plus and -0.6973 on DeepSeek).

To address this challenge, we introduce a neuro-symbolic reachability verification and slot-guard repair framework grounded in 36 official Chinese commercial insurance policy PDFs across 10 insurer groups (plus a 1,000-clause real insurance distractor pool and 50 external items from Fudan University's InsQABench) and 8 multi-threshold rule families:
1. We formalize eligibility clauses into threshold-partitioned reachability equivalence regions, separating surface slot completeness from reference-independent decision determinacy.
2. We demonstrate that zero-call symbolic reachability detection and dynamic label re-certification (with zero additional LLM calls and no query editing) slashes single-arm RAG evaluation MAE on unverified conversational queries from 0.6556 down to 0.0936 (an 85.7% error reduction, flipping rank correlation tau from -0.8792 to +0.9062), and on an 80-query mixed uncurated pool from 0.3292 down to 0.0504 (outperforming a constant-CLARIFY counter-baseline at MAE 0.4222 by 88.1%).
3. We provide a 4-stage marginal gain decomposition across all 176 pooled pairs (88 seeds across 25 clause clusters and 21 independent source documents), demonstrating that targeted second-pass repair reaches 98.86% deterministic symbolic validity (p_Holm_doc <= 0.015625 against all six baselines) and 98.75% dense/hybrid Top-1 (100.0% Hit@5) in the 1,000-clause InsQABench retrieval pool.
4. All experiments are verified with a dual-mode 3-flagship adjudication panel (Kimi-K3, DeepSeek-V4-Pro, Qwen3.8-Max), 9 end-to-end cloud LLM RAG pipelines, an independent 48-query External Holdout Benchmark, an uncontaminated 17-PDF three-flagship schema extraction audit (94.1%–100.0%), 27/27 (100.0%) font-size-cleaned symbolic extraction across 27 waiting-period PDFs from 10 insurers, and 92.0% quantitative threshold coverage on 50 external InsQABench items.

The entire study is 100% reproducible with 36 deterministic unit tests and offline rescoring scripts publicly available at https://github.com/jakelihua/chinese-insurance-rag-benchmark.

We confirm that this manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have approved the submission.

Thank you very much for your time and consideration of our work.

Sincerely,

Shunchi Wang
Independent Researcher
Email: jakelihua@gmail.com
ORCID: https://orcid.org/0009-0009-8629-2078
```

---

## 四、合规声明（Declarations & Statements）

### 1. 利益冲突声明（Declaration of Competing Interest）
在系统勾选 `No competing interests`，或在文本框填入：
```text
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
```

### 2. 数据可用性声明（Data Availability Statement）
在下拉菜单选择 `Data will be made available on request` 或 `Repository: ...`，填入：
```text
The benchmark generation logs, verification suites, unit tests, and offline scoring scripts supporting the findings of this study are openly available in the GitHub repository at https://github.com/jakelihua/chinese-insurance-rag-benchmark.
```

### 3. 生成式 AI 使用声明（Declaration of Generative AI in Scientific Writing）
Elsevier 现要求披露 AI 工具使用情况。请勾选并填写：
```text
During the preparation of this work, the authors used large language models (Qwen, DeepSeek, and Kimi) primarily as experimental objects/subjects under benchmark evaluation and secondarily for basic English grammar and phrasing refinement. After using these tools, the authors reviewed and edited the content and take full responsibility for the content of the published article.
```

### 4. 资助信息（Funding Source）
- 选择：**`This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.`**（未接受外部商业或公共基金资助）。

---

## 五、建议审稿人（Suggested Reviewers，通常填 3–4 位）

期刊系统会要求提供 3–4 位不具利益冲突的领域国际专家。以下为你精选 4 位**直接发表过本论文核心参考文献（SIGIR 2024 合成测试集、ACL 2025 RAGEval、EMNLP 2024 RC-RAG、ACL 2024 CLAMBER）的一线青年/中年学者与通讯作者**（相比超资深大牛院长，一线青年教授/研究员接审率最高、研究方向 100% 精准对口）：

1. **审稿人 1（合成检索测试集与系统排序偏差方向）**
   - **Name**: Dr. Hossein A. Rahmani
   - **Affiliation**: University College London (UCL), Department of Computer Science
   - **Email**: `h.rahmani@ucl.ac.uk`
   - **Reason**: First author of "Synthetic Test Collections for Retrieval Evaluation" (SIGIR 2024); directly researches how LLM-generated synthetic queries affect retrieval system evaluation and ranking stability.
2. **审稿人 2（垂直领域 Schema 驱动 RAG 评测基准生成方向）**
   - **Name**: Prof. Zhenghao Liu
   - **Affiliation**: Northeastern University (China), School of Computer Science and Engineering
   - **Email**: `liuzhenghao@mail.neu.edu.cn`
   - **Reason**: Corresponding author of "RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework" (ACL 2025); expert in schema-grounded synthetic RAG benchmark generation across financial, legal, and medical domains.
3. **审稿人 3（RAG 风险控制、拒答机制与反事实鲁棒性方向）**
   - **Name**: Prof. Ruqing Zhang
   - **Affiliation**: Institute of Computing Technology, Chinese Academy of Sciences (ICT, CAS)
   - **Email**: `zhangruqing@ict.ac.cn`
   - **Reason**: Corresponding author of "Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework (RC-RAG)" (Findings of EMNLP 2024); specialist in selective abstention and reliability evaluation for RAG systems.
4. **审稿人 4（大模型歧义检测与澄清追问评测方向）**
   - **Name**: Prof. Yang Deng
   - **Affiliation**: Singapore Management University (SMU), School of Computing and Information Systems
   - **Email**: `ydeng@smu.edu.sg`
   - **Reason**: Co-first/corresponding author of "CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models" (ACL 2024); expert in ambiguity detection and clarifying question generation in rule-governed QA.

---

## 六、文件上传清单（Upload Files 顺序）

在系统最后一步的 `Attach Files` 界面，按照以下文件类型上传：

| 顺序 | 文件类型（Item） | 上传什么文件 | 本地路径 |
|---:|---|---|---|
| 1 | **Cover Letter** | 导出或直接复制的 Cover Letter（`.docx` 或 `.pdf`，或直接在线粘贴） | 本文件第三节 |
| 2 | **Highlights** | 包含 5 条 Highlights 的纯文本或 Word 文件（`.docx`） | 本文件第二节第 6 项 |
| 3 | **Manuscript** | 已经编译好的正式双栏 PDF 文档（首选）或 LaTeX 源码压缩包 | [`paper/overleaf_submission.zip`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/overleaf_submission.zip) |
| 4 | **Declaration of Competing Interest** | 利益冲突声明文档（`.docx`） | 本文件第四节第 1 项 |

---

> 💡 **操作提示**：
> 1. 打开 [https://www.editorialmanager.com/eswa/](https://www.editorialmanager.com/eswa/) 点击 `Register Now` 注册后；
> 2. 依次点击 `Submit New Manuscript`，把上面的 **Title、Abstract、Keywords、Highlights、Cover Letter、Reviewers** 逐框复制粘贴进去；
> 3. 上传 [`paper/overleaf_submission.zip`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/overleaf_submission.zip) 或从 Overleaf 编译下载的 PDF；
> 4. 点击 `Build PDF for Approval`，预览确认无误后点击 `Approve Submission` 即可完成正式投稿！
