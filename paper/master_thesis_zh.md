# 面向中文保险RAG的合成问句决策保持验证与修复研究
## Decision-Preserving Verification and Repair of Synthetic Queries for Chinese Insurance RAG

**作者：** Shunchi Wang（Independent Researcher，`jakelihua@gmail.com`）  
**代码与可复现数据仓库：** [https://github.com/jakelihua/chinese-insurance-rag-benchmark](https://github.com/jakelihua/chinese-insurance-rag-benchmark)  
**基线版本与验证器标识：** `v3.0-frozen` / `v3.1-hardened` / `v3.1-untouched-holdout`（35 项确定性单元测试全部通过，支持无本地大模型/离线 CPU 一键复算）

---

## 中文摘要

在金融保险、临床医疗与劳动合规等规则密集型垂直领域构建检索增强生成（Retrieval-Augmented Generation, RAG）系统时，工程团队常因缺乏合规脱敏的大规模真实用户日志，转而使用大语言模型从源文档段落合成测试问句，并依据检索命中率或表面流畅度筛选样本。然而，在商业健康险核保与理赔等受合同规则约束的场景中，一条合成问句即便能将源条款排在检索首位，仍可能在口语化改写中遗漏或篡改决定拒赔与否的关键资格条件，导致参考标签失真并影响下游系统选型。

本文围绕“面向中文保险 RAG 的合成问句决策保持验证与修复”展开研究。在不微调任何模型权重的前提下，基于 26 份公开披露的中文商业保险正式合同条款 PDF（累计 328 条受控样本）与多阈值合成规则，跨越 Qwen 与 DeepSeek 模型家族完成以下工作：

1. **条件漂移机制与检索指标盲区（RQ1）**：我们将表面槽位完整度与无参考依赖的结论可确定性分离，通过遍历缺失槽位在各阈值分区上的可达裁决集合进行验证。实验表明，无约束口语改写（`ordinary_style`）虽能保持较高检索命中率（多阈值集 Top-1 达 100.0%），但独立结论保持率仅为 0.0%–45.0%（确认集与扩展集上为 0.0%–2.5%）；隐去产品与保险公司名称后，9 条款池上的密集检索 Top-1 准确率从 75.0% 降至 31.2%。
2. **解析不确定性分离与双实现一致性核查（RQ2）**：引入纯文本检测函数 `detect_parse_uncertainty`，将输出划分为结论保持（`PRESERVED_DECISION`）、条件篡改（`CONDITION_MUTATED`）、问句歧义需澄清（`CLARIFY_REQUIRED`）与解析器语法/分支超界（`PARSE_UNCERTAIN`）四类状态。在 120 例构造性诊断集上，验证器在支持语法范围内的误放行率与误拒绝率均为 0.0%，跨单位或日历日期表达（15/15）被标记为 `PARSE_UNCERTAIN`。在 400 条确认集生成记录的共享规则库双实现一致性核查（`shared_rule_base_dual_implementation_consistency_check`）中，纯槽位语义核查无误放行（TP = 296，FP = 0）；联合可用性核查中的 6 例假阳性均来自 160-Token `structured_json_rewrite` 的半句截断，而 `slot_guard_repair` 的误放行为 0/80。
3. **跨模型确认集、结构化基线消融与硬化后扩展集评测（RQ3）**：在 40 种子确认集（$N=80$ 对）上，符号槽位守卫修复（`slot_guard_repair`）的联合可用率为 97.5%（78/80），优于单通条件提示 `condition_prompt`、等预算自检 `generic_self_refine_budget_matched` 及确定性首行 JSON 渲染 `structured_json_schema_render`（均为 87.5%，70/80，McNemar $p=0.0078125$）。相较 320-token 结构化基线 `structured_json_untruncated_320`（91.25%，73/80），结果呈正向改善，但差异未达到统计显著性（McNemar $p=0.0625$）。在规则冻结后构建的 24 种子内部硬化后扩展集（$N=48$ 对）上，`slot_guard_repair` 达到 100.0%（48/48）联合可用率。
4. **隔离参考答案的下游流水线选型分析（RQ4）**：5 个候选检索+规则决策流水线均通过不接收 `seed` 字典的纯函数 `predict(question, retrieved_clause, inference_config)` 与预计算 BGE 向量缓存执行推理。当未验证口语改写直接沿用种子标签时，97.5%–100.0% 的样本发生标签污染并逆转系统排序（Kendall's $\tau \in [-0.8367, -0.7746]$）；经验证过滤或槽位修复后，标签污染率降至 0.0%，系统排序恢复至 $\tau = 1.0000$。本方法仍依赖手工编译的等待期与多阈值规则模式，尚未实现开放域复杂条款的自动规则抽取或外部多专家盲评。

**关键词：** 检索增强生成（RAG）；合成评测基准；保险条款核保理赔；决策保持验证；符号槽位守卫修复；系统选型偏差

---

## Abstract

Constructing evaluation benchmarks for vertical-domain Retrieval-Augmented Generation (RAG) systems often relies on instruction-tuned large language models (LLMs) to synthesize user queries from source documents. In rule-governed insurance underwriting and claims workflows, however, a synthetic query can rank the source clause at Top-1 while omitting or mutating eligibility conditions that determine whether a claim is covered, excluded, or requires clarification.

This thesis studies decision-preserving verification and repair of synthetic queries across 26 official Chinese insurance policy PDFs (328 controlled cases) and a multi-threshold rule testbed using Qwen and DeepSeek models without weight fine-tuning:

1. **Condition Drift vs. Retrieval Scores (RQ1)**: By separating surface slot completeness from reference-independent decision determinacy over threshold-partitioned equivalence regions, we show that unconstrained conversational rewriting (`ordinary_style`) reaches up to 100.0% Top-1 retrieval while preserving standalone decisions on only 0.0%–45.0% of development seeds and 0.0%–2.5% of confirmation and extension seeds. Masking product and insurer names drops 9-clause dense retrieval Top-1 accuracy from 75.0% to 31.2%.
2. **Separating `PARSE_UNCERTAIN` from `CLARIFY_REQUIRED` and Dual-Implementation Consistency Check (RQ2)**: Using a text-only gate (`detect_parse_uncertainty`) to distinguish parser grammar limits (`PARSE_UNCERTAIN`) from question ambiguity (`CLARIFY_REQUIRED`), the verifier records 0.0% false-accept and 0.0% false-reject rates on supported day-unit syntax in a 120-case diagnostic suite (routing all 15 month/week/date cases to `PARSE_UNCERTAIN`). In a 400-query shared-rule-base dual-implementation consistency check (`shared_rule_base_dual_implementation_consistency_check`), slot-only semantic consistency records 0 false accepts (TP = 296, FP = 0), while all 6 joint-usability false accepts come from 160-token truncations in `structured_json_rewrite` (0/80 on `slot_guard_repair`).
3. **Cross-Model Confirmation, Structured Baselines, and Post-Hardening Extension Set (RQ3)**: On the 40-seed confirmation set ($N=80$ pairs), `slot_guard_repair` reaches 97.5% (78/80) joint usability, outperforming `condition_prompt`, budget-matched `generic_self_refine`, and `structured_json_schema_render` (all 87.5%, 70/80, McNemar $p=0.0078125$). Compared with 320-token untruncated JSON rewriting (`structured_json_untruncated_320`, 91.25%, 73/80), it shows a positive trend ($p=0.0625$). On the 24-seed internal post-hardening extension set ($N=48$ pairs), `slot_guard_repair` reaches 100.0% (48/48) joint usability.
4. **Downstream Pipeline Selection Under Reference-Free Inference (RQ4)**: Evaluated via the reference-free interface `predict(question, retrieved_clause, inference_config)` and cached BGE dense embeddings across 80 reference queries, unverified `ordinary_style` rewrites suffer 97.5%–100.0% label corruption and invert system rankings ($\tau \in [-0.8367, -0.7746]$). Both `verify_only_filter` and `slot_guard_repair` reduce label corruption to 0.0% and recover the reference ranking ($\tau = 1.0000$), while our framework remains scoped to hand-compiled waiting-period schemas and programmatic consistency checks rather than automated legal extraction or external human annotation.

**Keywords:** Retrieval-Augmented Generation (RAG); Synthetic Benchmark Evaluation; Insurance Underwriting and Claims Rules; Decision-Preserving Verification; Symbolic Slot-Guard Repair; System Selection Bias

---

## 第 1 章 绪论

### 1.1 研究背景与问题提出

在金融保险、临床路径审核与劳动人事合规等严肃企业级场景中，检索增强生成（Retrieval-Augmented Generation, RAG）已成为连接大语言模型与专业规章条款的核心架构。然而，在实际算法工程落地时，研发团队普遍面临一个棘手的评测死局：手头往往只有几十份正式业务条款或百余条专家编写的种子样例，缺乏经过隐私脱敏与合规确权的大规模真实用户提问日志；若将仅有的少量种子全部用于提示工程或微调，便失去了客观评测系统迭代效果的独立测试集 [1, 3, 4]。因此，利用指令微调大语言模型从源文档段落反推合成测试提问（Synthetic Query Generation），已成为冷启动阶段构建 RAG 评测基准的标准范式。

现有学术界与工业界对合成测试集的研究，主要关注合成提问与真实提问在检索器排序分数（如 MRR@10、nDCG、Hit@k）上的分布偏移，或利用大模型评判器（LLM-as-a-Judge）对问题流畅度与表面可回答性打分 [1, 2, 5, 8, 9]。但在保险核保与理赔裁决这类**规则驱动（Rule-Governed）**场景中，这种以检索相关性和表面流畅度为核心的质量控制存在结构性盲区：

一份商业医疗保险合同对“等待期内出险是否免责”的裁决，严格取决于多个离散且相互耦合的资格条件槽位（Eligibility Slots）：
1. **产品与条款身份（`policy_id`）**：不同险种的等待期从 7 天、15 天、25 天、30 天、60 天、90 天到 180 天不等；
2. **保障责任类别（`benefit_id`）**：同一份合同内部不同责任可设置完全不同的等待期与豁免规则（例如《平安住院安心医疗保险》第四条第（一）款规定“一般住院医疗津贴”等待期为 30 日且意外伤害免等待期，而第（二）款规定“癌症住院医疗津贴”等待期为 90 日且无意外豁免）；
3. **投保状态与豁免批注（`enrollment`、`approved_waiver`）**：首次投保适用等待期，而连续续保或经保险人审核同意免等待期时可豁免；
4. **出险时间与原因（`policy_day`、`cause`）**：出险发生在保单生效第几天，以及源于疾病还是意外伤害。

当大模型按照常规提示将结构化种子场景改写为“自然、简短的口语化用户提问”时，极易发生两类致命的**决策漂移（Decision Drift）**：
- **第一类：关键条件遗漏导致的歧义退化（Ambiguity Degradation / `CLARIFY_REQUIRED`）**。模型为了去模板化而省略险种简称、责任类别、续保状态或出险天数，使原本结论唯一的测试题退化为“客观上同时存在赔付与拒赔两种可能、必须向用户追问澄清”的残缺问句。反之，若模型省略的是**非关键背景槽位**（例如在等待期已满的第 45 天省略出险原因是疾病还是意外），则并不影响“等待期不免责”的唯一裁决。单纯的字段表面匹配无法区分“关键遗漏”与“无害省略”，而检索指标对两者均完全失明。
- **第二类：条件与数值阈值篡改导致的结论翻转（Condition Mutation / `DECISION_CHANGED`）**。生成模型在改写时经常将用户实际经过天数（如第 7 天、第 45 天或第 120 天）错误覆盖为条款原文中的法定阈值（如 14 天或 90 天），或者反转责任极性（如将“本人主要责任”改写为“非主要责任”）。这类篡改反而进一步抬高了问句与源条款的词法重叠度，使检索得分更高，但参考答案已被颠倒。

更严重的是，如果将这些丢失关键条件或发生事实篡改的合成问句直接放入评测集，并朴素沿用原种子的二元裁决标签（赔付/拒赔）去评测下游 RAG 系统，那些遵循条款、在条件不足时正确发起追问（`CLARIFY`）的合规系统反而会被判为零分，而盲目猜测拒赔的激进系统却能拿到高分，从而导致**下游系统选型排序发生反转**。

### 1.2 核心研究问题（Research Questions）

围绕上述工程与科学痛点，本学位论文固定一条清晰的研究主线，重点回答以下四个层层递进的研究问题：

- **RQ1（条件漂移机制与检索指标盲区）**：在真实可追溯的中文商业保险合同条款与多阈值规则上，合成改写会产生哪些关键条件遗漏、数值阈值篡改与条款身份丢失？为什么常规词法检索（BM25）、密集向量检索（BGE）及混合检索（RRF）指标无法反映这些决策失效？
- **RQ2（形式化验证器可靠性、状态分离与双实现一致性核查）**：在显式界定条款与任务边界的前提下，如何构建不依赖参考种子元数据（`seed_frame`）的阈值分区可达性验证器，并将“问句本身缺少关键条件需澄清（`CLARIFY_REQUIRED`）”与“超出解析器语法辖域或命中未建模责任分支（`PARSE_UNCERTAIN`）”严格区分开来？在条款映射审计、120 例构造性黄金诊断集与 400 条确认集生成记录的共享规则库双实现一致性核查（`shared_rule_base_dual_implementation_consistency_check`）下，验证器的纯语义误放行率与联合格式截断误放行率分别是多少？
- **RQ3（跨模型家族 40 种子确认集、公平结构化基线与 24 种子硬化后未触碰扩展留出集评测）**：在 11 份零别名确认集保险条款 PDF（40 种子，160 例）与 6 份规则硬化后新增的未触碰扩展留出集保险条款 PDF（24 种子，48 对，10 个条款簇）上，跨越 `Qwen-Plus` 与 `DeepSeek` 两大家族模型，符号槽位守卫修复（`slot_guard_repair`）相比单通条件提示（`condition_prompt`）、160-Token 与 320-Token 结构化 JSON 改写、基于首行 JSON 的确定性 Schema 渲染后处理（`structured_json_schema_render`）、无符号反馈大模型自检（`generic_self_refine`）以及仅验证过滤（`verify_only_filter`），在双终点（`format_and_slot` 与 `slot_only`）可用率、场景覆盖率、Token 消耗与延迟上的收益及统计显著性（含按种子/条款聚类置换检验与 4 主基线 / 6 探索性基线 Holm-Bonferroni 校正）如何？
- **RQ4（严格隔离推理接口下的下游检索+符号规则决策流水线选型影响验证）**：在推理函数 `predict(question, retrieved_clause, inference_config)` 与参考真值完全隔离、且 80 条配对参考基准问句具备 100% 纯 BGE 向量缓存的前提下，未验证的合成问句是否会因“伪金标”污染而扭曲系统排名？经本文验证过滤或槽位修复后的合成基准能否恢复与参考真值基准一致的系统选型排序？

### 1.3 主要工作与研究贡献

本文的主要研究工作与贡献概括如下：
1. **提出解耦“表面槽位完整度”与“无参考依赖结论可确定性”的形式化验证及正反双见证认证框架**：在 **26 份官方中文保险条款 PDF**（锚定 SHA256、物理页码与字符级包围盒坐标，严格分离 `pdf_contract_clause` 与 `synthetic_seed_frame`）与 **4 段多阈值规则**上，构建了遍历阈值等价区间的可达性枚举器 `enumerate_admissible_decisions`，为每一条需澄清样本生成一对导向相反结论的具体补全见证（Dual Completion Witnesses），并引入纯文本驱动的 `detect_parse_uncertainty` 硬门控，将语法/分支超界（`PARSE_UNCERTAIN`）与问句歧义（`CLARIFY_REQUIRED`）解耦。
2. **构建条款映射审计、120 例构造性黄金诊断集与 400 条生成记录的共享规则库双实现一致性核查，量化验证器的精度、召回率与语法边界**：记录了 20 份商业保险合同手工规则编译的作者工程耗时估算（均值 **11.72 分钟/份**，总计 **234.5 分钟**）；在 120 例 8 类别黄金诊断集上取得 **0.0% 误放行率（FP = 0/60）** 与支持语法辖域内 **0.0% 误拒绝率（FN = 0/45）**；在 400 条跨家族生成记录的正向可达性双实现一致性核查中，纯语义可达性一致性取得 **TP = 296、FP = 0（语义误放行率 0.0%）、TN = 86、FN = 18**，联合格式与语义可用性取得 **TP = 290、FP = 6（FAR = 6.38%，Precision = 97.97%）、TN = 88、FN = 16（FRR = 5.23%，Recall = 94.77%）**，其中全部 6 例 FP 均源于 `structured_json_rewrite` 的 160-Token 截断残句（`Qwen-Plus` 5 例、`DeepSeek` 1 例），而 `slot_guard_repair` 与 `condition_prompt` 的误放行均为 **0/80（Precision = 100.0%）**。
3. **提出符号槽位守卫闭环修复机制（`slot_guard_repair`），并在 40 种子确认集、公平结构化基线与 24 种子硬化后未触碰扩展留出集上完成跨模型家族（`Qwen-Plus` & `DeepSeek`）实证**：证明了注入缺失/篡改槽位诊断的二轮修复在 `1.5B` 小模型及 `Qwen-Plus`、`DeepSeek` 云端大模型上均将解析器检出的阈值篡改率降至 **0.0%**，在 40 种子确认集与 24 种子硬化后未触碰扩展留出集上分别达到 **97.5%（78/80）** 与 **100.0%（48/48）** 的端到端可用率（确认集上相对 `condition_prompt` 与等预算 `generic_self_refine` 的 McNemar 均为 **$p=0.0078125$**，40 种子聚类置换分别为 **$p=0.0078125$** 与 **$p=0.015625$** [4 主基线 Holm 校正后均 $\le 0.03125$]；未触碰扩展留出集上相对 `condition_prompt` 为 48/48 vs. 39/48，McNemar **$p=0.00390625$**，24 种子聚类 **$p=0.0078125$**，10 条款聚类 **$p=0.03125$**），同时显著优于确定性首行 JSON 渲染 `structured_json_schema_render`（70/80，$p=0.0078125$；留出集 39/48，$p=0.00390625$），并相对放宽至 320 tokens 的 `structured_json_untruncated_320`（73/80 与 43/48，均为 5W/0L，$p=0.0625$）保持一致的点估计正向优势。
4. **在严格隔离推理接口（`predict(question, retrieved_clause, inference_config)`）与 100% 纯 BGE 缓存的检索+符号规则决策流水线上实证揭示“伪金标”导致的系统排名反转（Kendall's $\tau \in [-0.8367, -0.7746]$）并验证选型恢复（$\tau = 1.0000$）**：在 5 个由真实检索器（BGE/BM25）与确定性符号决策模块组成的候选流水线上，量化了未验证口语改写造成的选型偏差（真实选型遗憾 0.6250），证明本文验证与修复流水线在保持 97.5% 场景覆盖率的同时实现 0.0% 标签污染与零 Top-1 选型遗憾。

### 1.4 论文组织结构

全文共分为七章与三个附录：第 1 章阐述研究背景、核心研究问题与主要贡献；第 2 章综述相关工作并说明数据溯源、合规边界与零微调实验设定；第 3 章形式化定义条款条件帧、阈值分区可达性验证器、`PARSE_UNCERTAIN` 状态分离及槽位守卫修复算法；第 4 章报告 5 份开发集保险条款跨尺度评测、4 份后验扩展集责任消歧与标题掩码检索，以及 16 组多阈值规则上的等调用预算对照与 264 条诊断日志；第 5 章报告共享规则库双实现一致性核查、120 例黄金诊断集、40 种子确认集、公平结构化基线与 24 种子硬化后未触碰扩展留出集在 `Qwen-Plus` 与 `DeepSeek` 上的跨模型泛化与工程成本权衡；第 6 章报告下游 5 个候选 RAG+决策系统的评测与选型反转恢复实验；第 7 章总结全文、讨论方法局限性并展望未来工作；附录 A 报告社区问答语料级检索器选型与校准诊断，附录 B 给出全部 26 份官方商业保险合同条款的溯源清单，附录 C 给出离线复现命令与 35 项单元测试说明。

---

## 第 2 章 相关工作与理论基础

### 2.1 检索增强生成与合成测试集偏差

合成测试集已被广泛用于缓解信息检索与 RAG 系统的冷启动训练与评测难题。Promptagator [16] 与 InPars-v2 [17] 利用少样本提示大模型生成合成查询以训练密集检索器；ARES [2] 与 RAGAs [18] 则构建了自动化 RAG 评测框架。围绕“合成测试集能否替代真实用户查询进行检索器选型”，Rahmani 等人 [1] 与 van Elburg 等人 [5] 系统分析了大模型合成测试集在检索系统排序上的相关性与生成偏差。Kucia 与 Gawlik [8] 在高校院系信息系统的 RAG 评测中对比了 1,851 条大模型合成问题与 322 条真实学生调研问题在密集检索、词法检索（BM25）及混合检索（RRF）上的表现，证明合成提问会显著高估绝对检索分并导致系统排名相对真实用户提问发生偏移。由于合成与真实查询之间的语料级检索排名偏移已在上述文献中得到证实，本文将公开社区问答语料 `baoxianzhidao_filter.csv` 上的检索实验（附录 A）明确定位为剥离“源文档子采样方差”与“同源词法复制效应”的前置补充诊断，而将正文核心聚焦于**条款约束下的资格条件保持与决策验证**。

### 2.2 场景驱动合成、问题质量评估与模糊澄清

为使合成提问贴合垂直领域结构，RAGEval [3] 从金融、医疗、法律文档中提取领域模式（Schema）以生成场景化评测集，DataMorgana [4] 则通过参数化用户画像与问题分类学定制问答基准。在通用问题生成质量评估方面，QGEval [9] 提出了涵盖流畅度、清晰度、简洁度、相关性、一致性、可回答性与答案一致性等 7 个维度的评测基准。然而，开环的模式提示（Open-Loop Schema Prompting）与通用表面可回答性打分并未将源条款中的数值阈值与豁免条件编译为形式化可达性检验——正如本文第 4 章与第 5 章实验所示，即便在提示词中逐一列出全部槽位，生成模型仍会在 5.0%–31.2% 的样本中用条款法定阈值覆盖用户实际天数，或漏掉关键身份槽位。

在规则遵循与模糊提问澄清方面，ShARC [10]、CLAMBER [11] 以及 AskBench [12] 重点研究**下游对话或问答助手**在面对信息缺失的用户提问时如何识别歧义并主动发起追问；RC-RAG [14] 从选择性预测与风险控制角度出发，利用反事实提示估计 RAG 置信度并在误导性上下文下拒绝回答。在保险垂直领域，InsuranceQA [7] 提供了早期社区问答对，InsQABench [15] 则构建了面向真实保险产品规则的大模型问答评测基准。本文的研究切入点位于下游问答助手之前：我们聚焦于**上游评测基准合成阶段**，检验从已知规则条款生成的合成测试提问是否忠实保持了种子场景的完整资格条件与唯一可达结论，并通过显式枚举缺失槽位的正反补全见证（Counterexample Witnesses）将“非关键槽位省略但结论唯一保持”与“关键槽位丢失导致必须追问”严格区分开来。

### 2.3 小样本统计校准与预测驱动推断

预测驱动推断（Prediction-Powered Inference, PPI）[6]、ARES [2] 以及 PRECISE [13] 结合合成数据、大模型自动评判器与少量人工标注校准集，为 RAG 评测与排序构建统计有效的置信区间。当合成查询仅从语料库的一小部分源文档生成、且缺乏全库配对真实提问时，未配对收缩估计与边缘查询特征重加权难以消除文档子采样偏差与词法重叠膨胀；我们在附录 A 中如实报告了这些重加权负结果，并验证了小样本校准下的启发式配对 Bootstrap 候选集合暂缓选型规则。

### 2.4 数据溯源、合规边界与零微调实验设定

本研究涉及的全部数据集、条款文件与模型设定均遵循严格的可审计与合规边界（汇总见表 2.1）：

| 数据集 / 模型资产 | 来源与规模 | 在本研究中的角色 | 溯源核验与许可边界说明 |
|---|---|---|---|
| **5 份开发集官方保险条款 PDF** | 平安、太保、泰康公开披露条款（`pa_p0781`、`pa_h865`、`cpic_86017`、`cpic_82859`、`tk_group_2018`） | 第 4.1 节等待期规则验证与跨尺度评测（20 种子，80 受控样本，120 条改写） | 公开可下载官方原件（记录 SHA256 + PDF 页码与包围盒坐标）；未确权开放重分发许可，仓库仅分发坐标锚点与短摘录。 |
| **4 份后验扩展集保险条款 PDF** | 平安、太保公开披露条款（`pa_2011_36`、`pa_p0254`、`pa_p0321`、`cpic_93896`） | 第 4.2 节补充口语简称与责任级消歧（`benefit_id`）后的回顾性评估（16 种子，64 样本，112 条生成） | 用于检验同一合同内不同责任等待期（30 天 vs. 90 天）与标题掩码检索；明确界定为后验精化评估而非零样本盲测。 |
| **11 份严格冻结验证集保险条款 PDF** | 平安、太保公开披露条款（`pa_h716`、`pa_2651`、`pa_l0001`、`pa_1714`、`pa_1072`、`pa_596`、`cpic_107082`、`cpic_106702`、`cpic_60201`、`cpic_79920`、`cpic_85050`） | 第 5–6 章 `v3.0-frozen` 严格冻结盲测（`aliases=[]`，24 保险种子 + 16 多阈值种子 = 40 种子 / 160 样本） | 在冻结 `v3.0-frozen` 解析器后新引入，零新增别名、零解析器修改，用于非循环三层审计与跨家族模型验证。 |
| **4 段多阈值合成规则** | 作者参考临床路径与劳动法规编写的受控合成规则（`med_egfr_tki`、`med_vte_prophylaxis`、`law_work_injury`、`law_noncompete`） | 第 4.3 节、第 5–6 章多阈值组合规则压力测试（16 种子，64 受控样本） | 受控合成规则测试床，包含竞争性时间窗口（10/14/30/35/90/365 天）与布尔豁免条件，不代表真实临床或司法判例库。 |
| **120 例构造性黄金诊断集** | 独立构造的 8 类别黄金正负例测试集（`data_v3/gold_diagnostic_suite.jsonl`） | 第 5.2 节非循环评估验证器自身的 TP/FP/TN/FN 与 `PARSE_UNCERTAIN` 拦截率 | 含 60 例黄金正例与 60 例黄金负例，覆盖口语同义、非关键省略、对比纠错、关键遗漏、阈值篡改、否定耗时、多值冲突与跨单位日期。 |
| **`baoxianzhidao_filter.csv`** | 开源社区镜像（8,362 原始行 $\to$ 3,746 唯一回答，3,744 有效问答对） | 附录 A 连通分量防泄漏划分（750 校准 + 2,994 测试）与检索器校准诊断 | 历史百度知道社区问答（非精算真值），存在潜在大模型预训练暴露与未确权许可限制，不重分发原始 CSV。 |
| **生成与检索模型** | 本地 `Qwen2.5-1.5B-Instruct`、`BAAI/bge-small-zh-v1.5`；云端 API `Qwen-Plus`（`response_model="qwen-plus"`）、`Qwen-Max`、`DeepSeek`（`deepseek-chat` 接口，返回 `response_model="deepseek-flash"`） | 推理阶段合成改写、二轮槽位修复、密集/MaxP 向量检索与下游评测 | 全程零权重微调（Inference-Only），云端模型均设 `temperature=0`；历史确认集/留出集 JSONL 的 `raw_response`、Token 与延迟于生成时记录，逐行 `api_meta` 字典由作者运行级记录按家族回填（`metadata_provenance="family_level_author_run_record_backfilled_per_row_without_request_id"`，不含逐请求 HTTP `request_id`）。 |

---

## 第 3 章 条款约束下的决策保持形式化验证与槽位守卫修复方法

### 3.1 合同资格条件帧与原子级条款判定基准

设受控规则集合为 $\mathcal{P} = \mathcal{P}_{\text{ins}} \cup \mathcal{P}_{\text{multi}}$，其中 $\mathcal{P}_{\text{ins}}$ 包含 20 份中文商业保险正式合同条款，$\mathcal{P}_{\text{multi}}$ 包含 4 段多阈值组合规则。每份保险条款的等待期免责与豁免规则均通过 SHA256 校验和、PDF 物理页码与字符级包围盒坐标（`bbox_pdf_points`）锚定至官方原件。

在保险等待期判定域 $\mathcal{P}_{\text{ins}}$ 中，每个核保理赔咨询场景形式化表示为 7 元结构化条件帧 $z \in \mathcal{Z}_{\text{ins}}$：
$$z = (\texttt{policy\_id}, \; \texttt{enrollment}, \; \texttt{policy\_day}, \; \texttt{cause}, \; \texttt{plan\_wait\_days}, \; \texttt{approved\_waiver}, \; \texttt{benefit\_id})$$
各槽位定义如下：
- $\texttt{policy\_id} \in \mathcal{P}_{\text{ins}}$：目标保单条款唯一标识；
- $\texttt{enrollment} \in \{\texttt{first}, \texttt{renewal}\}$：投保类型（首次投保 vs. 连续续保/重新投保）；
- $\texttt{policy\_day} \in \mathbb{N}^+$：出险或住院确诊发生在合同生效后的第几天；
- $\texttt{cause} \in \{\texttt{disease}, \texttt{accident}\}$：出险原因（疾病 vs. 意外伤害）；
- $\texttt{plan\_wait\_days} \in \{7, 15, 25, 30, 60, 90, 180\}$：投保计划或责任对应的等待期阈值天数；
- $\texttt{approved\_waiver} \in \{\text{True}, \text{False}\}$：续保或团体增员时是否获得保险人免除等待期的特别约定或批注；
- $\texttt{benefit\_id} \in \{\texttt{general\_inpatient}, \texttt{cancer\_inpatient}, \texttt{None}\}$：同一合同内具有不同等待期规定的具体保障责任标识。

在显式界定的“等待期单项免责判定”范围内，每份合同条款对应一个确定性裁决函数 $f: \mathcal{Z}_{\text{ins}} \to \{\texttt{WAIT\_EXCLUDED}, \texttt{WAIT\_NOT\_EXCLUDED}\}$。例如，在《平安住院安心医疗保险》（`pa_2011_36`，第 1 页第四条“保险责任”）中，第（一）项“一般住院医疗津贴”规定首次投保等待期为 **30 日**，续保或因意外伤害住院治疗无等待期（$\texttt{accident\_exempt}=\text{True}$）；第（二）项“癌症住院医疗津贴”规定首次投保等待期为 **90 日**，续保无等待期，且无意外伤害免等待期条款（$\texttt{accident\_exempt}=\text{False}$）。当 $\texttt{policy\_id}=\texttt{pa\_2011_36}$、首次投保且生效第 45 天因病住院时：
- 若 $\texttt{benefit\_id}=\texttt{general\_inpatient}$，因 $45 > 30$ 裁决为 $\texttt{WAIT\_NOT\_EXCLUDED}$；
- 若 $\texttt{benefit\_id}=\texttt{cancer\_inpatient}$，因 $45 \le 90$ 裁决为 $\texttt{WAIT\_EXCLUDED}$。

在多阈值组合规则域 $\mathcal{P}_{\text{multi}}$ 中，每个场景表示为 6 元条件帧 $z = (\texttt{clause\_id}, \texttt{entity\_id}, \texttt{condition\_flag}, \texttt{elapsed\_day}, \texttt{threshold\_days}, \texttt{override\_flag})$，对应裁决函数 $f_{\text{multi}}: \mathcal{Z}_{\text{multi}} \to \{\texttt{ELIGIBLE}, \texttt{EXCLUDED}\}$。此外，在数据集的 `atomic_rubrics` 字段中，我们将锚定至 PDF 原件页码与坐标框的合同规则依据（`source_type='pdf_contract_clause'`）与来自受控测试种子的场景设定参数（`source_type='synthetic_seed_frame'`，如具体出险天数 `policy_day=45`）显式分离，确保条款原文证据与合成场景参数边界清晰。

### 3.2 表面槽位完整度与无参考依赖的可达性枚举

给定完整种子条件帧 $z$ 及其参考裁决 $y^* = f(z)$，当大语言模型将种子提问 $q(z)$ 改写为合成提问 $q'$ 时，确定性槽位提取器 $g(q')$ 从 $q'$ 的表面文本中提取观测条件帧 $\hat{z} = g(q')$。为消除合成文本中的“书名号剧透”，我们在生成后与评测前统一剥离书名号（`《》`）与显式法条编号前缀。槽位提取器 $g(\cdot)$ 集成了三项关键抗干扰机制：
1. **对比与括号内联纠错消解（Contrastive & Inline Correction Override Resolution）**：对形如“不是/并非/并不是 A，而是/实际是 B”以及括号/内联更正句式“A（更正/修正/勘误/实际是：B）”或“A，更正为 B”，在提取枚举槽位与数值天数前优先剥离被否定的前半句或被更正的旧值“A”，仅保留真实生效的后半句“B”；若更正内容“B”为跨时间单位（如“两个月”）或日历起止日期，则旧值“A”同样被抹除并由语法不确定性检测器捕获；
2. **否定耗时拦截（Negated Duration Rejection）**：拒绝将“不到/未满/不足 $N$ 天”提取为确定性经过天数 `policy_day = N`；
3. **同句多天数冲突拦截（Conflicting Day-Number Guard）**：当剔除等待期阈值表达后同一句中仍存在多个不同天数（如“第 20 天或第 45 天”）时，拒绝任选其一，将其置为未决并触发不确定性硬门控标记。

若仅检查 $\hat{z}$ 是否逐一保留了 $z$ 的所有非空字段（记为**全槽位表面匹配 `all_slots_preserved`**），会将“省略了不影响最终裁决的非关键槽位”（例如在已满等待期的第 45 天省略出险原因是疾病还是意外）误判为错误。为此，我们设计了**不读取参考种子帧 `seed_frame`** 的阈值分区可达性枚举器 `enumerate_admissible_decisions`：

给定观测帧 $\hat{z} = g(q')$，枚举器遍历所有与 $\hat{z}$ 已观测字段相容的合法补全帧 $z' \in \text{Completions}(\hat{z})$，其中未观测天数遍历覆盖所有条款等待期阈值 $\{7, 15, 25, 30, 60, 90, 180\}$（多阈值域为 $\{10, 14, 30, 35, 90, 365\}$）内外的有限商集代表元集合 $\mathcal{D}_{\text{rep}}$，计算可达裁决集合：
$$\mathcal{A}(\hat{z}) = \bigl\{\, f(z') \;\big|\; z' \in \text{Completions}(\hat{z}) \,\bigr\}$$

### 3.3 解析不确定性 `PARSE_UNCERTAIN` 与问句歧义 `CLARIFY_REQUIRED` 的纯文本硬门控分离

在早期版本中，凡是导致 $|\mathcal{A}(\hat{z})| > 1$ 的样本均被笼统归入 `DEGRADED_TO_CLARIFY`。然而从严谨的形式化验证角度看，导致提取器未能锁定唯一结论的原因有两个截然不同的层面：一是**问句本身客观缺少关键业务条件**（任何人类核保员或下游系统看到该问句都必须追问）；二是**问句使用了超出正则解析器语法边界的表达或涉及未建模的特殊责任分支**。

为此，我们在冻结版本 `v3.0-frozen` 中引入了**完全基于输入问句文本 $q'$（绝不读取参考种子帧 `seed_frame`）**的语法与模式不确定性检测函数 `detect_parse_uncertainty(q')`，并将其作为阻断 `PRESERVED_DECISION` 放行的硬门控（Hard Gate），将每条合成问句 $q'$ 的验证结果精确划分为四类互斥状态：
1. **条件篡改（`CONDITION_MUTATED` / `DECISION_CHANGED`）**：存在某个已提取槽位 $k$ 使得 $\hat{z}_k \neq \texttt{None}$ 且 $\hat{z}_k \neq z_k$（例如将实际第 45 天篡改为 90 天，或将首次投保篡改为续保），或者虽然 $|\mathcal{A}(\hat{z})| = 1$ 但唯一可达结论 $\mathcal{A}(\hat{z}) \neq \{y^*\}$；
2. **结论唯一保持（`PRESERVED_DECISION`）**：无任何槽位篡改，纯文本检测 `detect_parse_uncertainty(q')` 为假，且可达裁决集合唯一锁定原结论 $\mathcal{A}(\hat{z}) = \{y^*\}$；
3. **解析器语法/模式不确定（`PARSE_UNCERTAIN`）**：`detect_parse_uncertainty(q')` 检出问句文本包含超出确定性语法边界的表述——具体包括：(a) 使用“个月/周/星期”或具体日历日期（如“3月1日投保、3月20日住院”）表达经过时间（含括号内联更正为月份或日期的情形）；(b) 句子中无肯定经过天数且包含模糊否定耗时（如“未满/不到/不足 $N$ 天”）；(c) 同一句中出现多个冲突的经过天数（如“第 20 天或第 45 天”）或投保极性冲突；(d) 问句文本同时提及 `pa_h716`《平安安康住院费用（B，2017）医疗保险》与 `v3.0-frozen` 未建模的特定器官/疾病关键词（“腺样体肥大、疝气、扁桃腺或女性生殖器官”）且未处于超过最大等待期阈值（$>90$ 日）的无条件免除区间。此时验证器强制阻断 `PRESERVED_DECISION` 并保守拒绝放行该问句，明确标记为解析器不确定（`PARSE_UNCERTAIN`），不将其混同于问句本身歧义；
4. **问句歧义需澄清（`CLARIFY_REQUIRED` / `DEGRADED_TO_CLARIFY`）**：在解析器完全支持的确定性语法辖域内（`detect_parse_uncertainty(q')` 为假），因遗漏关键资格条件导致可达集合同时包含正反两种裁决 $|\mathcal{A}(\hat{z})| = 2$，且枚举器自动输出一对正反补全见证 $(z^+, z^-)$ 使得 $f(z^+) \neq f(z^-)$。

### 3.4 符号槽位守卫闭环修复 `slot_guard_repair` 与仅验证过滤 `verify_only_filter`

基于上述形式化验证器，我们在数据合成阶段定义了两种可直接落地的质量控制工作流，并与无符号反馈的大模型自检进行严格对照：

- **仅验证过滤工作流（`verify_only_filter`）**：对首轮条件受控提示（`condition_prompt`）生成的初稿 $q^{(1)}$ 运行形式化验证器；若判定为 `PRESERVED_DECISION` 则接纳进入最终评测基准池，若判定为 `CONDITION_MUTATED`、`CLARIFY_REQUIRED` 或 `PARSE_UNCERTAIN` 则直接丢弃该样本（不触发二轮 LLM 调用）。该策略保证了放行池零污染，但会损失复杂种子的场景覆盖率。
- **符号槽位守卫闭环修复工作流（`slot_guard_repair`）**：对首轮 `condition_prompt` 初稿 $q^{(1)}$ 运行槽位提取器 $g(q^{(1)})$ 与验证器；仅当检出存在缺失槽位 $\mathcal{M} = \{k \mid z_k \neq \texttt{None} \land \hat{z}_k = \texttt{None}\}$ 或篡改槽位 $\mathcal{X} = \{k \mid \hat{z}_k \neq \texttt{None} \land \hat{z}_k \neq z_k\}$ 时，精确触发一次二轮修复调用，将具体缺失/冲突槽位名称及其目标地面真值 $(\mathcal{M}, \mathcal{X}, \{z_k\}_{k \in \mathcal{M} \cup \mathcal{X}})$ 显式注入修复提示词，引导模型生成修复问句 $q^{(2)}$ 并再次交由验证器核验。
- **等调用预算无符号反馈自检基线（`generic_self_refine_budget_matched`）**：在与 `slot_guard_repair` **完全相同的触发种子集合与完全相等的 LLM 调用次数预算**下，向模型提供源条款、种子问句与初稿 $q^{(1)}$，要求模型自行检查并修正遗漏或错误的条件，但**不提供符号解析器输出的缺失/篡改槽位列表**，从而严格剥离“增加二轮调用预算”与“符号槽位诊断反馈”的独立贡献。

---

## 第 4 章 开发集与后验扩展集保险条款实验及多阈值等预算对照

### 4.1 5 份开发集保险条款受控基准与跨模型尺度评测

我们在 5 份公开披露的中文商业保险正式合同条款（平安 `pa_p0781`、`pa_h865`，太保 `cpic_86017`、`cpic_82859`，泰康 `tk_group_2018`）上构建了 **20 组基础种子场景**与 **80 条受控测试样本**（`data_v3/controlled_cases.jsonl`），每组种子包含 4 类变体：`complete`（条件完备）、`surface_Preserve`（表面同义改写）、`missing_critical`（删去关键条件，均附带正反双见证 $z^+, z^-$）以及 `counterfactual_flip`（翻转关键条件使结论反转）。验证器在该 80 条受控样本上的正反双见证认证通过率为 **100.0%（80/80）**。

表 4.1 报告了在本地小模型 `Qwen2.5-1.5B-Instruct`（`float32` 确定性贪婪解码）及云端大模型 `Qwen-Plus`、`Qwen-Max`（`temperature=0`）上，无约束口语改写（`ordinary_style`）与显式条件受控提示（`condition_prompt`）在 20 组种子（共 120 条生成记录，`data_v3/multi_scale_condition_summary.json`）上的表现：

| 模型尺度 | 提示方式 | 独立全槽位保留率 (/20) | 独立结论可确定率 (/20) | 已知保单下结论可确定率 (/20) | 退化为 `CLARIFY` (/20) | 条件篡改率 (/20) | 相对种子新颖度 $1-\kappa_2$ | `bm25` / `bge` / `rrf` Top-1 (5条款池) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-1.5B` | `ordinary_style` | 20.0% (4/20) | 30.0% (6/20) | 35.0% (7/20) | 60.0% (12/20) | 10.0% (2/20) | 0.2587 | 90.0% / 75.0% / 85.0% |
| `Qwen2.5-1.5B` | `condition_prompt` | 90.0% (18/20) | **95.0%** (19/20) | **95.0%** (19/20) | 0.0% (0/20) | 5.0% (1/20) | 0.1489 | 100.0% / 80.0% / 100.0% |
| `Qwen-Plus` | `ordinary_style` | 0.0% (0/20) | 0.0% (0/20) | 25.0% (5/20) | **100.0%** (20/20) | 0.0% (0/20) | **0.5174** | 60.0% / 60.0% / 60.0% |
| `Qwen-Plus` | `condition_prompt` | **100.0%** (20/20) | **100.0%** (20/20) | **100.0%** (20/20) | 0.0% (0/20) | 0.0% (0/20) | 0.2055 | 100.0% / 80.0% / 100.0% |
| `Qwen-Max` | `ordinary_style` | 45.0% (9/20) | 45.0% (9/20) | 50.0% (10/20) | 55.0% (11/20) | 0.0% (0/20) | 0.3822 | 60.0% / 65.0% / 60.0% |
| `Qwen-Max` | `condition_prompt` | 95.0% (19/20) | **100.0%** (20/20) | **100.0%** (20/20) | 0.0% (0/20) | 0.0% (0/20) | 0.2464 | 100.0% / 80.0% / 100.0% |

实验表明：(1) **增大模型参数规模不能自动消除口语改写中的关键条件遗漏**——`Qwen-Plus` 在 `ordinary_style` 下为了生成地道口语问句（新颖度高达 0.5174），100% 省略了险种全称且在 75% 的样本中漏掉投保状态或天数，导致独立结论保持率为 **0.0%（0/20）**（已知保单下亦仅 25.0%）；(2) **表面槽位完整度不等于结论可确定性**——`Qwen-Max` 在 `condition_prompt` 下 `s13` 虽省略了出险原因是疾病还是意外（全槽位保留 19/20），但因首次投保第 45 天已超过 30 天等待期，无论疾病还是意外均不免责，可达性枚举器正确判定其结论 100% 保持（20/20）。

### 4.2 4 份后验扩展集保险条款、责任级等待期消歧与标题掩码检索

为检验同一合同内部不同保障责任具有不同等待期时的消歧能力及检索对险种标题的依赖，我们将评测扩展至另外 4 份官方保险条款 PDF（`pa_2011_36`《平安住院安心医疗保险》、`pa_p0254`《平安住院医疗保险》、`pa_p0321`《平安肺癌疾病保险》、`cpic_93896`《附加（广西地区）学生幼儿住院医疗保险（2025版）》，共 16 组种子、64 条受控样本，`data_v3/equal_budget_and_heldout_summary.json`）。由于我们在共享解析器中补充了口语产品简称与 `benefit_id` 责任消歧槽位（区分 `pa_2011_36` 一般住院津贴 30 天等待期与癌症住院津贴 90 天等待期），本节实验明确界定为**后验精化评估（Post-Refinement Evaluation）**：

| 方法 / 基线 (`1.5B`) | 调用次数 | 格式有效 (/16) | 独立全槽位匹配 (/16) | 独立结论保持率 (/16) | 已知保单结论保持率 (/16) | 退化为 `CLARIFY` (/16) | 条件篡改 (/16) | 新颖度 $1-\kappa_2$ | 9条款标准检索 Top-1 (`bm25`/`bge`) | 9条款标题掩码 Top-1 (`bm25`/`bge`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed_passthrough` | 0 | 100.0% (16/16) | 100.0% (16/16) | 100.0% (16/16) | 100.0% (16/16) | 0.0% (0/16) | 0.0% (0/16) | 0.0000 | 75.0% / 75.0% | 75.0% / 31.2% |
| `ordinary_style` | 16 | 100.0% (16/16) | 0.0% (0/16) | 0.0% (0/16) | 31.2% (5/16) | 100.0% (16/16) | 0.0% (0/16) | **0.5380** | 56.2% / 43.8% | 62.5% / 56.2% |
| `condition_prompt` | 16 | 100.0% (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | 0.0% (0/16) | 0.0% (0/16) | 0.1251 | 75.0% / 75.0% | 75.0% / 31.2% |
| `structured_json_rewrite` | 16 | 81.2% (13/16) | 62.5% (10/16) | 62.5% (10/16) | 93.8% (15/16) | 37.5% (6/16) | 0.0% (0/16) | 0.2512 | 75.0% / 68.8% | 81.2% / 43.8% |
| `generic_self_refine` (等预算) | 16 | 100.0% (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | 0.0% (0/16) | 0.0% (0/16) | 0.1251 | 75.0% / 75.0% | 75.0% / 31.2% |
| `generic_self_refine` (全量) | 32 | 100.0% (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | 0.0% (0/16) | 0.0% (0/16) | 0.1303 | 75.0% / 75.0% | 75.0% / 31.2% |
| `slot_guard_repair` | 16 | 100.0% (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | **100.0%** (16/16) | 0.0% (0/16) | 0.0% (0/16) | 0.1251 | 75.0% / 75.0% | 75.0% / 31.2% |

结果说明：(1) 在单条款保险规则上，`condition_prompt` 首轮已达 **16/16（100.0%）**，因此 `slot_guard_repair` 二轮修复触发次数为 0（$p=1.0$）；而双行 `structured_json_rewrite` 因 5 例在首行 JSON 写出险种名却在第二行 `QUESTION:` 漏掉险种名，独立保持率降至 10/16（62.5%，$p=0.03125$）；(2) 在 9 条款合并池上隐去产品名称与保险公司名称后，保留标题问句的 `bge_small_zh` Top-1 从 **75.0%（12/16）骤降至 31.2%（5/16）**，证明常规高检索分高度依赖产品名称字符串锚点。

### 4.3 16 组多阈值合成规则上的等调用预算修复对照

在包含多个竞争性数值窗口的 16 组（64 例）多阈值规则测试床（`med_egfr_tki` 14 天洗脱期、`med_vte_prophylaxis` 膝关节 10 天 vs. 髋关节 35 天、`law_work_injury` 用人单位 30 天 vs. 职工 365 天、`law_noncompete` 90 天欠付补偿与宽限期）上，`Qwen2.5-1.5B-Instruct` 的等调用预算（26 次调用）对照结果如表 4.3 所示：

| 方法 / 基线 (`1.5B`) | 总调用次数 | 格式有效 (/16) | 独立全槽位匹配 (/16) | 独立结论保持率 (总计 / 医疗8 / 劳动8) | 给定源规则结论保持率 (/16) | 退化为 `CLARIFY` (/16) | 条件篡改率 (/16) | 新颖度 $1-\kappa_2$ | 标准 / 掩码检索 Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed_passthrough` | 0 | 100.0% (16/16) | 100.0% (16/16) | 100.0% (16/16) (8/8, 8/8) | 100.0% (16/16) | 0.0% (0/16) | 0.0% (0/16) | 0.0000 | 100% / 100% |
| `ordinary_style` | 16 | 100.0% (16/16) | 6.2% (1/16) | 6.2% (1/16) (1/8, 0/8) | 43.8% (7/16) | 81.2% (13/16) | 12.5% (2/16) | **0.5726** | 100% / 100% |
| `condition_prompt` | 16 | 100.0% (16/16) | 37.5% (6/16) | 43.8% (7/16) (5/8, 2/8) | 43.8% (7/16) | 25.0% (4/16) | 31.2% (5/16) | 0.3376 | 100% / 100% |
| `structured_json_rewrite` | 16 | 0.0% (0/16) | 6.2% (1/16) | 31.2% (5/16) (3/8, 2/8) | 31.2% (5/16) | 37.5% (6/16) | 31.2% (5/16) | 0.6064 | 100% / 93.8% |
| `generic_self_refine` (等预算) | **26** | 93.8% (15/16) | 37.5% (6/16) | 50.0% (8/16) (5/8, 3/8) | 50.0% (8/16) | 25.0% (4/16) | 25.0% (4/16) | 0.3151 | 100% / 100% |
| `generic_self_refine` (全量) | 32 | 93.8% (15/16) | 37.5% (6/16) | 50.0% (8/16) (5/8, 3/8) | 50.0% (8/16) | 25.0% (4/16) | 25.0% (4/16) | 0.3133 | 100% / 100% |
| **`slot_guard_repair` (本文)** | **26** | **100.0% (16/16)** | **68.8% (11/16)** | **75.0% (12/16) (8/8, 4/8)** | **93.8% (15/16)** | **25.0% (4/16)** | **0.0% (0/16)** | **0.3342** | **100% / 100%** |

核心结论：(1) 全部 6 种非 JSON 方法在标准与标题掩码下的 `bm25_char2` 和 `bge_small_zh` Top-1 准确率均为 **100.0%（16/16）**，但 `ordinary_style` 独立结论保持率仅 **6.2%（1/16）**；(2) 在完全相同的 **26 次调用预算**（共享 10 题触发日程）下，无解析器反馈的 `generic_self_refine_budget_matched` 仅将保持率从 7/16 微升至 **8/16（50.0%）**，残留 **4/16（25.0%）** 阈值覆盖篡改；而 `slot_guard_repair` 将独立全槽位匹配率提升至 **11/16（68.8%）**，独立结论保持率提升至 **12/16（75.0%，双侧精确 McNemar $p=0.125$）**，给定源规则身份（`granted_clause`）下达 **15/16（93.8%，$p=0.015625$）**，并将解析器检出的阈值篡改率降至 **0/16（0.0%）**。

### 4.4 264 条确定性验证器诊断日志与 29 条边界关注清单

在 `data_v3/item_by_item_semantic_audit.jsonl` 中，我们导出了前四章全部 264 条生成记录的确定性槽位提取轨迹与 29 条边界关注清单（16 条多阈值双行 JSON 在 45–111 tokens 处提前终止生成、7 条含业务短语但省略完整规程标题、5 条首行 JSON 含险种名但问句行省略、1 条触及 160-token 上限截断），所有记录均显式标定为自动解析器诊断解释而非外部人工盲评。

---

## 第 5 章 三层核查体系、40 种子确认集与 24 种子未触碰留出集跨模型泛化实验

为解决第 4.2 节属于后验评估、第 4.3 节样本量有限（$N=16$）以及如何客观评估解析器自身边界与跨条款泛化能力，我们在 `src/frozen_confirmation_and_rag_study.py` 中构建了三层核查体系、40 种子跨模型确认集、公平结构化基线消融以及在全部规则硬化（`v3.1-hardened`）彻底冻结后新采集的 **24 种子未触碰留出集（`v3.1-untouched-holdout`）**。云端生成跨越两大家族模型：阿里云 DashScope `Qwen-Plus`（请求 `qwen-plus`，响应 `response_model="qwen-plus"`）与 DeepSeek 官方接口（请求 `https://api.deepseek.com/chat/completions` 的 `model="deepseek-chat"`，服务端实际返回 `response_model="deepseek-flash"`、`system_fingerprint="e72d3c7f663d3e1383b00b3884cb4501"`）。

### 5.1 第一层审计：26 份官方保险条款与 4 段多阈值规则映射审计及作者工程耗时估算

我们将官方商业保险合同从 9 份扩展至 **26 份 PDF 原件**，包含 11 份零别名确认集条款（`pa_h716`《平安安康住院费用（B，2017）医疗保险》、`pa_2651`《平安安医保（易核版）医疗保险》、`pa_l0001`《平安安医保（星耀版）医疗保险》、`pa_1714`《平安附加守护全能（2023）提前给付重大疾病保险》、`pa_1072`《平安e生保长期医疗保险（费率可调）》、`pa_596`《平安e生保（尊享版）医疗保险》、`cpic_107082`《太保个人中高端C款医疗保险（H2026）》、`cpic_106702`《太保互联网母婴医疗保险》、`cpic_60201`《附加团体疾病医疗保险（B款）（H2014）》、`cpic_79920`《深圳专属团体医疗保险》、`cpic_85050`《心安·怡住院费用医疗保险（H2017A）》）以及在 `v3.1-hardened` 冻结后新采集的 **6 份未触碰留出集平安正式条款**（`pa_2650`《平安e生保（悦享版）医疗保险》、`pa_1071`《平安粤港澳大湾区深圳专属一年期重大疾病保险》、`pa_1073`《平安i药保B款特药费用医疗保险》、`pa_595`《平安e生保（加享版）医疗保险》、`pa_597`《平安e生保（颐享版）医疗保险》、`pa_l0002`《平安安医保e享长期医疗保险（费率可调）（互联网）》）。

在 `data_v3/clause_rule_mapping_audit.json` 与 `data_v3/untouched_holdout_sources.json` 中，我们逐一审计了原件 SHA256、物理页码、坐标包围盒、等待期天数、意外伤害豁免、续保豁免以及**未建模除外边界（Unmodeled Exclusions）**：
- **单份合同作者工程编译耗时估算**：前 20 份合同平均 **11.72 分钟/份**（中位数 **11.00 分钟/份**，范围 9.5–18.0 分钟/份，20 份合同总计 **234.5 分钟 [约 3.91 小时]**）。需要明确声明的是，该耗时为论文作者在阅读 PDF 原件、提取等待期与豁免槽位并编写 Python 字典配置过程中的工程记录估算（`author_engineering_time_estimate_not_human_subject_stopwatch_experiment`），而非跨多受试者的外部人工计时实验。
- **零别名约束与未建模分支真实文本检测**：对 11 份确认集条款与 6 份未触碰留出集条款，全部设置 `aliases = []`。特别地，`pa_h716`（《平安安康住院费用（B，2017）医疗保险》第 3 页）包含一条特定器官/疾病分支规则——**“被保险人因腺样体肥大、疝气、扁桃腺的疾病或女性生殖器官的疾病需要住院治疗或特殊门诊治疗的，自本主险合同生效之日起 90 日为等待期；因其他疾病需要住院治疗或特殊门诊治疗的，自本主险合同生效之日起 30 日为等待期”**。`detect_parse_uncertainty(text)` 仅从问句文本表面识别《平安安康住院费用（B，2017）》及“腺样体/疝气/扁桃腺/女性生殖器官”关键词，不再读取任何原种子元数据，并将 `is_parse_uncertain` 设为阻断 `PRESERVED_DECISION` 放行的硬门控。

### 5.2 第二层与第三层核查：120 例构造性黄金诊断集与 400 条生成记录的共享规则库双实现一致性核查

我们在 `data_v3/gold_diagnostic_suite.jsonl`（汇总于 `data_v3/gold_diagnostic_summary.json`）中构造了 **120 例黄金诊断集**（8 大类别，每类 15 例，含 60 例黄金正例 `PRESERVED_DECISION` 与 60 例黄金负例 `DECISION_CHANGED` / `CLARIFY_REQUIRED`），并对 400 条跨模型生成记录构建了独立存储的静态标签表 `data_v3/frozen_confirmation_independent_audit_labels.json` 及正向可达性核查函数 `evaluate_independent_text_semantics`。

需要重点说明的是，`evaluate_independent_text_semantics` 不再通过翻转 `seed['reference_decision']` 走捷径，而是直接将提取出的槽位送入正向可达性枚举器计算可达裁决集合：当问句在同一等待期区间内发生未跨阈值的数值偏离（例如在 30 天等待期条款下将“保单生效第 20 天”改写为“保单生效第 21 天”），正向可达性枚举准确得出唯一可达裁决仍为 `WAIT_EXCLUDED`，并将其判定为**事实槽位偏移但裁决不变（`FACT_MUTATED_SAME_DECISION`）**，仅当跨过 30 天等待期阈值（如改为“保单生效第 45 天”）时才判定为**裁决翻转（`DECISION_CHANGED`，`WAIT_NOT_EXCLUDED`）**。同时，由于 `evaluate_independent_text_semantics` 与被测验证器共享底层手工编译的条款规则表、且不存在外部多标注员双盲人工复核日志，我们将其诚实界定为**共享规则库的双实现一致性核查（`shared_rule_base_dual_implementation_consistency_check`）**而非外部人工金标，并分别报告纯语义可达性一致性与联合格式可用性两套指标（表 5.1）：

| 诊断类别 / 核查层级 | 样本数 | 参考真值类别 | 真阳性 (TP) | 假阳性 (FP / 误放行) | 真阴性 (TN) | 假阴性 (FN / 保守误拒) | `PARSE_UNCERTAIN` 触发数 | 类别准确率 / 放行精确率 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **类别 1**：`colloquial_exact_synonym`（口语同义改写） | 15 | 正例 (`PRESERVED`) | 15 | **0** | 0 | 0 | 0 | 100.0% / 100.0% |
| **类别 2**：`non_pivotal_omission_determinate`（省略非关键槽位但结论唯一） | 15 | 正例 (`PRESERVED`) | 15 | **0** | 0 | 0 | 0 | 100.0% / 100.0% |
| **类别 3**：`contrastive_correction_valid`（“不是 A 而是 B”及括号更正消解） | 15 | 正例 (`PRESERVED`) | 15 | **0** | 0 | 0 | 0 | 100.0% / 100.0% |
| **类别 4**：`pivotal_omission_ambiguous`（遗漏关键槽位需澄清） | 15 | 负例 (`CLARIFY`) | 0 | **0** | 15 | 0 | 0 | 100.0% / — |
| **类别 5**：`numeric_mutation_decision_flip`（数值篡改致结论翻转） | 15 | 负例 (`CHANGED`) | 0 | **0** | 15 | 0 | 0 | 100.0% / — |
| **类别 6**：`negated_duration_trap`（“未满/不到 $N$ 天”否定陷阱） | 15 | 负例 (`CLARIFY`) | 0 | **0** | 15 | 0 | 15 | 100.0% / — |
| **类别 7**：`conflicting_multi_value_trap`（“第 20 天或第 45 天”多值冲突） | 15 | 负例 (`CLARIFY`) | 0 | **0** | 15 | 0 | 15 | 100.0% / — |
| **类别 8**：`unsupported_syntax_determinate`（跨单位“个月/周”或日历日期） | 15 | 正例 (`PRESERVED`) | 0 | **0** | 0 | 15 | **15 (100% 拦截)** | 保守拦截为 `PARSE_UNCERTAIN` |
| **合计（第二层：类别 1–7 支持语法辖域，105 例）** | **105** | 45 正 / 60 负 | **45** | **0 (FAR = 0.0%)** | **60** | **0 (FRR = 0.0%)** | 30 | **精确率 100.0% / 召回率 100.0%** |
| **合计（第二层：全量 8 类别黄金诊断集，120 例）** | **120** | 60 正 / 60 负 | **45** | **0 (FAR = 0.0%)** | **60** | **15 (FRR = 25.0%)** | 45 | **精确率 100.0% / 召回率 75.0%** |
| **合计（第三层 A：400 条生成记录纯语义可达性一致性 `semantic_reachability`）** | **400** | 314 正 / 86 负 | **296** | **0 (FAR = 0.0%)** | **86** | **18 (FRR = 5.73%)** | 10 | **精确率 100.0% / 召回率 94.27%** |
| **合计（第三层 B：400 条生成记录联合格式与语义可用性 `joint_format_and_semantic`）** | **400** | 306 正 / 94 负 | **290** | **6 (FAR = 6.38%)** | **88** | **16 (FRR = 5.23%)** | 10 | **精确率 97.97% / 召回率 94.77%** |
| — 其中：`slot_guard_repair` 子集（$n=80$） | 80 | 79 正 / 1 负 | **78** | **0 (FAR = 0.0%)** | **1** | **1 (FRR = 1.27%)** | 1 | **精确率 100.0% / 召回率 98.73%** |
| — 其中：`structured_json_rewrite` 子集（$n=80$，160-Token上限） | 80 | 72 正 / 8 负 | **71** | **6 (Qwen 5例 + DeepSeek 1例截断)** | **2** | **1 (FRR = 1.39%)** | 2 | **精确率 92.21% / 召回率 98.61%** |

表 5.1 揭示了三项客观事实：
1. **120 例构造性诊断集与 400 条生成记录纯语义核查均实现零语义误放行（Semantic False Accept Rate = 0.0%，Precision = 100.0%）**：在类别 1–7 支持语法辖域（105 例）上准确率为 100.0%；在类别 8（跨单位“个月/周”或日历日期表达经过时间，15 例）以及跨单位括号更正上，`detect_parse_uncertainty` 将其 **100% 捕获并拦截为 `PARSE_UNCERTAIN`**。在 400 条生成记录的纯语义可达性核查（`semantic_reachability_consistency`）中，取得 **TP = 296、FP = 0、TN = 86、FN = 18**（注：`data_v3/frozen_confirmation_three_layer_audit.jsonl` 中的 `v3_0_raw_outcome` 字段为保持向后兼容而记录当前验证器 `v3.1-posthoc-hardened` 的输出；在 40 种子确认集上，`slot_guard_repair` 在 `v3.0-frozen` 与 `v3.1-posthoc-hardened` 下的可用率同为 78/80 = 97.50%）。
2. **联合格式与语义可用性核查中的 6 例假阳性（`FP = 6`）来自 `structured_json_rewrite` 的 160-Token 截断残句（`Qwen-Plus` 5 例、`DeepSeek` 1 例）**：在 `structured_json_rewrite`（`max_new_tokens=160` 上限）基线中，由于首行 JSON 槽位字典挤占了生成配额，有 6 条已被提取出完整槽位的输出在问句末尾被截断（`Qwen-Plus` 5 例、`DeepSeek` 1 例，`format_valid = False` 且缺少句末问号；另有 2 例截断输出同时因缺失关键槽位已被验证器拒斥为 `TN`）。相比之下，**`slot_guard_repair`（$n=80$）、`condition_prompt`（$n=80$）与 `generic_self_refine_budget_matched`（$n=80$）上的误放行均为 0（`FP = 0`，放行精确率 100.0%）**。
3. **16 例（纯语义 18 例）假阴性反映了关键词提取器的保守边界**：其中 2 例为 `fconf-ins-003`（`pa_h716` 特定器官分支）触发 `PARSE_UNCERTAIN` 保守拦截，其余出现在 `condition_prompt` 与 `generic_self_refine` 中——模型使用了“该产品/本合同”指代险种全称或口语换述，严格无别名验证器将其保守拒斥并触发二轮槽位修复，从而在 `slot_guard_repair` 后将召回率拉升至 **98.73%（78/79）**。

### 5.3 40 种子确认集、公平结构化基线与 24 种子硬化后未触碰扩展留出集实验

在 40 种子确认集（24 个零别名保险种子 + 16 个多阈值种子，共 160 条受控样本，跨越 15 个条款簇）上，我们除评测主实验 8 种方法外，专门构建了两项消除 160-Token 截断的**公平结构化基线**（`data_v3/fair_structured_baseline_generations.jsonl`）：
- (1) **`structured_json_untruncated_320`（独立云端 API 重生成，`max_new_tokens=320`）**：在保持双行提示（首行 JSON + 第二行 `QUESTION:`）完全不变的前提下对全部 80 个样本（40 种子 $\times$ 2 模型家族）独立执行 `max_new_tokens=320` 生成请求（80 次 API 调用，其中 26/80 条 `raw_response` 与 160-Token 运行一致，平均总消耗 533.5 tokens/seed），彻底消除了截断；
- (2) **`structured_json_schema_render`（基于 160-Token 首行 JSON 的确定性模板渲染消融）**：不额外发起 API 调用（80/80 复用 `structured_json_rewrite` 的 `raw_response`，继承平均总消耗 530.6 tokens/seed），直接提取首行 JSON 字典并通过确定性函数 `render_question_from_json_schema` 渲染为自然语言单句提问，以剥离自由文本 `QUESTION:` 行生成错误与开放式 JSON 槽位值/模板渲染错误。

进一步，在 `v3.1-posthoc-hardened` 全部解析规则彻底冻结后，我们在同仓库内从未参与过任何规则或提示调优的 **24 种子内部未触碰扩展留出集**（从本地原始条款池中提取的 6 份未见平安正式保险 PDF `pa_2650`、`pa_1071`、`pa_1073`、`pa_595`、`pa_597`、`pa_l0002` 共 12 个保险种子 + 在 4 个多阈值规则模板上用全新参数组合实例化的 12 个多阈值种子，合计 10 个条款簇、$N=48$ 跨模型配对样本，`data_v3/untouched_holdout_generations.jsonl`）上执行了零改动泛化评测（表 5.2）：

| 模型家族 | 方法 / 基线 | 总调用次数 | 输入 / 输出 / 总 Token | 平均延迟 (ms/seed) | 独立全槽位匹配 (/40) | 槽位结论保持 `slot_only` (总计 / 保险24 / 多阈值16) | 给定条款结论保持 (/40) | 需澄清 `CLARIFY` (/40) | 解析不确定 `PARSE_UNCERTAIN` (/40) | 条件篡改 `MUTATED` (/40) | 格式合规 `format_valid` (/40) | 联合格式与结论可用 `format_and_slot` (/40) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen-Plus` | `seed_passthrough` | 0 | 0 / 0 / 0 | 0.0 | 95.0% (38/40) | 97.5% (39/40) (23/24, 16/16) | 97.5% (39/40) | 0 | 1 | 0 | 40 / 40 | 97.5% (39/40) / 0.0% |
| `Qwen-Plus` | `ordinary_style` | 40 | 5,641 / 1,175 / 6,816 | 2,219.5 | 0.0% (0/40) | 0.0% (0/40) (0/24, 0/16) | 30.0% (12/40) | 38 | 0 | 2 | 40 / 40 | 0.0% (0/40) / **100.0%** |
| `Qwen-Plus` | `condition_prompt` | 40 | 18,123 / 2,885 / 21,008 | 3,228.0 | 87.5% (35/40) | 92.5% (37/40) (23/24, 14/16) | 92.5% (37/40) | 0 | 1 | 2 | 40 / 40 | 92.5% (37/40) / 7.5% |
| `Qwen-Plus` | `structured_json_rewrite` (160上限) | 40 | 16,427 / 5,739 / 22,166 | 4,755.7 | 92.5% (37/40) | 95.0% (38/40) (23/24, 15/16) | 95.0% (38/40) | 0 | 1 | 1 | **33 / 40 (7 题截断)** | **82.5% (33/40)** (槽位 95.0%) |
| `Qwen-Plus` | `structured_json_untruncated_320` (320上限) | 40 | 16,427 / 5,895 / 22,322 | 4,918.8 | 92.5% (37/40) | 95.0% (38/40) (22/24, 16/16) | 95.0% (38/40) | 1 | 1 | 0 | **40 / 40 (0 截断)** | **95.0% (38/40)** / 5.0% |
| `Qwen-Plus` | `structured_json_schema_render` (复用160首行JSON渲染) | 40 (复用) | 16,427 / 5,739 / 22,166 | 4,755.8 | 65.0% (26/40) | 90.0% (36/40) (22/24, 14/16) | 92.5% (37/40) | 3 | 1 | 0 | **40 / 40 (0 截断)** | **90.0% (36/40)** / 10.0% |
| `Qwen-Plus` | `verify_only_filter` | 40 | 18,123 / 2,885 / 21,008 | 3,228.0 | 87.5% (35/40) | 92.5% (保留 37/40, 丢弃 3 题) | 92.5% (37/40) | 0 | 1 | 2 | 40 / 40 | **92.5% (丢 7.5%) / 0.0%** |
| `Qwen-Plus` | `generic_self_refine` (等预算) | **47** | 21,585 / 3,549 / 25,134 | 3,879.9 | 90.0% (36/40) | 90.0% (36/40) (22/24, 14/16) | 90.0% (36/40) | 0 | 1 | 4 | 40 / 40 | 90.0% (36/40) / 10.0% |
| `Qwen-Plus` | `generic_self_refine` (全量) | 80 | 37,923 / 5,882 / 43,805 | 6,410.9 | 82.5% (33/40) | 85.0% (34/40) (22/24, 12/16) | 85.0% (34/40) | 2 | 1 | 4 | 39 / 40 | 82.5% (33/40) (槽位 85.0%) |
| `Qwen-Plus` | **`slot_guard_repair` (本文)** | **47** | **20,503 / 3,477 / 23,980** | **3,812.2** | **92.5% (37/40)** | **97.5% (39/40) (23/24, 16/16)** | **97.5% (39/40)** | **0** | **1** | **0** | **40 / 40** | **97.5% (39/40) / 0.0%** |
| `DeepSeek` | `seed_passthrough` | 0 | 0 / 0 / 0 | 0.0 | 95.0% (38/40) | 97.5% (39/40) (23/24, 16/16) | 97.5% (39/40) | 0 | 1 | 0 | 40 / 40 | 97.5% (39/40) / 0.0% |
| `DeepSeek` | `ordinary_style` | 40 | 4,807 / 1,213 / 6,020 | 1,206.0 | 0.0% (0/40) | 2.5% (1/40) (1/24, 0/16) | 40.0% (16/40) | 39 | 0 | 0 | 40 / 40 | 2.5% (1/40) / **97.5%** |
| `DeepSeek` | `condition_prompt` | 40 | 16,109 / 2,426 / 18,535 | 1,331.4 | 67.5% (27/40) | 82.5% (33/40) (21/24, 12/16) | 90.0% (36/40) | 6 | 1 | 0 | 40 / 40 | 82.5% (33/40) / 17.5% |
| `DeepSeek` | `structured_json_rewrite` (160上限) | 40 | 14,725 / 5,559 / 20,284 | 1,534.6 | 90.0% (36/40) | 97.5% (39/40) (23/24, 16/16) | 97.5% (39/40) | 0 | 1 | 0 | **39 / 40 (1 题截断)** | **95.0% (38/40)** (槽位 97.5%) |
| `DeepSeek` | `structured_json_untruncated_320` (320上限) | 40 | 14,725 / 5,636 / 20,361 | 1,567.2 | 82.5% (33/40) | 87.5% (35/40) (19/24, 16/16) | 90.0% (36/40) | 3 | 2 | 0 | **40 / 40 (0 截断)** | **87.5% (35/40)** / 12.5% |
| `DeepSeek` | `structured_json_schema_render` (复用160首行JSON渲染) | 40 (复用) | 14,725 / 5,559 / 20,284 | 1,534.6 | 67.5% (27/40) | 85.0% (34/40) (20/24, 14/16) | 87.5% (35/40) | 3 | 2 | 1 | **40 / 40 (0 截断)** | **85.0% (34/40)** / 15.0% |
| `DeepSeek` | `verify_only_filter` | 40 | 16,109 / 2,426 / 18,535 | 1,331.4 | 67.5% (27/40) | 82.5% (保留 33/40, 丢弃 7 题) | 90.0% (36/40) | 6 | 1 | 0 | 40 / 40 | **82.5% (丢 17.5%) / 0.0%** |
| `DeepSeek` | `generic_self_refine` (等预算) | **53** | 21,719 / 3,311 / 25,030 | 1,747.2 | 75.0% (30/40) | 85.0% (34/40) (21/24, 13/16) | 85.0% (34/40) | 3 | 1 | 2 | 40 / 40 | 85.0% (34/40) / 15.0% |
| `DeepSeek` | `generic_self_refine` (全量) | 80 | 33,668 / 5,008 / 38,676 | 2,620.0 | 72.5% (29/40) | 85.0% (34/40) (21/24, 13/16) | 85.0% (34/40) | 3 | 1 | 2 | 40 / 40 | 85.0% (34/40) / 15.0% |
| `DeepSeek` | **`slot_guard_repair` (本文)** | **53** | **19,820 / 3,250 / 23,070** | **1,774.3** | **95.0% (38/40)** | **97.5% (39/40) (23/24, 16/16)** | **97.5% (39/40)** | **0** | **1** | **0** | **40 / 40** | **97.5% (39/40) / 0.0%** |

### 5.4 双终点统计显著性检验（含种子级/条款级聚类置换与 4 组 / 6 组 Holm-Bonferroni 校正）及 24 种子扩展留出集分析

为避免混合“单纯槽位保持（`slot_only`）”与“联合格式完整且结论保持（`format_and_slot`）”两个终点，我们在 `src/frozen_confirmation_and_rag_study.py` 中对所有对比同时计算了两个终点的配对 McNemar 检验、**以种子为聚类单元（$N_{\text{seed}}=40$ 或 $24$）**及**以条款文档为聚类单元（$N_{\text{clause}}=15$ 或 $10$）**的双侧精确符号翻转置换检验，并同时给出**预设 4 组主基线对比**（`ordinary_style`、`condition_prompt`、`generic_self_refine_budget_matched`、`structured_json_rewrite`）以及**纳入两项公平结构化基线后的 6 组探索性对比**的 Holm-Bonferroni 多重比较校正 $p$ 值（表 5.3）：

| 对比组合 (`slot_guard_repair` vs. 对比基线) | 数据集与终点 | 观测单元数 | `slot_guard_repair` 胜率 | 对比基线胜率 | 胜负对 / 胜负种子簇 / 胜负文档簇 | 双侧精确 McNemar $p$ (4组Holm / 6组Holm) | 种子聚类置换 $p$ (4组Holm / 6组Holm) | 条款聚类置换 $p$ (4组Holm / 6组Holm) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| vs. `ordinary_style` | 40 种子确认集 (`format_and_slot`) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 1.25% (1/80) | 对 `77:0` / 种子 `39:0` / 文档 `15:0` | **$p = 1.32 \times 10^{-23}$** ($5.29 \times 10^{-23}$ / $7.94 \times 10^{-23}$) | **$p = 3.64 \times 10^{-12}$** ($1.46 \times 10^{-11}$ / $2.18 \times 10^{-11}$) | **$p = 0.000061$** (**0.000244** / **0.000366**) |
| vs. `condition_prompt` (等价于 `verify_only_filter`) | `DeepSeek` 单模型 (`format_and_slot`) | 40 (40 种子 / 15 文档) | **97.50% (39/40)** | 82.50% (33/40) | 对 `6:0` | **$p = 0.03125$** | — | — |
| vs. `condition_prompt` (等价于 `verify_only_filter`) | 40 种子确认集 (`format_and_slot` & `slot_only`) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 87.50% (70/80) | 对 `8:0` / 种子 `8:0` / 文档 `5:0` | **$p = 0.0078125$** (**0.023438** / **0.039063**) | **$p = 0.0078125$** (**0.023438** / **0.039063**) | $p = 0.0625$ (0.125 / 0.1875) |
| vs. `generic_self_refine_budget_matched` (等预算) | 40 种子确认集 (`format_and_slot` & `slot_only`) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 87.50% (70/80) | 对 `8:0` / 种子 `7:0` / 文档 `6:0` | **$p = 0.0078125$** (**0.023438** / **0.039063**) | **$p = 0.015625$** (**0.03125** / 0.0625) | **$p = 0.03125$** (0.09375 / 0.15625) |
| vs. `generic_self_refine_all` (全量 80 次调用) | 40 种子确认集 (`format_and_slot` / `slot_only`) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 83.75% (67/80) / 85.00% (68/80) | 对 `11:0` (槽位 `10:0`) / 种子 `10:0` / 文档 `7:0` | **$p = 0.000977$** (槽位 0.001953) | **$p = 0.001953$** (槽位 0.003906) | **$p = 0.015625$** |
| vs. `structured_json_rewrite` (160上限) | 40 种子确认集 (**`slot_only` 纯槽位**) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 96.25% (77/80) | 对 `1:0` / 种子 `1:0` / 文档 `1:0` | $p = 1.0000$ | $p = 1.0000$ | $p = 1.0000$ |
| vs. `structured_json_rewrite` (160上限) | 40 种子确认集 (**`format_and_slot` 联合可用**) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 88.75% (71/80: Qwen 33 + DS 38) | 对 `7:0` / 种子 `6:0` / 文档 `5:0` | **$p = 0.015625$** (**0.023438** / **0.039063**) | **$p = 0.03125$** (**0.03125** / 0.09375) | $p = 0.0625$ (0.125 / 0.1875) |
| vs. `structured_json_untruncated_320` (320上限) | 40 种子确认集 (`format_and_slot` & `slot_only`) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 91.25% (73/80: Qwen 38 + DS 35) | 对 `5:0` / 种子 `5:0` / 文档 `5:0` | $p = 0.0625$ (— / 0.0625) | $p = 0.0625$ (— / 0.09375) | $p = 0.0625$ (— / 0.1875) |
| vs. `structured_json_schema_render` (复用160首行JSON渲染) | 40 种子确认集 (`format_and_slot` & `slot_only`) | 80 (40 种子 / 15 文档) | **97.50% (78/80)** | 87.50% (70/80: Qwen 36 + DS 34) | 对 `8:0` / 种子 `6:0` / 文档 `6:0` | **$p = 0.0078125$** (— / **0.039063**) | **$p = 0.03125$** (— / 0.09375) | **$p = 0.03125$** (— / 0.15625) |
| **vs. `condition_prompt` (未触碰扩展留出集)** | **24 种子留出集 (`format_and_slot`)** | **48 (24 种子 / 10 文档)** | **100.0% (48/48)** | **81.25% (39/48: Qwen 20 + DS 19)** | **对 `9:0` / 种子 `8:0` / 文档 `6:0`** | **$p = 0.00390625$** | **$p = 0.0078125$** | **$p = 0.03125$** |
| **vs. `structured_json_rewrite` (160上限, 留出集)** | **24 种子留出集 (`format_and_slot`)** | **48 (24 种子 / 10 文档)** | **100.0% (48/48)** | **77.08% (37/48; 槽位 45/48)** | **对 `11:0` / 种子 `8:0` / 文档 `6:0`** | **$p = 0.00097656$** | **$p = 0.0078125$** | **$p = 0.03125$** |
| **vs. `structured_json_schema_render` (留出集)** | **24 种子留出集 (`format_and_slot`)** | **48 (24 种子 / 10 文档)** | **100.0% (48/48)** | **81.25% (39/48: Qwen 20 + DS 19)** | **对 `9:0` / 种子 `6:0` / 文档 `6:0`** | **$p = 0.00390625$** | **$p = 0.03125$** | **$p = 0.03125$** |
| **vs. `structured_json_untruncated_320` (留出集)** | **24 种子留出集 (`format_and_slot`)** | **48 (24 种子 / 10 文档)** | **100.0% (48/48)** | **89.58% (43/48: Qwen 22 + DS 21)** | **对 `5:0` / 种子 `5:0` / 文档 `3:0`** | $p = 0.0625$ | $p = 0.0625$ | $p = 0.25$ |
| **vs. `generic_self_refine_budget_matched` (留出集)** | **24 种子留出集 (`format_and_slot`)** | **48 (24 种子 / 10 文档)** | **100.0% (48/48)** | **89.58% (43/48: Qwen 22 + DS 21)** | **对 `5:0` / 种子 `4:0` / 文档 `3:0`** | $p = 0.0625$ | $p = 0.125$ | $p = 0.25$ |

结合表 5.2、表 5.3 与 `data_v3/frozen_confirmation_summary.json` 中的失效机制分解（`failure_mechanism_decomposition`），可以得出三项结论：
1. **40 种子确认集上的统计比较**：在联合可用率（`format_and_slot`）主终点上，经预设 4 主基线 Holm-Bonferroni 校正后，`slot_guard_repair`（97.50%，78/80）在配对 McNemar 与 40 种子聚类符号翻转检验下均优于 `condition_prompt`、等预算 `generic_self_refine_budget_matched` 与 160-Token `structured_json_rewrite`（$p_{\text{Holm}} \le 0.03125$）。若将两项探索性结构化基线一并纳入 6 组比较，配对级 McNemar 中前 5 组仍满足 $p_{\text{Holm}} = 0.039063 < 0.05$；在 15 条款文档级聚类下，由于分歧样本分布在 5–6 个条款簇内，单次符号翻转 $p$ 值为 $0.03125$–$0.0625$，经 15 簇多重比较校正后除 `ordinary_style`（$p_{\text{Holm}} = 0.000244$）外呈边缘显著趋势（$p_{\text{Holm}} \in [0.09375, 0.125]$）。
2. **解除截断与确定性 JSON 模板渲染的残留失效分析**：
   - 将 `structured_json_rewrite` 的生成上限从 160 放宽至 `max_new_tokens=320`（`structured_json_untruncated_320`）后，消除了 `Qwen-Plus` 上的截断（从 33/40 升至 38/40，总体联合可用率升至 **91.25%，73/80**），说明 160-Token 基线的格式损失主要源于输出长度限制。修复方法在确认集上的联合可用率为 97.5%（78/80）。相较 320-token 结构化基线（91.25%，73/80），结果呈正向改善，但差异未达到统计显著性（5W/0L，McNemar $p=0.0625$）。对 `structured_json_untruncated_320` 的 7 例未通过样本逐条核查表明：其首行 JSON 的 `day` 字段与种子天数全部一致（`json_day_matches_seed = 80/80`，0 例 JSON 天数覆写），未通过原因在于：**4 例**（`deepseek:fconf-ins-007, 009, 015, 017`）在自由文本 `QUESTION:` 行中附带写出了合同规则“（合同约定意外伤害无等待期）”，与题干中的“因非意外疾病住院治疗”同现，触发疾病/意外极性冲突拦截；**2 例**（`qwen_plus:fconf-ins-003`、`deepseek:fconf-ins-003`）命中 `pa_h716` 未建模器官分支；**1 例**（`qwen_plus:fconf-ins-006`）缺少显式豁免审批词。
   - 在复用 160-Token 首行 JSON 直接执行确定性模板渲染的 `structured_json_schema_render`（联合可用率 **87.50%，70/80**，`slot_guard_repair` 净胜 8:0，McNemar $p=0.0078125$，种子与条款聚类均为 $p=0.03125$）中，首行 JSON 的 `day` 字段同样与种子天数一致（`json_day_matches_seed = 80/80`）。在 `deepseek:fconf-cross-011`（种子 `status='secondary', elapsed_day=105, threshold_days=90, approved_waiver=False`）与 `012`（种子 `status='primary', elapsed_day=105, threshold_days=90, approved_waiver=True`）上，JSON `"day"` 字段保持整数 `105`，被 `render_question_from_json_schema` 渲染为通用短语“目前已达第105天”，因缺少竞业限制规则提取 `elapsed_day` 所需的领域前缀导致 `parsed_day` 为 `None`（`rendered_elapsed_day_not_extracted = 2`）；同时在 `011` 上模型将 JSON `"status"` 槽位写成“高级技术人员且负有保密义务”（`extracted.status='primary'`, `mutated_slots=['status']`, `missing_slots=['elapsed_day', 'threshold_days']`, `outcome='CONDITION_MUTATED'`, `fine_outcome='DECISION_CHANGED'`），而在 `012` 上从“超过90天（3个月）”提取出 `threshold_days=90` 且在 `approved_waiver=True` 下保持 `PRESERVED_DECISION`。其 10 例未通过样本的分布为：**4 例**（`qwen_plus:fconf-ins-009`、`deepseek:fconf-ins-007, 009, 017`）因模型在非豁免种子的 JSON `"waiver"` 槽位中填入合同背景规则“意外伤害无等待期”，被模板拼入问句后触发疾病/意外极性冲突；**3 例**（`qwen_plus:fconf-cross-014`、`deepseek:fconf-cross-014`、`deepseek:fconf-cross-011`）因多阈值 5 字段 JSON Schema（`clause, status, condition, day, waiver`）未单列 `threshold_days` 槽位及模板渲染短语未被领域正则提取；**2 例**（`fconf-ins-003`）为 `pa_h716` 未建模分支；**1 例**（`qwen_plus:fconf-cross-005`）因 JSON `"clause"` 槽位缩写为“膝关节置换”导致条款名未识别。这说明即便用确定性模板替代自由文本造句，未经规则验证的 JSON 槽位仍可能因合同规则回声、Schema 字段缺省与模板短语匹配缺口产生 12.5% 的未通过率。
3. **24 种子硬化后内部扩展留出集（48 对跨模型样本，10 个条款簇）上的表现**：在 6 份未见平安正式合同 PDF 与 12 个新参数组合的多阈值种子上，`slot_guard_repair` 在 `Qwen-Plus` 与 `DeepSeek` 上均取得 **24/24（100.0%）** 联合可用率，在配对 McNemar、24 种子聚类与 10 条款文档聚类下均优于 `condition_prompt`（39/48，$p=0.00390625 / 0.0078125 / 0.03125$）、160-Token `structured_json_rewrite`（37/48，$p=0.00097656 / 0.0078125 / 0.03125$）与 `structured_json_schema_render`（39/48，$p=0.00390625 / 0.03125 / 0.03125$），并相对 `structured_json_untruncated_320` 与 `generic_self_refine_budget_matched`（均为 43/48 = 89.58%，5W/0L，McNemar $p=0.0625$）保持一致的正向改善趋势。

---

## 第 6 章 推理与评测严格隔离的下游检索+符号规则决策流水线选型影响研究

### 6.1 严格隔离参考真值的推理接口 `predict(question, retrieved_clause, inference_config)` 与 5 个候选流水线设计

为确保下游候选系统的推理过程绝不读取隐藏的评测参考答案（`seed['reference_decision']`）或预期源文档 ID（`seed['policy_id']` / `seed['clause_id']`），我们在 `src/frozen_confirmation_and_rag_study.py` 中将下游推理严格封装为顶层纯函数 `predict(question, retrieved_clause, inference_config)`：
- **输入参数严格受限**：`predict()` 仅接收测试问句字符串 `question`、检索器返回的 Top-1 条款文档 ID `retrieved_clause` 以及部署时公开可得的条款规则配置 `inference_config`（系统名称、领域类别与 20 份保险条款规则表），**函数签名与闭包中完全不包含 `seed` 字典**；
- **条款不匹配主动澄清完全基于问句文本提取**：当 `S1_dense_rule_clarify` 或 `S2_bm25_rule_clarify` 判断检索到的 `retrieved_clause` 是否与用户所问险种一致时，仅将从 `question` 文本表面提取出的显式险种标识 `explicit_id` 与 `retrieved_clause` 比对（`clause_mismatch = (explicit_id != retrieved_clause)`），若问句丢失险种名称（`explicit_id is None`）或检索错配，则按合规规则主动输出 `CLARIFY`，并在检出条款 `retrieved_clause` 的规则下执行正向可达性枚举，绝不读取 `seed['reference_decision']`；
- **隐藏标签翻转不变量验证**：反事实隔离探针证实，固定问句与检索结果，将隐藏的 `seed['reference_decision']` 从 `WAIT_EXCLUDED` 强行翻转为 `WAIT_NOT_EXCLUDED` 时，`predict()` 与 `predict_system_from_text` 的输出 100% 保持为 `WAIT_EXCLUDED` 不变；
- **100% 纯 `BAAI/bge-small-zh-v1.5` 向量缓存覆盖与零词法重叠域外门控披露**：全部 80 条配对参考基准问句（48 条保险 + 32 条多阈值）、全部 160 条受控测试问句及全部 640 条云端合成问句均已预计算真实 `bge-small-zh-v1.5` 向量并存入 `data_v3/bge_embedding_cache.npz`；`retrieve_bge_top1` 移除了缓存缺失时的词法回退（若遇到未缓存问句直接抛出 `KeyError`），同时保留了针对完全无关乱码探针（如“苹果香蕉葡萄？”）的前置域外拒识门控——当查询与全部候选条款的字符二元 BM25 最高分 $\le 0$（零词法重叠）时直接返回 `(None, 0.0)` 触发 `CLARIFY`，对所有具有非零词法重叠的领域内问句则 100% 按 `bge-small-zh-v1.5` 余弦相似度排序（已在 `retrieval_backend_audit.lexical_zero_overlap_guard` 中显式披露）。

5 个候选检索+符号规则决策流水线定义如下：
1. **`S1_dense_rule_clarify`（密集检索 `bge_small_zh` + 文本槽位提取与合规主动澄清）**：对测试问句 $q$ 查表获取真实 `bge_small_zh` 向量并检索 Top-1 条款 $\hat{d}$，当 `explicit_id == retrieved_clause`、语法确定且在 $\hat{d}$ 下可达裁决唯一时输出该裁决，否则主动输出 `CLARIFY`；
2. **`S2_bm25_rule_clarify`（字符二元 `bm25_char2` 检索 + 文本槽位提取与合规主动澄清）**：对测试问句 $q$ 运行字符二元 `bm25_char2` 检索 Top-1 条款 $\hat{d}$，并执行相同的纯文本槽位提取与主动澄清逻辑；
3. **`S3_dense_aggressive_deny`（密集检索 `bge_small_zh` + 激进拒赔/不予通过策略）**：对 $q$ 运行 `bge_small_zh` 检索 Top-1 条款，在条件缺失或条款不匹配时**从不发起 `CLARIFY` 追问**，默认输出“等待期免责拒赔 / 不予通过”（`WAIT_EXCLUDED` / `EXCLUDED`）；
4. **`S4_dense_aggressive_approve`（密集检索 `bge_small_zh` + 激进赔付/予以通过策略）**：对 $q$ 运行 `bge_small_zh` 检索 Top-1 条款，在条件缺失或条款不匹配时从不发起追问，默认输出“正常赔付 / 予以通过”（`WAIT_NOT_EXCLUDED` / `ELIGIBLE`）；
5. **`S5_dense_threshold_echo_heuristic`（密集检索 `bge_small_zh` + 表面数字阈值比较启发式）**：对 $q$ 运行 `bge_small_zh` 检索 Top-1 条款，当问句缺少关键条件时不发起追问，仅根据问句表面提取到的天数与检出文档主等待期（或 30 天默认阈值）做简单大小比较进行二元猜测。

在由 40 个完整条件种子问句与 40 个受控缺失关键槽位的需澄清问句构成的 **80 条配对参考测试集（Oracle Ground-Truth Benchmark）**上，5 个候选流水线的真实准确率为：`S1_dense_rule_clarify` 为 **97.50%（78/80）**，`S2_bm25_rule_clarify` 为 **93.75%（75/80）**，`S5_dense_threshold_echo_heuristic` 为 **41.25%（33/80）**，`S3_dense_aggressive_deny` 为 **35.00%（28/80）**，`S4_dense_aggressive_approve` 为 **31.25%（25/80）**。

### 6.2 “伪金标”标签污染导致的下游系统排名反转实证

表 6.1 对比了在 80 题配对参考真值基准与各合成基准构建策略（基于全部 40 个确认集种子，`combined_40_seeds`）下，5 个候选流水线的表观准确率、相对真实系统排序的等级相关系数（Kendall's $\tau$ 与 Spearman's $\rho$）以及真实选型遗憾（Selection Regret，定义为真实最优系统准确率 97.50% 减去所选系统真实准确率）：

| 评测基准构建策略 | 生成模型 | 保留样本数 / 场景覆盖率 | 基准标签污染率 | 各候选系统表观得分 (`S1` / `S2` / `S3` / `S4` / `S5`) | 评测选出的冠军系统 | 相对真实排序 Kendall's $\tau$ / Spearman's $\rho$ | 真实选型遗憾 (Regret $\downarrow$) |
|---|---|---:|---:|---|---|---:|---:|
| **Oracle 配对参考基准 (40 完备 + 40 需澄清)** | 标准参考集 | 80 / 100.0% | **0.0%** | **97.5%** / 93.8% / 35.0% / 31.2% / 41.2% | **`S1_dense_rule_clarify`** | **1.0000 / 1.0000** | **0.0000** |
| 确定性完备种子直通 (`seed_passthrough`) | 确定性种子 | 40 / 100.0% | **0.0%** | **95.0%** / 87.5% / 65.0% / 57.5% / 80.0% | **`S1_dense_rule_clarify`** | **1.0000 / 1.0000** | **0.0000** |
| 未验证口语改写 (`ordinary_style`) | `Qwen-Plus` | 40 / 100.0% | **100.0%** | 0.0% / 0.0% / **50.0%** / 50.0% / 50.0% | `S3_dense_aggressive_deny` (错误) | **-0.7746 / -0.8660** | **0.6250** |
| 未验证口语改写 (`ordinary_style`) | `DeepSeek` | 40 / 100.0% | **97.5%** | 0.0% / 2.5% / **50.0%** / 50.0% / 50.0% | `S3_dense_aggressive_deny` (错误) | **-0.8367 / -0.8944** | **0.6250** |
| 未验证条件提示 (`condition_prompt`) | `Qwen-Plus` | 40 / 100.0% | 7.5% | 92.5% / 90.0% / 67.5% / 60.0% / 80.0% | `S1_dense_rule_clarify` | 1.0000 / 1.0000 | 0.0000 |
| 未验证条件提示 (`condition_prompt`) | `DeepSeek` | 40 / 100.0% | 17.5% | 82.5% / 77.5% / 60.0% / 60.0% / 72.5% | `S1_dense_rule_clarify` (低估 15.0%) | 0.9487 / 0.9747 | 0.0000 |
| 未验证 JSON 改写 (`structured_json_rewrite`) | `Qwen-Plus` | 40 / 100.0% | 17.5% (截断+篡改) | 92.5% / 90.0% / 67.5% / 60.0% / 80.0% | `S1_dense_rule_clarify` (含 7 题残句) | 1.0000 / 1.0000 | 0.0000 |
| 未验证 JSON 改写 (`structured_json_rewrite`) | `DeepSeek` | 40 / 100.0% | 2.5% (截断) | 97.5% / 92.5% / 65.0% / 62.5% / 80.0% | `S1_dense_rule_clarify` | 1.0000 / 1.0000 | 0.0000 |
| 未验证等预算自检 (`generic_self_refine`) | `Qwen-Plus` | 40 / 100.0% | 10.0% | 90.0% / 87.5% / 65.0% / 60.0% / 77.5% | `S1_dense_rule_clarify` | 1.0000 / 1.0000 | 0.0000 |
| 未验证等预算自检 (`generic_self_refine`) | `DeepSeek` | 40 / 100.0% | 15.0% | 85.0% / 80.0% / 60.0% / 62.5% / 75.0% | `S1_dense_rule_clarify` | 0.8000 / 0.9000 | 0.0000 |
| **仅验证过滤 (`verify_only_filter`)** | `Qwen-Plus` | 37 / 92.5% (丢 3 题) | **0.0%** | **97.3%** / 94.6% / 67.6% / 62.2% / 81.1% | **`S1_dense_rule_clarify`** | **1.0000 / 1.0000** | **0.0000** |
| **仅验证过滤 (`verify_only_filter`)** | `DeepSeek` | 33 / 82.5% (丢 7 题) | **0.0%** | **100.0%** / 93.9% / 66.7% / 63.6% / 75.8% | **`S1_dense_rule_clarify`** | **1.0000 / 1.0000** | **0.0000** |
| **槽位守卫修复 (`slot_guard_repair`)** | `Qwen-Plus` | **39 / 97.5%** | **0.0%** | **97.4%** / 94.9% / 66.7% / 61.5% / 79.5% | **`S1_dense_rule_clarify`** | **1.0000 / 1.0000** | **0.0000** |
| **槽位守卫修复 (`slot_guard_repair`)** | `DeepSeek` | **39 / 97.5%** | **0.0%** | **100.0%** / 94.9% / 66.7% / 64.1% / 79.5% | **`S1_dense_rule_clarify`** | **1.0000 / 1.0000** | **0.0000** |

### 6.3 选型反转机制剖析与验证结论

表 6.1 反映了合成基准评测中的三项现象：
1. **未验证口语改写导致澄清系统被判低分、默认拒赔系统排在首位**：在 `ordinary_style` 下，97.5%–100.0% 的问句因遗漏险种标题与关键条件而退化为 `CLARIFY` 问句。在沿用原种子二元标签的评分方式下，检出条件缺失并输出 `CLARIFY` 的 `S1_dense_rule_clarify` 与 `S2_bm25_rule_clarify` 表观得分仅为 **0.0%–2.5%**；而在条件缺失时一律输出拒赔的 `S3_dense_aggressive_deny`（两模型上均为 **50.0%**）排在第一，造成 **0.6250** 的选型遗憾与负相关排序（`Qwen-Plus` 上 $\tau = -0.7746, \rho = -0.8660$；`DeepSeek` 上 $\tau = -0.8367, \rho = -0.8944$）。
2. **密集检索（`S1`）与词法检索（`S2`）在口语化改写上的表现差异**：`S1` 与 `S2` 直接在合成问句文本上执行 `bge_small_zh` 与 `bm25_char2` 检索。当模型把条款标准术语改写为口语表达时，`bge_small_zh` 保持了较高的 Top-1 条款命中率，因此在 `slot_guard_repair` 上 `S1`（97.4%–100.0%）高于 `S2`（94.9%），与参考基准上的排序一致。
3. **验证过滤与槽位修复恢复下游参考排序**：通过 `slot_guard_repair` 修复缺失与篡改槽位并过滤 1 例 `PARSE_UNCERTAIN` 样本后，在 `Qwen-Plus` 与 `DeepSeek` 上均将放行池标签污染率降至 **0.0%**，保留了 **97.5%（39/40）** 的场景覆盖率，并恢复了参考系统排序（Kendall's $\tau = 1.0000$，Spearman's $\rho = 1.0000$，选型遗憾 0.0000）。

---

## 第 7 章 总结、局限性与未来展望

### 7.1 全文工作总结

本文针对中文商业保险与多阈值规则场景下 RAG 合成评测基准中的决策漂移问题，研究了条款约束下的合成问句决策保持验证与槽位守卫修复方法。主要结论如下：
1. **表面流畅度与检索 Top-1 不能替代决策保持验证**：跨 26 份官方保险条款与多阈值规则的实验表明，无约束口语改写在保持较高检索命中率的同时，使 55%–100% 的合成问句发生关键条件遗漏或阈值篡改。
2. **分离 `PARSE_UNCERTAIN` 与 `CLARIFY_REQUIRED` 并结合格式完整性检查可控制放行池误差**：在 120 例构造性诊断集与 400 条生成记录的纯槽位语义核查上，验证器未出现语义误放行（FP = 0）；联合可用性核查中的 6 例假阳性均来自 160-Token `structured_json_rewrite` 的半句截断，而 `slot_guard_repair`（$n=80$）上的误放行为 0 例，召回率为 98.73%（78/79）。
3. **符号槽位守卫修复（`slot_guard_repair`）提高了确认集与扩展集上的联合可用率**：在 40 种子确认集上，`slot_guard_repair` 的联合可用率为 **97.5%（78/80）**，优于 `condition_prompt`、等预算 `generic_self_refine` 及 `structured_json_schema_render`（均为 87.5%，70/80，McNemar $p=0.0078125$），以及 160-Token `structured_json_rewrite`（88.75%，71/80，McNemar $p=0.015625$）。相较 320-token 结构化基线 `structured_json_untruncated_320`（91.25%，73/80），结果呈正向改善，但差异未达到统计显著性（McNemar $p=0.0625$）。在 24 种子内部硬化后扩展留出集上，`slot_guard_repair` 的联合可用率达到 **100.0%（48/48）**。
4. **避免未验证合成标签导致的下游流水线排序反转**：在隔离参考真值的推理接口 `predict(question, retrieved_clause, inference_config)` 与预计算 BGE 向量缓存下，验证过滤与槽位修复将评测集标签污染从 97.5%–100.0% 降至 **0.0%**，使下游 5 个候选系统的排序从 $\tau \in [-0.8367, -0.7746]$ 恢复至 **$\tau = 1.0000$（选型遗憾 0.0000）**。

### 7.2 适用边界与方法局限性

1. **半自动条款规则编译的工程耗时与复杂责任边界**：如第 5.1 节如实披露，前 20 份商业保险合同的规则抽取与模式编译平均耗时 **11.72 分钟/份**（20 份共 234.5 分钟），且该数据为论文作者自身的工程编译记录估算而非多受试者计时实验。此外，当前形式化验证器聚焦于等待期、时间窗口与布尔豁免规则，尚未覆盖跨年度累计免赔额阶梯扣减、既往症分级责任比例核算及长篇医疗目录外用药除外责任的自动编译。
2. **共享规则库双实现一致性核查与真实多专家盲评的差距**：第 5.2 节对 400 条生成记录的第三层核查采用了正向可达性评估器（`evaluate_independent_text_semantics`），虽然从代码上切断了对 `r['outcome']` 的读取并修复了同等待期区间内数值偏离（`FACT_MUTATED_SAME_DECISION`）的裁决判定，但仍与被测验证器共享手工编译的底层条款规则表、且缺乏外部多标注员双盲人工标注日志，因此必须诚实定位为**共享规则库双实现一致性核查（`shared_rule_base_dual_implementation_consistency_check`）**而非外部人工真值审核。
3. **正则槽位提取器的语法辖域、版本演进与扩展留出集边界**：120 例黄金诊断集表明，当问句使用“两个月”“三周”或日历日期差值表达经过时间，或涉及未建模的特定器官/疾病分支（如 `pa_h716` 的腺样体肥大/疝气 90 日 vs. 其他疾病 30 日等待期）时，解析器需依赖 `detect_parse_uncertainty` 进行保守拦截；同时，我们将修复括号内联更正与多值冲突后的版本明确区分标识为 `v3.1-posthoc-hardened`，并通过额外构建的 24 种子未触碰扩展留出集（`v3.1-untouched-holdout`）单独检验了硬化后的泛化能力。需要诚实指出的是，该 24 种子扩展留出集由同仓库作者在冻结 `v3.1-posthoc-hardened` 后从本地前期缓存的平安条款池（6 份未见 PDF）及现有 4 个多阈值规则模板的新参数组合中构建，属于**硬化后内部未见种子扩展评测**，而非由外部独立第三方机构盲采集的外部基准；同时，由于确认集与留出集的条款簇数量有限（分别为 15 个与 10 个），条款文档级聚类符号翻转检验在多重比较校正后的统计功效受限，相关结论已按点估计优势与边缘显著趋势如实报告。
4. **受控下游流水线与开放式生成 RAG 的区别**：第 6 章评测的 5 个候选系统为严格隔离参考真值的“真实检索器（`bge_small_zh` / `bm25_char2`）+ 符号规则决策器”流水线（`predict(question, retrieved_clause, inference_config)`），旨在受控隔离评测标签污染对系统排名的影响，尚未直接评测以大模型自由生成长文本作为最终回答器（End-to-End LLM Generator）时的开放域答案忠实度。
5. **历史社区语料的固有局限**：附录 A 使用的 `baoxianzhidao_filter.csv` 源自历史百度知道问答，答案属于社区弱标签而非保险精算裁决，且存在潜在预训练暴露风险，故仅用作语料级检索排序与抽样混杂诊断。

### 7.3 未来研究方向

未来工作将沿三个方向推进：(1) 引入确定性日历算术与时间单位归一化算子（将“周/月/具体起止日期”精确折算为天数区间），进一步压降 `PARSE_UNCERTAIN` 的保守拒绝率；(2) 探索“大模型辅助起草规则 Schema + 人类精算师复核 + 形式化求解器验证”的半自动条款编译流水线，开展多受试者外部计时实验，并将验证范围扩展至免赔额与赔付比例核算；(3) 基于本仓库已开源的双盲网页标注套件（`reviewer_A.html` / `reviewer_B.html`），组织外部核保与理赔人员完成大规模多标注者盲评，并将下游评测扩展至端到端大模型生成式 RAG 问答系统。

---

## 参考文献

[1] Hossein A. Rahmani, Nick Craswell, Emine Yilmaz, Bhaskar Mitra, and Daniel Campos. 2024. Synthetic Test Collections for Retrieval Evaluation. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024)*, pages 2647–2651.  
[2] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. In *Proceedings of NAACL 2024*, pages 338–354.  
[3] Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang, Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework. In *Proceedings of ACL 2025 (Volume 1: Long Papers)*, pages 8520–8544.  
[4] Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin, Liane Lewin-Eytan, and Yoelle Maarek. 2025. Generating Q&A Benchmarks for RAG Evaluation with DataMorgana. In *Proceedings of ACL 2025 (Volume 6: Industry Track)*.  
[5] Jonas van Elburg, Peter van der Putten, and Maarten Marx. 2025. Can we Evaluate RAGs with Synthetic Data? *arXiv preprint arXiv:2508.11758*.  
[6] Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. 2023. Prediction-Powered Inference. *Science*, 382(6671):669–674.  
[7] Minwei Feng, Bing Xiang, Michael R. Glass, Lidan Wang, and Bowen Zhou. 2015. Applying Deep Learning to Answer Selection: A Study and An Open Task. In *Proceedings of IEEE ASRU 2015*, pages 813–820.  
[8] Filip J. Kucia and Barbara M. Gawlik. 2026. Beyond Benchmark Scores: How Synthetic and Authentic Query Distributions Diverge in RAG Evaluation. *arXiv preprint arXiv:2609.14579 (CIKM 2026 Short Research Paper)*.  
[9] Weiping Fu, Bifan Wei, Jianxiang Hu, Zhongmin Cai, and Jun Liu. 2024. QGEval: Benchmarking Multi-dimensional Evaluation for Question Generation. In *Proceedings of EMNLP 2024*.  
[10] Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer Singh, Tim Rocktäschel, Mike Sheldon, Guillaume Bouchard, and Sebastian Riedel. 2018. Interpretation of Natural Language Rules in Conversational Machine Reading. In *Proceedings of EMNLP 2018*, pages 2087–2097.  
[11] Tong Zhang, Peixin Qin, Yang Deng, Chen Huang, Wenqiang Lei, Junhong Liu, Dingnan Jin, Hongru Liang, and Tat-Seng Chua. 2024. CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models. In *Proceedings of ACL 2024 (Volume 1: Long Papers)*.  
[12] Jiale Zhao, Ke Fang, and Lu Cheng. 2026. When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification. In *Findings of ACL 2026*, pages 17120–17140.  
[13] Abhishek Divekar. 2026. Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference. *arXiv preprint arXiv:2606.05308 (ACL 2026 GEM Workshop)*.  
[14] Lu Chen, Ruqing Zhang, Jiafeng Guo, Yixing Fan, and Xueqi Cheng. 2024. Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework. In *Findings of EMNLP 2024*.  
[15] Jing Ding, Kai Feng, Binbin Lin, Jiarui Cai, Qiushi Wang, Yu Xie, Xiaojin Zhang, Zhongyu Wei, and Wei Chen. 2025. InsQABench: Benchmarking Chinese Insurance Domain Question Answering with Large Language Models. *arXiv preprint arXiv:2501.10943*.  
[16] Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, and Ming-Wei Chang. 2023. Promptagator: Few-shot Dense Retrieval From 8 Examples. In *Proceedings of ICLR 2023*.  
[17] Vitor Jeronymo, Luiz Bonifacio, Hugo Abonizio, Marzieh Fadaee, Roberto Lotufo, Jakub Zavrel, and Rodrigo Nogueira. 2023. InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval. *arXiv preprint arXiv:2301.01820*.  
[18] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. 2024. RAGAs: Automated Evaluation of Retrieval Augmented Generation. In *Proceedings of EACL 2024: System Demonstrations*.

---

## 附录 A：社区问答语料级检索器选型与校准诊断

本附录报告公开保险社区问答语料 `baoxianzhidao_filter.csv`（3,746 条唯一回答、3,725 个防泄漏连通分量：750 条真实校准提问与 2,994 条保留真实测试提问）上的检索器选型与统计校准诊断结果：
1. **全量真实测试集上的密集检索优势与零截断 `Full-MaxP` 复核**：在 2,994 条真实测试提问上，`bge_small_zh` 显著优于混合检索 `rrf_bm25_bge`（MRR@10 **0.5216 vs. 0.5030**，配对差值 $+0.0187$，95% Bootstrap CI $[+0.0077, +0.0296]$）与词法检索 `bm25_char2`（0.3817）。对 509 条超长文档切分 1,271 个滑动窗口执行零截断 `Full-MaxP` 评测，仍确认 `bge_small_zh` 领先（**0.5165 vs. 0.5007**，$\Delta = +0.0158$，95% CI $[+0.0049, +0.0267]$）。
2. **源文档子采样方差与同源词法复制分解**：在 `1.5B`（$n=209$）与 `Qwen-Plus`（$n=153$）对齐的源文档子集上，真实测试点估计已因小样本抽样方差偏向 `rrf_bm25_bge`（$-0.0279$ 与 $-0.0312$，95% CI 跨零）；固定配对文档后的双重差分 $\Delta\Delta_{\text{RRF}-\text{BGE}}$ 在全部 4 组设定下 95% CI 均跨零（$+0.0123$、$-0.0259$、$-0.0020$、$-0.0298$），证明小样本合成集上的排名翻转同时混杂了源文档子采样方差与词法重叠膨胀。
3. **边缘密度重加权负结果与小预算启发式候选集合暂缓选型**：边缘 TF-IDF 倾向性重加权（`synthetic_density_weighted`）在 $B \in \{25, 50, 100, 200\}$ 下错误率均为 100%；而基于 `real_only` 200 次配对 Bootstrap 的启发式候选集合暂缓选型规则在 $B=25$ 下将错误确定选型率从 60% 降至 **0.0%**（暂缓率 80%，平均候选集大小 2.60，最优系统经验包含率 100%）。

---

## 附录 B：26 份官方商业保险合同条款清单与 SHA256 溯源表

全部 26 份官方商业保险条款 PDF 的完整元数据、页码、包围盒坐标与作者工程编译耗时均记录于 `data_v3/sources.json`、`data_v3/heldout_clauses_sources.json`、`data_v3/frozen_confirmation_sources.json`、`data_v3/untouched_holdout_sources.json` 与 `data_v3/clause_rule_mapping_audit.json`：

| 序号 | 条款 ID | 划分阶段 | 官方条款名称 | 保险公司 | 等待期规则摘要 | 原件 SHA256 前缀 | 编译耗时 |
|---:|---|---|---|---|---|---|---:|
| 1 | `pa_p0781` | `dev_calibration_5` | 平安关爱相伴团体医疗保险（B款） | 平安 | 90日 (意外免, 无续保免) | `55a4e49a07eb...` | 11.0 min |
| 2 | `pa_h865` | `dev_calibration_5` | 平安员工及家属住院医疗保险 | 平安 | 30日 (意外免, 续保/获批免) | `112aea1a036c...` | 10.5 min |
| 3 | `cpic_86017` | `dev_calibration_5` | 太保互联网C款门急诊医疗保险 | 太平洋 | 7/15/25日 (意外不免, 续保/获批免) | `54aca8a0b54c...` | 15.0 min |
| 4 | `cpic_82859` | `dev_calibration_5` | 太保互联网个人一年期住院E款医疗保险 | 太平洋 | 30日 (意外免, 续保/获批免) | `8bfda0ffd63a...` | 11.0 min |
| 5 | `tk_group_2018` | `dev_calibration_5` | 泰康悦享环球（尊享版）高端医疗保险 | 泰康 | 30日 (意外免, 无续保免) | `0d207b7c2bf7...` | 11.5 min |
| 6 | `pa_2011_36` | `extension_set_4` | 平安住院安心医疗保险 | 平安 | 30/90日 (一般30日意外免; 癌症90日意外不免, 续保免) | `10916a68758a...` | 18.0 min |
| 7 | `pa_p0254` | `extension_set_4` | 平安住院医疗保险 | 平安 | 90日 (意外免, 续保/获批免) | `d6ad23849fec...` | 10.0 min |
| 8 | `pa_p0321` | `extension_set_4` | 平安肺癌疾病保险 | 平安 | 30日 (原位癌除外, 意外不免, 续保/获批免) | `e04e0b452858...` | 12.5 min |
| 9 | `cpic_93896` | `extension_set_4` | 附加（广西地区）学生幼儿住院医疗保险（2025版） | 太平洋 | 30日 (意外免, 无续保免) | `309d8f029fb8...` | 11.0 min |
| 10 | `pa_h716` | `frozen_confirmation_11` | 平安安康住院费用（B，2017）医疗保险 | 平安 | 30/90日 (其他疾病30日; 腺样体肥大/疝气等90日, 意外免) | `c6597d19f042...` | 17.5 min |
| 11 | `pa_2651` | `frozen_confirmation_11` | 平安安医保（易核版）医疗保险 | 平安 | 30日 (意外免, 续保/获批免) | `7fa649061fe0...` | 11.5 min |
| 12 | `pa_l0001` | `frozen_confirmation_11` | 平安安医保（星耀版）医疗保险 | 平安 | 30日 (意外免, 续保/获批免) | `85e18de2d813...` | 9.5 min |
| 13 | `pa_1714` | `frozen_confirmation_11` | 平安附加守护全能（2023）提前给付重大疾病保险 | 平安 | 90日 (意外免, 无续保免) | `4ca9f0ac3cd8...` | 10.0 min |
| 14 | `pa_1072` | `frozen_confirmation_11` | 平安e生保长期医疗保险（费率可调） | 平安 | 90日 (意外免, 续保/获批免) | `f043311c2257...` | 11.0 min |
| 15 | `pa_596` | `frozen_confirmation_11` | 平安e生保（尊享版）医疗保险 | 平安 | 30日 (意外免, 续保/获批免) | `88b784c61f52...` | 10.5 min |
| 16 | `cpic_107082` | `frozen_confirmation_11` | 太保个人中高端C款医疗保险（H2026） | 太平洋 | 30日 (意外免, 续保/获批免) | `98562a559e39...` | 11.0 min |
| 17 | `cpic_106702` | `frozen_confirmation_11` | 太保互联网母婴医疗保险 | 太平洋 | 15日 (意外免, 无续保免) | `fef1c0203561...` | 9.5 min |
| 18 | `cpic_60201` | `frozen_confirmation_11` | 附加团体疾病医疗保险（B款）（H2014） | 太平洋 | 30日 (意外不免, 续保/获批免) | `80050c1912de...` | 11.0 min |
| 19 | `cpic_79920` | `frozen_confirmation_11` | 深圳专属团体医疗保险 | 太平洋 | 30日 (意外免, 续保/获批免) | `b3deb2a5103f...` | 10.5 min |
| 20 | `cpic_85050` | `frozen_confirmation_11` | 心安·怡住院费用医疗保险（H2017A） | 太平洋 | 90日 (意外免, 无续保免) | `a16bd72ab6ac...` | 12.0 min |
| 21 | `pa_2650` | `untouched_holdout_6` | 平安e生保（悦享版）医疗保险 | 平安 | 30日 (意外免, 续保/获批免) | `9d93c137aac3...` | — (留出集) |
| 22 | `pa_1071` | `untouched_holdout_6` | 平安粤港澳大湾区深圳专属一年期重大疾病保险 | 平安 | 90日 (意外免, 续保/获批免) | `fe815af25df2...` | — (留出集) |
| 23 | `pa_1073` | `untouched_holdout_6` | 平安i药保B款特药费用医疗保险 | 平安 | 30日 (意外不免, 无续保免) | `cbf26584a59f...` | — (留出集) |
| 24 | `pa_595` | `untouched_holdout_6` | 平安e生保（加享版）医疗保险 | 平安 | 30日 (意外免, 续保/获批免) | `b72df94a33b2...` | — (留出集) |
| 25 | `pa_597` | `untouched_holdout_6` | 平安e生保（颐享版）医疗保险 | 平安 | 30日 (意外免, 续保/获批免) | `2d810dc3af77...` | — (留出集) |
| 26 | `pa_l0002` | `untouched_holdout_6` | 平安安医保e享长期医疗保险（费率可调）（互联网） | 平安 | 90日 (意外免, 续保/获批免) | `554884992560...` | — (留出集) |

---

## 附录 C：离线可复现命令与 35 项确定性单元测试说明

本仓库所有验证、重评分、双实现一致性核查、未触碰留出集评测与下游 RAG 选型实验均支持在无本地大模型、无外部 `../../work/` 目录、纯离线 CPU 环境（`HF_HUB_OFFLINE=1`）下毫秒/秒级精确复现：

```bash
# 1. 运行全部 35 项确定性单元测试（含推理解耦探针、同区间天数偏移、100% BGE 缓存、并发线程安全与未触碰留出集断言）
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. 依次运行 5 个自包含离线重评分与审计脚本（无需调用 LLM API，产物与版本库 100% 逐字节一致）
python3 src/analyze_v3.py
python3 src/cross_domain_study.py --score-only
python3 src/equal_budget_and_heldout_study.py --score-only
python3 src/build_item_semantic_audit.py
python3 src/frozen_confirmation_and_rag_study.py --score-only
```
