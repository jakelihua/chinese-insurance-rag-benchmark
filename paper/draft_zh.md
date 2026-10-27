# 合成查询能否保持保险核保理赔结论？

**作者：** Shunchi Wang（Independent Researcher，`jakelihua@gmail.com`）  
**代码与数据仓库：** [https://github.com/jakelihua/chinese-insurance-rag-benchmark](https://github.com/jakelihua/chinese-insurance-rag-benchmark)

---

## 摘要

检索增强生成（RAG）基准常通过大模型从源条款段落合成用户提问。在保险核保与理赔等规则密集型场景中，一次改写即便能将源条款排在检索榜首，仍可能漏掉或篡改决定拒赔与否的关键资格条件。本文基于9份公开披露的中文商业保险正式合同条款（5份开发集、4份扩展集，共144条经双反例见证认证的受控样本）与16组多阈值合成规则，研究源条款约束下的合成查询决策保持问题。我们区分表面槽位完整度与形式化结论可确定性：枚举缺失槽位的所有阈值等价区间，检验观测到的提问条件能否唯一锁定原参考结论、发生条件篡改，还是退化为必须追问澄清的歧义题。在开发集与扩展集保险条款上，无约束口语改写（涵盖三个模型尺度）虽然获得较高词法新颖度，但独立结论可确定率仅为 0%–45%；当隐去产品全名与保险公司名称后，9条款池上的密集检索 Top-1 准确率从 75.0% 降至 31.2%。显式条件守恒提示在开发集与补充产品简称及责任消歧后的扩展集保险条款上首轮即恢复了 95%–100% 的结论可确定率（二轮修复 0 次触发）；而在多阈值合成规则的 26 次等调用预算对照中，符号槽位守卫修复实现 75.0%（12/16）的结论保持率与 0/16 的解析器检出篡改，高于无解析器反馈的大模型自检基线（50.0%，8/16，残留 4/16 篡改；双侧精确 McNemar $p=0.125$）。

---

## 1 引言

构建垂直领域检索增强生成（RAG）评测基准时，工程团队通常缺乏经过脱敏与合规授权的大规模真实用户日志，因此普遍让指令微调大模型根据源文档段落反推合成提问 [1, 3, 4]。现有研究多从检索系统排序相关性或表面文本流畅度角度评估合成查询 [1, 5]。然而在商业健康险核保、理赔初审等受合同规则严格约束的场景中，检索命中源条款只是第一步：一个合格的评测提问还必须完整保留决定业务结论的**关键资格条件（Pivotal Eligibility Conditions）**。

一份医疗险合同对“等待期内出险是否免责”的判定，取决于多个相互作用的业务槽位：具体产品与条款版本（等待期从 7 天、15 天、25 天、30 天到 90 天不等）、投保状态（首次投保还是连续续保且获批免等待期）、出险距保单生效的天数、出险原因（意外伤害还是疾病），以及所申请的具体保障责任（例如同一合同中“一般住院医疗津贴”为 30 天等待期且意外伤害免等待期，而“癌症住院医疗津贴”为 90 天等待期且无意外豁免）。当大模型为了追求“简短、自然的口语提问”而省略产品简称、隐去申请责任类别，或把“生效第 20 天”改写为“刚买第二天”时，即便检索器仍能凭残存词块召回源条款，原提问的参考结论也已被破坏——要么从确定拒赔/赔付退化为必须追问澄清（`CLARIFY`）的歧义提问，要么发生反事实结论翻转。

本文将评测重心从检索分数相关性推进到**源条款约束下的决策保持验证（Clause-Grounded Decision Preservation）**。所有实验均在推理阶段评估现成预训练模型与嵌入模型，不对任何大模型或向量模型权重进行预训练或微调。本文围绕三个具体问题展开实证检验：

- **RQ1（口语改写下的条件漂移与检索盲区）**：在可追溯的官方保险合同条款上，普通口语化改写与显式条件守恒提示在多大程度上保持独立核保理赔结论？常规词法与密集检索指标能否识别丢失关键资格条件的退化提问？
- **RQ2（扩展合同条款、责任级等待期消歧与检索锚点依赖）**：当扩展到另外 4 份官方保险正式条款（在共享解析器中补充口语产品简称与同一合同内“一般住院医疗津贴 [30天]”和“癌症住院医疗津贴 [90天]”的责任消歧槽位 `benefit_id`，并去掉书名号与法条编号）时，统一解析器与可达性验证器表现如何？隐去产品名称后检索准确率如何变化？
- **RQ3（多阈值规则下的等调用预算修复对照）**：当单条规则同时包含多个数值阈值与布尔前提时，在严格相等的 LLM 调用次数预算下，基于符号槽位与可达性反馈的修复机制（`slot_guard_repair`）相比单通结构化 JSON 改写（`structured_json_rewrite`）和无符号反馈的大模型自检（`generic_self_refine`）有何表现与统计边界？

此外，为完整交代本研究从社区问答检索诊断演进至合同条款形式化验证的过程，我们在**附录 A** 中报告了公开保险社区问答语料（`baoxianzhidao_filter.csv`，3,746 条回答、2,994 条真实测试提问）上的检索器选型反转、源文档抽样混杂分解、探索性源重叠度门控以及小预算候选集合暂缓选型（Candidate-Set Abstention）诊断实验。

---

## 2 相关工作

### 2.1 合成检索基准与排名偏移分析
合成测试集已被广泛用于缓解信息检索与 RAG 系统的冷启动评测难题。Rahmani 等人 [1] 与 van Elburg 等人 [5] 分析了大模型合成测试集在检索系统排名上的相关性与生成偏差。Kucia 与 Gawlik [8] 在高校院系信息系统的 RAG 评测中对比了 1,851 条大模型合成问题与 322 条真实学生调研问题在密集检索、词法检索（BM25）及混合检索（RRF）上的表现，表明合成提问会高估绝对检索分并导致系统排名相对真实用户提问发生偏移。由于合成与真实查询之间的检索排名偏移已在上述文献中得到证实，本文将社区问答检索实验（附录 A）定位为剥离“源文档子采样方差”与“同源词法复制效应”的补充诊断。

### 2.2 场景驱动合成、问题质量评估与规则澄清
为使合成提问贴合领域结构，RAGEval [3] 从金融、医疗、法律文档中提取领域模式（Schema）以生成评测集，DataMorgana [4] 则通过参数化用户画像与问题分类学定制问答基准。在通用问题生成质量评估方面，QGEval [9] 提出了涵盖流畅度、清晰度、简洁度、相关性、一致性、可回答性与答案一致性等 7 个维度的评测基准。在规则遵循与模糊提问澄清方面，ShARC [10]、CLAMBER [11] 以及 AskBench [12] 重点研究**下游对话或问答助手**在面对信息缺失的用户提问时如何识别歧义并主动发起追问；RC-RAG [14] 从选择性预测与风险控制角度出发，利用反事实提示估计 RAG 置信度并在误导性上下文下拒绝回答。在保险垂直领域，InsuranceQA [7] 提供了早期社区问答对，InsQABench [15] 则构建了面向真实保险产品规则的大模型问答评测基准。

本文的研究切入点位于下游问答助手之前：我们聚焦于**上游评测数据合成阶段**，检验从已知规则条款生成的合成测试提问是否忠实保持了种子场景的完整资格条件与唯一可达结论，并通过显式枚举缺失槽位的正反补全见证（Counterexample Witnesses）将“非关键槽位省略但结论唯一保持”与“关键槽位丢失导致必须追问”严格区分开来。

### 2.3 小样本统计校准与预测驱动推断
预测驱动推断（PPI）[6]、ARES [2] 以及 PRECISE [13] 结合合成数据、大模型自动评判器与少量人工标注校准集，为 RAG 评测与排序构建统计有效的置信区间。当合成查询仅从语料库的一小部分源文档生成、且缺乏全库配对真实提问时，未配对收缩估计与边缘查询特征重加权难以消除文档子采样偏差与词法重叠膨胀；我们在附录 A 中如实报告了这些重加权负结果，并验证了小样本校准下的配对 Bootstrap 候选集合暂缓选型规则。

---

## 3 条款约束下的决策保持验证方法

### 3.1 合同资格条件帧与原子级条款判定基准
设受控保险条款集合为 $\mathcal{P}$。每份条款的等待期免责与豁免规则均锚定至官方 PDF 原件的物理页码、字符级包围盒坐标（`bbox_pdf_points`）与 SHA256 校验和。每个咨询场景表示为结构化条件帧 $z \in \mathcal{Z}$：
$$z = (\texttt{policy\_id}, \; \texttt{enrollment}, \; \texttt{policy\_day}, \; \texttt{cause}, \; \texttt{plan\_wait\_days}, \; \texttt{approved\_waiver}, \; \texttt{benefit\_id})$$
其中 $\texttt{enrollment} \in \{\texttt{first}, \texttt{renewal}\}$ 表示首次投保或续保/重新投保，$\texttt{policy\_day} \in \mathbb{N}^+$ 表示出险发生在保单生效第几天，$\texttt{cause} \in \{\texttt{disease}, \texttt{accident}\}$ 表示疾病或意外伤害，$\texttt{plan\_wait\_days}$ 为投保计划或责任对应的等待期天数，$\texttt{approved\_waiver} \in \{\text{True}, \text{False}\}$ 表示续保或重新投保时是否满足免等待期条件，$\texttt{benefit\_id} \in \{\texttt{general\_inpatient}, \texttt{cancer\_inpatient}\}$ 区分一般住院医疗津贴与癌症住院医疗津贴责任。

在本文显式界定的“等待期单项免责判定”范围内，每份合同条款对应一个确定性规则函数 $f: \mathcal{Z} \to \{\texttt{WAIT\_EXCLUDED}, \texttt{WAIT\_NOT\_EXCLUDED}\}$。特别地，在扩展集条款《平安住院安心医疗保险》（`pa_2011_36`，第 1 页第四条“保险责任”）中，合同对两项住院津贴责任规定了不同的等待期与豁免条件：第（一）项“一般住院医疗津贴”规定首次投保等待期为 **30 日**，续保或因意外伤害住院治疗无等待期（$\texttt{accident\_exempt}=\text{True}$）；第（二）项“癌症住院医疗津贴”规定首次投保等待期为 **90 日**，续保无等待期，且无意外伤害免等待期条款（$\texttt{accident\_exempt}=\text{False}$）。因此，当 $\texttt{policy\_id}=\texttt{pa\_2011_36}$、首次投保且生效第 45 天时，若申请一般住院医疗津贴（$\texttt{benefit\_id}=\texttt{general\_inpatient}$），因 $45 > 30$ 判定为 $\texttt{WAIT\_NOT\_EXCLUDED}$；若申请癌症住院医疗津贴（$\texttt{benefit\_id}=\texttt{cancer\_inpatient}$），因 $45 \le 90$ 判定为 $\texttt{WAIT\_EXCLUDED}$；在首次投保疾病出险且无同时豁免两项责任的续保免等待期条件、由等待期直接决定赔付结果的第 31–90 天场景下，若提问未指明申请哪项津贴（或将两项津贴混为一谈），则可达集合同时包含 `['WAIT_EXCLUDED', 'WAIT_NOT_EXCLUDED']`，被拒绝并归类为 `DEGRADED_TO_CLARIFY`（若已存在对两项责任均适用的明确续保免等待期约定，则省略责任类别不影响等待期结论）；若提问将一般住院津贴错配为 90 天等待期或将癌症住院津贴错配为 30 天等待期，则被直接判定为 `CONDITION_MUTATED`。此外，在数据集的 `atomic_rubrics` 字段中，我们将锚定至 PDF 原件页码与坐标框的合同规则依据（`source_type='pdf_contract_clause'`）与来自受控测试种子的场景设定参数（`source_type='synthetic_seed_frame'`，如具体出险天数 `policy_day=45`）显式分离，确保条款原文证据与合成场景参数边界清晰。

### 3.2 表面槽位完整度与无参考依赖的结论可确定性
给定完整种子帧 $z$ 及其参考结论 $y^* = f(z)$，当生成模型将种子提问 $q(z)$ 改写为合成提问 $q'$ 时，槽位提取器 $g(q')$ 从 $q'$ 的表面文本中提取观测条件帧 $\hat{z} = g(q')$。为贴合真实业务咨询并消除合成文本中的“书名号剧透”，我们在生成后与评测前统一剥离书名号（`《》`）与显式法条编号前缀，并在共享解析器中支持口语产品简称（如“平安住院安心医疗险”“太保C款门急诊险”“太平洋广西学生幼儿住院医疗险”等）与自然时间表述（如“30天等待期”“等待期30天”）。

如果仅检查 $q'$ 是否逐一保留了种子帧的所有非空字段（记为全槽位表面匹配 `all_slots_preserved`），会把“省略了不影响最终结论的冗余槽位”错误判为不可回答。为此，我们设计了不读取参考种子帧 `seed_frame` 的可达性枚举器 `enumerate_admissible_decisions`：

1. **全槽位存在且未篡改**：若 $\hat{z}$ 已包含目标条款所需的全部架构槽位且与 $z$ 无冲突，则唯一可达集合为 $\mathcal{A}(\hat{z}) = \{f(\hat{z})\}$。
2. **存在缺失槽位时的阈值区间完备枚举**：若 $\hat{z}$ 缺失部分槽位（例如未说明出险天数、未说明投保计划档位或未指明保单名称），枚举器在不访问真实种子帧 $z$ 的前提下，遍历所有候选保单、合法投保状态、原因、责任类型以及覆盖所有等待期阈值内外的代表性天数区间 $\texttt{policy\_day} \in \{5, 7, 10, 15, 20, 25, 30, 45, 60, 90, 120\}$，计算所有相容完整帧 $z' \in \text{Completions}(\hat{z})$ 的结论集合：
   $$\mathcal{A}(\hat{z}) = \left\{ f(z') \;\middle|\; z' \in \text{Completions}(\hat{z}) \right\}$$

据此，合成提问 $q'$ 被严格划分为三类互斥状态：
- **结论保持（`PRESERVED_DECISION`）**：已提取槽位与种子帧无矛盾，且所有合法补全导出唯一结论并等于参考结论，即 $\mathcal{A}(\hat{z}) = \{y^*\}$。这既涵盖全槽位完整保留的情形，也涵盖非关键槽位省略但结论依然唯一确定的情形（例如续保免等待期已获批，或出险天数已超过最长等待期档位时省略出险原因）。
- **退化为必须追问（`DEGRADED_TO_CLARIFY`）**：已提取槽位虽无篡改，但因丢失关键槽位导致合法补全同时包含排除与不排除两种相反结论，即 $|\mathcal{A}(\hat{z})| > 1$（或无法唯一锁定目标条款规则）。在构建基准集的 `missing_critical` 样本时，每道题均强制附带两个导出相反结论的具体补全实例作为**双反例见证（Counterexample Witnesses）**。
- **条件篡改 / 反事实翻转（`CONDITION_MUTATED` / `COUNTERFACTUAL_FLIP`）**：已提取槽位与种子帧发生实质冲突（如将“首次投保”改为“续保”、将“生效第 12 天”改为“生效第 60 天”），或导出与参考结论相反的唯一结论 $\mathcal{A}(\hat{z}) = \{\neg y^*\}$。

### 3.3 对比生成策略与等调用预算修复
我们在受控实验中共对比 7 种生成与修复设定：
1. **种子直通基线（`seed_passthrough`，0 次 LLM 调用）**：直接使用剥离书名号与条款编号后的受控种子提问，作为验证解析器与检索上限的参考锚点。
2. **普通口语改写（`ordinary_style`，每题 1 次调用）**：提示模型“把原问题改写成用户口语化提问，简短自然，保持原本意图”，并在输出端剥离书名号与法条编号。
3. **显式条件守恒提示（`condition_prompt`，每题 1 次调用）**：在提示词中明确要求保留产品名称（或清晰简称）、投保状态、生效天数、出险原因、申请责任范围及计划等待期天数，只允许口语化换述，禁止增删条件或将未知补全为已知。
4. **单通结构化 JSON 改写（`structured_json_rewrite`，每题 1 次调用）**：要求模型在单次生成中先输出一行提取各条件槽位的 JSON 对象，换行后再输出以 `QUESTION:` 开头的口语化提问（最大生成上限 160 tokens）。
5. **等调用预算无反馈大模型自检（`generic_self_refine_budget_matched`）**：先以 `condition_prompt` 生成初稿；在不运行符号槽位解析器的前提下，为与槽位守卫修复消耗完全相同的二轮调用次数，对触发二轮预算的样本提示模型自行核对原题与初稿是否遗漏或篡改条件并输出修正版。
6. **全量二轮无反馈大模型自检（`generic_self_refine_all`，每题固定 2 次调用）**：对全部 16 组种子无条件执行“初稿生成 + 大模型自我核对修正”两轮调用（共 32 次调用）。
7. **符号槽位守卫闭环修复（`slot_guard_repair`）**：先以 `condition_prompt` 生成初稿（第 1 次调用），立即运行确定性槽位提取器 $g(q')$ 与可达性枚举器 `enumerate_admissible_decisions`。仅当检测到缺失关键槽位、条件篡改或结论不可确定时，自动将具体缺失/冲突字段名拼入修复提示词发起第 2 次生成调用；若初稿已通过形式化验证则直接放行（0 次额外调用）。

---

## 4 实验设置与数据来源

### 4.1 九份官方保险条款（5份开发集 + 4份扩展集）与多阈值规则集
如表 1 所示，我们从平安、太平洋、泰康三家保险集团公开披露渠道采集 9 份正式保险合同条款 PDF（5 份用于开发集规则与初始解析器构建，4 份作为补充口语产品简称与责任级等待期消歧槽位 `benefit_id` 后的扩展验证集），同时构造 4 段受临床指南与劳动法启发的多阈值合成规则用于压力测试。需要如实说明的是，扩展集 4 份条款并非在完全未改动的旧开发集解析器上进行的零样本盲测，而是在共享解析器 `extract_fields_surface_normalized` 中增补了这 4 份产品的口语简称词表与 `pa_2011_36` 的责任区分槽位后开展的**补充规则精化后回顾评估（Retrospective Post-Refinement Evaluation）**。出于版权合规，代码仓库不重分发原始商业合同 PDF，仅保存文件 SHA256、页码、字符级坐标框（`bbox_pdf_points`）及短摘录供核验。

| 数据集划分 | 条款代号 | 正式条款名称 | 证据页码 | 研究责任范围（第四条 保险责任） | 等待期（天） | 意外豁免 | 续保免等待期 | PDF SHA256（前12位） |
|---|---|---|---:|---|---|---:|---:|---|
| 开发集 (5份) | `pa_p0781` | 平安关爱相伴团体医疗保险（B款） | 第4页 | 特定疾病或轻症疾病医疗 | `[90]` | 是 | 否 | `55a4e49a07eb` |
| 开发集 (5份) | `pa_h865` | 平安员工及家属住院医疗保险 | 第3页 | 一般住院医疗 | `[30]` | 是 | 是 | `112aea1a036c` |
| 开发集 (5份) | `cpic_86017` | 太保互联网C款门急诊医疗保险 | 第4页 | 2.5.2互联网药品费用医疗 | `[7, 15, 25]` | 否 | 是 | `54aca8a0b54c` |
| 开发集 (5份) | `cpic_82859` | 太保互联网个人一年期住院E款医疗保险 | 第3页 | 一般住院医疗 | `[30]` | 是 | 是 | `8bfda0ffd63a` |
| 开发集 (5份) | `tk_group_2018` | 泰康悦享环球（尊享版）高端医疗保险 | 第4页 | 非精神心理一般住院医疗 | `[30]` | 是 | 否 | `0d207b7c2bf7` |
| **扩展集 (4份)** | `pa_2011_36` | 平安住院安心医疗保险 | 第1页 | （一）一般住院医疗津贴 / （二）癌症住院医疗津贴 | `[30, 90]` | 一般:是 / 癌症:否 | 是 | `10916a68758a` |
| **扩展集 (4份)** | `pa_p0254` | 平安住院医疗保险 | 第1页 | 疾病住院医疗保险金 | `[90]` | 是 | 是 | `d6ad23849fec` |
| **扩展集 (4份)** | `pa_p0321` | 平安肺癌疾病保险 | 第1页 | 初次发生原发性肺癌（不包括原位癌） | `[30]` | 否 | 是 | `e04e0b452858` |
| **扩展集 (4份)** | `cpic_93896` | 附加（广西地区）学生幼儿住院医疗保险（2025版） | 第1、3页 | 一、住院医疗费用保险金（第6条不保证续保，仅测首投） | `[30]` | 是 | 是（首投评测） | `309d8f029fb8` |

- **受控基准样本规模**：5 份开发集条款包含 20 组种子、80 条受控样本（`complete`、`surface`、`missing_critical`、`counterfactual` 各 20 条）；4 份扩展集条款包含 16 组种子、64 条受控样本；多阈值合成规则集（涵盖靶向药医保报销 `med_egfr_tki`、术后静脉血栓预防 `med_vte_prophylaxis`、上下班工伤认定 `law_work_injury`、竞业限制补偿解除 `law_noncompete` 四类多条件交集规则）包含 16 组种子、64 条受控样本。
- **推理环境与复现控制**：本地模型采用 `Qwen/Qwen2.5-1.5B-Instruct`（MPS `float32/eager` 贪心解码，规避 `float16` 下溢问题），云端对比模型采用 `Qwen-Plus` 与 `Qwen-Max`（`temperature=0`）。检索评估在 9 份正式条款池（以及多阈值 4 段规则池）上同时运行字符二元 `bm25_char2` 与 `BAAI/bge-small-zh-v1.5`，并测试标准检索（`standard_retrieval`）与掩去所有产品名称及保险公司品牌词的标题掩码检索（`title_masked_retrieval`）。

---

## 5 实验结果与分析

### 5.1 开发集保险条款：跨模型条件漂移与表面槽位 vs. 结论可确定性
表 2 汇总了 5 份开发集保险条款（20 组完整种子）在本地 `Qwen2.5-1.5B-Instruct`、云端 `Qwen-Plus` 与 `Qwen-Max` 三个模型尺度下的配对改写结果（共 120 条生成提问）。

| 模型尺度 | 改写提示策略 | 全槽位表面保留率 (`all_slots_preserved`) | 独立结论可确定率 (`standalone_decision`) | 已知源保单下结论可确定率 (`granted_policy`) | 退化为追问 (`CLARIFY`) | 数值/条件篡改 |
|---|---|---:|---:|---:|---:|---:|
| `Qwen2.5-1.5B` | `ordinary_style` | 30.0% (6/20) | **30.0%** (6/20) | 55.0% (11/20) | 55.0% (11/20) | 15.0% (3/20) |
| `Qwen2.5-1.5B` | `condition_prompt` | 95.0% (19/20) | **95.0%** (19/20) | 100.0% (20/20) | 5.0% (1/20) | 0.0% (0/20) |
| `Qwen-Plus` | `ordinary_style` | 0.0% (0/20) | **0.0%** (0/20) | 75.0% (15/20) | 100.0% (20/20) | 0.0% (0/20) |
| `Qwen-Plus` | `condition_prompt` | 95.0% (19/20) | **100.0%** (20/20) | 100.0% (20/20) | 0.0% (0/20) | 0.0% (0/20) |
| `Qwen-Max` | `ordinary_style` | 20.0% (4/20) | **45.0%** (9/20) | 80.0% (16/20) | 55.0% (11/20) | 0.0% (0/20) |
| `Qwen-Max` | `condition_prompt` | 100.0% (20/20) | **100.0%** (20/20) | 100.0% (20/20) | 0.0% (0/20) | 0.0% (0/20) |

实验揭示了三个核心现象：
1. **扩大模型参数规模不能自动阻止普通口语改写丢失关键资格条件**：在 `ordinary_style` 下，`Qwen-Plus` 与 `Qwen-Max` 为了生成地道的简短客服提问，分别在 20/20（100%）与 11/20（55%）的样本中省去了险种名称或计划等待期档位（如改写为“我刚投保 20 天生病住院能报销吗？”），导致独立提问无法锁定适用条款而退化为 `CLARIFY`。即便假设评测系统已知目标保单（`granted_policy`），仍有 20%–45% 的样本因漏掉投保档位或出险时间而无法得出唯一结论。
2. **表面槽位缺失不等于结论不可确定**：在 `Qwen-Max` 的 `ordinary_style` 输出中，全槽位表面保留率为 20.0%（4/20），但独立结论可确定率达到 **45.0%（9/20）**；类似地，`Qwen-Plus` 在 `condition_prompt` 下全槽位保留率为 95.0%（19/20），而独立结论可确定率达到 **100.0%（20/20）**。其原因在于部分样本虽然省略了“出险原因（疾病或意外）”，但提问保留了“连续投保且经保险公司批准免除等待期”或“保单生效第 45 天（已超过 30 天等待期）”，枚举器遍历 `{disease, accident}` 两个分支后均导出相同的 `WAIT_NOT_EXCLUDED` 结论，从而被正确归入 `PRESERVED_DECISION`。这验证了区分 `all_slots_preserved` 与 `enumerate_admissible_decisions` 的必要性。
3. **显式条件守恒提示在单阈值保险条款上即可大幅恢复结论保持率**：切换为 `condition_prompt` 后，`Qwen-Plus` 与 `Qwen-Max` 在这 20 组开发集样本上均实现 20/20（100%）的独立结论保持与 0/20 的条件篡改；本地 `1.5B` 模型也从 30.0% 提升至 95.0%（唯一 1 例未达标样本 `condition-008` 漏写了险种名称 `policy_id`，在已知源保单下为 20/20），并将 `ordinary_style` 下出现的 3 例日期数值篡改（如将“生效第 20 天”误写为“生效第二天”）降至 0/20。

### 5.2 扩展集四份正式条款评估、责任细分与标题掩码检索诊断
为检验上述发现在另外 4 份正式保险条款（包含同一合同内区分一般住院与癌症住院等待期的 `pa_2011_36`、明确排除原位癌的原发性肺癌险 `pa_p0321` 等）以及去掉书名号与法条编号的自然口语提问上是否成立，我们在 4 份扩展集正式条款（16 组种子、64 条受控样本）上评估了 7 种生成方法。统一验证器对 64 条受控样本（含 `pa_2011_36` 一般住院津贴 30 天等待期 vs. 癌症住院津贴 90 天等待期的差异化规则）实现 **64/64（100.0%）** 预期分类。表 3 报告了扩展集上的生成与检索表现。

| 生成方法 | 总调用次数 | 全槽位保留 (`standalone`) | 独立结论保持 (`standalone`) | 已知保单结论保持 (`granted`) | 退化追问 (`CLARIFY`) | 条件篡改 | 模板新颖度 ($1-\kappa_2$) | 9条款标准检索 Top-1 (BM25 / BGE) | 9条款标题掩码 Top-1 (BM25 / BGE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed_passthrough` | 0 | 100.0% (16/16) | **100.0%** (16/16) | 100.0% (16/16) | 0/16 | 0/16 | 0.0000 | 75.0% / 75.0% | 75.0% / 31.2% |
| `ordinary_style` | 16 | 0.0% (0/16) | **0.0%** (0/16) | 31.2% (5/16) | 16/16 | 0/16 | **0.5380** | 56.2% / 43.8% | 62.5% / 56.2% |
| `condition_prompt` | 16 | 100.0% (16/16) | **100.0%** (16/16) | 100.0% (16/16) | 0/16 | 0/16 | 0.1251 | 75.0% / 75.0% | 75.0% / 31.2% |
| `structured_json_rewrite` | 16 | 62.5% (10/16) | **62.5%** (10/16) | 93.8% (15/16) | 6/16 | 0/16 | 0.2797 | 75.0% / 62.5% | 75.0% / 31.2% |
| `generic_self_refine_budget_matched` | 16 | 100.0% (16/16) | **100.0%** (16/16) | 100.0% (16/16) | 0/16 | 0/16 | 0.1251 | 75.0% / 75.0% | 75.0% / 31.2% |
| `generic_self_refine_all` | 32 | 100.0% (16/16) | **100.0%** (16/16) | 100.0% (16/16) | 0/16 | 0/16 | 0.1190 | 75.0% / 75.0% | 75.0% / 31.2% |
| `slot_guard_repair` | 16 | 100.0% (16/16) | **100.0%** (16/16) | 100.0% (16/16) | 0/16 | 0/16 | 0.1251 | 75.0% / 75.0% | 75.0% / 31.2% |

扩展集实验得出三项重要结论：
1. **口语新颖度与业务结论保持之间存在尖锐权衡**：`ordinary_style` 取得了最高的模板新颖度（$1-\kappa_2 = 0.5380$，条款字符包含率仅 0.1917），但 16 条生成提问全部隐去了具体产品标识，导致独立结论保持率为 **0/16（0.0%）**；即便在评测时已知源保单（`granted_policy`），也仅有 **5/16（31.2%）** 能唯一确定结论，其余 11/16 因漏掉生效天数、投保状态或申请责任类别而无法判定。
2. **标题掩码检索诊断了检索器对产品名称锚点的强依赖**：在 9 份正式条款构成的检索池中，对于保留了产品名称的提问（`seed_passthrough` 与 `condition_prompt`），掩去产品全称、口语简称与保险公司名称（`title_masked_retrieval`）后，密集检索 `bge_small_zh` 的 Top-1 准确率从 **75.0%（12/16）骤降至 31.2%（5/16）**（`bm25_char2` 保持 75.0%），因为多份住院医疗险在去除产品名后具有高度相似的 30 天或 90 天等待期条款表述；相反，对于原本在提问中就省去了产品名的 `ordinary_style`，从索引文档中移除这些未匹配的标题字符串改变了文档表示与排序结果，使 Top-1 从 56.2% / 43.8% 微升至 62.5% / 56.2%。这表明常规密集检索的高分在很大程度上依赖表层产品名称锚点，而非对资格条件的精确匹配。
3. **在保险合同集上单轮 `condition_prompt` 已达饱和，闭环修复未产生额外增益**：在扩展集 4 份保险条款上，`condition_prompt` 首轮生成即达到 **16/16（100.0%）** 的全槽位与结论保持率，因此 `slot_guard_repair` 触发了 **0 次**二轮修复调用（总计 16 次调用，输出与 `condition_prompt` 完全一致，双侧精确 McNemar $p=1.0$）。这说明在单份保险合同等待期规则上，单轮显式自然语言条件约束已足够充分，闭环修复在保险合同集上并无额外增益（其相对 `structured_json_rewrite` 10/16 的差异 [双侧精确 McNemar $p=0.03125$] 反映的是单轮自然语言提示相对双行 JSON 格式的优势，而非二轮修复的作用）。

### 5.3 多阈值合成规则集上的等调用预算（Equal-Call-Budget）对照
当单条规则同时约束多个竞争性数值窗口与布尔前提（如 `med_egfr_tki` 结合 T790M 状态、14 天洗脱期与急诊特批；`med_vte_prophylaxis` 对比膝关节 10 天与髋关节 35 天窗口；`law_work_injury` 对比用人单位 30 天与职工亲属 365 天工伤申报期限；`law_noncompete` 结合 90 天竞业补偿违约期与补充宽限期）时，单轮提示词容易出现数值阈值覆盖与局部条件遗漏。表 4 展示了 16 组多阈值种子在 7 种方法下的等调用预算对照结果。

| 生成方法 | 总调用次数 | 格式有效率 (`format_valid`) | 触及 160-Token 上限 | 全槽位保留 (`all_slots`) | 独立结论保持 (`preserved_decision`) | 退化追问 (`CLARIFY`) | 条件/阈值篡改 (`mutated`) | 模板新颖度 ($1-\kappa_2$) | 4规则检索 Top-1 (标准 & 掩码) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed_passthrough` | 0 | 100.0% (16/16) | 0/16 | 100.0% (16/16) | **100.0%** (16/16) | 0.0% (0/16) | 0.0% (0/16) | 0.0000 | 100.0% / 100.0% |
| `ordinary_style` | 16 | 100.0% (16/16) | 0/16 | 6.2% (1/16) | **6.2%** (1/16) | 81.2% (13/16) | 12.5% (2/16) | **0.5726** | 100.0% / 100.0% |
| `condition_prompt` | 16 | 100.0% (16/16) | 0/16 | 37.5% (6/16) | **43.8%** (7/16) | 25.0% (4/16) | 31.2% (5/16) | 0.3376 | 100.0% / 100.0% |
| `structured_json_rewrite` | 16 | **0.0%** (0/16) | 0/16 | 6.2% (1/16) | **31.2%** (5/16) | 37.5% (6/16) | 31.2% (5/16) | 0.6064 | 100.0% / 93.8% |
| `generic_self_refine_budget_matched` | **26** | 93.8% (15/16) | 0/16 | 37.5% (6/16) | **50.0%** (8/16) | 25.0% (4/16) | 25.0% (4/16) | 0.3151 | 100.0% / 100.0% |
| `generic_self_refine_all` | 32 | 93.8% (15/16) | 0/16 | 37.5% (6/16) | **50.0%** (8/16) | 25.0% (4/16) | 25.0% (4/16) | 0.3133 | 100.0% / 100.0% |
| `slot_guard_repair` | **26** | 100.0% (16/16) | 0/16 | **68.8%** (11/16) | **75.0%** (12/16) | 25.0% (4/16) | **0.0% (0/16)** | 0.3342 | 100.0% / 100.0% |

多阈值压力测试揭示了三点关键机制：
1. **100% 检索准确率完全掩盖了 93.8% 的决策失效**：除 JSON 格式失败的基线外，其余 6 种方法在标准检索与标题掩码检索下的 `bm25_char2` 和 `bge_small_zh` Top-1 准确率均为 **100.0%（16/16）**。然而 `ordinary_style` 的结论保持率仅为 **6.2%（1/16）**（13/16 退化为 `CLARIFY`，2/16 发生条件篡改，如 `cross-011` 将“本人主要责任”反转为“不是主要责任”）。
2. **单通 `structured_json_rewrite` 的失效源于首行 JSON 后提前终止生成而非长度截断**：在多阈值规则集的 `1.5B` 模型生成日志中，`structured_json_rewrite` 的双行格式有效率为 **0/16（0.0%）**。核对原始 `generated_tokens` 字段可见，该集合上 **0/16** 的样本触及 160-token 上限，实际生成长度分布在 **45–111 tokens** 之间；模型在输出第一行 JSON 槽位字典并闭合花括号后即提前终止生成，未继续输出第二行 `QUESTION:` 字段（相比之下，在扩展集 4 份保险条款上该基线有 15/16 包含 `QUESTION:` 行，2/16 触及 160-token 上限）。由于双段式输出格式在小模型上混入了格式遵循失败，我们的核心对比以单问句自然语言基线（`condition_prompt` 与 `generic_self_refine`）为准。
3. **等调用预算下符号槽位反馈优于无反馈大模型自检，但小样本统计功效需诚实报告**：`condition_prompt` 初稿在 16 组种子中有 10 组未通过形式化验证（7/16 保持结论，4/16 退化为 `CLARIFY`，5/16 发生将用户实际天数 7/45/120 天改写为法条阈值 14/90 天的阈值覆盖篡改），因此 `slot_guard_repair` 精确触发了 10 次二轮修复调用（总计 $16+10=26$ 次调用）。在共享完全相同的 10 题触发日程与 **26 次调用预算**下，无解析器反馈的 `generic_self_refine_budget_matched`（以及无条件对全部 16 题执行二轮自检、共 32 次调用的 `generic_self_refine_all`）仅将结论保持率从 7/16 微升至 **8/16（50.0%）**，且仍残留 **4/16（25.0%）** 的阈值篡改；而向模型反馈具体缺失/冲突槽位名称与目标值的 `slot_guard_repair` 将全槽位保留率提升至 **11/16（68.8%）**，将结论保持率提升至 **12/16（75.0%，其中医疗规则 8/8、劳动法规则 4/8）**，并将解析器检出的阈值篡改率降至 **0/16（0.0%）**。在 $N=16$ 的样本规模下，`slot_guard_repair` 相对 `generic_self_refine_budget_matched` 在配对样本上赢 4 题（`cross-002`、`cross-003`、`cross-004`、`cross-009`）、输 0 题（双侧精确 McNemar $p=0.125$），相对单轮 `condition_prompt` 赢 5 题、输 0 题（双侧精确 McNemar $p=0.0625$）。我们如实报告这些精确 $p$ 值：在 $\alpha=0.05$ 水平下，16 组种子的样本量尚不足以声称达到独立双侧统计显著性。

### 5.4 解析器确定性诊断日志与 29 条边界关注清单（Watchlist）
由于本研究未开展独立多标注员人工盲评，我们不在文中声称任何人工语义一致率或解析器假阳性/假阴性率。为便于逐条复查确定性槽位提取器 $g(q')$ 与可达性枚举器的判定依据，我们在 `data_v3/item_by_item_semantic_audit.jsonl`（汇总于 `data_v3/item_by_item_semantic_audit_summary.json`）中导出了全部 264 条核心生成记录（开发集 40 条、扩展集 112 条、多阈值等预算对照 112 条）的**确定性验证器诊断日志与人工抽检关注清单**（显式标记为 `automated_verifier_explanation_and_manual_inspection_watchlist; NOT_external_semantic_validation_or_human_annotation`）。

该清单自动筛出 **29 条** 处于双段式格式或表面规程标题边界的样本供人工复核，精确对应四类确定性现象：
- **多阈值集双行 JSON 早停（`json_stopped_after_json_line`，16 条）**：多阈值规则集上 `structured_json_rewrite` 的全部 16 条样本仅输出第一行 JSON 字段（生成长度 45–111 tokens，0/16 触及 160-token 上限）即停止生成，未输出第二行 `QUESTION:` 问句。
- **包含业务短语但省略完整规程标题（`domain_phrase_without_formal_clause_title`，7 条）**：多阈值竞业限制规则 `law_noncompete` 上的 4 条 `ordinary_style`（`cross-013`、`014`、`015`、`016`）与 3 条 `slot_guard_repair`（`cross-013`、`014`、`016`）在问句中写出了“竞业限制经济补偿/解除竞业限制协议”等业务短语，但未写出《劳动合同法第二十四条竞业限制经济补偿与单方解除合规指引》完整规程标题，因此在多规则池独立评测下因缺失 `clause_id`（`cross-014`、`016` 同时缺失 `threshold_days`）被保守归入 `DEGRADED_TO_CLARIFY`（连同 `cross-010` 构成了 `slot_guard_repair` 在劳动法子集上的 4 例未保持样本）。
- **首行 JSON 含险种名但第二行问句省略（`json_header_has_policy_but_question_omits_it`，5 条）**：扩展集保险条款上 `structured_json_rewrite` 的 5 条样本（`heldout-001`、`007`、`010`、`011`、`012`）在第一行 JSON 中包含保单名称，但在第二行 `QUESTION:` 问句中省略了保单名称，导致仅解析问句行时缺失 `policy_id`。
- **双行 JSON 触及生成上限截断（`json_truncated_at_token_cap`，1 条）**：扩展集 `heldout-008__structured_json_rewrite` 触及 160-token 上限导致末尾问句被截断，未形成完整问句。
此外，对于上一轮审核指出的 `cross-009`（工伤申报第 45 天 vs. 用人单位 30 天法定时限），确定性诊断日志已如实反映原句文本：`condition_prompt`、`generic_self_refine_budget_matched` 与 `generic_self_refine_all` 在问句中写出了“30天内提出申请”（保留 `threshold_days=30`）但遗漏了“距事故发生第45天”（缺失 `elapsed_day=45`），而 `slot_guard_repair` 补齐了“距事故发生第45天”与“法定时限30天”，从而实现 `PRESERVED_DECISION`。

---

## 6 局限性与适用边界

1. **手工编码规则编译器、扩展集后验属性与单项责任范围限制**：本研究中的等待期规则编译器与槽位提取器由作者针对 9 份保险条款与 4 段多阈值规则手工构建；其中 4 份扩展集保险条款是在共享解析器中补充口语产品简称与 `benefit_id` 责任消歧槽位后开展的回顾性评估，而非对冻结解析器的零样本盲测。此外，真实商业医疗险理赔还涉及既往症认定、免赔额扣减、社保结算比例及多责任组合赔付，当前等待期模型不能直接外推为全自动条款编译或全流程理赔系统。
2. **受控实验样本规模与统计功效**：扩展集与多阈值压力测试各包含 16 组种子（64 条受控样本）。尽管这足以暴露 `ordinary_style`（0%–6.2% 结论保持）与 `structured_json_rewrite` 的结构性失效，但 `slot_guard_repair` 相对 `generic_self_refine_budget_matched` 的 12/16 vs. 8/16 差异在双侧精确 McNemar 检验下 $p=0.125$（相对 `condition_prompt` 7/16 为 $p=0.0625$），而在扩展集保险条款上 `condition_prompt` 已达 16/16（二轮修复 0 触发，$p=1.0$），未来需在更大规模的多阈值种子集上扩大统计检验功效。
3. **社区语料授权与预训练污染风险**：附录 A 使用的 `baoxianzhidao_filter.csv` 源自历史百度知道社区问答镜像，其“最佳回答”属于网友弱标签而非精算真值，且存在潜在的大模型预训练语料重叠与未确权开放许可限制；本仓库不重分发原始第三方数据与商业条款 PDF。
4. **缺乏独立多标注员人工盲评**：代码库提供了双盲人工标注网页工具包（`data_v3/independent_review/reviewer_A.html` 与 `reviewer_B.html`）及 264 条验证器诊断日志与关注清单，但独立人类标注员的双盲标注与 Cohen's $\kappa$ 评测尚未开展，本文不作任何人工语义一致率主张。

---

## 7 结论

本文围绕规则密集型垂直领域 RAG 评测中的合成查询质量控制问题，证明了高检索命中率（乃至 100% Top-1）并不能保证合成提问保留源条款的业务判定结论。通过将评估拆解为表面槽位完整度与无参考依赖的形式化结论可确定性，并在 9 份可追溯中文商业保险正式条款（5 份开发集、4 份扩展集）与多阈值规则集上开展受控实验，我们展示了无约束口语改写如何因擦除险种标识、责任类别与时间阈值而使绝大多数合成提问退化为必须追问的歧义题，同时阐明了显式条件守恒提示在单合同保险条款上的直接有效性（无需二轮修复）以及符号槽位守卫修复在多阈值组合规则上的纠偏机制与小样本统计边界。

---

## 附录 A：社区问答语料级检索器选型与校准诊断

为完整呈现本研究的前置探索，本附录报告公开保险社区问答语料 `baoxianzhidao_filter.csv` 上的词法/密集/混合检索器评测、源文档抽样混杂分解、探索性源重叠度门控以及小预算候选集合暂缓选型实验。

### A.1 防泄漏连通分量划分与词法代理偏差
对 8,362 条原始记录执行最佳回答筛选（`is_best == '1'`）、长度过滤与隐私正则清洗后，得到 3,744 个问题组与 3,746 条唯一回答文档。为严防近似重复回答跨集泄漏，我们使用并查集（Union-Find）将标题相同、共享同一回答文档或标题 2–4 元字符 TF-IDF 余弦相似度 $\ge 0.90$ 的问题强制合并为 3,725 个连通分量，按固定种子 `20261025` 切分为 **750 条真实校准池提问**与 **2,994 条保留真实测试提问**（跨划分共享回答数为 0，最大标题字符余弦相似度为 0.8965）。

在 6 个字符级词法检索器（`tfidf_char1`、`tfidf_char2`、`tfidf_char1_3`、`tfidf_char3_5`、`bm25_char1`、`bm25_char2`）上，真实测试集与抽取式关键词代理均选中 `bm25_char2`（MRR@10 分别为 0.3817 与 0.9335），而直接截取回答首句前 60 字符的前缀复制诊断集（`source_prefix_diagnostic`）则错误选中长字符 n-gram 模型 `tfidf_char3_5`（代理 MRR@10 为 0.9330，但真实测试仅 0.2727，产生 **0.1090** 的真实选型遗憾）。

### A.2 跨模型合成选型反转、源抽样混杂分解与 Full-MaxP 检验
从回答库中防污染抽取 256 个源回答，分别使用本地 `Qwen2.5-1.5B-Instruct`（249 个共同有效源）与云端 `Qwen-Plus`（179 个共同有效源）生成 `zero_shot` 与 `few_shot_3` 合成提问，并引入 `BAAI/bge-small-zh-v1.5` 与固定混合检索 `rrf_bm25_bge`（$k=60, \text{top-}100$）：

| 候选检索系统 | 全量真实测试 MRR@10 ($N=2{,}994$) | 全量真实测试 Hit@5 | 相对 `bm25_char2` 配对差值（95% Bootstrap CI） | 1.5B 零样本 / 少样本 MRR@10 ($N=249$) | `Qwen-Plus` 零样本 / 少样本 MRR@10 ($N=179$) |
|---|---:|---:|---|---:|---:|
| `bm25_char2` | 0.3817 | 0.4826 | $\phantom{+}0.0000 \; [\phantom{+}0.0000, \phantom{+}0.0000]$ | 0.5013 / 0.3225 | 0.6528 / 0.5747 |
| `bge_small_zh` | **0.5216** | **0.6453** | $+0.1399 \; [+0.1250, +0.1551]$ | 0.5202 / 0.4024 | 0.7005 / 0.6794 |
| `rrf_bm25_bge` | 0.5030 | 0.6075 | $+0.1212 \; [+0.1115, +0.1308]$ | **0.5647** / **0.4124** | **0.7371** / **0.6813** |

1. **全量真实测试集上的密集检索优势与 `Full-MaxP` 复核**：在全量 2,994 条真实提问上，纯密集检索 `bge_small_zh` 稳定优于 `rrf_bm25_bge`（MRR@10 0.5216 vs. 0.5030，配对差值 $+0.0187$，10,000 次配对 Bootstrap 95% 区间 $[+0.0077, +0.0296]$）。对 509 条超过 512 tokens 的长回答切分 762 个额外尾部窗口（共 1,271 个滑动窗口）执行零截断 `Full-MaxP` 复核，`bge_small_zh` 仍以 **0.5165 vs. 0.5007**（配对差值 $+0.0158$，95% CI $[+0.0049, +0.0267]$）领先 `rrf_bm25_bge`。
2. **源文档抽样混杂与同源词法复制分解**：表面上看，`1.5B` 与 `Qwen-Plus` 的合成提问均将最优系统反转至 `rrf_bm25_bge`。但将真实测试题与合成题严格对齐到相同目标文档（`1.5B` 对应 207 篇文档/209 条真实题；`Qwen-Plus` 对应 151 篇文档/153 条真实题）后发现：在这两个小样本真实子集上，真实 RRF 点估计已微弱高于真实 BGE（0.4975 vs. 0.4696 与 0.4980 vs. 0.4668，95% 区间均跨零）。固定配对文档后，虽然零样本合成显著抬高了提问对源回答的二元字符包含率 $\kappa_2(q', d^*)$（0.3253 $\to$ 0.4775 与 0.3167 $\to$ 0.4638）并推高 `bm25_char2` 得分（配对增益 $+0.0925$ 与 $+0.2457$），但 RRF 相对 BGE 优势的双重差分增量 $\Delta\Delta_{\text{RRF}-\text{BGE}}$ 在全部 4 组设定下的 95% 置信区间均跨零（例如 `1.5B zero_shot` 为 $+0.0123$ $[-0.0375, +0.0649]$，`Qwen-Plus zero_shot` 为 $-0.0020$ $[-0.0479, +0.0438]$）。这证明小样本合成集上的排名翻转混杂了源文档子采样方差，不能简单归结为生成风格单方面放大了 RRF 相对 BGE 的差距。

### A.3 密度重加权负结果、探索性源重叠度门控与候选集合暂缓选型
在 360 组同预算校准实验（校准预算 $B \in \{25, 50, 100, 200\}$，5 次重复采样）中：
- **边缘特征密度比重加权的负结果**：基于问题长度或 1–3 元字符 TF-IDF 逻辑回归倾向性权重的重加权估计（`synthetic_density_weighted`）在所有预算下选中全量真实冠军 `bge_small_zh` 的比例均为 **0%**，因为问题表面的边缘特征分布无法消除合成提问与源文档之间的内部词法耦合。
- **探索性关系型源重叠度门控（`source_overlap_gated`）**：筛除 $\kappa_2(q', d^*)$ 高于校准集分位点或冷启动分位点 $q \in [0.10, 0.30]$ 的高重叠合成题后，保留的低重叠子集在 `1.5B` 与 `Qwen-Plus` 上均恢复选中 `bge_small_zh`（而相同样本量的 1,000 次随机子采样仅在 5.9% 与 18.2% 中选中 BGE）。但鉴于小样本文档子集本身存在抽样方差，我们将该门控明确定位为**事后诊断启发式（Exploratory Diagnostic Heuristic）**。
- **小预算候选集合暂缓选型（Candidate-Set Abstention）**：当直接使用真实校准样本 `real_only` 进行 200 次配对 Bootstrap 构建 95% 置信候选集合 $\widehat{\mathcal{C}}_{0.95}$ 时，在低预算 $B=25$ 下将错误确定选型率从点估计的 60%–100% 降至 **0.0%**（暂缓单选比例 80%，平均候选集大小 2.60 个系统，全量最优系统覆盖率 **100%**）。

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
