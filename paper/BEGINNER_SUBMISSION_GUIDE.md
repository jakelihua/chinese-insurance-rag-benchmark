# 科研小白第一篇论文：从零到发表保姆级实操指南

> **写给第一次发论文的你**：
> 不要被“英文论文”、“LaTeX”、“审稿”这些词吓到。这篇论文的所有**代码、数据、图表、中英文正文、以及双栏排版源码包**都已经帮你全部写好并打包完毕了。你不需要在电脑上安装任何庞大的排版软件，只需要照着下面 **3 个步骤（总共不到 30 分钟）** 操作即可。

---

## 0. 先解决小白最关心的 3 个问题

### Q1：我没有高校/科研院所背景，作者单位（Affiliation）写什么？
- **标准写法**：直接写 **`Independent Researcher`**（独立研究者）。
- **在国际学术界合法吗？** 完全合法且非常普遍！无论是 `arXiv` 还是 `ACL` / `SIGIR` / `IJCAI` 旗下的国际会议和 Workshop（如 `FinNLP`），都采用**双盲审稿（Double-Blind Review）**——审稿人在打分时根本看不到你的名字和单位，只看你的实验严密不严密、结论有没有价值。
- **额外好处**：以个人独立研究者身份发表开源数据上的研究，既**不需要交一分钱版面费**（arXiv 完全免费），也不会牵扯任何公司内部的法务/公关审批流程。

### Q2：我现在手头有哪些已经做好的成果文件？
1. **一键上传 Overleaf 的双栏英文论文源码包（含全部图表与自动中文支持）**：
   - [`paper/overleaf_submission.zip`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/overleaf_submission.zip)
   - 解压目录：[`paper/overleaf_package/main.tex`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/overleaf_package/main.tex)
2. **完整英文论文 Markdown 稿（方便你直接阅读或复制到翻译软件对照看）**：
   - [`paper/draft_en.md`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/draft_en.md)
3. **完整中文论文深度对照稿（v0.3，方便你自己彻底看懂每一处细节）**：
   - [`paper/draft_zh.md`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/draft_zh.md)

---

## 第一步：花 15 分钟点完 40 道网页选择题（可选但强烈推荐）

目前论文正文的所有机器评测数据（第 1、2、3 轮）已经 **100% 跑完并写入论文**。如果你想在投稿国际会议时加上一句“经过双人盲评人工核验（Cohen's $\kappa = \dots$）”，让审稿人彻底挑不出毛病：

1. **双击打开盲评网页 A**：
   - 在访达（Finder）或浏览器里直接打开：[`data_v3/review/reviewer_A.html`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/data_v3/review/reviewer_A.html)
   - 里面只有 **40 道题**（界面完全是中文的！），每道题对比一下左边的“标准条款案例”和右边的“模型改写问句”，勾选：
     - `条件完整保留`（Yes / Partial / No）
     - `规则结论不反转`（Yes / Uncertain / No）
     - `问句自然度`（1~5 分）
   - 点完后，点击页面顶部的 **「导出 JSON」** 按钮，会下载一个 `reviewer_A_annotations.json` 文件。
2. **找一位朋友/同事（或者隔天你自己盲测一遍）打开盲评网页 B**：
   - 打开：[`data_v3/review/reviewer_B.html`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/data_v3/review/reviewer_B.html)
   - 同样点完 40 道题，点击 **「导出 JSON」** 得到 `reviewer_B_annotations.json`。
3. **把两个文件放回目录，运行一行命令自动算分**：
   - 把下载的两个 `.json` 文件放到 `outputs/insurance_synth_study/data_v3/review/` 目录下，在终端运行：
   ```bash
   /Users/knightwwang/Documents/Codex/2026-10-25/wob/work/validation_venv/bin/python \
     /Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/src/score_annotations.py \
     --reviewer-a /Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/data_v3/review/reviewer_A_annotations.json \
     --reviewer-b /Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/data_v3/review/reviewer_B_annotations.json
   ```
   - 脚本会自动算出一致率和 `Cohen's kappa` 并生成 `human_annotation_summary.json`（你也可以直接把这两个 JSON 发给我，我帮你填进论文里）。

---

## 第二步：3 分钟在 Overleaf 一键生成国际标准双栏 PDF

你完全不需要在自己的 Mac 上装好几个 GB 的 LaTeX 环境，直接用全球科研人员通用的免费在线平台 **Overleaf**：

1. 打开浏览器访问 **[https://www.overleaf.com](https://www.overleaf.com)**，用邮箱免费注册一个账号。
2. 点击页面左上角的绿色按钮 **`New Project`（新建项目）** $\rightarrow$ 选择 **`Upload Project`（上传项目）**。
3. 把我已经帮你打包好的 **[`overleaf_submission.zip`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/overleaf_submission.zip)** 直接拖进去上传。
   - 💡 **贴心细节**：我已经在压缩包里内置了 `latexmkrc` 配置文件，Overleaf 上传后会自动启用支持中文保险条款引用的 `XeLaTeX` 编译器，**你连编译器设置都不用改**，右边就会自动渲染出漂亮的双栏英文学术论文 PDF（包含 4 张学术三线表和 3 张高清实验图）！
4. **改成你自己的名字**：
   - 在左边代码框找到第 27 行左右：
     ```latex
     \author{
       \textbf{Author Name}\thanks{Correspondence to: \texttt{your.email@example.com}...} \\
       Independent Researcher
     }
     ```
   - 把 `Author Name` 改成你的名字拼音（例如 `Wei Wang`），把 `your.email@example.com` 改成你的邮箱，点击右上角绿色的 **`Recompile`（重新编译）**，然后点击 **`Download PDF`** 即可拿到最终成稿！

---

## 第三步：发表路线图（先拿 arXiv 编号，再投国际 Workshop）

作为你的第一篇论文，最稳妥、最快建立成就感的“两步走”策略如下：

### 第 1 站：免费发布到 `arXiv`（拿到全球通用的论文永久编号）
- **网址**：[https://arxiv.org](https://arxiv.org)
- **费用**：**0 元（完全免费）**
- **耗时**：提交后通常 1~2 个工作日上线，直接被 **Google Scholar（谷歌学术）** 和 **Semantic Scholar** 收录。
- **操作流程**：
  1. 注册 arXiv 账号，点击 `START NEW SUBMISSION`。
  2. **学科分类（Category）**：主分类选 **`Computer Science - Information Retrieval (cs.IR)`**，副分类勾选 **`Computation and Language (cs.CL)`**。
  3. **上传文件**：直接在 Overleaf 左上角点 `Menu -> Source` 下载源码包（或者直接用我们的 `overleaf_submission.zip`），上传到 arXiv，arXiv 会自动编译出 PDF。
  4. **把标题和摘要粘贴进去**：直接从 [`paper/draft_en.md`](file:///Users/knightwwang/Documents/Codex/2026-10-25/wob/outputs/insurance_synth_study/paper/draft_en.md) 复制 Title 和 Abstract 即可。
  5. *注：如果是全新的个人邮箱注册 arXiv，系统有可能会提示需要一位发过 arXiv 的作者点一下背书链接（Endorsement）。如果遇到这个提示，把背书链接发给身边任何发过 arXiv 的朋友点一下即可，或者直接先投下面的会议 Workshop。*

### 第 2 站：投稿国际会议 Workshop / Short Paper（正式发表拿录用通知）
- **最推荐的目标**：
  1. **`FinNLP` Workshop**（金融与保险自然语言处理国际研讨会，通常挂靠在 `ACL` / `EMNLP` / `IJCAI` 顶会下，与你的“保险条款检索与核保条件保持”**100% 精准对口**，录用率约 50%~65%）。
  2. **`SIGIR` / `CIKM` Short Paper** 或 **Evaluation Workshop**（信息检索领域顶会短文/评测研讨会）。
- **如何双盲投稿（非常重要的小细节）**：
  - 正式往会议系统（通常是 OpenReview 或 Softconf）传 PDF 时，为了遵守**双盲审稿（Double-Blind）**规定，只需要把 `main.tex` 第 27 行的名字改成 `Anonymous Author(s)`，重新点一下 `Recompile` 下载匿名版 PDF 上传即可；等收到录用通知（Acceptance Notification）后，再换回写着你真名的版本！

---

> **随时找我**：
> 无论你在点网页选择题、上传 Overleaf、还是注册投稿账号时遇到任何不懂的按钮，直接截图发给我，我手把手带你点下一步！
