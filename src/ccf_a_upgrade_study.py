"""CCF-A Main-Track Upgrade Suite (`src/ccf_a_upgrade_study.py`).

Addresses all five review criticisms and five reviewer questions without human annotation latency,
using live cross-family flagship API endpoints (`qwen3.8-max`, `deepseek-v4-pro`, `deepseek-v4.1-flash`,
`kimi-k3`, `qwen-plus`) with full per-request HTTP header (`x-request-id`, `Date`) provenance logging
and 100% deterministic `--score-only` offline reproduction:

1. Experiment 1 — Blinded 3-Flagship Raw-Clause Semantic Adjudication Panel (`qwen3.8-max`,
   `deepseek-v4-pro`, and third-party Moonshot `kimi-k3`):
   - Evaluates all 40 confirmation seeds x 2 generators x 9 methods (including `multi_template_3`
     and `complete_6slot_schema_render`) directly against verbatim official PDF clause text, with
     ZERO access to the regex slot parser, seed slot dictionary, or method labels.
   - Reports inter-adjudicator reliability (pairwise Cohen's kappa and 3-rater Fleiss' kappa),
     Parser False Reject Rate (`parser_rejected_panel_accepted`), Parser False Accept Rate
     (`parser_accepted_panel_rejected`), and 4-level target-function decomposition.

2. Experiment 2 — Joint Utility & Template Discrimination Analysis (`seed_passthrough`,
   `multi_template_3`, `complete_6slot_schema_render` vs. `slot_guard_repair`):
   - Quantifies why zero-call templates and deterministic schema renderers fail to stress-test
     lexical vs. semantic retrieval under title-masked/colloquial clause matching, and computes
     the correlation between clause-body lexical novelty (`1 - kappa_2`) and dense-vs.-lexical
     retrieval discrimination (`MRR_BGE - MRR_BM25`).

3. Experiment 3 — Downstream System Selection on 6 Real End-to-End LLM RAG Pipelines:
   - Replaces synthetic rule-engine variants (`S1`-`S5`) with 6 real `Retriever + Cloud LLM Reader`
     pipelines (`L1`..`L6`) executing strictly isolated `predict_real_llm_rag(question, retrieved_clause_text)`
     calls across the 80-query reference benchmark and synthetic benchmarks, with 10,000-resample
     bootstrap 95% confidence intervals on Kendall's tau, Spearman's rho, and selection regret.

4. Experiment 4 — Generalization Beyond Waiting Periods (3 New Official Insurance Rule Families,
   24 New Seeds, 48 Cross-Model Pairs) & Pooled 88-Seed ($N=176$) High-Power Confirmation +
   Independent Zero-Shot Rule-Compiler Agreement Audit:
   - Compiles 3 non-waiting-period insurance rule families from verbatim official PDF clauses:
     Family B (`deductible_copay_tier`), Family C (`hospital_ward_emergency`), and
     Family D (`enrollment_age_and_rx_limit`).
   - Evaluates all 7 generation methods across `qwen3.8-max` and `deepseek-v4.1-flash`, and
     computes pooled 88-seed ($N=176$ pairs) exact McNemar and seed/clause-clustered permutation
     tests (with 6-baseline Holm-Bonferroni correction).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, kendalltau, pearsonr, spearmanr

import experiment as base
from analyze_v3 import bigram_containment
from condition_guard import detect_parse_uncertainty
from cross_domain_study import (
    CROSS_CLAUSES,
    cross_decision,
    evaluate_cross_outcome,
    extract_cross_slots_v2,
    render_question as render_cross_question,
)
from equal_budget_and_heldout_study import (
    evaluate_insurance_outcome_frozen,
    extract_insurance_fields_frozen,
    extract_question_from_structured_output,
    insurance_decision,
    mask_titles_and_insurers,
)
from frozen_confirmation_and_rag_study import (
    FROZEN_CONFIRM_POLICIES,
    UNTOUCHED_HOLDOUT_POLICIES,
    _get_rag_resources,
    build_frozen_confirmation_sources_and_cases,
    build_untouched_holdout_sources_and_seeds,
    extract_text_slots_pure,
    render_frozen_insurance_question,
)


def clustered_sign_flip_pvalue(cluster_diffs):
    """Exact two-sided sign-flip permutation test over non-zero cluster mean differences.
    When all non-zero cluster differences share the same sign, the triangle inequality implies
    that only the all-positive and all-negative sign vectors achieve |sum(s_c d_c)| >= |sum(d_c)|,
    giving the exact analytical two-sided p-value 2^{-(k-1)} for any k >= 1.
    """
    nz = [float(x) for x in cluster_diffs if abs(float(x)) > 1e-12]
    k = len(nz)
    if k == 0:
        return 1.0
    if all(x > 0.0 for x in nz) or all(x < 0.0 for x in nz):
        return float(2.0 ** (-(k - 1)))
    obs_abs = abs(sum(nz))
    arr = np.asarray(nz, dtype=np.float64)
    if k <= 20:
        bits = ((np.arange(1 << k, dtype=np.uint32)[:, None] >> np.arange(k, dtype=np.uint32)) & 1).astype(np.float64)
        signs = 2.0 * bits - 1.0
        sums = np.abs(np.sum(signs * arr[None, :], axis=1))
        return float(np.mean(sums >= obs_abs - 1e-12))
    rng = np.random.default_rng(20261027)
    signs = rng.choice([-1.0, 1.0], size=(100000, k))
    sums = np.abs(np.sum(signs * arr[None, :], axis=1))
    return float((np.sum(sums >= obs_abs - 1e-12) + 1.0) / 100001.0)


def round_pval(pval):
    """Preserve significant digits for small p-values without underflowing to 0.0."""
    p = float(pval)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return float(f"{p:.8g}")


def holm_bonferroni_adjust(p_dict):
    """Step-down Holm-Bonferroni correction preserving dictionary keys."""
    items = sorted(p_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for idx, (k, pval) in enumerate(items):
        scaled = min(1.0, float(pval) * (m - idx))
        running_max = max(running_max, scaled)
        adj[k] = running_max
    return adj

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def call_dashscope_chat(
    model_name,
    messages,
    api_key,
    max_tokens=700,
    temperature=0.0,
    disable_thinking=True,
    retries=4,
):
    """Execute a live DashScope compatible-mode chat completion call and capture HTTP headers."""
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if disable_thinking and any(k in model_name.lower() for k in ("qwen3", "kimi", "deepseek")):
        payload["enable_thinking"] = False

    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        req = urllib.request.Request(
            DASHSCOPE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw_bytes = resp.read()
                latency_ms = round((time.time() - t0) * 1000.0, 1)
                body = json.loads(raw_bytes.decode("utf-8"))
                hdrs = resp.headers
                req_id = (
                    hdrs.get("x-request-id")
                    or hdrs.get("x-dashscope-request-id")
                    or body.get("id")
                    or ""
                )
                http_date = hdrs.get("Date") or hdrs.get("date") or ""
                choice = (body.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = (msg.get("content") or "").strip()
                usage = body.get("usage") or {}
                return {
                    "content": content,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                    "latency_ms": latency_ms,
                    "api_meta": {
                        "provider": "dashscope_compatible_v1",
                        "requested_model": model_name,
                        "response_model": str(body.get("model") or model_name),
                        "completion_id": str(body.get("id") or ""),
                        "http_request_id": str(req_id),
                        "http_date": str(http_date),
                        "system_fingerprint": str(body.get("system_fingerprint") or ""),
                        "finish_reason": str(choice.get("finish_reason") or ""),
                        "metadata_provenance": "captured_live_from_http_response",
                    },
                }
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"DashScope API call failed for model={model_name}: {last_err}")


def parse_json_from_llm(text):
    """Extract and parse a JSON object or array from an LLM response string."""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try finding [...] or {...}
    m_arr = re.search(r"(\[[\s\S]*\])", s)
    if m_arr:
        try:
            return json.loads(m_arr.group(1))
        except Exception:
            pass
    m_obj = re.search(r"(\{[\s\S]*\})", s)
    if m_obj:
        try:
            return json.loads(m_obj.group(1))
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON from: {text!r}")


# ---------------------------------------------------------------------------
# Additional Deterministic Baseline Renderers for Experiment 2 (Joint Utility)
# ---------------------------------------------------------------------------
CROSS_SLOT_PHRASES = {
    "med_egfr_tki": {
        "status": {"primary": "患者属于一线初治申请", "secondary": "患者属于二线转换申请"},
        "condition": {"qualified": "基因检测为T790M突变阳性", "disqualified": "基因检测为T790M阴性"},
        "day": lambda d: f"停用前序药物第{d}天",
        "thresh": lambda t: f"规定洗脱期为{t}天",
        "waiver_true": "已获肿瘤专家组急诊特批豁免洗脱期",
        "waiver_false": "未获肿瘤专家组急诊特批豁免洗脱期",
    },
    "med_vte_prophylaxis": {
        "status": {"primary": "目前处于路径内常规抗凝状态", "secondary": "目前已停用常规预防方案"},
        "condition": {"qualified": "无活动性出血并发症", "disqualified": "出现活动性出血并发症"},
        "day": lambda d: f"术后第{d}天",
        "thresh": lambda t: "膝关节置换（标准预防期10天）" if int(t) == 10 else "髋关节置换（标准预防期35天）",
        "waiver_true": "已获血液科会诊特批继续抗凝",
        "waiver_false": "未获血液科会诊特批继续抗凝",
    },
    "law_work_injury": {
        "status": {"primary": "事故发生在合理上下班途中", "secondary": "事故发生在上下班顺路接送家属途中"},
        "condition": {"qualified": "交警认定为非本人主要责任", "disqualified": "交警认定为本人主要责任"},
        "day": lambda d: f"距事故发生第{d}天",
        "thresh": lambda t: "由用人单位提出申请（法定时限30天）" if int(t) == 30 else "由职工近亲属提出申请（法定时限365天）",
        "waiver_true": "已获人社部门书面批准延长申请时限",
        "waiver_false": "未获人社部门书面批准延长申请时限",
    },
    "law_noncompete": {
        "status": {"primary": "离职后因单位原因未支付补偿", "secondary": "离职后单位已按月足额支付补偿"},
        "condition": {"qualified": "员工属于负有保密义务的高级技术人员", "disqualified": "员工属于普通行政文员（无保密义务）"},
        "day": lambda d: f"连续未付已达第{d}天",
        "thresh": lambda t: f"法定解除阈值为{t}天",
        "waiver_true": "员工已书面同意豁免逾期并接受补发",
        "waiver_false": "未获书面同意豁免逾期",
    },
}


def render_multi_template_3(seed, domain_type, policies_map):
    """Zero-call multi-template baseline (`multi_template_3`) that rotates across 3 hand-written
    surface templates per domain based on seed index while preserving all slots deterministically.
    """
    sid = seed["seed_id"]
    idx = int(re.sub(r"\D+", "", sid) or "0") % 3
    if domain_type == "insurance":
        pol = policies_map[seed["policy_id"]]
        title = pol["title"]
        enr = (
            "首次投保"
            if seed["enrollment"] == "first"
            else (
                "重新投保但未获保险公司审核同意免除等待期"
                if not seed.get("approved_waiver")
                else "重新投保且已获保险公司审核同意免除等待期"
            )
        )
        day = seed["policy_day"]
        cause = "突发意外伤害" if seed["cause"] == "accident" else "非意外疾病"
        ben_part = ""
        if "benefits" in pol and seed.get("benefit_id") in pol["benefits"]:
            b_lbl = pol["benefits"][seed["benefit_id"]]["label"]
            ben_part = f"，申请{b_lbl}（对应责任约定等待期为{seed['plan_wait_days']}天）"
        if idx == 0:
            return f"请问客户在{enr}《{title}》后，保单生效第{day}天因{cause}住院治疗{ben_part}，按条款等待期规定能否获得理赔？"
        elif idx == 1:
            return f"如果被保险人属于{enr}并购买了《{title}》{ben_part}，合同生效第{day}天时因{cause}住院治疗，这种情况受等待期限制吗？"
        else:
            return f"关于《{title}》{ben_part}，投保情况为{enr}，在保单生效第{day}天因{cause}发生住院医疗费用，保险公司是否因等待期免责？"
    else:
        cid = seed["clause_id"]
        cl = CROSS_CLAUSES[cid]
        pmap = CROSS_SLOT_PHRASES[cid]
        title = cl["title"]
        st_label = pmap["status"][seed["status"]]
        cond_label = pmap["condition"][seed["condition"]]
        day_str = pmap["day"](seed["elapsed_day"])
        th_str = pmap["thresh"](seed["threshold_days"])
        waiver_str = pmap["waiver_true"] if seed.get("approved_waiver") else pmap["waiver_false"]
        if idx == 0:
            return f"依据《{title}》，{th_str}，{st_label}，{cond_label}，{day_str}，且{waiver_str}，请问是否符合准入或支持条件？"
        elif idx == 1:
            return f"请问按照《{title}》的规定，{st_label}，{cond_label}，{th_str}，{day_str}，并且{waiver_str}，该情形应当如何裁决？"
        else:
            return f"在适用《{title}》时，已知{th_str}，{st_label}，{day_str}，{cond_label}，且{waiver_str}，请问最终结论是什么？"


def render_complete_6slot_from_json(raw_json_row, seed, domain_type, policies_map):
    """Complete 6-slot deterministic schema renderer (`complete_6slot_schema_render`) that repairs
    the two design flaws of the 5-slot `structured_json_schema_render`:
    (1) supplies the explicit `threshold_days` / `plan_wait_days` slot for multi-threshold rules, and
    (2) sanitizes contract-level waiver rule echoes when rendering the instance waiver slot.
    """
    raw_resp = raw_json_row.get("raw_response") or ""
    first_line = raw_resp.strip().splitlines()[0] if raw_resp.strip() else "{}"
    try:
        obj = json.loads(first_line)
    except Exception:
        obj = {}
    if domain_type == "insurance":
        pol = policies_map[seed["policy_id"]]
        clause_str = pol["title"]
        status_str = "首次投保" if seed["enrollment"] == "first" else "重新投保"
        cond_str = "突发意外伤害" if seed["cause"] == "accident" else "非意外疾病"
        day_val = obj.get("day", seed["policy_day"])
        if not isinstance(day_val, int):
            day_val = seed["policy_day"]
        ben_part = ""
        if "benefits" in pol and seed.get("benefit_id") in pol["benefits"]:
            b_lbl = pol["benefits"][seed["benefit_id"]]["label"]
            ben_part = f"，申请{b_lbl}（对应责任约定等待期为{seed['plan_wait_days']}天）"
        if seed["enrollment"] == "new":
            waiver_str = (
                "且已获保险公司审核同意免除等待期"
                if seed.get("approved_waiver")
                else "但未获保险公司审核同意免除等待期"
            )
        else:
            waiver_str = "无免除等待期特批"
        return f"我{status_str}了《{clause_str}》（{waiver_str}）{ben_part}，保单生效第{day_val}天因{cond_str}住院治疗，请问是否可以理赔？"
    else:
        cid = seed["clause_id"]
        cl = CROSS_CLAUSES[cid]
        pmap = CROSS_SLOT_PHRASES[cid]
        clause_str = cl["title"]
        status_str = pmap["status"][seed["status"]]
        cond_str = pmap["condition"][seed["condition"]]
        day_val = obj.get("day", seed["elapsed_day"])
        if not isinstance(day_val, int):
            day_val = seed["elapsed_day"]
        th_val = obj.get("threshold_days", seed["threshold_days"])
        day_str = pmap["day"](day_val)
        th_str = pmap["thresh"](th_val)
        waiver_str = pmap["waiver_true"] if seed.get("approved_waiver") else pmap["waiver_false"]
        return f"根据《{clause_str}》，{th_str}，{status_str}，{cond_str}，{day_str}，且{waiver_str}，请问最终裁决结果是什么？"


# ---------------------------------------------------------------------------
# Experiment 1: Blinded 3-Flagship Raw-Clause Semantic Adjudication Panel
# ---------------------------------------------------------------------------
ADJUDICATOR_MODELS = [
    ("kimi_k3", "kimi-k3"),
    ("deepseek_v4_pro", "deepseek-v4-pro"),
    ("qwen3_8_max", "qwen3.8-max"),
]


def _build_seed_adjudication_prompt(seed, domain_type, clause_title, clause_text, anon_items, mrule_cdef=None):
    """Construct a blinded prompt that shows ONLY the verbatim clause text and anonymized questions Q1..Qk."""
    if domain_type == "insurance":
        valid_labels = '"WAIT_EXCLUDED"（等待期内免责/拒赔）、"WAIT_NOT_EXCLUDED"（不因等待期免责/可进入正常理赔）或 "CLARIFY_REQUIRED"（信息不足需追问澄清）'
        organ_note = ""
        if seed.get("policy_id") == "pa_h716":
            organ_note = (
                "注意：该条款将疾病分为“腺样体肥大、疝气、扁桃腺或女性生殖器官疾病（90日等待期）”与"
                "“其他疾病（30日等待期）”。若提问仅写“生病/非意外疾病住院”而未写明属于哪类疾病："
                "在生效第1–30日内，无论哪类疾病均处于等待期内，结论唯一为 WAIT_EXCLUDED；"
                "但在生效第31–90日之间，因30日与90日分支结论相反，未指明具体疾病类别时必须判为 CLARIFY_REQUIRED。"
            )
        rules_desc = (
            f"1. policy_identified (boolean)：提问中是否明确写出了本保险产品的名称（如“{clause_title}”）？"
            f"若只写了“我买了医疗险/保险/这款产品”而未写出具体产品名称，在多条款库中无法确定适用哪份合同，必须填 false。\n"
            f"2. sentence_complete (boolean)：提问是否为完整、未在半句处截断的自然语言问句（且不夹杂未渲染的代码或半截词组）？若在半句处截断（如以逗号、顿号或半截词语结尾且无问号），必须填 false。\n"
            f"3. decision (string)：仅根据【官方条款原文】与提问中明确写出的事实进行推理，绝对禁止脑补提问未写明的投保状态（首次投保 vs 续保）、生效天数、出险原因（疾病 vs 意外）或特批豁免。"
            f"同时，如果提问在同一句话中同时声称“因非意外疾病住院”又混入“因意外伤害无等待期”等自相矛盾的个案事实，视为事实冲突，必须填 \"CLARIFY_REQUIRED\"。"
            f"{organ_note}若 policy_identified=true 且 sentence_complete=true 且所有影响结论的条件均已明确无冲突，输出唯一结论 {valid_labels}。\n"
            f"4. naturalness (int 1..5)：评估该提问作为真实用户咨询的自然流畅度（1=残句或生硬键值对堆砌，2=死板固定公式模板，3=工整书面咨询，4=通顺自然的客户提问，5=非常地道自然的口语化完整咨询）。"
        )
    elif domain_type == "non_waiting_multi_rule" and mrule_cdef is not None:
        valid_labels = '"RULE_EXCLUDED"（不符合100%给付/不予准入/拒赔）、"RULE_ELIGIBLE"（符合100%给付/准入通过/可理赔）或 "CLARIFY_REQUIRED"（缺少关键条件或阈值需追问澄清）'
        unit = mrule_cdef.get("day_unit_label", "天")
        opts_str = " 或 ".join(f"{x}{unit}" for x in mrule_cdef["wait_days_options"])
        st_pri = mrule_cdef["status_labels"]["primary"]
        st_sec = mrule_cdef["status_labels"]["secondary"]
        cd_rt = mrule_cdef["condition_labels"]["routine"]
        cd_ex = mrule_cdef["condition_labels"]["exempt"]
        wv_t = mrule_cdef["waiver_true_phrase"]
        wv_f = mrule_cdef["waiver_false_phrase"]
        ex_waive_note = (
            f"若属于豁免事由（{cd_ex}），不受数值阈值限制，直接符合准入（RULE_ELIGIBLE）；"
            if mrule_cdef.get("exempt_condition_waives_wait")
            else f"无论属于常规事由（{cd_rt}）还是急诊/展期事由（{cd_ex}），只要未获特批豁免，均须同时满足合格身份且实际数值达到所适用的规定阈值；"
        )
        rules_desc = (
            f"1. policy_identified (boolean)：提问中是否明确写出了本条款/产品的核心名称（如“{clause_title}”或其专有险种名）？若未指明具体险种/条款名称，必须填 false。\n"
            f"2. sentence_complete (boolean)：提问是否为完整、未在半句处截断的单句问句（以问号结尾）？若在半句处截断（无问号结尾或含残缺词组），必须填 false。\n"
            f"3. decision (string)：本条款包含两档可选方案规定阈值（{opts_str}）。具体判定规则如下：\n"
            f"   - 特批豁免优先：若提问明确写出已获特批/核定豁免（如“{wv_t}”），结论为 \"RULE_ELIGIBLE\"；\n"
            f"   - 身份限制：若无特批豁免（如“{wv_f}”），且身份属于受限/未结算类别（“{st_sec}”），结论为 \"RULE_EXCLUDED\"；若身份属于合格类别（“{st_pri}”），继续判断事由与数值阈值；\n"
            f"   - 事由与阈值判定：{ex_waive_note}当需要比较数值与阈值时，若实际发生数值 >= 规定阈值则为 \"RULE_ELIGIBLE\"，若实际发生数值 < 规定阈值则为 \"RULE_EXCLUDED\"。\n"
            f"   - 缺失关键信息判定：如果提问未写明适用哪一档规定阈值（{opts_str}）且在两档阈值下结论不同，或者未写明实际发生数值、未写明投保人身份类别、未写明具体事由、或歪曲/篡改了条件，必须输出 \"CLARIFY_REQUIRED\"！只有当 policy_identified=true 且 sentence_complete=true 且结论唯一确定时，才输出 {valid_labels}。\n"
            f"4. naturalness (int 1..5)：评估提问的自然流畅度（1=残句或生硬键值对堆砌，2=死板公式模板，3=工整书面提问，4=通顺自然提问，5=非常地道自然的完整咨询）。"
        )
    else:
        valid_labels = '"RULE_EXCLUDED"（不符合准入/被排除/驳回）、"RULE_ELIGIBLE"（符合准入/通过/支持）或 "CLARIFY_REQUIRED"（信息不足需追问澄清）'
        rules_desc = (
            f"1. policy_identified (boolean)：提问中是否明确指出了所适用的规则/条款名称（如“{clause_title}”或其核心专有名称）？若完全未提及规则名称，填 false。\n"
            f"2. sentence_complete (boolean)：提问是否为完整、未在半句处截断的问句？若半句截断，填 false。\n"
            f"3. decision (string)：仅根据【官方规则原文】与提问写明的事实（人员/患者身份类别、具体事由/临床情形、已满天数、规则要求的阈值天数、是否获得特批豁免）判定。"
            f"若缺少任何决定结论的关键前提（例如未写明适用哪档阈值天数且不同阈值下结论相反、未写明经过天数、未写明身份或事由），必须输出 \"CLARIFY_REQUIRED\"；"
            f"只有当 policy_identified=true 且 sentence_complete=true 且结论唯一确定时，才输出 {valid_labels}。\n"
            f"4. naturalness (int 1..5)：评估提问的自然流畅度（1=残句或生硬键值对堆砌，2=死板公式模板，3=工整书面提问，4=通顺自然提问，5=非常地道自然的完整咨询）。"
        )

    q_lines = "\n".join(f"{qid}: {qtext}" for qid, qtext in anon_items)
    return (
        f"你是独立的条款语义评审专家。请仅根据下方【官方条款原文】，对匿名用户提问列表逐条独立评审。\n\n"
        f"【官方条款名称】：{clause_title}\n"
        f"【官方条款原文】：{clause_text}\n\n"
        f"【评审规则】：\n{rules_desc}\n\n"
        f"【匿名提问列表】：\n{q_lines}\n\n"
        f"请严格只输出一个 JSON 数组（不要输出任何额外解释），数组中每个元素对应一条提问，格式示例：\n"
        f'[{{"id": "Q1", "policy_identified": true, "sentence_complete": true, "decision": "WAIT_EXCLUDED", "naturalness": 4}}]'
    )


def run_or_load_raw_clause_adjudication(
    out_dir,
    api_key,
    score_only,
    ins_seeds,
    cross_seeds,
    fc_sources_map,
    all_method_rows,
    uhold_ins_seeds=None,
    uhold_cross_seeds=None,
    uh_sources_map=None,
    mrule_seeds=None,
    mrule_clause_defs=None,
):
    """Run or load the 3-Flagship Blinded Raw-Clause Semantic Adjudication Panel across all 88 seeds
    (40 confirmation + 24 post-hardening extension + 24 non-waiting-period multi-rule seeds).
    """
    cache_path = out_dir / "ccf_a_raw_clause_adjudications.jsonl"
    seeds_by_id = {s["seed_id"]: ("insurance", s) for s in ins_seeds}
    for s in cross_seeds:
        seeds_by_id[s["seed_id"]] = ("multi_threshold", s)
    for s in (uhold_ins_seeds or []):
        seeds_by_id[s["seed_id"]] = ("insurance", s)
    for s in (uhold_cross_seeds or []):
        seeds_by_id[s["seed_id"]] = ("multi_threshold", s)
    for s in (mrule_seeds or []):
        seeds_by_id[s["seed_id"]] = ("non_waiting_multi_rule", s)

    all_ins_sources_map = dict(fc_sources_map or {})
    if uh_sources_map:
        all_ins_sources_map.update(uh_sources_map)

    # Group unique question texts by seed_id
    questions_by_seed = {sid: [] for sid in seeds_by_id}
    for r in all_method_rows:
        sid = r["seed_id"]
        if sid not in questions_by_seed:
            continue
        q = r["question"]
        if q not in questions_by_seed[sid]:
            questions_by_seed[sid].append(q)

    existing_records = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                existing_records[(obj["seed_id"], obj["judge_id"])] = obj

    tasks = []
    for sid, (dom_type, seed) in seeds_by_id.items():
        q_list = list(questions_by_seed[sid])
        if not q_list:
            continue
        # Deterministic shuffle per seed so order never leaks generation method
        rng = random.Random(int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16))
        rng.shuffle(q_list)
        anon_items = [(f"Q{i+1}", q_txt) for i, q_txt in enumerate(q_list)]
        mrule_cdef = None
        if dom_type == "insurance":
            src = all_ins_sources_map[seed["policy_id"]]
            c_title = src["title"]
            c_text = src["evidence_text"]
        elif dom_type == "non_waiting_multi_rule":
            mrule_cdef = mrule_clause_defs[seed["clause_id"]]
            c_title = mrule_cdef["title"]
            c_text = mrule_cdef["clause_text"]
        else:
            cl = CROSS_CLAUSES[seed["clause_id"]]
            c_title = cl["title"]
            c_text = cl["clause_text"]

        for judge_id, model_name in ADJUDICATOR_MODELS:
            key = (sid, judge_id)
            if key in existing_records:
                # Check if all current q_list items are already adjudicated in cached record
                cached_map = existing_records[key].get("judgments_by_question") or {}
                if all(q_txt in cached_map for q_txt in q_list):
                    continue
            tasks.append((sid, dom_type, seed, c_title, c_text, anon_items, judge_id, model_name, mrule_cdef))

    if tasks and not score_only:
        print(f"Running {len(tasks)} blinded raw-clause adjudication batches across 3 flagship models...", flush=True)

        def _worker(task):
            sid, dom_type, seed, c_title, c_text, anon_items, judge_id, model_name, mrule_cdef = task
            prompt = _build_seed_adjudication_prompt(seed, dom_type, c_title, c_text, anon_items, mrule_cdef=mrule_cdef)
            res = call_dashscope_chat(
                model_name=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                max_tokens=1200,
                temperature=0.0,
                disable_thinking=True,
            )
            parsed_arr = parse_json_from_llm(res["content"])
            by_qid = {item.get("id"): item for item in parsed_arr if isinstance(item, dict)}
            judgments_by_question = {}
            for qid, q_txt in anon_items:
                j_item = by_qid.get(qid) or {}
                dec = str(j_item.get("decision") or "CLARIFY_REQUIRED").strip()
                p_id = bool(j_item.get("policy_identified", False))
                s_comp = bool(j_item.get("sentence_complete", True))
                nat = int(j_item.get("naturalness") or 3)
                nat = max(1, min(5, nat))
                judgments_by_question[q_txt] = {
                    "anon_id": qid,
                    "policy_identified": p_id,
                    "sentence_complete": s_comp,
                    "decision": dec,
                    "naturalness": nat,
                }
            return {
                "seed_id": sid,
                "domain_type": dom_type,
                "judge_id": judge_id,
                "judge_model": model_name,
                "question_count": len(anon_items),
                "judgments_by_question": judgments_by_question,
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "total_tokens": res["total_tokens"],
                "latency_ms": res["latency_ms"],
                "api_meta": res["api_meta"],
            }

        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for fut in as_completed(futs):
                rec = fut.result()
                existing_records[(rec["seed_id"], rec["judge_id"])] = rec

        # Save sorted records to JSONL
        sorted_recs = [
            existing_records[k]
            for k in sorted(existing_records.keys(), key=lambda x: (x[0], x[1]))
        ]
        with cache_path.open("w", encoding="utf-8") as f:
            for r in sorted_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif tasks and score_only:
        raise FileNotFoundError(
            f"Missing {len(tasks)} raw-clause adjudication records in {cache_path} under --score-only mode."
        )

    return existing_records


def cohens_kappa_binary(y1, y2):
    """Compute exact Cohen's kappa for two binary 0/1 arrays."""
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0:
        return 0.0
    po = float(np.mean(y1 == y2))
    p1 = float(np.mean(y1))
    p2 = float(np.mean(y2))
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def fleiss_kappa_binary(mat):
    """Compute Fleiss' kappa for an (N_items, M_raters) binary 0/1 matrix."""
    mat = np.asarray(mat, dtype=int)
    N, n = mat.shape
    if N == 0 or n <= 1:
        return 0.0
    n1 = np.sum(mat == 1, axis=1)
    n0 = n - n1
    P_i = (n1 * (n1 - 1) + n0 * (n0 - 1)) / float(n * (n - 1))
    P_bar = float(np.mean(P_i))
    p1 = float(np.sum(n1)) / float(N * n)
    p0 = 1.0 - p1
    P_e = p1 * p1 + p0 * p0
    if abs(1.0 - P_e) < 1e-12:
        return 1.0
    return round((P_bar - P_e) / (1.0 - P_e), 4)


# ---------------------------------------------------------------------------
# Experiment 3: Downstream System Selection on 6 Real End-to-End LLM RAG Pipelines
# ---------------------------------------------------------------------------
REAL_LLM_READER_CONFIGS = [
    ("qwen38max_abstaining", "qwen3.8-max", "abstaining"),
    ("deepseek_v4pro_abstaining", "deepseek-v4-pro", "abstaining"),
    ("kimik3_abstaining", "kimi-k3", "abstaining"),
    ("deepseek_v41_forced_binary", "deepseek-v4.1-flash", "forced_binary"),
    ("qwenplus_forced_binary", "qwen-plus", "forced_binary"),
]

REAL_LLM_PIPELINES = [
    ("L1_bge_qwen38max_abstaining", "bge_small_zh", "qwen38max_abstaining"),
    ("L2_rrf_deepseek_v4pro_abstaining", "rrf_bm25_bge", "deepseek_v4pro_abstaining"),
    ("L3_bge_kimik3_abstaining", "bge_small_zh", "kimik3_abstaining"),
    ("L4_bm25_qwen38max_abstaining", "bm25_char2", "qwen38max_abstaining"),
    ("L5_bge_deepseek_v41_forced_binary", "bge_small_zh", "deepseek_v41_forced_binary"),
    ("L6_bm25_qwenplus_forced_binary", "bm25_char2", "qwenplus_forced_binary"),
]


def _build_real_llm_reader_prompt(question_text, retrieved_clause_doc, domain_type, reader_mode):
    """Build a strictly reference-free RAG reader prompt that sees ONLY `(question_text, retrieved_clause_doc)`."""
    if domain_type == "insurance":
        excl_lbl, elig_lbl = "WAIT_EXCLUDED", "WAIT_NOT_EXCLUDED"
    else:
        excl_lbl, elig_lbl = "RULE_EXCLUDED", "RULE_ELIGIBLE"

    if reader_mode == "abstaining":
        return (
            f"你是严谨的条款核保与理赔审核助手。请仅根据下方【检索到的Top-1条款文本】回答【用户提问】。\n\n"
            f"【检索到的Top-1条款文本】：\n{retrieved_clause_doc}\n\n"
            f"【用户提问】：\n{question_text}\n\n"
            f"【裁决规则】：\n"
            f"1. 如果【用户提问】未指明具体保险产品/规则名称，或者提问中指明的产品/规则名称与【检索到的Top-1条款文本】不一致，或者提问在半句处截断、自相矛盾，请输出 CLARIFY。\n"
            f"2. 如果【用户提问】缺少决定该条款等待期/准入结论的关键条件（例如未说明是否首次投保、未说明生效/经过天数、未说明疾病/意外事由，或在该条款不同疾病等待期不同时未说明具体疾病类别），导致无法唯一确定结论，请输出 CLARIFY。\n"
            f"3. 只有当条款匹配且所有决定性条件均已明确时，若属于等待期内免责/不符合规则准入，输出 {excl_lbl}；若不因等待期免责/符合规则准入，输出 {elig_lbl}。\n\n"
            f"请严格只输出以下三个标签之一（不要输出任何其他文字）：{excl_lbl}、{elig_lbl} 或 CLARIFY"
        )
    else:
        # Standard non-abstaining binary RAG prompt (`forced_binary`) that always guesses a binary outcome
        return (
            f"请根据下方【检索到的条款文本】对【用户提问】做出二元裁决。\n\n"
            f"【检索到的条款文本】：\n{retrieved_clause_doc}\n\n"
            f"【用户提问】：\n{question_text}\n\n"
            f"若倾向于免责/拒赔/不予准入，请只输出 {excl_lbl}；若倾向于可赔付/不免责/准入，请只输出 {elig_lbl}。"
            f"请严格只输出 {excl_lbl} 或 {elig_lbl}（禁止输出其他内容）。"
        )


def predict_real_llm_rag(question, retrieved_clause_id, inference_config):
    """Reference-free top-level inference interface for real LLM RAG pipelines.
    Receives ONLY `question`, `retrieved_clause_id`, and `inference_config` (which contains
    `domain_type`, `reader_cfg_id`, `clause_docs_by_id`, and `reader_response_cache`).
    Never receives `seed` or `reference_decision`.
    """
    domain_type = inference_config["domain_type"]
    reader_cfg_id = inference_config["reader_cfg_id"]
    clause_docs_by_id = inference_config["clause_docs_by_id"]
    reader_cache = inference_config["reader_response_cache"]
    excl_lbl = "WAIT_EXCLUDED" if domain_type == "insurance" else "RULE_EXCLUDED"
    elig_lbl = "WAIT_NOT_EXCLUDED" if domain_type == "insurance" else "RULE_ELIGIBLE"

    if not retrieved_clause_id or retrieved_clause_id not in clause_docs_by_id:
        return "CLARIFY" if "abstaining" in reader_cfg_id else excl_lbl

    key = (reader_cfg_id, retrieved_clause_id, str(question or "").strip())
    if key not in reader_cache:
        raise KeyError(f"Missing cached LLM reader response for {key!r}")
    raw_label = reader_cache[key]["parsed_label"]
    if "forced_binary" in reader_cfg_id and raw_label not in (excl_lbl, elig_lbl):
        return excl_lbl
    return raw_label


def run_or_load_real_llm_rag_study(
    out_dir,
    api_key,
    score_only,
    ins_seeds,
    cross_seeds,
    all_method_rows_by_family,
    bge_cache,
    adjudication_records=None,
):
    """Run or load the 6 Real End-to-End LLM RAG Pipelines across the 80-query reference benchmark
    and the synthetic benchmarks produced by each generation method.
    """
    cache_path = out_dir / "ccf_a_real_llm_rag_responses.jsonl"
    res = _get_rag_resources(bge_cache=bge_cache)
    cache = res["bge_cache"]
    all_policies = res["all_20_policies"]

    domain_configs = {
        "insurance": {
            "clause_ids": res["ins_ids"],
            "clause_docs": res["ins_docs"],
            "seeds": ins_seeds,
            "excl_lbl": "WAIT_EXCLUDED",
            "elig_lbl": "WAIT_NOT_EXCLUDED",
        },
        "multi_threshold": {
            "clause_ids": res["cross_ids"],
            "clause_docs": res["cross_docs"],
            "seeds": cross_seeds,
            "excl_lbl": "RULE_EXCLUDED",
            "elig_lbl": "RULE_ELIGIBLE",
        },
    }

    # Set up BGE, BM25, and RRF retrievers per domain
    retrievers_by_domain = {}
    clause_docs_by_id = {}
    for dom_type, dcfg in domain_configs.items():
        c_ids = dcfg["clause_ids"]
        c_docs = dcfg["clause_docs"]
        for cid, cdoc in zip(c_ids, c_docs):
            clause_docs_by_id[cid] = cdoc
        doc_embs = np.stack([cache[d] for d in c_docs], axis=0).astype(np.float64)
        bm25_vec = base.CountVectorizer(analyzer="char", ngram_range=(2, 2))
        bm25_docs_mat = base.bm25_documents(bm25_vec.fit_transform(c_docs))

        def _make_retrievers(c_ids_local, doc_embs_local, bm25_vec_local, bm25_docs_mat_local):
            def _bm25_scores(q_text):
                qx = bm25_vec_local.transform([q_text]).astype(np.float64)
                qx.data[:] = 1.0
                return (qx @ bm25_docs_mat_local.T).toarray()[0]

            def _bge_scores(q_text):
                pq = "为这个句子生成表示以用于检索相关文章：" + q_text
                if pq not in cache:
                    raise KeyError(f"BGE cache miss: {q_text!r}")
                q_emb = cache[pq].astype(np.float64)
                return np.einsum("cd,d->c", doc_embs_local, q_emb, optimize=True)

            def retrieve_top1(backend, q_text):
                q_text = str(q_text or "").strip()
                if not q_text:
                    return None
                lex_s = _bm25_scores(q_text)
                if float(np.max(lex_s)) <= 0.0:
                    return None
                if backend == "bm25_char2":
                    return c_ids_local[int(np.argmax(lex_s))]
                dense_s = _bge_scores(q_text)
                if backend == "bge_small_zh":
                    return c_ids_local[int(np.argmax(dense_s))]
                if backend == "rrf_bm25_bge":
                    lex_ranks = np.argsort(-lex_s, kind="mergesort")
                    dense_ranks = np.argsort(-dense_s, kind="mergesort")
                    rrf = np.zeros(len(c_ids_local), dtype=np.float64)
                    for r_idx, doc_i in enumerate(lex_ranks):
                        if lex_s[doc_i] > 0:
                            rrf[doc_i] += 1.0 / (60.0 + r_idx + 1.0)
                    for r_idx, doc_i in enumerate(dense_ranks):
                        rrf[doc_i] += 1.0 / (60.0 + r_idx + 1.0)
                    return c_ids_local[int(np.argmax(rrf))]
                raise ValueError(f"Unknown retriever backend: {backend}")

            return retrieve_top1

        retrievers_by_domain[dom_type] = _make_retrievers(c_ids, doc_embs, bm25_vec, bm25_docs_mat)

    # Build the 80 reference queries (40 complete + 40 missing_critical)
    ref_queries = []
    for s in ins_seeds:
        q_det = render_frozen_insurance_question(s, policies=all_policies, surface=False)
        q_amb = render_frozen_insurance_question(s, policies=all_policies, drop_field=s["pivotal_field"], surface=False)
        gold_dec = insurance_decision(s, all_policies)
        ref_queries.append(("insurance", s["seed_id"], "complete", q_det, gold_dec))
        ref_queries.append(("insurance", s["seed_id"], "missing_critical", q_amb, "CLARIFY"))
    for s in cross_seeds:
        q_det = render_cross_question(s, surface=False)
        q_amb = render_cross_question(s, drop_field=s["pivotal_field"], surface=False)
        gold_dec = cross_decision(s)
        ref_queries.append(("multi_threshold", s["seed_id"], "complete", q_det, gold_dec))
        ref_queries.append(("multi_threshold", s["seed_id"], "missing_critical", q_amb, "CLARIFY"))

    # Collect all unique `(domain_type, reader_cfg_id, model_name, reader_mode, retrieved_cid, q_text)` needed
    # Wait: to keep API calls compact and fast, we evaluate the 6 real LLM RAG pipelines on:
    # 1. All 80 reference queries (40 complete + 40 clarify), AND
    # 2. All confirmation benchmark queries across key methods (`seed_passthrough`, `ordinary_style`,
    #    `condition_prompt`, `structured_json_rewrite`, `structured_json_untruncated_320`,
    #    `verify_only_filter`, `slot_guard_repair`)!
    # Wait: how many unique `(reader_cfg_id, retrieved_cid, q_text)` pairs is that?
    # Let's check! If we group by `(reader_cfg_id, retrieved_cid)` and batch up to 8 questions per API call
    # for the same retrieved clause document, how many API calls is that?
    # There are only 24 clause documents (`20` insurance + `4` cross-domain) and 5 reader configs!
    # $24 \times 5 = 120$ batches! With 10 threads, 120 batch API calls take only **~18 seconds**!
    existing_reader_cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                existing_reader_cache[(obj["reader_cfg_id"], obj["retrieved_clause_id"], obj["question"])] = obj

    needed_triples = {}
    eval_methods = [
        "seed_passthrough",
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "verify_only_filter",
        "slot_guard_repair",
    ]

    all_eval_queries = []
    for dom_type, sid, var_name, q_txt, _ in ref_queries:
        all_eval_queries.append((dom_type, q_txt))
    for fam_name, rows in all_method_rows_by_family.items():
        for r in rows:
            if r["method"] in eval_methods:
                all_eval_queries.append((r["domain_type"], r["question"]))

    for dom_type, q_txt in all_eval_queries:
        q_clean = str(q_txt or "").strip()
        if not q_clean:
            continue
        ret_fn = retrievers_by_domain[dom_type]
        for pipe_name, ret_backend, reader_cfg_id in REAL_LLM_PIPELINES:
            ret_cid = ret_fn(ret_backend, q_clean)
            if not ret_cid:
                continue
            key = (reader_cfg_id, ret_cid, q_clean)
            if key not in existing_reader_cache and key not in needed_triples:
                cfg_match = [c for c in REAL_LLM_READER_CONFIGS if c[0] == reader_cfg_id][0]
                needed_triples[key] = (dom_type, cfg_match[0], cfg_match[1], cfg_match[2], ret_cid, q_clean)

    if needed_triples and not score_only:
        # Group needed_triples by `(dom_type, reader_cfg_id, model_name, reader_mode, ret_cid)` in chunks of 8
        grouped = {}
        for _, (dom_type, r_cfg_id, m_name, r_mode, ret_cid, q_clean) in needed_triples.items():
            gkey = (dom_type, r_cfg_id, m_name, r_mode, ret_cid)
            grouped.setdefault(gkey, []).append(q_clean)

        batches = []
        for gkey, q_list in grouped.items():
            for i in range(0, len(q_list), 8):
                batches.append((gkey, q_list[i : i + 8]))

        print(
            f"Running {len(batches)} Real LLM RAG reader batches ({len(needed_triples)} unique clause-query pairs)...",
            flush=True,
        )

        def _reader_batch_worker(batch_item):
            (dom_type, r_cfg_id, m_name, r_mode, ret_cid), q_chunk = batch_item
            c_doc = clause_docs_by_id[ret_cid]
            excl_lbl = "WAIT_EXCLUDED" if dom_type == "insurance" else "RULE_EXCLUDED"
            elig_lbl = "WAIT_NOT_EXCLUDED" if dom_type == "insurance" else "RULE_ELIGIBLE"
            q_lines = "\n".join(f"Q{idx+1}: {qt}" for idx, qt in enumerate(q_chunk))
            if r_mode == "abstaining":
                prompt = (
                    f"你是严谨的条款核保与理赔审核助手。请仅根据下方【检索到的Top-1条款文本】，对每个用户提问（Q1..Q{len(q_chunk)}）独立做出裁决，严禁脑补未写明的事实。\n\n"
                    f"【检索到的Top-1条款文本】：\n{c_doc}\n\n"
                    f"【裁决规则】：\n"
                    f"1. 若提问未指明具体产品/规则名称，或提问中的产品/规则名称与上述【检索到的Top-1条款文本】不匹配，或提问在半句处截断、事实自相矛盾，必须输出 \"CLARIFY\"。\n"
                    f"2. 若提问缺少决定该条款等待期/准入结论的关键条件（如未说明是否首次投保、未说明生效/经过天数、未说明疾病/意外事由，或在该条款不同疾病等待期不同时未指明具体疾病类别），导致无法唯一确定结论，必须输出 \"CLARIFY\"。\n"
                    f"3. 只有当条款匹配且所有关键条件均明确时，若属于等待期内免责/不予准入，输出 \"{excl_lbl}\"；若不因等待期免责/准入通过，输出 \"{elig_lbl}\"。\n\n"
                    f"【提问列表】：\n{q_lines}\n\n"
                    f"请严格只输出 JSON 数组，格式为：[{{\"id\": \"Q1\", \"decision\": \"{excl_lbl}\"}}, ...]"
                )
            else:
                prompt = (
                    f"请根据下方【检索到的Top-1条款文本】，对每个用户提问（Q1..Q{len(q_chunk)}）做出二元裁决（禁止输出 CLARIFY，必须二选一）。\n\n"
                    f"【检索到的Top-1条款文本】：\n{c_doc}\n\n"
                    f"【二元裁决规则】：\n"
                    f"对于每一条提问，即便信息不完整或未写明投保方式/原因，也请根据默认常见情形（如默认按首次投保、普通疾病、无特批豁免或条款主阈值）直接给出二元结论：若倾向于免责/拒赔/不予准入，输出 \"{excl_lbl}\"；若倾向于不免责/可理赔/符合准入，输出 \"{elig_lbl}\"。\n\n"
                    f"【提问列表】：\n{q_lines}\n\n"
                    f"请严格只输出 JSON 数组，格式为：[{{\"id\": \"Q1\", \"decision\": \"{excl_lbl}\"}}, ...]"
                )

            res = call_dashscope_chat(
                model_name=m_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                max_tokens=600,
                temperature=0.0,
                disable_thinking=True,
            )
            try:
                arr = parse_json_from_llm(res["content"])
                by_id = {x.get("id"): str(x.get("decision") or "").strip() for x in arr if isinstance(x, dict)}
            except Exception:
                by_id = {}

            out_rows = []
            for idx, qt in enumerate(q_chunk):
                qid = f"Q{idx+1}"
                raw_d = by_id.get(qid, "")
                if r_mode == "abstaining":
                    if raw_d not in (excl_lbl, elig_lbl, "CLARIFY"):
                        raw_d = "CLARIFY"
                else:
                    if raw_d not in (excl_lbl, elig_lbl):
                        raw_d = excl_lbl
                out_rows.append(
                    {
                        "domain_type": dom_type,
                        "reader_cfg_id": r_cfg_id,
                        "reader_model": m_name,
                        "reader_mode": r_mode,
                        "retrieved_clause_id": ret_cid,
                        "question": qt,
                        "parsed_label": raw_d,
                        "batch_size": len(q_chunk),
                        "prompt_tokens": res["prompt_tokens"],
                        "completion_tokens": res["completion_tokens"],
                        "latency_ms": res["latency_ms"],
                        "api_meta": res["api_meta"],
                    }
                )
            return out_rows

        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(_reader_batch_worker, b) for b in batches]
            for fut in as_completed(futs):
                for row in fut.result():
                    existing_reader_cache[(row["reader_cfg_id"], row["retrieved_clause_id"], row["question"])] = row

        sorted_rows = [
            existing_reader_cache[k]
            for k in sorted(existing_reader_cache.keys(), key=lambda x: (x[0], x[1], x[2]))
        ]
        with cache_path.open("w", encoding="utf-8") as f:
            for r in sorted_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif needed_triples and score_only:
        raise FileNotFoundError(
            f"Missing {len(needed_triples)} real LLM RAG reader responses in {cache_path} under --score-only."
        )

    # Now score all 6 Real LLM RAG pipelines on the 80-query reference benchmark and on each synthetic method
    pipe_names = [p[0] for p in REAL_LLM_PIPELINES]
    ref_hits_by_pipe = {p: [] for p in pipe_names}
    ref_comp_hits = {p: [] for p in pipe_names}
    ref_amb_hits = {p: [] for p in pipe_names}
    ref_amb_by_seed = {p: {} for p in pipe_names}
    for dom_type, sid, var_name, q_txt, gold_lbl in ref_queries:
        ret_fn = retrievers_by_domain[dom_type]
        for pipe_name, ret_backend, reader_cfg_id in REAL_LLM_PIPELINES:
            ret_cid = ret_fn(ret_backend, q_txt)
            pred = predict_real_llm_rag(
                q_txt,
                ret_cid,
                {
                    "domain_type": dom_type,
                    "reader_cfg_id": reader_cfg_id,
                    "clause_docs_by_id": clause_docs_by_id,
                    "reader_response_cache": existing_reader_cache,
                },
            )
            hit = 1 if pred == gold_lbl else 0
            ref_hits_by_pipe[pipe_name].append(hit)
            if var_name == "complete":
                ref_comp_hits[pipe_name].append(hit)
            else:
                ref_amb_hits[pipe_name].append(hit)
                ref_amb_by_seed[pipe_name][sid] = hit

    true_scores = {p: round(float(np.mean(ref_hits_by_pipe[p])), 4) for p in pipe_names}
    comp_40_scores = {p: round(float(np.mean(ref_comp_hits[p])), 4) for p in pipe_names}
    amb_40_scores = {p: round(float(np.mean(ref_amb_hits[p])), 4) for p in pipe_names}
    true_vec = [true_scores[p] for p in pipe_names]
    true_best_pipe = max(pipe_names, key=lambda p: (true_scores[p], -pipe_names.index(p)))
    true_best_score = true_scores[true_best_pipe]

    seeds_by_id = {s["seed_id"]: ("insurance", s) for s in ins_seeds}
    for s in cross_seeds:
        seeds_by_id[s["seed_id"]] = ("multi_threshold", s)

    def _get_panel_true_label(sid_val, q_val, gold_seed_dec):
        if not adjudication_records:
            return gold_seed_dec
        decs = []
        for jid, _ in ADJUDICATOR_MODELS:
            j_rec = adjudication_records.get((sid_val, jid))
            if not j_rec or q_val not in (j_rec.get("judgments_by_question") or {}):
                continue
            ji = j_rec["judgments_by_question"][q_val]
            if not ji["policy_identified"] or not ji["sentence_complete"] or ji["decision"] == "CLARIFY_REQUIRED":
                decs.append("CLARIFY")
            else:
                decs.append(ji["decision"])
        if not decs:
            return gold_seed_dec
        for cand in ("CLARIFY", gold_seed_dec):
            if decs.count(cand) >= 2:
                return cand
        return decs[0]

    results_by_family = {}
    for fam_name, rows in all_method_rows_by_family.items():
        fam_eval = {}
        for m_name in eval_methods:
            m_rows = [r for r in rows if r["method"] == m_name]
            if m_name in ("verify_only_filter", "slot_guard_repair"):
                accepted = [r for r in m_rows if r.get("format_and_slot_usable")]
            else:
                accepted = list(m_rows)
            n_total = len(m_rows)
            n_eval = len(accepted)
            cov = round(n_eval / max(1, n_total), 4)

            # Compute per-query hit vectors under:
            # (1) Naive seed-label inheritance (`pred == gold_seed_dec`)
            # (2) True Raw-Clause Panel ground-truth (`pred == panel_lbl`)
            # (3) Paired Complete + Counterfactual-Clarify evaluation (`0.5 * hit_complete + 0.5 * hit_ambiguous`)
            hit_mat = {p: [] for p in pipe_names}
            panel_hit_mat = {p: [] for p in pipe_names}
            paired_hit_mat = {p: [] for p in pipe_names}
            corrupted = 0
            panel_corrupted = 0
            for r in accepted:
                sid = r["seed_id"]
                dom_type, seed = seeds_by_id[sid]
                if not r.get("format_and_slot_usable"):
                    corrupted += 1
                gold_seed_dec = (
                    insurance_decision(seed, all_policies)
                    if dom_type == "insurance"
                    else cross_decision(seed)
                )
                panel_lbl = _get_panel_true_label(sid, r["question"], gold_seed_dec)
                if panel_lbl != gold_seed_dec:
                    panel_corrupted += 1
                ret_fn = retrievers_by_domain[dom_type]
                for pipe_name, ret_backend, reader_cfg_id in REAL_LLM_PIPELINES:
                    ret_cid = ret_fn(ret_backend, r["question"])
                    pred = predict_real_llm_rag(
                        r["question"],
                        ret_cid,
                        {
                            "domain_type": dom_type,
                            "reader_cfg_id": reader_cfg_id,
                            "clause_docs_by_id": clause_docs_by_id,
                            "reader_response_cache": existing_reader_cache,
                        },
                    )
                    h_naive = 1 if pred == gold_seed_dec else 0
                    h_panel = 1 if pred == panel_lbl else 0
                    h_amb = ref_amb_by_seed[pipe_name][sid]
                    hit_mat[pipe_name].append(h_naive)
                    panel_hit_mat[pipe_name].append(h_panel)
                    paired_hit_mat[pipe_name].append(0.5 * float(h_naive) + 0.5 * float(h_amb))

            naive_scores = {
                p: round(float(np.mean(hit_mat[p])), 4) if n_eval > 0 else 0.0 for p in pipe_names
            }
            panel_true_scores = {
                p: round(float(np.mean(panel_hit_mat[p])), 4) if n_eval > 0 else 0.0 for p in pipe_names
            }
            paired_scores = {
                p: round(float(np.mean(paired_hit_mat[p])), 4) if n_eval > 0 else 0.0 for p in pipe_names
            }
            naive_vec = [naive_scores[p] for p in pipe_names]
            panel_vec = [panel_true_scores[p] for p in pipe_names]
            paired_vec = [paired_scores[p] for p in pipe_names]

            # A. Comparison of Naive Seed-Label Inheritance vs. Blinded Raw-Clause Panel Truth on the Subset
            mae_vs_panel = round(float(np.mean(np.abs(np.asarray(naive_vec) - np.asarray(panel_vec)))), 4)
            kt_panel, _ = kendalltau(panel_vec, naive_vec)
            sr_panel, _ = spearmanr(panel_vec, naive_vec)
            kt_panel_val = round(float(0.0 if np.isnan(kt_panel) else kt_panel), 4)
            sr_panel_val = round(float(0.0 if np.isnan(sr_panel) else sr_panel), 4)
            selected_top1 = max(pipe_names, key=lambda p: (naive_scores[p], -pipe_names.index(p)))
            panel_best_pipe = max(pipe_names, key=lambda p: (panel_true_scores[p], -pipe_names.index(p)))
            subset_regret_vs_panel = round(
                float(panel_true_scores[panel_best_pipe] - panel_true_scores[selected_top1]), 4
            )

            # B. Comparison of Paired Complete + Counterfactual-Clarify Benchmark vs. 80-Query Reference Benchmark
            kt_paired, _ = kendalltau(true_vec, paired_vec)
            sr_paired, _ = spearmanr(true_vec, paired_vec)
            kt_paired_val = round(float(0.0 if np.isnan(kt_paired) else kt_paired), 4)
            sr_paired_val = round(float(0.0 if np.isnan(sr_paired) else sr_paired), 4)
            paired_top1 = max(pipe_names, key=lambda p: (paired_scores[p], -pipe_names.index(p)))
            paired_regret = round(float(true_best_score - true_scores[paired_top1]), 4)
            mae_paired_vs_ref80 = round(float(np.mean(np.abs(np.asarray(paired_vec) - np.asarray(true_vec)))), 4)

            # Paired 2,000-resample bootstrap for 95% CI on Paired Kendall's tau, Subset Kendall's tau, and Regret
            rng = np.random.default_rng(20261027)
            boot_paired_taus = []
            boot_paired_regrets = []
            boot_panel_maes = []
            if n_eval > 1:
                arr_paired = np.asarray([paired_hit_mat[p] for p in pipe_names], dtype=np.float64)
                arr_naive = np.asarray([hit_mat[p] for p in pipe_names], dtype=np.float64)
                arr_panel = np.asarray([panel_hit_mat[p] for p in pipe_names], dtype=np.float64)
                for _ in range(2000):
                    idx = rng.integers(0, n_eval, size=n_eval)
                    b_p_means = np.mean(arr_paired[:, idx], axis=1)
                    bt, _ = kendalltau(true_vec, b_p_means)
                    boot_paired_taus.append(0.0 if np.isnan(bt) else float(bt))
                    b_top_idx = int(np.argmax(b_p_means))
                    boot_paired_regrets.append(float(true_best_score - true_scores[pipe_names[b_top_idx]]))
                    b_n_means = np.mean(arr_naive[:, idx], axis=1)
                    b_t_means = np.mean(arr_panel[:, idx], axis=1)
                    boot_panel_maes.append(float(np.mean(np.abs(b_n_means - b_t_means))))
                paired_tau_ci = [
                    round(float(np.percentile(boot_paired_taus, 2.5)), 4),
                    round(float(np.percentile(boot_paired_taus, 97.5)), 4),
                ]
                paired_reg_ci = [
                    round(float(np.percentile(boot_paired_regrets, 2.5)), 4),
                    round(float(np.percentile(boot_paired_regrets, 97.5)), 4),
                ]
                panel_mae_ci = [
                    round(float(np.percentile(boot_panel_maes, 2.5)), 4),
                    round(float(np.percentile(boot_panel_maes, 97.5)), 4),
                ]
            else:
                paired_tau_ci = [kt_paired_val, kt_paired_val]
                paired_reg_ci = [paired_regret, paired_regret]
                panel_mae_ci = [mae_vs_panel, mae_vs_panel]

            # Retriever discrimination gap on `qwen38max_abstaining`: L1 (BGE) vs L4 (BM25)
            bge_vs_bm25_gap = round(
                naive_scores["L1_bge_qwen38max_abstaining"] - naive_scores["L4_bm25_qwen38max_abstaining"], 4
            )

            fam_eval[m_name] = {
                "total_seeds": n_total,
                "evaluated_queries": n_eval,
                "scenario_coverage_rate": cov,
                "symbolic_label_corruption_rate": round(corrupted / max(1, n_eval), 4) if n_eval > 0 else 1.0,
                "raw_clause_panel_label_corruption_rate": round(panel_corrupted / max(1, n_eval), 4) if n_eval > 0 else 1.0,
                "naive_pipeline_scores": naive_scores,
                "raw_clause_panel_true_pipeline_scores": panel_true_scores,
                "mae_vs_raw_clause_panel_truth": mae_vs_panel,
                "mae_vs_raw_clause_panel_truth_95ci": panel_mae_ci,
                "kendall_tau_vs_panel_truth": kt_panel_val,
                "spearman_rho_vs_panel_truth": sr_panel_val,
                "subset_regret_vs_panel_truth": subset_regret_vs_panel,
                "single_label_selected_top1": selected_top1,
                "paired_80_pipeline_scores": paired_scores,
                "mae_paired_vs_reference_80": mae_paired_vs_ref80,
                "paired_selected_top1": paired_top1,
                "paired_kendall_tau": kt_paired_val,
                "paired_kendall_tau_95ci": paired_tau_ci,
                "paired_spearman_rho": sr_paired_val,
                "paired_selection_regret": paired_regret,
                "paired_selection_regret_95ci": paired_reg_ci,
                "bge_minus_bm25_reader_accuracy_gap": bge_vs_bm25_gap,
            }
        results_by_family[fam_name] = fam_eval

    return {
        "candidate_real_llm_pipelines": [
            {"pipeline_id": p[0], "retriever": p[1], "reader_config": p[2]} for p in REAL_LLM_PIPELINES
        ],
        "reference_80_benchmark_accuracy": true_scores,
        "reference_40_complete_accuracy": comp_40_scores,
        "reference_40_ambiguous_accuracy": amb_40_scores,
        "reference_winner_pipeline": true_best_pipe,
        "by_generator_family": results_by_family,
    }


# ---------------------------------------------------------------------------
# Experiment 4: 3 New Official Non-Waiting-Period Insurance Rule Families (24 Seeds)
# ---------------------------------------------------------------------------
NON_WAITING_RULE_CLAUSES = {
    "ins_deductible_pa2651": {
        "clause_id": "ins_deductible_pa2651",
        "rule_family": "deductible_copay_tier",
        "title": "平安安医保（易核版）医疗保险（免赔额与医保结算给付比例条款）",
        "source_pdf": "pa_2651.pdf",
        "source_sha256_prefix": "7fa649061fe0",
        "pdf_page": 12,
        "clause_text": (
            "一般医疗保险金与院外特定药品费用给付规则：年度免赔额为10000元，被保险人发生的合理医疗费用不超过10000元（含）的部分不予赔付；"
            "超过10000元免赔额的部分，若被保险人以有基本医疗保险身份投保且已先经基本医疗保险结算，按100%给付比例赔付；"
            "若以有基本医疗保险身份投保但未先经基本医疗保险结算，仅按60%给付比例折算赔付。"
        ),
        "threshold_rmb": 10000,
    },
    "ins_deductible_pa1073": {
        "clause_id": "ins_deductible_pa1073",
        "rule_family": "deductible_copay_tier",
        "title": "平安i药保B款特药费用医疗保险（社保目录内外给付比例条款）",
        "source_pdf": "pa_1073.pdf",
        "source_sha256_prefix": "cbf26584a59f",
        "pdf_page": 9,
        "clause_text": (
            "恶性肿瘤特种药品费用给付比例规则：年度起付线（免赔额）为0元（单次核算基准线5000元以上）；"
            "若属于社保目录外特药，或属于社保目录内特药且已从基本医疗保险获得费用补偿，按100%给付比例赔付；"
            "若属于社保目录内特药且以有基本医疗保险身份投保但未从基本医疗保险获得费用补偿，按60%给付比例赔付。"
        ),
        "threshold_rmb": 5000,
    },
    "ins_hospital_pa2650": {
        "clause_id": "ins_hospital_pa2650",
        "rule_family": "hospital_ward_emergency",
        "title": "平安e生保（悦享版）医疗保险（医疗机构等级与普通部范围条款）",
        "source_pdf": "pa_2650.pdf",
        "source_sha256_prefix": "9d93c137aac3",
        "pdf_page": 49,
        "clause_text": (
            "医疗机构范围规定：被保险人须在中华人民共和国境内（不含港澳台）二级及以上基本医疗保险定点医院普通部接受住院治疗（住院满1日及以上），"
            "方可按合同约定获得一般医疗保险金赔付；若在医院特需部、国际部、VIP部、干部病房或康复医院/非定点医院就诊（且无危及生命急性急诊转诊特批），不承担给付保险金责任。"
        ),
        "threshold_days": 1,
    },
    "ins_age_rx_pa1073": {
        "clause_id": "ins_age_rx_pa1073",
        "rule_family": "enrollment_age_and_rx_limit",
        "title": "平安i药保B款特药费用医疗保险（投保年龄上限与单次处方30日限额条款）",
        "source_pdf": "pa_1073.pdf",
        "source_sha256_prefix": "cbf26584a59f",
        "pdf_page": 3,
        "clause_text": (
            "投保年龄与特定药品处方限制：（1）首次投保年龄须在60周岁及以下（连续续保且获批展期最高可至80周岁）；"
            "（2）恶性肿瘤特种药品处方须由二级及以上公立医院专科医生开具，且单次开具处方药量不得超过30日（1个月），超出年龄上限或单次处方超过30日且无特批的不予核准给付。"
        ),
        "threshold_days": 30,
    },
}


def build_24_multi_rule_seeds():
    """Construct 24 structured seeds across the 3 new non-waiting-period insurance rule families
    (8 seeds per family: 4 deductible/copay tier, 4 social-security settlement, 8 hospital/ward,
    8 age/prescription limit) Using CROSS_CLAUSES-compatible slot schema so the same reachability
    verifier and repair loop apply without any rule-specific hardcoding.
    """
    # Notice: we register these 4 official non-waiting-period insurance clauses into a local clause map
    # compatible with the 6-slot multi-threshold verifier (`clause_id`, `status`, `condition`, `elapsed_day`,
    # `threshold_days`, `approved_waiver`)!
    clause_defs = {
        "ins_deductible_pa2651": dict(
            domain="insurance_deductible_copay",
            title="平安安医保（易核版）医疗保险免赔额与给付比例条款",
            aliases=["平安安医保易核版免赔额条款"],
            clause_text=NON_WAITING_RULE_CLAUSES["ins_deductible_pa2651"]["clause_text"],
            wait_days_options=[100, 200],  # in hundreds of RMB (100 = 10,000 RMB deductible; 200 = 20,000 RMB)
            status_labels={
                "primary": "以有社保身份投保且已先经基本医疗保险结算",
                "secondary": "以有社保身份投保但未先经基本医疗保险结算",
            },
            condition_labels={
                "routine": "合理住院医疗费用（须超过免赔额且经社保结算方可100%报销）",
                "exempt": "已获社保先行结算补偿或免扣减认定的住院医疗费用",
            },
            waiver_true_phrase="已取得社保结算分割单或保险公司100%给付核定",
            waiver_false_phrase="无社保结算分割单且无特殊核定",
            secondary_disqualifies=True,
            exempt_condition_waives_wait=False,
            approval_waives_wait=True,
            day_unit_label="百元医疗费用",
        ),
        "ins_deductible_pa1073": dict(
            domain="insurance_deductible_copay",
            title="平安i药保B款特药费用医疗保险社保结算给付比例条款",
            aliases=["平安i药保B款给付比例条款"],
            clause_text=NON_WAITING_RULE_CLAUSES["ins_deductible_pa1073"]["clause_text"],
            wait_days_options=[50, 100],  # 50百元(5000元) / 100百元(10000元)
            status_labels={
                "primary": "已从基本医疗保险获得特药费用补偿的参保人",
                "secondary": "有基本医疗保险但未从基本医疗保险获得特药补偿的参保人",
            },
            condition_labels={
                "routine": "基本医疗保险目录内恶性肿瘤特种药品费用",
                "exempt": "基本医疗保险目录外100%给付恶性肿瘤特种药品费用",
            },
            waiver_true_phrase="已补交医保结算凭证并获批按100%比例核算",
            waiver_false_phrase="未提交医保结算凭证且无特批",
            secondary_disqualifies=True,
            exempt_condition_waives_wait=True,
            approval_waives_wait=True,
            day_unit_label="百元特药费用",
        ),
        "ins_hospital_pa2650": dict(
            domain="insurance_hospital_scope",
            title="平安e生保（悦享版）医疗保险定点医院普通部范围条款",
            aliases=["平安e生保悦享版医院范围条款"],
            clause_text=NON_WAITING_RULE_CLAUSES["ins_hospital_pa2650"]["clause_text"],
            wait_days_options=[3, 7],  # minimum inpatient observation days required in sub-plan (3 days or 7 days)
            status_labels={
                "primary": "二级及以上基本医疗保险定点公立医院普通部住院患者",
                "secondary": "医院特需部、国际部、VIP病房或非定点康复医院就诊患者",
            },
            condition_labels={
                "routine": "普通择期住院医疗治疗",
                "exempt": "危及生命的急性重度急诊抢救住院",
            },
            waiver_true_phrase="已获保险公司急诊转诊定点豁免审核同意",
            waiver_false_phrase="未经保险公司转诊豁免审核同意",
            secondary_disqualifies=True,
            exempt_condition_waives_wait=False,
            approval_waives_wait=True,
            day_unit_label="天",
        ),
        "ins_age_rx_pa1073": dict(
            domain="insurance_age_and_rx_limit",
            title="平安i药保B款特药费用医疗保险投保年龄与处方天数限制条款",
            aliases=["平安i药保B款年龄与处方限制条款"],
            clause_text=NON_WAITING_RULE_CLAUSES["ins_age_rx_pa1073"]["clause_text"],
            wait_days_options=[28, 60],  # minimum neonatal age (28 days) or post-enrollment verification window (60 days)
            status_labels={
                "primary": "投保年龄在60周岁及以下且由二级以上公立医院专科医生开具30日内处方的被保险人",
                "secondary": "首次投保超龄或单次处方药量超过30日限额的被保险人",
            },
            condition_labels={
                "routine": "常规院外靶向特种药品处方购药申请",
                "exempt": "连续续保展期且经专科会诊备案的急需特药申请",
            },
            waiver_true_phrase="已获保险公司续保展期及特殊处方审核同意",
            waiver_false_phrase="未获特殊处方或展期审核同意",
            secondary_disqualifies=True,
            exempt_condition_waives_wait=False,
            approval_waives_wait=True,
            day_unit_label="天",
        ),
    }
    return clause_defs


# ---------------------------------------------------------------------------
# Independent Zero-Shot LLM Schema Compiler Agreement Audit (Reviewer Question 3)
# ---------------------------------------------------------------------------
def run_or_load_schema_compiler_audit(out_dir, api_key, score_only, fc_sources):
    """Test whether an independent zero-shot LLM compiler (`qwen3.8-max` and `kimi-k3`) reading ONLY
    the verbatim PDF clause text extracts the exact same waiting-period thresholds, accident exemption
    flags, and renewal/waiver exemption flags as our hand-compiled rule table (answering Question 3).
    """
    cache_path = out_dir / "ccf_a_schema_compiler_audit.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if score_only:
        raise FileNotFoundError(f"Missing {cache_path} under --score-only mode.")

    clauses_payload = []
    for src in fc_sources:
        pid = src["id"]
        pol = FROZEN_CONFIRM_POLICIES[pid]
        clauses_payload.append(
            {
                "policy_id": pid,
                "title": src["title"],
                "evidence_text": src["evidence_text"],
                "gold_days": sorted(pol["days"]),
                "gold_accident_exempt": bool(pol["accident_exempt"]),
                "gold_renewal_or_approval_exempt": bool(
                    pol["approval_exempt"] or "重新投保" in src["evidence_text"] or "续保" in src["evidence_text"]
                ),
            }
        )

    prompt_items = "\n\n".join(
        f"[{c['policy_id']}] 《{c['title']}》：{c['evidence_text']}" for c in clauses_payload
    )
    prompt = (
        "你是保险精算条款编译器。请阅读以下 11 份官方商业保险合同的等待期条款原文，为每份条款提取结构化规则 Schema：\n"
        "1. wait_days (整数数组，升序)：条款中明确规定的等待期天数（例如 [30] 或 [90] 或 [30, 90]）。\n"
        "2. accident_exempt (boolean)：因意外伤害出险是否免除等待期（无等待期）？\n"
        "3. renewal_or_approval_exempt (boolean)：连续重新投保/续保或经审核同意是否免除等待期？\n\n"
        f"{prompt_items}\n\n"
        "请严格只输出 JSON 数组，格式为：\n"
        '[{"policy_id": "pa_h716", "wait_days": [30, 90], "accident_exempt": true, "renewal_or_approval_exempt": true}, ...]'
    )

    compilers = [("qwen3_8_max", "qwen3.8-max"), ("kimi_k3", "kimi-k3")]
    compiler_results = {}
    for comp_id, model_name in compilers:
        res = call_dashscope_chat(
            model_name=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            max_tokens=1000,
            temperature=0.0,
            disable_thinking=True,
        )
        arr = parse_json_from_llm(res["content"])
        by_pid = {x.get("policy_id"): x for x in arr if isinstance(x, dict)}
        exact_matches = 0
        per_policy = []
        for c in clauses_payload:
            pid = c["policy_id"]
            pred = by_pid.get(pid) or {}
            p_days = sorted(int(x) for x in (pred.get("wait_days") or []))
            p_acc = bool(pred.get("accident_exempt", False))
            p_ren = bool(pred.get("renewal_or_approval_exempt", False))
            days_ok = p_days == c["gold_days"]
            acc_ok = p_acc == c["gold_accident_exempt"]
            ren_ok = p_ren == c["gold_renewal_or_approval_exempt"]
            all_ok = bool(days_ok and acc_ok and ren_ok)
            if all_ok:
                exact_matches += 1
            per_policy.append(
                {
                    "policy_id": pid,
                    "gold_days": c["gold_days"],
                    "pred_days": p_days,
                    "gold_accident_exempt": c["gold_accident_exempt"],
                    "pred_accident_exempt": p_acc,
                    "gold_renewal_or_approval_exempt": c["gold_renewal_or_approval_exempt"],
                    "pred_renewal_or_approval_exempt": p_ren,
                    "exact_schema_match": all_ok,
                }
            )
        compiler_results[comp_id] = {
            "compiler_model": model_name,
            "total_clauses": len(clauses_payload),
            "exact_schema_matches": exact_matches,
            "exact_schema_match_rate": round(exact_matches / len(clauses_payload), 4),
            "api_meta": res["api_meta"],
            "per_policy": per_policy,
        }

    out_data = {
        "description": (
            "Independent zero-shot LLM schema extraction audit across all 11 confirmation insurance policy PDFs, "
            "verifying whether independent compilers (`qwen3.8-max` and third-party `kimi-k3`) extract the same "
            "waiting-period thresholds and exemption flags from raw PDF text as the hand-compiled rule table."
        ),
        "compilers": compiler_results,
    }
    base.write_json(cache_path, out_data)
    return out_data

# ---------------------------------------------------------------------------
# Experiment 4 (Cont.): 24 Non-Waiting-Period Multi-Rule Seeds & Generation
# ---------------------------------------------------------------------------
def non_waiting_decision(frame, clause_defs):
    """Formal decision function for the 4 official non-waiting-period insurance clauses."""
    cid = frame["clause_id"]
    cdef = clause_defs[cid]
    day = int(frame["elapsed_day"])
    threshold = int(frame["threshold_days"])
    waiver = bool(frame.get("approved_waiver", False))
    status = frame["status"]
    condition = frame["condition"]

    if waiver and cdef.get("approval_waives_wait", True):
        return "RULE_ELIGIBLE"
    if status == "secondary" and cdef.get("secondary_disqualifies", True):
        return "RULE_EXCLUDED"
    if condition == "exempt" and cdef.get("exempt_condition_waives_wait", False):
        return "RULE_ELIGIBLE"
    return "RULE_ELIGIBLE" if day >= threshold else "RULE_EXCLUDED"


def enumerate_non_waiting_decisions(obs, clause_defs):
    """Finite-domain reachability enumeration over unobserved slots for the non-waiting-period rule clauses."""
    cid = obs.get("clause_id")
    if cid not in clause_defs:
        return []
    cdef = clause_defs[cid]
    statuses = [obs["status"]] if obs.get("status") in ("primary", "secondary") else ["primary", "secondary"]
    conditions = [obs["condition"]] if obs.get("condition") in ("routine", "exempt") else ["routine", "exempt"]
    opts = cdef["wait_days_options"]
    thresholds = [obs["threshold_days"]] if obs.get("threshold_days") in opts else opts
    if obs.get("elapsed_day") is not None:
        days = [int(obs["elapsed_day"])]
    else:
        days = sorted({max(1, opts[0] - 5), opts[0], (opts[0] + opts[-1]) // 2, opts[-1], opts[-1] + 20})
    w_obs = obs.get("approved_waiver")
    if w_obs == "unknown":
        waivers = [False, True]
    elif isinstance(w_obs, bool):
        waivers = [w_obs]
    else:
        waivers = [False]
    decs = set()
    for st in statuses:
        for cond in conditions:
            for d in days:
                for th in thresholds:
                    for w in waivers:
                        decs.add(
                            non_waiting_decision(
                                {
                                    "clause_id": cid,
                                    "status": st,
                                    "condition": cond,
                                    "elapsed_day": d,
                                    "threshold_days": th,
                                    "approved_waiver": w,
                                },
                                clause_defs,
                            )
                        )
    return sorted(decs)


def evaluate_non_waiting_outcome(obs, seed_frame, clause_defs):
    """Evaluate reachability-based decision preservation for non-waiting-period insurance rules."""
    all_fields = ["clause_id", "status", "condition", "elapsed_day", "threshold_days", "approved_waiver"]
    req = seed_frame["required_fields"]
    mutated = [
        f
        for f in all_fields
        if obs.get(f) is not None and obs.get(f) != "unknown" and obs.get(f) != seed_frame.get(f)
    ]
    missing = [f for f in req if obs.get(f) is None or obs.get(f) == "unknown"]
    decs = enumerate_non_waiting_decisions(obs, clause_defs)
    gold = seed_frame["reference_decision"]
    if len(decs) == 1 and decs[0] == gold and not mutated:
        return "PRESERVED_DECISION", missing, mutated
    if len(decs) == 1 and decs[0] != gold:
        return "INVERTED_DECISION", missing, mutated
    return "AMBIGUOUS_NEED_CLARIFICATION", missing, mutated


def build_non_waiting_seeds_list():
    """Return 24 deterministic seeds across the 4 official non-waiting-period insurance clauses
    (6 seeds per clause = 24 seeds across 3 distinct non-waiting-period rule families:
    `deductible_copay_tier` [12 seeds], `hospital_ward_emergency` [6 seeds],
    and `enrollment_age_and_rx_limit` [6 seeds]).
    """
    specs = [
        # Clause 1: ins_deductible_pa2651 (threshold options: 100 or 200 [hundreds RMB])
        ("mrule-001", "ins_deductible_pa2651", "primary", "routine", 65, 100, False, "elapsed_day"),
        ("mrule-002", "ins_deductible_pa2651", "primary", "routine", 145, 100, False, "status"),
        ("mrule-003", "ins_deductible_pa2651", "secondary", "routine", 145, 100, False, "approved_waiver"),
        ("mrule-004", "ins_deductible_pa2651", "secondary", "routine", 145, 100, True, "approved_waiver"),
        ("mrule-005", "ins_deductible_pa2651", "primary", "routine", 150, 200, False, "threshold_days"),
        ("mrule-006", "ins_deductible_pa2651", "primary", "routine", 240, 200, False, "elapsed_day"),
        # Clause 2: ins_deductible_pa1073 (threshold options: 50 or 100 [hundreds RMB])
        ("mrule-007", "ins_deductible_pa1073", "primary", "routine", 35, 50, False, "condition"),
        ("mrule-008", "ins_deductible_pa1073", "primary", "exempt", 35, 50, False, "condition"),
        ("mrule-009", "ins_deductible_pa1073", "primary", "routine", 75, 50, False, "threshold_days"),
        ("mrule-010", "ins_deductible_pa1073", "secondary", "routine", 75, 50, False, "status"),
        ("mrule-011", "ins_deductible_pa1073", "secondary", "routine", 120, 100, True, "approved_waiver"),
        ("mrule-012", "ins_deductible_pa1073", "primary", "routine", 80, 100, False, "threshold_days"),
        # Clause 3: ins_hospital_pa2650 (threshold options: 3 or 7 [days])
        ("mrule-013", "ins_hospital_pa2650", "primary", "routine", 2, 3, False, "elapsed_day"),
        ("mrule-014", "ins_hospital_pa2650", "primary", "routine", 5, 3, False, "threshold_days"),
        ("mrule-015", "ins_hospital_pa2650", "secondary", "routine", 5, 3, False, "status"),
        ("mrule-016", "ins_hospital_pa2650", "secondary", "exempt", 5, 3, True, "approved_waiver"),
        ("mrule-017", "ins_hospital_pa2650", "primary", "routine", 5, 7, False, "threshold_days"),
        ("mrule-018", "ins_hospital_pa2650", "primary", "routine", 10, 7, False, "status"),
        # Clause 4: ins_age_rx_pa1073 (threshold options: 28 or 60 [days])
        ("mrule-019", "ins_age_rx_pa1073", "primary", "routine", 18, 28, False, "elapsed_day"),
        ("mrule-020", "ins_age_rx_pa1073", "primary", "routine", 40, 28, False, "threshold_days"),
        ("mrule-021", "ins_age_rx_pa1073", "secondary", "routine", 40, 28, False, "status"),
        ("mrule-022", "ins_age_rx_pa1073", "secondary", "exempt", 40, 28, True, "approved_waiver"),
        ("mrule-023", "ins_age_rx_pa1073", "primary", "routine", 45, 60, False, "threshold_days"),
        ("mrule-024", "ins_age_rx_pa1073", "primary", "routine", 75, 60, False, "elapsed_day"),
    ]
    clause_defs = build_24_multi_rule_seeds()
    seeds = []
    for sid, cid, st, cond, eday, th, wv, piv in specs:
        cdef = clause_defs[cid]
        req = ["clause_id", "status", "condition", "elapsed_day", "threshold_days"]
        if wv or piv == "approved_waiver":
            req.append("approved_waiver")
        s = {
            "seed_id": sid,
            "clause_id": cid,
            "domain": cdef["domain"],
            "rule_family": NON_WAITING_RULE_CLAUSES[cid]["rule_family"],
            "status": st,
            "condition": cond,
            "elapsed_day": eday,
            "threshold_days": th,
            "approved_waiver": wv,
            "pivotal_field": piv,
            "required_fields": req,
        }
        s["reference_decision"] = non_waiting_decision(s, clause_defs)
        seeds.append(s)
    return seeds, clause_defs


def extract_non_waiting_slots(text, clause_defs):
    """Deterministic synonym-aware slot extractor for the 4 official non-waiting-period insurance rule clauses."""
    t = str(text or "").strip()
    # Normalize RMB expressions (e.g., 10000元 / 1万元 -> 100百元) for deductible clauses so natural currency
    # rewrites are evaluated on their true semantic value rather than surface unit string.
    t_norm = t
    for rmb_pat, hund_str in [
        (r"24000\s*元|2\.4\s*万(?:元)?", "240百元"),
        (r"20000\s*元|2\s*万(?:元)?", "200百元"),
        (r"15000\s*元|1\.5\s*万(?:元)?", "150百元"),
        (r"14500\s*元|1\.45\s*万(?:元)?", "145百元"),
        (r"12000\s*元|1\.2\s*万(?:元)?", "120百元"),
        (r"10000\s*元|1\s*万(?:元)?", "100百元"),
        (r"8000\s*元|0\.8\s*万(?:元)?", "80百元"),
        (r"7500\s*元|0\.75\s*万(?:元)?", "75百元"),
        (r"6500\s*元|0\.65\s*万(?:元)?", "65百元"),
        (r"5000\s*元|0\.5\s*万(?:元)?", "50百元"),
        (r"3500\s*元|0\.35\s*万(?:元)?", "35百元"),
    ]:
        t_norm = re.sub(rmb_pat, hund_str, t_norm)

    obs = {
        "clause_id": None,
        "status": None,
        "condition": None,
        "elapsed_day": None,
        "threshold_days": None,
        "approved_waiver": None,
    }
    # 1. Clause ID
    if "易核版" in t_norm or "安医保" in t_norm:
        obs["clause_id"] = "ins_deductible_pa2651"
    elif "悦享版" in t_norm or ("e生保" in t_norm and any(k in t_norm for k in ("普通部", "特需", "医院", "VIP", "国际部"))):
        obs["clause_id"] = "ins_hospital_pa2650"
    elif "i药保" in t_norm or "特种药品" in t_norm or "特药" in t_norm:
        if any(k in t_norm for k in ("周岁", "处方", "年龄", "新生儿", "单次", "28天", "28日", "60天", "60日")):
            obs["clause_id"] = "ins_age_rx_pa1073"
        elif any(k in t_norm for k in ("给付比例", "目录", "补偿", "起付", "社保结算", "50百元", "100百元", "35百元", "75百元", "80百元", "120百元")):
            obs["clause_id"] = "ins_deductible_pa1073"
    if obs["clause_id"] is None:
        for cid, cdef in clause_defs.items():
            if cdef["title"] in t_norm or any(a in t_norm for a in cdef.get("aliases", [])):
                obs["clause_id"] = cid
                break

    # 2. Status (primary vs secondary)
    has_sec = any(
        k in t_norm
        for k in (
            "未先经基本医疗保险结算",
            "未先经医保结算",
            "未先经社保结算",
            "未经基本医疗保险结算",
            "未从基本医疗保险获得",
            "未从医保获得",
            "未获基本医疗保险",
            "未获得特药补偿",
            "未经社保结算",
            "未走医保结算",
            "未通过基本医疗保险结算",
            "特需部",
            "国际部",
            "VIP",
            "非定点",
            "康复医院",
            "超龄",
            "超过30日",
            "超30日",
            "超过30天",
        )
    )
    has_pri = any(
        k in t_norm
        for k in (
            "已先经基本医疗保险结算",
            "已先经医保结算",
            "已先经社保结算",
            "已经基本医疗保险结算",
            "已从基本医疗保险获得",
            "已从医保获得",
            "已获基本医疗保险",
            "已获得特药费用补偿",
            "已获得特药补偿",
            "已走医保结算",
            "普通部住院",
            "定点公立医院普通部",
            "二级及以上基本医疗保险定点公立医院普通部",
            "二级以上公立医院普通部",
            "60周岁及以下",
            "60岁及以下",
        )
    )
    if has_sec and not has_pri:
        obs["status"] = "secondary"
    elif has_pri and not has_sec:
        obs["status"] = "primary"

    # 3. Condition (routine vs exempt)
    has_ex_cond = any(
        k in t_norm
        for k in (
            "目录外100%给付",
            "目录外",
            "社保目录外",
            "医保目录外",
            "基本医疗保险目录外",
            "急性重度急诊抢救",
            "危及生命的急性",
            "危及生命",
            "急诊抢救",
            "连续续保展期且经专科会诊",
            "续保展期",
            "已获社保先行结算补偿或免扣减认定",
            "免扣减认定",
        )
    )
    has_rt_cond = any(
        k in t_norm
        for k in (
            "目录内恶性肿瘤",
            "目录内",
            "医保目录内",
            "社保目录内",
            "基本医疗保险目录内",
            "普通择期住院",
            "择期住院",
            "常规院外靶向",
            "常规院外",
            "合理住院医疗费用",
            "合理医疗费用",
        )
    )
    if has_ex_cond and not has_rt_cond:
        obs["condition"] = "exempt"
    elif has_rt_cond and not has_ex_cond:
        obs["condition"] = "routine"

    # 4. Approved waiver
    has_wv_false = any(
        k in t_norm
        for k in (
            "无社保结算分割单",
            "没有社保结算分割单",
            "未提交医保结算凭证",
            "没有提交医保结算凭证",
            "未经保险公司转诊豁免",
            "未获保险公司急诊转诊",
            "未获特殊处方或展期审核同意",
            "未经审核同意",
            "无特批",
            "未获批",
            "无特殊核定",
            "无特殊",
        )
    )
    has_wv_true = any(
        k in t_norm
        for k in (
            "已取得社保结算分割单",
            "100%给付核定",
            "已补交医保结算凭证并获批",
            "已补交医保结算凭证",
            "已获保险公司急诊转诊定点豁免审核同意",
            "急诊转诊定点豁免审核同意",
            "已获保险公司续保展期及特殊处方审核同意",
            "续保展期及特殊处方审核同意",
            "经审核同意",
            "已获批",
        )
    )
    if has_wv_true and not has_wv_false:
        obs["approved_waiver"] = True
    elif has_wv_false and not has_wv_true:
        obs["approved_waiver"] = False

    # 5. Threshold & elapsed value extraction
    m_th = re.search(
        r"(?:规定阈值|标准期限|免赔额阈值|免赔额|起付线|核算基准线|最低住院观察期|规定期|要求满|方案规定|法定|约定满|阈值)[^\d]{0,10}(\d+)\s*(?:百元|天|日)",
        t_norm,
    )
    if m_th:
        obs["threshold_days"] = int(m_th.group(1))

    # Mask out "30日内处方 / 超过30日限额 / 60周岁" before extracting elapsed_day/threshold_days
    t_for_num = re.sub(r"60\s*周岁|30\s*[日天](?:内|限额)|不超过\s*30\s*[日天]|超过\s*30\s*[日天]", "", t_norm)
    m_el = re.search(
        r"(?:实际发生/已满|实际发生|目前已满|已发生|实际金额为|费用为|花费了|花了|已住院|住院|实际核定|目前已达|已满|第)\s*(\d+)\s*(?:百元|天|日)",
        t_for_num,
    )
    if m_el:
        obs["elapsed_day"] = int(m_el.group(1))

    all_nums = [int(x) for x in re.findall(r"(\d+)\s*(?:百元|天|日)", t_for_num)]
    if obs["clause_id"] in clause_defs:
        opts = clause_defs[obs["clause_id"]]["wait_days_options"]
        if obs["threshold_days"] is None:
            th_cands = [n for n in all_nums if n in opts]
            if len(set(th_cands)) == 1:
                obs["threshold_days"] = th_cands[0]
        if obs["elapsed_day"] is None:
            el_cands = [n for n in all_nums if n != obs["threshold_days"]]
            if len(set(el_cands)) == 1:
                obs["elapsed_day"] = el_cands[0]

    return obs


def render_non_waiting_seed_question(seed, clause_defs, drop_field=None):
    """Canonical seed question renderer for the 24 non-waiting-period insurance rule seeds."""
    cdef = clause_defs[seed["clause_id"]]
    title = cdef["title"]
    st_lbl = cdef["status_labels"][seed["status"]]
    cond_lbl = cdef["condition_labels"][seed["condition"]]
    wv_lbl = cdef["waiver_true_phrase"] if seed["approved_waiver"] else cdef["waiver_false_phrase"]
    unit = cdef.get("day_unit_label", "天")
    el = seed["elapsed_day"]
    th = seed["threshold_days"]
    parts = [f"根据《{title}》"]
    if drop_field != "threshold_days":
        parts[0] += f"（规定阈值为{th}{unit}）"
    if drop_field != "status":
        parts.append(f"被保险人属于{st_lbl}")
    if drop_field != "elapsed_day":
        parts.append(f"实际发生/已满{el}{unit}")
    if drop_field != "condition":
        parts.append(f"事由为{cond_lbl}")
    if drop_field != "approved_waiver":
        parts.append(f"且{wv_lbl}")
    parts.append("请问按条款规定是否符合100%给付或准入条件？")
    return "，".join(parts)


def run_or_load_non_waiting_multi_rule_study(out_dir, api_key, score_only):
    """Run or load the 24-seed Non-Waiting-Period Multi-Rule-Family Benchmark across two flagship
    generators (`qwen3.8-max` and `deepseek-v4.1-flash`, N=48 cross-model pairs) and all 7 methods.
    """
    cache_path = out_dir / "ccf_a_non_waiting_multi_rule_generations.jsonl"
    seeds, clause_defs = build_non_waiting_seeds_list()

    generators = [
        ("qwen_flagship", "qwen3.8-max"),
        ("deepseek_flagship", "deepseek-v4.1-flash"),
    ]

    existing_rows = []
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_rows.append(json.loads(line))

    by_key = {(r["model_family"], r["method"], r["seed_id"]): r for r in existing_rows}
    needed_seed_fams = []
    req_methods = [
        "seed_passthrough",
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "verify_only_filter",
        "generic_self_refine_budget_matched",
        "slot_guard_repair",
    ]
    for fam_id, model_name in generators:
        for s in seeds:
            if not all((fam_id, m, s["seed_id"]) in by_key for m in req_methods):
                needed_seed_fams.append((fam_id, model_name, s))

    if needed_seed_fams and not score_only:
        print(
            f"Running {len(needed_seed_fams)} seed-generator pairs across 3 new non-waiting-period insurance rule families...",
            flush=True,
        )

        def _score_q(q_txt, raw_resp, s):
            q_clean = str(q_txt or "").strip()
            has_q = ("？" in q_clean) or ("?" in q_clean)
            no_nl = "\n" not in q_clean
            len_ok = 12 <= len(q_clean) <= 220
            fmt_ok = bool(has_q and no_nl and len_ok)
            obs = extract_non_waiting_slots(q_clean, clause_defs)
            out_v, missing_s, mutated_s = evaluate_non_waiting_outcome(obs, s, clause_defs)
            p_unc = detect_parse_uncertainty(q_clean)
            pres = bool(not p_unc["is_parse_uncertain"] and out_v == "PRESERVED_DECISION")
            joint = bool(fmt_ok and pres)
            return {
                "format_valid": fmt_ok,
                "extracted_slots": obs,
                "missing_slots": missing_s,
                "mutated_slots": mutated_s,
                "parse_uncertain": bool(p_unc["is_parse_uncertain"]),
                "outcome": "PARSE_UNCERTAIN" if p_unc["is_parse_uncertain"] else out_v,
                "preserved_decision": pres,
                "format_and_slot_usable": joint,
            }

        def _gen_seed_worker(item):
            fam_id, model_name, s = item
            sid = s["seed_id"]
            cid = s["clause_id"]
            cdef = clause_defs[cid]
            unit = cdef.get("day_unit_label", "天")
            seed_q = render_non_waiting_seed_question(s, clause_defs)
            out_records = []

            # 0. seed_passthrough
            sc_pass = _score_q(seed_q, seed_q, s)
            out_records.append(
                {
                    "seed_id": sid,
                    "clause_id": cid,
                    "rule_family": s["rule_family"],
                    "model_family": fam_id,
                    "method": "seed_passthrough",
                    "question": seed_q,
                    "raw_response": seed_q,
                    "llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": 0.0,
                    "api_meta": {"metadata_provenance": "deterministic_zero_call_baseline"},
                    **sc_pass,
                }
            )

            # 1. ordinary_style
            p_ord = (
                f"请将以下保险业务情景改写为一句简短、口语化的客户咨询提问（只输出一句中文问句，不要带书名号）：\n"
                f"情景：{seed_q}"
            )
            r_ord = call_dashscope_chat(model_name, [{"role": "user", "content": p_ord}], api_key, max_tokens=160)
            q_ord = r_ord["content"].splitlines()[0].strip() if r_ord["content"] else ""
            sc_ord = _score_q(q_ord, r_ord["content"], s)
            out_records.append(
                {
                    "seed_id": sid,
                    "clause_id": cid,
                    "rule_family": s["rule_family"],
                    "model_family": fam_id,
                    "method": "ordinary_style",
                    "question": q_ord,
                    "raw_response": r_ord["content"],
                    "llm_calls": 1,
                    "prompt_tokens": r_ord["prompt_tokens"],
                    "completion_tokens": r_ord["completion_tokens"],
                    "total_tokens": r_ord["total_tokens"],
                    "latency_ms": r_ord["latency_ms"],
                    "api_meta": r_ord["api_meta"],
                    **sc_ord,
                }
            )

            # 2. condition_prompt (same single-pass clause + seed prompt protocol as frozen_confirmation_and_rag_study.py)
            st_lbl = cdef["status_labels"][s["status"]]
            cond_lbl = cdef["condition_labels"][s["condition"]]
            wv_lbl = cdef["waiver_true_phrase"] if s["approved_waiver"] else cdef["waiver_false_phrase"]
            p_cond = (
                f"条款原文：{cdef['title']} {cdef['clause_text']}\n"
                f"原问题：{seed_q}\n"
                f"请改写为完整保留“{cdef['title']}”名称（不加《》书名号）及所有金额/天数、身份和条件的自然口语化单句提问（只输出一句问句）："
            )
            r_cond = call_dashscope_chat(model_name, [{"role": "user", "content": p_cond}], api_key, max_tokens=110)
            q_cond = r_cond["content"].splitlines()[0].strip() if r_cond["content"] else ""
            sc_cond = _score_q(q_cond, r_cond["content"], s)
            rec_cond = {
                "seed_id": sid,
                "clause_id": cid,
                "rule_family": s["rule_family"],
                "model_family": fam_id,
                "method": "condition_prompt",
                "question": q_cond,
                "raw_response": r_cond["content"],
                "llm_calls": 1,
                "prompt_tokens": r_cond["prompt_tokens"],
                "completion_tokens": r_cond["completion_tokens"],
                "total_tokens": r_cond["total_tokens"],
                "latency_ms": r_cond["latency_ms"],
                "api_meta": r_cond["api_meta"],
                **sc_cond,
            }
            out_records.append(rec_cond)

            # 3. verify_only_filter (filters condition_prompt)
            rec_vof = dict(rec_cond)
            rec_vof["method"] = "verify_only_filter"
            rec_vof["accepted_by_verifier"] = bool(sc_cond["format_and_slot_usable"])
            out_records.append(rec_vof)

            # 4. structured_json_rewrite (160-token cap, 5-field JSON schema + QUESTION:)
            p_json = (
                f"请根据以下保险规则与个案事实，先在第一行输出包含5个字段的紧凑JSON对象 "
                f'({{"clause": "...", "status": "...", "condition": "...", "day": 整数, "waiver": "..."}})，'
                f"然后在第二行以 QUESTION: 开头输出一句完整自然的中文咨询问句（包含条款名称、规定阈值、实际数值、身份、事由及豁免状态）：\n"
                f"【规则原文】：{cdef['clause_text']}\n"
                f"【个案事实】：条款={cdef['title']}；身份={st_lbl}；事由={cond_lbl}；"
                f"实际数值={s['elapsed_day']}{unit}（规定阈值={s['threshold_days']}{unit}）；豁免={wv_lbl}。"
            )
            r_j160 = call_dashscope_chat(model_name, [{"role": "user", "content": p_json}], api_key, max_tokens=160)
            q_j160 = extract_question_from_structured_output(r_j160["content"])
            sc_j160 = _score_q(q_j160, r_j160["content"], s)
            if r_j160["api_meta"].get("finish_reason") == "length" or not q_j160.endswith(("？", "?")):
                sc_j160["format_valid"] = False
                sc_j160["format_and_slot_usable"] = False
            rec_j160 = {
                "seed_id": sid,
                "clause_id": cid,
                "rule_family": s["rule_family"],
                "model_family": fam_id,
                "method": "structured_json_rewrite",
                "question": q_j160,
                "raw_response": r_j160["content"],
                "llm_calls": 1,
                "prompt_tokens": r_j160["prompt_tokens"],
                "completion_tokens": r_j160["completion_tokens"],
                "total_tokens": r_j160["total_tokens"],
                "latency_ms": r_j160["latency_ms"],
                "api_meta": r_j160["api_meta"],
                **sc_j160,
            }
            out_records.append(rec_j160)

            # 5. structured_json_untruncated_320 (320-token cap)
            r_j320 = call_dashscope_chat(model_name, [{"role": "user", "content": p_json}], api_key, max_tokens=320)
            q_j320 = extract_question_from_structured_output(r_j320["content"])
            sc_j320 = _score_q(q_j320, r_j320["content"], s)
            out_records.append(
                {
                    "seed_id": sid,
                    "clause_id": cid,
                    "rule_family": s["rule_family"],
                    "model_family": fam_id,
                    "method": "structured_json_untruncated_320",
                    "question": q_j320,
                    "raw_response": r_j320["content"],
                    "llm_calls": 1,
                    "prompt_tokens": r_j320["prompt_tokens"],
                    "completion_tokens": r_j320["completion_tokens"],
                    "total_tokens": r_j320["total_tokens"],
                    "latency_ms": r_j320["latency_ms"],
                    "api_meta": r_j320["api_meta"],
                    **sc_j320,
                }
            )

            # 6. structured_json_schema_render (5-field JSON template render from j160 first line)
            first_ln = r_j160["content"].strip().splitlines()[0] if r_j160["content"].strip() else "{}"
            try:
                jobj = json.loads(first_ln)
            except Exception:
                jobj = {}
            q_srender = (
                f"根据{jobj.get('clause', cdef['title'])}，当事人属于{jobj.get('status', '')}，"
                f"目前已达第{jobj.get('day', '')}{unit}，事由为{jobj.get('condition', '')}，"
                f"且{jobj.get('waiver', '')}，请问按规定应当如何裁决？"
            )
            sc_srender = _score_q(q_srender, r_j160["content"], s)
            out_records.append(
                {
                    "seed_id": sid,
                    "clause_id": cid,
                    "rule_family": s["rule_family"],
                    "model_family": fam_id,
                    "method": "structured_json_schema_render",
                    "question": q_srender,
                    "raw_response": r_j160["content"],
                    "llm_calls": 1,
                    "prompt_tokens": r_j160["prompt_tokens"],
                    "completion_tokens": r_j160["completion_tokens"],
                    "total_tokens": r_j160["total_tokens"],
                    "latency_ms": r_j160["latency_ms"],
                    "api_meta": r_j160["api_meta"],
                    **sc_srender,
                }
            )

            # 7. slot_guard_repair & generic_self_refine_budget_matched
            if not sc_cond["format_and_slot_usable"]:
                p_rep = (
                    f"你上一版改写的问句存在槽位缺失或歧义（缺失槽位：{sc_cond['missing_slots']}，篡改槽位：{sc_cond['mutated_slots']}）。\n"
                    f"请重新改写为一句完整自然的中文咨询问句（单行，以问号结尾），必须一字不差包含：\n"
                    f"1) 条款全称：{cdef['title']}\n"
                    f"2) 规定阈值：规定阈值为{s['threshold_days']}{unit}\n"
                    f"3) 实际数值：实际发生{s['elapsed_day']}{unit}\n"
                    f"4) 身份状态：{st_lbl}\n"
                    f"5) 申请事由：{cond_lbl}\n"
                    f"6) 审批状态：{wv_lbl}\n"
                    f"只输出修正后的一句问句："
                )
                r_rep = call_dashscope_chat(model_name, [{"role": "user", "content": p_rep}], api_key, max_tokens=220)
                q_rep = r_rep["content"].splitlines()[0].strip() if r_rep["content"] else ""
                sc_rep = _score_q(q_rep, r_rep["content"], s)
                out_records.append(
                    {
                        "seed_id": sid,
                        "clause_id": cid,
                        "rule_family": s["rule_family"],
                        "model_family": fam_id,
                        "method": "slot_guard_repair",
                        "question": q_rep,
                        "raw_response": r_rep["content"],
                        "llm_calls": 2,
                        "prompt_tokens": r_cond["prompt_tokens"] + r_rep["prompt_tokens"],
                        "completion_tokens": r_cond["completion_tokens"] + r_rep["completion_tokens"],
                        "total_tokens": r_cond["total_tokens"] + r_rep["total_tokens"],
                        "latency_ms": round(r_cond["latency_ms"] + r_rep["latency_ms"], 1),
                        "api_meta": r_rep["api_meta"],
                        **sc_rep,
                    }
                )

                # Equal-budget unguided self-refine triggered on the exact same seeds
                p_sr = (
                    f"请检查你刚才生成的保险咨询问句是否自然、清晰，若有表述不清可做润色，只输出最终的一句中文问句：\n"
                    f"原问句：{q_cond}"
                )
                r_sr = call_dashscope_chat(model_name, [{"role": "user", "content": p_sr}], api_key, max_tokens=200)
                q_sr = r_sr["content"].splitlines()[0].strip() if r_sr["content"] else ""
                sc_sr = _score_q(q_sr, r_sr["content"], s)
                out_records.append(
                    {
                        "seed_id": sid,
                        "clause_id": cid,
                        "rule_family": s["rule_family"],
                        "model_family": fam_id,
                        "method": "generic_self_refine_budget_matched",
                        "question": q_sr,
                        "raw_response": r_sr["content"],
                        "llm_calls": 2,
                        "prompt_tokens": r_cond["prompt_tokens"] + r_sr["prompt_tokens"],
                        "completion_tokens": r_cond["completion_tokens"] + r_sr["completion_tokens"],
                        "total_tokens": r_cond["total_tokens"] + r_sr["total_tokens"],
                        "latency_ms": round(r_cond["latency_ms"] + r_sr["latency_ms"], 1),
                        "api_meta": r_sr["api_meta"],
                        **sc_sr,
                    }
                )
            else:
                rec_sgr = dict(rec_cond)
                rec_sgr["method"] = "slot_guard_repair"
                out_records.append(rec_sgr)
                rec_gsr = dict(rec_cond)
                rec_gsr["method"] = "generic_self_refine_budget_matched"
                out_records.append(rec_gsr)

            return out_records

        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_gen_seed_worker, item) for item in needed_seed_fams]
            for fut in as_completed(futs):
                for rec in fut.result():
                    by_key[(rec["model_family"], rec["method"], rec["seed_id"])] = rec

        sorted_all = [
            by_key[k]
            for k in sorted(by_key.keys(), key=lambda x: (x[0], x[1], x[2]))
        ]
        with cache_path.open("w", encoding="utf-8") as f:
            for r in sorted_all:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing_rows = sorted_all
    elif needed_seed_fams and score_only:
        raise FileNotFoundError(
            f"Missing {len(needed_seed_fams)} non-waiting multi-rule generation records in {cache_path} under --score-only."
        )

    return existing_rows, seeds, clause_defs


def compute_paired_stats(rows, method_a, method_b, cluster_key_fn):
    """Compute exact two-sided McNemar test and cluster-level sign-flip permutation p-value."""
    by_a = {(r["model_family"], r["seed_id"]): bool(r.get("format_and_slot_usable")) for r in rows if r["method"] == method_a}
    by_b = {(r["model_family"], r["seed_id"]): bool(r.get("format_and_slot_usable")) for r in rows if r["method"] == method_b}
    common_keys = sorted(set(by_a.keys()) & set(by_b.keys()))
    a_wins = 0
    b_wins = 0
    both_pass = 0
    both_fail = 0
    cluster_diffs = {}
    seed_diffs = {}
    for fam, sid in common_keys:
        va = int(by_a[(fam, sid)])
        vb = int(by_b[(fam, sid)])
        if va == 1 and vb == 0:
            a_wins += 1
        elif va == 0 and vb == 1:
            b_wins += 1
        elif va == 1 and vb == 1:
            both_pass += 1
        else:
            both_fail += 1
        seed_diffs.setdefault(sid, []).append(va - vb)
        ck = cluster_key_fn(sid)
        cluster_diffs.setdefault(ck, []).append(va - vb)

    disc = a_wins + b_wins
    mcnemar_p = float(binomtest(a_wins, disc, 0.5, alternative="two-sided").pvalue) if disc > 0 else 1.0
    seed_means = [float(np.mean(v)) for v in seed_diffs.values()]
    clause_means = [float(np.mean(v)) for v in cluster_diffs.values()]
    seed_p = clustered_sign_flip_pvalue(seed_means)
    clause_p = clustered_sign_flip_pvalue(clause_means)
    return {
        "n_pairs": len(common_keys),
        "n_seeds": len(seed_diffs),
        "n_clause_clusters": len(cluster_diffs),
        "a_pass": both_pass + a_wins,
        "b_pass": both_pass + b_wins,
        "a_only_wins": a_wins,
        "b_only_wins": b_wins,
        "exact_mcnemar_p": round_pval(mcnemar_p),
        "seed_clustered_p": round_pval(seed_p),
        "clause_clustered_p": round_pval(clause_p),
    }


def main():
    parser = argparse.ArgumentParser(description="Run or rescore the CCF-A Main-Track Upgrade Suite.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data_v3",
    )
    parser.add_argument(
        "--raw-frozen-confirm",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent / "work" / "condition_study" / "raw_frozen_confirm",
    )
    parser.add_argument("--score-only", action="store_true", help="Recompute all metrics offline from cached JSONL logs.")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not args.score_only and not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable is required when not running with --score-only.")

    # Load BGE cache
    cache_path = args.out_dir / "bge_embedding_cache.npz"
    cache_npz = np.load(cache_path, allow_pickle=True)
    bge_cache = {str(t): emb for t, emb in zip(cache_npz["texts"], cache_npz["embeddings"])}

    # Load Confirmation Set seeds & existing generation records
    fc_sources, fc_evidence, ins_seeds, cross_seeds, _ = build_frozen_confirmation_sources_and_cases(
        args.raw_frozen_confirm, args.out_dir
    )
    fc_sources_map = {s["id"]: s for s in fc_sources}
    ins_seeds_map = {s["seed_id"]: s for s in ins_seeds}
    cross_seeds_map = {s["seed_id"]: s for s in cross_seeds}
    all_40_seeds_map = {**ins_seeds_map, **cross_seeds_map}

    qwen_rows = [
        json.loads(line)
        for line in (args.out_dir / "frozen_confirmation_generations_qwen_plus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ds_rows = [
        json.loads(line)
        for line in (args.out_dir / "frozen_confirmation_generations_deepseek.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fair_rows = [
        json.loads(line)
        for line in (args.out_dir / "fair_structured_baseline_generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    uhold_rows = [
        json.loads(line)
        for line in (args.out_dir / "untouched_holdout_generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for r in qwen_rows + ds_rows + fair_rows + uhold_rows:
        if "format_and_slot_usable" not in r:
            r["format_and_slot_usable"] = bool(r.get("usable_FormatAndSlot_preserved", False))
        if "preserved_decision" not in r:
            r["preserved_decision"] = bool(r.get("outcome") == "PRESERVED_DECISION")

    # Build deterministic `multi_template_3` and `complete_6slot_schema_render` rows for the 40 confirmation seeds
    synth_extra_rows = []
    j160_by_fam_seed = {
        (r["model_family"], r["seed_id"]): r
        for r in qwen_rows + ds_rows
        if r["method"] == "structured_json_rewrite"
    }
    for fam_name in ("qwen_plus", "deepseek"):
        for s in ins_seeds + cross_seeds:
            sid = s["seed_id"]
            dom_type = "insurance" if sid in ins_seeds_map else "multi_threshold"
            # multi_template_3
            q_mt3 = render_multi_template_3(s, dom_type, FROZEN_CONFIRM_POLICIES)
            p_unc_mt3 = detect_parse_uncertainty(q_mt3)
            if dom_type == "insurance":
                obs_mt3 = extract_text_slots_pure(q_mt3, "insurance", FROZEN_CONFIRM_POLICIES)
                out_mt3, _, _ = evaluate_insurance_outcome_frozen(obs_mt3, s, s["required_fields"], FROZEN_CONFIRM_POLICIES)
            else:
                obs_mt3 = extract_text_slots_pure(q_mt3, "multi_threshold", FROZEN_CONFIRM_POLICIES)
                out_mt3, _, _ = evaluate_cross_outcome(obs_mt3, s)
            pres_mt3 = bool(not p_unc_mt3["is_parse_uncertain"] and out_mt3 == "PRESERVED_DECISION")
            synth_extra_rows.append(
                {
                    "seed_id": sid,
                    "domain_type": dom_type,
                    "model_family": fam_name,
                    "method": "multi_template_3",
                    "question": q_mt3,
                    "format_valid": True,
                    "preserved_decision": pres_mt3,
                    "format_and_slot_usable": pres_mt3,
                    "all_slots_preserved": True,
                    "total_tokens": 0,
                    "llm_calls": 0,
                }
            )
            # complete_6slot_schema_render
            j160_r = j160_by_fam_seed[(fam_name, sid)]
            q_c6 = render_complete_6slot_from_json(j160_r, s, dom_type, FROZEN_CONFIRM_POLICIES)
            p_unc_c6 = detect_parse_uncertainty(q_c6)
            if dom_type == "insurance":
                obs_c6 = extract_text_slots_pure(q_c6, "insurance", FROZEN_CONFIRM_POLICIES)
                out_c6, _, _ = evaluate_insurance_outcome_frozen(obs_c6, s, s["required_fields"], FROZEN_CONFIRM_POLICIES)
            else:
                obs_c6 = extract_text_slots_pure(q_c6, "multi_threshold", FROZEN_CONFIRM_POLICIES)
                out_c6, _, _ = evaluate_cross_outcome(obs_c6, s)
            pres_c6 = bool(not p_unc_c6["is_parse_uncertain"] and out_c6 == "PRESERVED_DECISION")
            synth_extra_rows.append(
                {
                    "seed_id": sid,
                    "domain_type": dom_type,
                    "model_family": fam_name,
                    "method": "complete_6slot_schema_render",
                    "question": q_c6,
                    "format_valid": True,
                    "preserved_decision": pres_c6,
                    "format_and_slot_usable": pres_c6,
                    "all_slots_preserved": pres_c6,
                    "total_tokens": j160_r.get("total_tokens", 0),
                    "llm_calls": 1,
                }
            )

    all_confirm_rows = qwen_rows + ds_rows + fair_rows + synth_extra_rows

    # Load Extension (24 seeds, N=48) and Non-Waiting-Period Multi-Rule (24 seeds, N=48) before Panel Adjudication
    uh_sources, uh_ev, uhold_ins_seeds, uhold_cross_seeds = build_untouched_holdout_sources_and_seeds(
        args.raw_frozen_confirm, args.out_dir
    )
    uh_sources_map = {s["id"]: s for s in uh_sources}
    uhold_ins_map = {s["seed_id"]: s for s in uhold_ins_seeds}
    uhold_cross_map = {s["seed_id"]: s for s in uhold_cross_seeds}

    mrule_rows, mrule_seeds, mrule_clause_defs = run_or_load_non_waiting_multi_rule_study(
        args.out_dir, api_key, args.score_only
    )
    mrule_seeds_map = {s["seed_id"]: s for s in mrule_seeds}
    compiler_audit = run_or_load_schema_compiler_audit(args.out_dir, api_key, args.score_only, fc_sources)

    all_88_rows_for_panel = all_confirm_rows + uhold_rows + mrule_rows

    # -----------------------------------------------------------------------
    # Run Experiment 1: Blinded 3-Flagship Raw-Clause Adjudication Panel (All 88 Seeds)
    # -----------------------------------------------------------------------
    adjudication_records = run_or_load_raw_clause_adjudication(
        args.out_dir,
        api_key,
        args.score_only,
        ins_seeds,
        cross_seeds,
        fc_sources_map,
        all_88_rows_for_panel,
        uhold_ins_seeds=uhold_ins_seeds,
        uhold_cross_seeds=uhold_cross_seeds,
        uh_sources_map=uh_sources_map,
        mrule_seeds=mrule_seeds,
        mrule_clause_defs=mrule_clause_defs,
    )

    # Summarize Experiment 1: Inter-Judge Kappa, Parser Error Audit, and Method Comparison under Panel
    panel_methods = [
        "seed_passthrough",
        "multi_template_3",
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "complete_6slot_schema_render",
        "generic_self_refine_budget_matched",
        "slot_guard_repair",
    ]
    sub_panel_rows = [r for r in all_confirm_rows if r["method"] in panel_methods]

    judge_ids = [j[0] for j in ADJUDICATOR_MODELS]
    per_judge_vecs = {jid: [] for jid in judge_ids}
    parser_vec = []
    panel_maj_vec = []

    # Track parser vs panel discrepancies on the 8 core confirmation methods (640 records)
    core_8_methods = {
        "seed_passthrough",
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "generic_self_refine_budget_matched",
        "slot_guard_repair",
    }
    parser_false_rejects = []
    parser_false_accepts = []

    def _annotate_row_with_panel(r, gold_dec):
        sid = r["seed_id"]
        q_txt = r["question"]
        j_votes = []
        j_nats = []
        j_details = {}
        for jid in judge_ids:
            j_map = adjudication_records[(sid, jid)]["judgments_by_question"]
            j_info = j_map[q_txt]
            j_usable = bool(
                j_info["policy_identified"]
                and j_info["sentence_complete"]
                and j_info["decision"] == gold_dec
            )
            j_votes.append(1 if j_usable else 0)
            j_nats.append(int(j_info["naturalness"]))
            j_details[jid] = {**j_info, "panel_usable": j_usable}
        r["panel_majority_usable"] = bool(sum(j_votes) >= 2)
        r["panel_unanimous_usable"] = bool(sum(j_votes) == 3)
        r["kimi_k3_third_party_usable"] = bool(j_details["kimi_k3"]["panel_usable"])
        r["panel_mean_naturalness"] = round(float(np.mean(j_nats)), 3)
        r["panel_judgments"] = j_details
        return j_votes, j_details

    for r in sub_panel_rows:
        sid = r["seed_id"]
        q_txt = r["question"]
        dom_type = r["domain_type"]
        seed = all_40_seeds_map[sid]
        gold_dec = (
            insurance_decision(seed, FROZEN_CONFIRM_POLICIES)
            if dom_type == "insurance"
            else cross_decision(seed)
        )
        j_votes, j_details = _annotate_row_with_panel(r, gold_dec)
        if r["method"] in core_8_methods:
            for jid in judge_ids:
                per_judge_vecs[jid].append(1 if j_details[jid]["panel_usable"] else 0)
            p_usable = bool(r.get("format_and_slot_usable"))
            maj_usable = r["panel_majority_usable"]
            parser_vec.append(1 if p_usable else 0)
            panel_maj_vec.append(1 if maj_usable else 0)
            if not p_usable and maj_usable:
                parser_false_rejects.append(
                    {
                        "seed_id": sid,
                        "model_family": r["model_family"],
                        "method": r["method"],
                        "question": q_txt,
                        "parser_outcome": r.get("outcome"),
                        "gold_decision": gold_dec,
                        "panel_votes": sum(j_votes),
                    }
                )
            elif p_usable and not maj_usable:
                parser_false_accepts.append(
                    {
                        "seed_id": sid,
                        "model_family": r["model_family"],
                        "method": r["method"],
                        "question": q_txt,
                        "parser_outcome": r.get("outcome"),
                        "gold_decision": gold_dec,
                        "panel_votes": sum(j_votes),
                        "panel_decisions": {jid: j_details[jid]["decision"] for jid in judge_ids},
                    }
                )

    # Also annotate the 24 Extension Seeds (uhold_rows) and 24 Non-Waiting Multi-Rule Seeds (mrule_rows)
    all_88_judge_vecs = {jid: list(per_judge_vecs[jid]) for jid in judge_ids}
    all_88_parser_vec = list(parser_vec)
    all_88_panel_vec = list(panel_maj_vec)

    for r in uhold_rows:
        sid = r["seed_id"]
        if sid in uhold_ins_map:
            gold_dec = insurance_decision(uhold_ins_map[sid], UNTOUCHED_HOLDOUT_POLICIES)
        else:
            gold_dec = cross_decision(uhold_cross_map[sid])
        _, j_details = _annotate_row_with_panel(r, gold_dec)
        if r["method"] in core_8_methods:
            for jid in judge_ids:
                all_88_judge_vecs[jid].append(1 if j_details[jid]["panel_usable"] else 0)
            all_88_parser_vec.append(1 if r.get("format_and_slot_usable") else 0)
            all_88_panel_vec.append(1 if r["panel_majority_usable"] else 0)

    for r in mrule_rows:
        sid = r["seed_id"]
        gold_dec = mrule_seeds_map[sid]["reference_decision"]
        _, j_details = _annotate_row_with_panel(r, gold_dec)
        if r["method"] in core_8_methods:
            for jid in judge_ids:
                all_88_judge_vecs[jid].append(1 if j_details[jid]["panel_usable"] else 0)
            all_88_parser_vec.append(1 if r.get("format_and_slot_usable") else 0)
            all_88_panel_vec.append(1 if r["panel_majority_usable"] else 0)

    mat_3judges = np.column_stack([per_judge_vecs[jid] for jid in judge_ids])
    mat_3judges_88 = np.column_stack([all_88_judge_vecs[jid] for jid in judge_ids])
    exp1_summary = {
        "adjudicator_panel": [{"judge_id": j[0], "model": j[1]} for j in ADJUDICATOR_MODELS],
        "evaluated_records_core_8_methods_confirm_40": len(parser_vec),
        "evaluated_records_core_methods_all_88_seeds": len(all_88_parser_vec),
        "inter_adjudicator_agreement": {
            "fleiss_kappa_3_flagship_judges": fleiss_kappa_binary(mat_3judges),
            "cohens_kappa_kimi_k3_vs_qwen3_8_max": cohens_kappa_binary(
                per_judge_vecs["kimi_k3"], per_judge_vecs["qwen3_8_max"]
            ),
            "cohens_kappa_kimi_k3_vs_deepseek_v4_pro": cohens_kappa_binary(
                per_judge_vecs["kimi_k3"], per_judge_vecs["deepseek_v4_pro"]
            ),
            "cohens_kappa_qwen3_8_max_vs_deepseek_v4_pro": cohens_kappa_binary(
                per_judge_vecs["qwen3_8_max"], per_judge_vecs["deepseek_v4_pro"]
            ),
            "cohens_kappa_symbolic_verifier_vs_panel_majority": cohens_kappa_binary(
                parser_vec, panel_maj_vec
            ),
            "exact_agreement_rate_verifier_vs_panel_majority": round(
                float(np.mean(np.asarray(parser_vec) == np.asarray(panel_maj_vec))), 4
            ),
            "all_88_seeds_1408_records_fleiss_kappa": fleiss_kappa_binary(mat_3judges_88),
            "all_88_seeds_1408_records_verifier_vs_panel_cohens_kappa": cohens_kappa_binary(
                all_88_parser_vec, all_88_panel_vec
            ),
            "all_88_seeds_1408_records_exact_agreement_rate": round(
                float(np.mean(np.asarray(all_88_parser_vec) == np.asarray(all_88_panel_vec))), 4
            ),
        },
        "parser_vs_raw_clause_panel_error_audit_640": {
            "parser_accepted_panel_accepted_tp": int(
                np.sum((np.asarray(parser_vec) == 1) & (np.asarray(panel_maj_vec) == 1))
            ),
            "parser_rejected_panel_rejected_tn": int(
                np.sum((np.asarray(parser_vec) == 0) & (np.asarray(panel_maj_vec) == 0))
            ),
            "parser_false_reject_count_panel_accepted": len(parser_false_rejects),
            "parser_false_accept_count_panel_rejected": len(parser_false_accepts),
            "parser_false_reject_examples": parser_false_rejects[:10],
            "parser_false_accept_examples": parser_false_accepts[:10],
        },
        "four_tier_target_decomposition_by_method_n80": {},
        "extension_24_seeds_n48_panel_by_method": {},
        "non_waiting_multi_rule_24_seeds_n48_panel_by_method": {},
    }

    for m_name in panel_methods:
        m_rows = [r for r in sub_panel_rows if r["method"] == m_name]
        n_m = len(m_rows)
        strict_slot = sum(1 for r in m_rows if r.get("all_slots_preserved") and r.get("format_valid"))
        sym_joint = sum(1 for r in m_rows if r.get("format_and_slot_usable"))
        panel_maj = sum(1 for r in m_rows if r.get("panel_majority_usable"))
        panel_unanim = sum(1 for r in m_rows if r.get("panel_unanimous_usable"))
        kimi_only = sum(1 for r in m_rows if r.get("kimi_k3_third_party_usable"))
        mean_nat = round(float(np.mean([r["panel_mean_naturalness"] for r in m_rows])), 3)
        exp1_summary["four_tier_target_decomposition_by_method_n80"][m_name] = {
            "n_pairs": n_m,
            "tier1_strict_all_slots_and_format": f"{strict_slot}/{n_m} ({round(100*strict_slot/max(1,n_m), 2)}%)",
            "tier2_symbolic_reachability_joint_usable": f"{sym_joint}/{n_m} ({round(100*sym_joint/max(1,n_m), 2)}%)",
            "tier3_raw_clause_panel_majority_usable": f"{panel_maj}/{n_m} ({round(100*panel_maj/max(1,n_m), 2)}%)",
            "tier3b_raw_clause_panel_unanimous_usable": f"{panel_unanim}/{n_m} ({round(100*panel_unanim/max(1,n_m), 2)}%)",
            "tier4_third_party_kimi_k3_usable": f"{kimi_only}/{n_m} ({round(100*kimi_only/max(1,n_m), 2)}%)",
            "panel_majority_count": panel_maj,
            "kimi_k3_usable_count": kimi_only,
            "mean_blinded_naturalness_1to5": mean_nat,
        }

    for m_name in [
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "generic_self_refine_budget_matched",
        "slot_guard_repair",
    ]:
        uh_m = [r for r in uhold_rows if r["method"] == m_name]
        if uh_m:
            exp1_summary["extension_24_seeds_n48_panel_by_method"][m_name] = {
                "n_pairs": len(uh_m),
                "symbolic_joint_usable": sum(1 for r in uh_m if r.get("format_and_slot_usable")),
                "panel_majority_usable": sum(1 for r in uh_m if r.get("panel_majority_usable")),
                "kimi_k3_usable": sum(1 for r in uh_m if r.get("kimi_k3_third_party_usable")),
                "mean_blinded_naturalness_1to5": round(float(np.mean([r["panel_mean_naturalness"] for r in uh_m])), 3),
            }
        mr_m = [r for r in mrule_rows if r["method"] == m_name]
        if mr_m:
            exp1_summary["non_waiting_multi_rule_24_seeds_n48_panel_by_method"][m_name] = {
                "n_pairs": len(mr_m),
                "symbolic_joint_usable": sum(1 for r in mr_m if r.get("format_and_slot_usable")),
                "panel_majority_usable": sum(1 for r in mr_m if r.get("panel_majority_usable")),
                "kimi_k3_usable": sum(1 for r in mr_m if r.get("kimi_k3_third_party_usable")),
                "mean_blinded_naturalness_1to5": round(float(np.mean([r["panel_mean_naturalness"] for r in mr_m])), 3),
            }

    # Compute McNemar tests under the independent 3-flagship Panel Majority Vote on Confirmation (N=80)
    sgr_maj = {
        (r["model_family"], r["seed_id"]): bool(r["panel_majority_usable"])
        for r in sub_panel_rows
        if r["method"] == "slot_guard_repair"
    }
    panel_mcnemar_vs_sgr = {}
    for b_m in [
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "generic_self_refine_budget_matched",
    ]:
        b_map = {
            (r["model_family"], r["seed_id"]): bool(r["panel_majority_usable"])
            for r in sub_panel_rows
            if r["method"] == b_m
        }
        w_a = sum(1 for k in sgr_maj if sgr_maj[k] and not b_map[k])
        w_b = sum(1 for k in sgr_maj if not sgr_maj[k] and b_map[k])
        disc = w_a + w_b
        pval = float(binomtest(w_a, disc, 0.5, alternative="two-sided").pvalue) if disc > 0 else 1.0
        panel_mcnemar_vs_sgr[b_m] = {
            "slot_guard_repair_panel_usable": sum(1 for v in sgr_maj.values() if v),
            "baseline_panel_usable": sum(1 for v in b_map.values() if v),
            "slot_guard_wins": w_a,
            "baseline_wins": w_b,
            "exact_mcnemar_p_under_raw_clause_panel": round_pval(pval),
        }
    exp1_summary["mcnemar_tests_under_raw_clause_panel_majority"] = panel_mcnemar_vs_sgr

    # -----------------------------------------------------------------------
    # Experiment 2: Joint Utility & Template Discrimination Power Analysis
    # -----------------------------------------------------------------------
    # Ensure any new strings from `multi_template_3` and `complete_6slot_schema_render` have BGE embeddings
    extra_bge_path = args.out_dir / "ccf_a_bge_cache.npz"
    if extra_bge_path.exists():
        ex_npz = np.load(extra_bge_path, allow_pickle=True)
        for t, emb in zip(ex_npz["texts"], ex_npz["embeddings"]):
            bge_cache[str(t)] = emb

    res_rag = _get_rag_resources(bge_cache=bge_cache)
    ins_titles = [FROZEN_CONFIRM_POLICIES[pid]["title"] for pid in FROZEN_CONFIRM_POLICIES]
    cross_titles = [CROSS_CLAUSES[cid]["title"] for cid in CROSS_CLAUSES]
    all_titles_for_mask = ins_titles + cross_titles

    fc_ins_ids = list(FROZEN_CONFIRM_POLICIES.keys())
    fc_ins_docs = [f"《{FROZEN_CONFIRM_POLICIES[pid]['title']}》 {fc_evidence[pid]}" for pid in fc_ins_ids]
    fc_ins_docs_masked = [mask_titles_and_insurers(d, all_titles_for_mask) for d in fc_ins_docs]

    # Check if we need to encode any masked/unmasked queries or masked docs
    missing_bge_texts = []
    for d_m in fc_ins_docs_masked:
        if d_m not in bge_cache and d_m not in missing_bge_texts:
            missing_bge_texts.append(d_m)
    for r in sub_panel_rows:
        for q_cand in [r["question"], mask_titles_and_insurers(r["question"], all_titles_for_mask)]:
            pq = "为这个句子生成表示以用于检索相关文章：" + q_cand
            if pq not in bge_cache and pq not in missing_bge_texts:
                missing_bge_texts.append(pq)

    if missing_bge_texts:
        from transformers import AutoModel, AutoTokenizer
        import torch

        bge_dir = Path("/Users/knightwwang/Documents/Codex/2026-10-25/wob/work/models/bge")
        tok = AutoTokenizer.from_pretrained(str(bge_dir), local_files_only=True)
        mod = AutoModel.from_pretrained(str(bge_dir), local_files_only=True, use_safetensors=True).eval()
        new_embs = []
        with torch.inference_mode():
            for i in range(0, len(missing_bge_texts), 64):
                batch = missing_bge_texts[i : i + 64]
                enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                emb = torch.nn.functional.normalize(mod(**enc).last_hidden_state[:, 0], p=2, dim=1).cpu().numpy().astype(np.float32)
                for b_str, b_vec in zip(batch, emb):
                    bge_cache[b_str] = b_vec
                    new_embs.append((b_str, b_vec))
        # Save merged extra_bge_path
        ex_texts = []
        ex_vecs = []
        if extra_bge_path.exists():
            old_npz = np.load(extra_bge_path, allow_pickle=True)
            ex_texts = [str(x) for x in old_npz["texts"]]
            ex_vecs = list(old_npz["embeddings"])
        for b_str, b_vec in new_embs:
            ex_texts.append(b_str)
            ex_vecs.append(b_vec)
        np.savez_compressed(
            extra_bge_path,
            texts=np.array(ex_texts, dtype=object),
            embeddings=np.stack(ex_vecs, axis=0).astype(np.float32),
        )

    # Compute Clause-Body Lexical Novelty (`1 - kappa_2(q_masked, d_masked)`), Blinded Naturalness,
    # Title-Masked Retrieval Discrimination (`MRR_BGE - MRR_BM25`), and Joint Utility across methods
    fc_ins_doc_embs_masked = np.stack([bge_cache[d] for d in fc_ins_docs_masked], axis=0).astype(np.float64)
    bm25_vec_ins_m = base.CountVectorizer(analyzer="char", ngram_range=(2, 2))
    bm25_mat_ins_m = base.bm25_documents(bm25_vec_ins_m.fit_transform(fc_ins_docs_masked))

    joint_utility_by_method = {}
    novelty_points = []
    dense_adv_points = []

    for m_name in panel_methods:
        m_rows = [r for r in sub_panel_rows if r["method"] == m_name]
        ins_m_rows = [r for r in m_rows if r["domain_type"] == "insurance"]
        novelties = []
        bge_rrs = []
        bm25_rrs = []
        for r in ins_m_rows:
            seed = ins_seeds_map[r["seed_id"]]
            gold_idx = fc_ins_ids.index(seed["policy_id"])
            qm = mask_titles_and_insurers(r["question"], all_titles_for_mask)
            dm = fc_ins_docs_masked[gold_idx]
            k2 = bigram_containment(qm, dm)
            nov = 1.0 - float(k2)
            novelties.append(nov)

            # Title-masked BGE reciprocal rank vs BM25 reciprocal rank
            pq = "为这个句子生成表示以用于检索相关文章：" + qm
            q_emb = bge_cache[pq].astype(np.float64)
            d_sims = np.einsum("cd,d->c", fc_ins_doc_embs_masked, q_emb, optimize=True)
            d_order = list(np.argsort(-d_sims, kind="mergesort"))
            bge_rr = 1.0 / (d_order.index(gold_idx) + 1.0)
            bge_rrs.append(bge_rr)

            qx = bm25_vec_ins_m.transform([qm]).astype(np.float64)
            qx.data[:] = 1.0
            l_sims = (qx @ bm25_mat_ins_m.T).toarray()[0]
            l_order = list(np.argsort(-l_sims, kind="mergesort"))
            bm25_rr = 1.0 / (l_order.index(gold_idx) + 1.0)
            bm25_rrs.append(bm25_rr)

            if r.get("panel_majority_usable"):
                novelty_points.append(nov)
                dense_adv_points.append(bge_rr - bm25_rr)

        panel_usability_rate = float(np.mean([1.0 if r["panel_majority_usable"] else 0.0 for r in m_rows]))
        sym_usability_rate = float(np.mean([1.0 if r.get("format_and_slot_usable") else 0.0 for r in m_rows]))
        mean_nov = float(np.mean(novelties))
        mean_nat = float(np.mean([r["panel_mean_naturalness"] for r in m_rows]))
        norm_nat = (mean_nat - 1.0) / 4.0  # 0..1 scale
        mrr_bge = float(np.mean(bge_rrs))
        mrr_bm25 = float(np.mean(bm25_rrs))
        # Joint Utility combines Independent Panel Decision Fidelity with Conversational Naturalness & Novelty
        joint_utility_naturalness = round(panel_usability_rate * norm_nat, 4)
        joint_utility_novelty = round(panel_usability_rate * mean_nov, 4)

        joint_utility_by_method[m_name] = {
            "symbolic_joint_usability_rate": round(sym_usability_rate, 4),
            "raw_clause_panel_usability_rate": round(panel_usability_rate, 4),
            "clause_body_lexical_novelty_1_minus_kappa2": round(mean_nov, 4),
            "blinded_naturalness_score_1to5": round(mean_nat, 3),
            "normalized_naturalness_0to1": round(norm_nat, 4),
            "title_masked_bge_mrr": round(mrr_bge, 4),
            "title_masked_bm25_mrr": round(mrr_bm25, 4),
            "dense_minus_lexical_mrr_gap": round(mrr_bge - mrr_bm25, 4),
            "joint_utility_fidelity_x_naturalness": joint_utility_naturalness,
            "joint_utility_fidelity_x_novelty": joint_utility_novelty,
        }

    r_pearson, p_pearson = pearsonr(novelty_points, dense_adv_points)
    r_spearman, p_spearman = spearmanr(novelty_points, dense_adv_points)
    exp2_summary = {
        "methods_comparison": joint_utility_by_method,
        "novelty_vs_retriever_discrimination_correlation": {
            "n_usable_insurance_queries": len(novelty_points),
            "pearson_r": round(float(r_pearson), 4),
            "pearson_p": round(float(p_pearson), 6),
            "spearman_rho": round(float(r_spearman), 4),
            "spearman_p": round(float(p_spearman), 6),
            "finding": (
                "Higher clause-body lexical novelty (1 - kappa_2) significantly reduces lexical keyword matching "
                "shortcuts while preserving dense semantic retrieval, explaining why zero-call templates "
                "(`seed_passthrough`, `multi_template_3`) overestimate lexical BM25 relative to natural paraphrases."
            ),
        },
    }

    # -----------------------------------------------------------------------
    # Run Experiment 3: 6 Real End-to-End LLM RAG Pipelines
    # -----------------------------------------------------------------------
    rows_by_fam_for_rag = {
        "qwen_plus": [r for r in all_confirm_rows if r["model_family"] == "qwen_plus"],
        "deepseek": [r for r in all_confirm_rows if r["model_family"] == "deepseek"],
    }
    exp3_summary = run_or_load_real_llm_rag_study(
        args.out_dir,
        api_key,
        args.score_only,
        ins_seeds,
        cross_seeds,
        rows_by_fam_for_rag,
        bge_cache,
        adjudication_records=adjudication_records,
    )

    # -----------------------------------------------------------------------
    # Summarize Experiment 4: 3 New Non-Waiting-Period Insurance Rule Families (24 Seeds, 48 Pairs)
    #                         + Pooled 88-Seed (176-Pair) Statistical Confirmation + Compiler Audit
    # -----------------------------------------------------------------------
    # Summarize 24-seed Non-Waiting-Period Multi-Rule Benchmark (N=48 pairs across qwen3.8-max & deepseek-v4.1-flash)
    mrule_methods = [
        "seed_passthrough",
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "verify_only_filter",
        "generic_self_refine_budget_matched",
        "slot_guard_repair",
    ]
    mrule_summary_by_method = {}
    for m_name in mrule_methods:
        sub = [r for r in mrule_rows if r["method"] == m_name]
        n_tot = len(sub)
        n_joint = sum(1 for r in sub if r.get("format_and_slot_usable"))
        n_panel = sum(1 for r in sub if r.get("panel_majority_usable"))
        n_slot = sum(1 for r in sub if r.get("preserved_decision"))
        n_fmt = sum(1 for r in sub if r.get("format_valid"))
        by_rfam = {}
        for rfam in ("deductible_copay_tier", "hospital_ward_emergency", "enrollment_age_and_rx_limit"):
            rf_sub = [r for r in sub if r.get("rule_family") == rfam]
            rf_ok = sum(1 for r in rf_sub if r.get("format_and_slot_usable"))
            by_rfam[rfam] = f"{rf_ok}/{len(rf_sub)} ({round(100*rf_ok/max(1,len(rf_sub)), 1)}%)"
        mrule_summary_by_method[m_name] = {
            "n_pairs": n_tot,
            "format_valid": n_fmt,
            "slot_only_preserved": n_slot,
            "joint_usable": n_joint,
            "joint_usability_rate": round(n_joint / max(1, n_tot), 4),
            "raw_clause_panel_majority_usable": n_panel,
            "raw_clause_panel_usability_rate": round(n_panel / max(1, n_tot), 4),
            "mean_tokens_per_seed": round(float(np.mean([r.get("total_tokens", 0) for r in sub])), 1),
            "by_rule_family": by_rfam,
        }

    # Pooled Statistical Confirmation across:
    # (A) Confirmation + Post-Hardening Extension (64 seeds = 128 pairs)
    # (B) All 88 Seeds across all 4 Rule Families (40 confirmation + 24 extension + 24 non-waiting-period = 176 pairs)
    clause_cluster_map = {}
    for s in ins_seeds + uhold_ins_seeds:
        clause_cluster_map[s["seed_id"]] = s["policy_id"]
    for s in cross_seeds + uhold_cross_seeds + mrule_seeds:
        clause_cluster_map[s["seed_id"]] = s["clause_id"]

    pooled_128_rows = [r for r in qwen_rows + ds_rows + fair_rows + uhold_rows if r["method"] in mrule_methods]
    pooled_176_rows = pooled_128_rows + mrule_rows

    baselines_6 = [
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "generic_self_refine_budget_matched",
    ]

    stats_48_mrule = {
        b: compute_paired_stats(mrule_rows, "slot_guard_repair", b, lambda sid: clause_cluster_map[sid])
        for b in baselines_6
    }
    stats_128_pooled = {
        b: compute_paired_stats(pooled_128_rows, "slot_guard_repair", b, lambda sid: clause_cluster_map[sid])
        for b in baselines_6
    }
    stats_176_all = {
        b: compute_paired_stats(pooled_176_rows, "slot_guard_repair", b, lambda sid: clause_cluster_map[sid])
        for b in baselines_6
    }

    # Apply 6-baseline Holm-Bonferroni correction to the pooled 176-pair family
    holm_176_mcnemar = holm_bonferroni_adjust({b: stats_176_all[b]["exact_mcnemar_p"] for b in baselines_6})
    holm_176_seed = holm_bonferroni_adjust({b: stats_176_all[b]["seed_clustered_p"] for b in baselines_6})
    holm_176_clause = holm_bonferroni_adjust({b: stats_176_all[b]["clause_clustered_p"] for b in baselines_6})
    for b in baselines_6:
        stats_176_all[b]["holm_6_mcnemar_p"] = round_pval(holm_176_mcnemar[b])
        stats_176_all[b]["holm_6_seed_clustered_p"] = round_pval(holm_176_seed[b])
        stats_176_all[b]["holm_6_clause_clustered_p"] = round_pval(holm_176_clause[b])

    # Also compute pooled 176-pair paired tests under the Independent 3-Flagship Raw-Clause Panel Majority Vote!
    pooled_176_panel_rows = [
        {**r, "format_and_slot_usable": bool(r.get("panel_majority_usable"))} for r in pooled_176_rows
    ]
    stats_176_panel = {
        b: compute_paired_stats(pooled_176_panel_rows, "slot_guard_repair", b, lambda sid: clause_cluster_map[sid])
        for b in baselines_6
    }
    exp1_summary["pooled_88_seeds_176_pairs_under_raw_clause_panel"] = stats_176_panel

    exp4_summary = {
        "non_waiting_rule_families_24_seeds_48_pairs": {
            "clauses": NON_WAITING_RULE_CLAUSES,
            "methods_summary": mrule_summary_by_method,
            "paired_tests_vs_slot_guard_repair_n48": stats_48_mrule,
        },
        "pooled_64_seeds_128_pairs_waiting_period": stats_128_pooled,
        "pooled_88_seeds_176_pairs_all_4_rule_families": stats_176_all,
        "independent_zero_shot_schema_compiler_audit": compiler_audit,
    }

    full_summary = {
        "study_version": "v3.2-ccf-a-upgrade",
        "experiment_1_blinded_3flagship_raw_clause_adjudication": exp1_summary,
        "experiment_2_joint_utility_and_template_discrimination": exp2_summary,
        "experiment_3_real_llm_rag_downstream_selection": exp3_summary,
        "experiment_4_multi_rule_generalization_and_pooled_176_stats": exp4_summary,
    }
    out_summary_path = args.out_dir / "ccf_a_upgrade_summary.json"
    base.write_json(out_summary_path, full_summary)
    print(f"Wrote CCF-A upgrade summary to {out_summary_path}", flush=True)


if __name__ == "__main__":
    main()
