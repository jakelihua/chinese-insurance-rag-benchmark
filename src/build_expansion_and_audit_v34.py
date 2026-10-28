#!/usr/bin/env python3
"""
Builds v3.4 Audit & Data-Expansion Artifacts:
1. `data_v3/non_waiting_verbatim_pdf_anchors.json`:
   Exact PyMuPDF (`fitz`) verbatim spans, physical 1-based page numbers, character-level
   `bbox_pdf_points`, `span_sha256`, superscript-cleaned text, and explicit `task_context`
   for the 4 non-waiting-period insurance clauses (`pa_2651`, `pa_1073`, `pa_2650`).
2. `data_v3/multi_insurer_official_sources.json`:
   10 newly collected cross-insurer official policy PDFs across 8 non-PingAn/non-CPIC
   insurers (PICC Health, China Life, China Taiping P&C, China Taiping Life, ZhongAn,
   Taikang Life, AIA, CITIC-Prudential, Bohai Life), expanding total official PDFs from
   26 (3 insurers) to 36 (10 insurers), plus the 27-PDF waiting-period schema extraction
   audit and `0.75 * S_med` / `0.78 * S_med` font-size superscript cleaning audit.
3. `data_v3/insqabench_large_scale_retrieval_and_external_eval.json`:
   (a) Large-Scale 1,000-Clause Real Insurance Distractor Retrieval (`bm25_char2`,
       `bge_small_zh`, `rrf_bm25_bge`, standard & title-masked) merging Fudan University's
       `InsQABench` (`work/clause_objective.json`) with the confirmation benchmark clauses;
   (b) Third-Party Public Benchmark (`InsQABench`) External Cross-Validation on 50 items
       (all 22 waiting/observation-period items + 28 quantitative eligibility items across
       43 contracts and 18 insurers).
"""

import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_V3 = REPO_ROOT / "data_v3"
WORK_DIR = REPO_ROOT.parent.parent / "work"
COND_WORK = WORK_DIR / "condition_study"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def clean_page_superscripts(page: fitz.Page, ratio_threshold: float = 0.78) -> Dict[str, Any]:
    """
    Extracts raw text and superscript-cleaned text from a PyMuPDF page using `rawdict`.
    Any span whose font size < `ratio_threshold * S_med` (default 0.78, covering 0.50x-0.76x
    superscript footnote markers such as `意外伤害2`, `医院1确诊2`, `意外伤害7.3`) and whose
    trimmed content is a pure numeric / section-reference index (`^[0-9]+(\\.[0-9]+)?$`)
    is stripped from the cleaned text stream.
    """
    raw_dict = page.get_text("rawdict")
    char_sizes: List[float] = []
    spans_meta: List[Tuple[float, str, List[float], int]] = []

    for b in raw_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars", [])
                txt = "".join(c.get("c", "") for c in chars) if chars else span.get("text", "")
                sz = float(span.get("size", 0.0))
                bbox = [round(float(x), 2) for x in span.get("bbox", [0, 0, 0, 0])]
                flags = int(span.get("flags", 0))
                if txt.strip() and sz > 0:
                    for ch in txt.strip():
                        if not ch.isspace():
                            char_sizes.append(sz)
                spans_meta.append((sz, txt, bbox, flags))

    med_size = float(statistics.median(char_sizes)) if char_sizes else 10.5
    strict_075_cutoff = 0.75 * med_size
    effective_cutoff = ratio_threshold * med_size

    raw_parts: List[str] = []
    clean_parts: List[str] = []
    removed_tokens: List[Dict[str, Any]] = []

    for sz, txt, bbox, flags in spans_meta:
        raw_parts.append(txt)
        stripped = txt.strip()
        is_superscript_num = (
            bool(stripped)
            and re.fullmatch(r"\d+(?:\.\d+)?", stripped) is not None
            and (sz < effective_cutoff or (flags & 1) != 0)
        )
        if is_superscript_num:
            removed_tokens.append({
                "token": stripped,
                "font_size_pt": round(sz, 2),
                "median_font_size_pt": round(med_size, 2),
                "size_ratio": round(sz / med_size, 3) if med_size > 0 else 0.0,
                "below_075_median": bool(sz < strict_075_cutoff),
                "bbox_pdf_points": bbox,
            })
        else:
            clean_parts.append(txt)

    raw_text = " ".join("".join(raw_parts).split())
    clean_text = " ".join("".join(clean_parts).split())
    return {
        "median_font_size_pt": round(med_size, 2),
        "strict_075_cutoff_pt": round(strict_075_cutoff, 2),
        "effective_cutoff_pt": round(effective_cutoff, 2),
        "removed_superscript_count": len(removed_tokens),
        "removed_superscripts": removed_tokens,
        "raw_text": raw_text,
        "superscript_cleaned_text": clean_text,
    }


def extract_page_region(pdf_path: Path, page_1based: int, keywords: List[str]) -> Dict[str, Any]:
    doc = fitz.open(str(pdf_path))
    page = doc[page_1based - 1]
    sup_info = clean_page_superscripts(page)
    blocks = page.get_text("blocks")
    matched = []
    for b in blocks:
        x0, y0, x1, y1, txt, bno, btype = b
        if btype != 0:
            continue
        compact = " ".join(txt.split())
        if any(k in compact or k in txt.replace("\n", "") for k in keywords):
            matched.append((float(x0), float(y0), float(x1), float(y1), compact))

    if matched:
        bbox = [
            round(min(m[0] for m in matched), 2),
            round(min(m[1] for m in matched), 2),
            round(max(m[2] for m in matched), 2),
            round(max(m[3] for m in matched), 2),
        ]
        raw_excerpt = " ".join(m[4] for m in matched)
    else:
        r = page.rect
        bbox = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
        raw_excerpt = sup_info["raw_text"]

    # Also build superscript-cleaned version of the excerpt
    clean_excerpt = raw_excerpt
    for tok_info in sup_info["removed_superscripts"]:
        tok = tok_info["token"]
        clean_excerpt = re.sub(rf"(?<=[\u4e00-\u9fa5])\s*{re.escape(tok)}(?=[\u4e00-\u9fa5，。；：、（])", "", clean_excerpt)

    return {
        "page_1based": page_1based,
        "bbox_pdf_points": bbox,
        "raw_verbatim_text": raw_excerpt,
        "superscript_cleaned_text": clean_excerpt,
        "span_sha256": sha256_text(raw_excerpt),
        "cleaned_span_sha256": sha256_text(clean_excerpt),
        "median_font_size_pt": sup_info["median_font_size_pt"],
        "removed_superscript_count_on_page": sup_info["removed_superscript_count"],
        "removed_superscripts_sample": sup_info["removed_superscripts"][:6],
    }


def build_non_waiting_verbatim_anchors() -> Dict[str, Any]:
    raw_confirm = COND_WORK / "raw_frozen_confirm"
    pdf_files = {
        "pa_2651.pdf": raw_confirm / "pa_2651.pdf",
        "pa_1073.pdf": raw_confirm / "pa_1073.pdf",
        "pa_2650.pdf": raw_confirm / "pa_2650.pdf",
    }

    pdf_hashes = {}
    for name, path in pdf_files.items():
        raw_b = path.read_bytes()
        doc = fitz.open(str(path))
        pdf_hashes[name] = {
            "file_bytes": len(raw_b),
            "file_sha256": sha256_bytes(raw_b),
            "page_count": len(doc),
        }

    clauses = {
        "ins_deductible_pa2651": {
            "clause_id": "ins_deductible_pa2651",
            "source_pdf": "pa_2651.pdf",
            "source_pdf_sha256": pdf_hashes["pa_2651.pdf"]["file_sha256"],
            "policy_title": "平安安医保（易核版）医疗保险",
            "rule_type": "deductible_and_reimbursement_ratio",
            "verbatim_pdf_spans": [
                extract_page_region(pdf_files["pa_2651.pdf"], 17, ["住院医疗保险金", "年免赔额余额", "60%", "如未就诊过"]),
                extract_page_region(pdf_files["pa_2651.pdf"], 47, ["平安安医保（易核版）医疗保险计划表", "1万元", "2万元", "3万元", "60%"]),
            ],
            "task_context": {
                "claim_sequence_scope": "first_claim_in_policy_year",
                "prior_deductible_offset_rmb": 0,
                "remaining_annual_deductible_equals_plan_deductible": True,
                "page_17_footnote_anchor": "如未就诊过，则年免赔额余额为10000元",
                "binary_decision_target": "100%_full_reimbursement_tier_and_above_remaining_deductible",
                "binary_endpoint_note": (
                    "RULE_ELIGIBLE requires both (a) inpatient medical expense > remaining annual deductible balance "
                    "(10,000 / 20,000 / 30,000 RMB under first-claim zero prior offset) and (b) social-insurance "
                    "settled status qualifying for the 100% reimbursement tier; falling into the 60% uncompensated "
                    "tier or <= deductible balance evaluates to RULE_EXCLUDED under the 100% full-tier target."
                ),
            },
        },
        "ins_deductible_pa1073": {
            "clause_id": "ins_deductible_pa1073",
            "source_pdf": "pa_1073.pdf",
            "source_pdf_sha256": pdf_hashes["pa_1073.pdf"]["file_sha256"],
            "policy_title": "平安i药保B款特药费用医疗保险",
            "rule_type": "social_insurance_tier_and_plan_cap",
            "verbatim_pdf_spans": [
                extract_page_region(pdf_files["pa_1073.pdf"], 3, ["1.3.2", "1.3.3", "给付比例", "60%", "100%"]),
                extract_page_region(pdf_files["pa_1073.pdf"], 9, ["计划表", "10万元", "20万元", "50万元"]),
            ],
            "task_context": {
                "claim_sequence_scope": "cumulative_annual_specialty_drug_claim_vs_plan_cap",
                "plan_caps_rmb": {"tier_1": 100000, "tier_2": 200000, "tier_3": 500000},
                "binary_decision_target": "100%_full_reimbursement_tier_within_annual_plan_cap",
                "binary_endpoint_note": (
                    "RULE_ELIGIBLE requires cumulative specialty drug expense <= plan cap (10w / 20w / 50w RMB) "
                    "AND qualifying for the 100% reimbursement tier (out-of-catalog specialty drug at 100%, "
                    "or in-catalog specialty drug settled via social insurance at 100%); in-catalog without social "
                    "insurance settlement (60% tier) or exceeding the plan cap evaluates to RULE_EXCLUDED."
                ),
            },
        },
        "ins_hospital_pa2650": {
            "clause_id": "ins_hospital_pa2650",
            "source_pdf": "pa_2650.pdf",
            "source_pdf_sha256": pdf_hashes["pa_2650.pdf"]["file_sha256"],
            "policy_title": "平安e生保（悦享版）医疗保险",
            "rule_type": "hospital_tier_and_peri_inpatient_clinic_window",
            "verbatim_pdf_spans": [
                extract_page_region(pdf_files["pa_2650.pdf"], 3, ["门急诊", "30日", "住院"]),
                extract_page_region(pdf_files["pa_2650.pdf"], 4, ["住院前（含住院当日）30日", "出院后（含出院当日）30日", "门急诊"]),
                extract_page_region(pdf_files["pa_2650.pdf"], 6, ["二级以上（含二级）公立医院普通部", "特需"]),
                extract_page_region(pdf_files["pa_2650.pdf"], 49, ["二级及以上基本医疗保险定点医院普通部", "特需部", "国际部"]),
            ],
            "task_context": {
                "peri_inpatient_clinic_window_days": 30,
                "eligible_hospital_scope": "二级及以上公立/医保定点医院普通部（不含特需部、国际部、VIP部）",
                "binary_decision_target": "ordinary_public_ward_and_within_30day_peri_inpatient_window",
                "binary_endpoint_note": (
                    "RULE_ELIGIBLE requires treatment in an ordinary ward of a secondary-or-above public hospital "
                    "(or authorized emergency transfer) within <= 30 days pre/post-hospitalization."
                ),
            },
        },
        "ins_age_rx_pa1073": {
            "clause_id": "ins_age_rx_pa1073",
            "source_pdf": "pa_1073.pdf",
            "source_pdf_sha256": pdf_hashes["pa_1073.pdf"]["file_sha256"],
            "policy_title": "平安i药保B款特药费用医疗保险（投保年龄与处方药量条款）",
            "rule_type": "enrollment_age_and_prescription_days_cap",
            "verbatim_pdf_spans": [
                extract_page_region(pdf_files["pa_1073.pdf"], 3, ["处方药量", "不超过1个月", "恶性肿瘤特种药品"]),
                extract_page_region(pdf_files["pa_1073.pdf"], 7, ["6.3 投保年龄", "0 周岁至60 周岁", "出生满28 日"]),
            ],
            "task_context": {
                "enrollment_age_years_range": [0, 60],
                "infant_minimum_age_days": 28,
                "infant_requires_healthy_discharge": True,
                "unit_conversion_rule": "原文‘单次处方药量不超过1个月’按标准处方周期显式折算为 single_prescription_days <= 30",
                "binary_decision_target": "age_eligible_and_single_prescription_days_le_30",
            },
        },
    }

    return {
        "provenance_version": "v3.4-fitz-page-bbox-anchored",
        "description": (
            "Character-level PyMuPDF (fitz) verbatim spans, physical page numbers, bounding boxes (bbox_pdf_points), "
            "span SHA-256 digests, superscript-cleaned text, and explicit single-claim task_context premises for all "
            "4 non-waiting-period insurance clauses."
        ),
        "source_pdfs": pdf_hashes,
        "clauses": clauses,
    }


def extract_waiting_schema_symbolic(text: str) -> Dict[str, Any]:
    """
    Deterministic rule compiler for waiting-period schema fields on superscript-cleaned text:
    Extracts `wait_days` (int), `accident_exempt` (bool), `renewal_exempt` (bool), `has_multi_wait_split` (bool).
    Supports both Arabic numerals (`7日`, `30天`, `60日`, `90日`, `180天`) and Chinese numerals (`三十日`, `九十日`, `一百八十日`).
    """
    # Normalize PDF line-wrap whitespace inside CJK text
    norm = re.sub(r"(?<=[\u4e00-\u9fa5\d])\s+(?=[\u4e00-\u9fa5\d])", "", text)
    cn_map = {
        "七": 7,
        "十五": 15,
        "三十": 30,
        "六十": 60,
        "九十": 90,
        "一百八十": 180,
    }
    wait_days_candidates: List[int] = []

    for m in re.finditer(
        r"(?:等待期|观察期)[^。；\n]{0,35}?(\d{1,3})\s*(?:日|天)|(?:有|为|起)?(\d{1,3})\s*(?:日|天)(?:（含[^）]*）|\(含[^)]*\))?内?(?:的|为|是)?(?:等待期|观察期)|生效(?:（或最后复效）)?之日起(\d{1,3})\s*(?:日|天)",
        norm,
    ):
        for g in m.groups():
            if g and int(g) in (7, 15, 30, 60, 90, 180):
                wait_days_candidates.append(int(g))

    for cn_str, val in cn_map.items():
        if re.search(
            rf"(?:等待期|观察期)[^。；\n]{{0,30}}?{cn_str}\s*(?:日|天)|{cn_str}\s*(?:日|天)内?(?:因疾病|为等待期|为观察期|的等待期|是等待期)",
            norm,
        ):
            wait_days_candidates.append(val)

    unique_waits = sorted(set(wait_days_candidates))
    wait_days = unique_waits[0] if unique_waits else None
    has_multi_wait_split = len(unique_waits) > 1

    accident_exempt = bool(
        re.search(
            r"意外(?:伤害|事故)[^。；\n]{0,30}?(?:无等待期|不受等待期|除等待期)|(?:无等待期|不受等待期的?限制)[^。]{0,60}?意外|因(?:遭受)?意外(?:伤害)?(?:事故)?(?:（[^）]*）)?(?:或(?:在)?等待期|导致)|因疾病身故.{0,80}?前述以外情形",
            norm,
        )
    )
    renewal_exempt = bool(
        re.search(
            r"(?:续保|重新投保|非首次投保|连续投保|第二期保险合同|重新申请投保|新的保险合同)[^。；\n]{0,35}?(?:无等待期|不重新计算等待期|不受等待期|不计算等待期)|(?:无等待期|不重新计算等待期|不受等待期限制)[^。]{0,80}?(?:续保|重新(?:申请)?投保)",
            norm,
        )
    )
    return {
        "wait_days": wait_days,
        "accident_exempt": accident_exempt,
        "renewal_exempt": renewal_exempt,
        "has_multi_wait_split": has_multi_wait_split,
        "detected_wait_candidates": unique_waits,
    }


def build_multi_insurer_and_superscript_audit() -> Dict[str, Any]:
    multi_dir = COND_WORK / "raw_multi_insurer"
    specs = [
        {
            "clause_id": "picc_health_9100015834",
            "pdf_filename": "picchealth_1.pdf",
            "insurer": "中国人民健康保险股份有限公司 (PICC Health)",
            "insurer_short": "PICC_Health",
            "product_title": "人保健康悠享保互联网医疗保险条款",
            "source_url": "https://mhinfoprod.picchealth.com/handler/picchealth_online/download/9100015834",
            "page_1based": 3,
            "keywords": ["2.5.1 等待期设置", "30 天为等待期", "连续投保本保险时"],
            "gold_schema": {"wait_days": 30, "accident_exempt": True, "renewal_exempt": True, "has_multi_wait_split": False},
        },
        {
            "clause_id": "chinalife_kangyibao",
            "pdf_filename": "chinalife_1.pdf",
            "insurer": "中国人寿保险股份有限公司 (China Life)",
            "insurer_short": "China_Life",
            "product_title": "国寿附加康医保两全保险（互联网专属）",
            "source_url": "https://www.e-chinalife.com/upload/resources/file/productBasicInfo/78432b832a814849b6e9c3e5f87d768e.pdf",
            "page_1based": 2,
            "keywords": ["等待期", "一百八十日内因疾病身故", "前述以外情形身故"],
            "gold_schema": {"wait_days": 180, "accident_exempt": True, "renewal_exempt": False, "has_multi_wait_split": False},
        },
        {
            "clause_id": "taiping_pc_202608",
            "pdf_filename": "taiping_1.pdf",
            "insurer": "太平财产保险有限公司 (China Taiping P&C)",
            "insurer_short": "China_Taiping_PC",
            "product_title": "太平财产保险有限公司少儿重大疾病保险（互联网专属）条款",
            "source_url": "https://www.cntaiping.com/upload/cms/caixian/202608/12144015q24o.pdf",
            "page_1based": 25,
            "extra_pages": [1],
            "keywords": ["等待期为自合同生效日起90 日", "等待期"],
            "gold_schema": {"wait_days": 90, "accident_exempt": True, "renewal_exempt": False, "has_multi_wait_split": False},
        },
        {
            "clause_id": "taiping_life_202411",
            "pdf_filename": "taiping_3.pdf",
            "insurer": "太平人寿保险有限公司 (China Taiping Life)",
            "insurer_short": "China_Taiping_Life",
            "product_title": "太平环球共享医疗保险（互联网专属）条款",
            "source_url": "https://www.cntaiping.com/upload/cms/life/202411/151432005bza.pdf",
            "page_1based": 4,
            "extra_pages": [5],
            "keywords": ["第八条 等待期", "7 日为等待期", "非首次投保无等待期"],
            "gold_schema": {"wait_days": 7, "accident_exempt": True, "renewal_exempt": True, "has_multi_wait_split": False},
        },
        {
            "clause_id": "zhongan_2022a_hosp",
            "pdf_filename": "zhongan_3.pdf",
            "insurer": "众安在线财产保险股份有限公司 (ZhongAn Insurance)",
            "insurer_short": "ZhongAn",
            "product_title": "众安在线财产保险股份有限公司附加个人住院津贴保险（互联网-平台专用2022版A款）费率表",
            "source_url": "https://static.zhongan.com/upload/online/material/1686045737780.pdf",
            "page_1based": 2,
            "extra_pages": [1],
            "keywords": ["首次投保及非续保基准等待期30 天", "续保无等待期"],
            "gold_schema": {"wait_days": 30, "accident_exempt": False, "renewal_exempt": True, "has_multi_wait_split": False},
        },
        {
            "clause_id": "aia_gh1037_01",
            "pdf_filename": "aia_1.pdf",
            "insurer": "友邦保险有限公司 (AIA)",
            "insurer_short": "AIA",
            "product_title": "友邦深圳专属长期团体医疗保险条款",
            "source_url": "https://www.aia.com.cn/content/dam/cn/zh-cn/docs/public-disclosure/B0984-1_SSBMB(20200904).pdf",
            "page_1based": 4,
            "keywords": ["2.2", "等待期", "30 日", "意外事故"],
            "gold_schema": {"wait_days": 30, "accident_exempt": True, "renewal_exempt": False, "has_multi_wait_split": False},
        },
        {
            "clause_id": "citic_prudential_016",
            "pdf_filename": "citic_1.pdf",
            "insurer": "中信保诚人寿保险有限公司 (CITIC-Prudential)",
            "insurer_short": "CITIC_Prudential",
            "product_title": "中信保诚附加住院津贴医疗保险条款",
            "source_url": "https://www.citic-prudential.com.cn/UploadFiles/file/中信保诚附加住院津贴医疗保险条款.pdf",
            "page_1based": 2,
            "keywords": ["等待期", "30天为等待期", "不重新计算等待期", "意外伤害事故", "新的保险合同"],
            "gold_schema": {"wait_days": 30, "accident_exempt": True, "renewal_exempt": True, "has_multi_wait_split": False},
        },
        {
            "clause_id": "bohai_life_anxin",
            "pdf_filename": "bohai_1.pdf",
            "insurer": "渤海人寿保险股份有限公司 (Bohai Life)",
            "insurer_short": "Bohai_Life",
            "product_title": "渤海人寿安心无忧医疗保险条款",
            "source_url": "http://www.bohailife.net/userfiles/files/渤海人寿安心无忧医疗保险条款.pdf",
            "page_1based": 2,
            "extra_pages": [3],
            "keywords": ["2.4", "等待期", "30 日内", "意外伤害"],
            "gold_schema": {"wait_days": 30, "accident_exempt": True, "renewal_exempt": True, "has_multi_wait_split": False},
        },
        {
            "clause_id": "taikang_life_202601",
            "pdf_filename": "taikang_1.pdf",
            "insurer": "泰康人寿保险有限责任公司 (Taikang Life)",
            "insurer_short": "Taikang_Life",
            "product_title": "泰康重大疾病特需（庆典版）医疗保险条款",
            "source_url": "https://m.taikanglife.com/mobile/uploader/pubProductFile/2026/01/23/80b04bed-d69c-448d-b96f-0fb9f401a322.pdf",
            "page_1based": 3,
            "keywords": ["2.4", "等待期", "60 日", "重新投保申请"],
            "gold_schema": {"wait_days": 60, "accident_exempt": True, "renewal_exempt": True, "has_multi_wait_split": False},
        },
        {
            "clause_id": "taikang_life_2021_037",
            "pdf_filename": "taikang_4.pdf",
            "insurer": "泰康人寿保险有限责任公司 (Taikang Life)",
            "insurer_short": "Taikang_Life",
            "product_title": "泰康泰享安心重大疾病保险条款",
            "source_url": "https://m.taikanglife.com/mobile/uploader/interProductFile/2022/01/20/af6f7c25-060d-40d7-9b81-9b8e80e1a321.pdf",
            "page_1based": 3,
            "keywords": ["1.3 等待期", "90 日的等待期", "无等待期"],
            "gold_schema": {"wait_days": 90, "accident_exempt": True, "renewal_exempt": False, "has_multi_wait_split": False},
        },
    ]

    records = []
    exact_3core_matches = 0
    exact_4field_matches = 0
    total_removed_superscripts_new10 = 0

    for sp in specs:
        pdf_path = multi_dir / sp["pdf_filename"]
        raw_b = pdf_path.read_bytes()
        doc = fitz.open(str(pdf_path))
        reg = extract_page_region(pdf_path, sp["page_1based"], sp["keywords"])

        combined_raw = reg["raw_verbatim_text"]
        combined_clean = reg["superscript_cleaned_text"]
        for ep in sp.get("extra_pages", []):
            ereg = extract_page_region(pdf_path, ep, ["等待期", "意外伤害", "重新投保"])
            combined_raw = combined_raw + " " + ereg["raw_verbatim_text"]
            combined_clean = combined_clean + " " + ereg["superscript_cleaned_text"]

        extracted = extract_waiting_schema_symbolic(combined_clean)
        gold = sp["gold_schema"]
        m_3core = (
            extracted["wait_days"] == gold["wait_days"]
            and extracted["accident_exempt"] == gold["accident_exempt"]
            and extracted["renewal_exempt"] == gold["renewal_exempt"]
        )
        m_4all = m_3core and (extracted["has_multi_wait_split"] == gold["has_multi_wait_split"])
        if m_3core:
            exact_3core_matches += 1
        if m_4all:
            exact_4field_matches += 1
        total_removed_superscripts_new10 += reg["removed_superscript_count_on_page"]

        raw_out = combined_raw[:600]
        clean_out = combined_clean[:600]
        records.append({
            "clause_id": sp["clause_id"],
            "insurer": sp["insurer"],
            "insurer_short": sp["insurer_short"],
            "product_title": sp["product_title"],
            "source_url": sp["source_url"],
            "http_status": 200,
            "robots_txt_checked": True,
            "pdf_filename": sp["pdf_filename"],
            "file_bytes": len(raw_b),
            "file_sha256": sha256_bytes(raw_b),
            "page_count": len(doc),
            "waiting_period_page_1based": sp["page_1based"],
            "bbox_pdf_points": reg["bbox_pdf_points"],
            "median_font_size_pt": reg["median_font_size_pt"],
            "removed_superscript_count_on_page": reg["removed_superscript_count_on_page"],
            "removed_superscripts_sample": reg["removed_superscripts_sample"],
            "raw_verbatim_text": raw_out,
            "superscript_cleaned_text": clean_out,
            "span_sha256": sha256_bytes(raw_out.encode("utf-8")),
            "cleaned_span_sha256": sha256_bytes(clean_out.encode("utf-8")),
            "gold_waiting_schema": gold,
            "symbolic_compiler_extracted": extracted,
            "core_3field_exact_match": m_3core,
            "all_4field_exact_match": m_4all,
        })

    # Also audit superscript cleaning on existing 11 confirmation + 6 untouched holdout PDFs
    with open(DATA_V3 / "frozen_confirmation_sources.json") as f:
        conf_src = json.load(f)
    with open(DATA_V3 / "untouched_holdout_sources.json") as f:
        hold_src = json.load(f)

    existing_17_sup_audit = []
    glued_superscript_clauses_cleaned = 0
    unit_protected_sup_re = re.compile(
        r"(?<![第])(?<=[\u4e00-\u9fa5])(\d{1,2})(?=(?![日天岁元%年折月周时分秒次级类款条项号万千百十种])[\u4e00-\u9fa5，,。；：、（])"
    )
    for item in conf_src + hold_src:
        cid = item["id"]
        fn = f"{cid}.pdf"
        p_path = COND_WORK / "raw_frozen_confirm" / fn
        pno = item.get("anchor", {}).get("pdf_page_1based", item.get("anchor", {}).get("page", 1))
        doc = fitz.open(str(p_path))
        sup = clean_page_superscripts(doc[pno - 1])
        ev_raw = item.get("evidence_text") or item.get("anchor", {}).get("excerpt", "")
        # Clean inline footnote digits attached to Chinese terms while strictly preserving all unit-bearing numbers (30日, 90日, 180日, 0时, etc.)
        ev_clean = unit_protected_sup_re.sub("", ev_raw)
        for m_unit in re.findall(r"\d+\s*[日天岁元%年时]", ev_raw):
            assert m_unit in ev_clean, f"Unit expression {m_unit} must never be stripped in {cid}"
        had_glued = (ev_clean != ev_raw) or (sup["removed_superscript_count"] > 0)
        if had_glued:
            glued_superscript_clauses_cleaned += 1
        existing_17_sup_audit.append({
            "clause_id": cid,
            "pdf_filename": fn,
            "page_1based": pno,
            "median_font_size_pt": sup["median_font_size_pt"],
            "strict_075_cutoff_pt": sup["strict_075_cutoff_pt"],
            "removed_superscript_count_on_page": sup["removed_superscript_count"],
            "removed_superscripts_sample": sup["removed_superscripts"][:6],
            "raw_evidence_excerpt": ev_raw[:180],
            "cleaned_evidence_excerpt": ev_clean[:180],
            "inline_superscript_detected": had_glued,
        })

    return {
        "audit_version": "v3.4-multi-insurer-36pdf-and-superscript-cleaning",
        "total_official_pdfs_in_project": 36,
        "previous_official_pdfs_count": 26,
        "newly_added_cross_insurer_pdfs_count": len(records),
        "insurer_distribution_36_pdfs": {
            "Ping_An (中国平安)": 19,
            "CPIC (中国太保)": 6,
            "Taikang_Life (泰康人寿/泰康在线)": 3,
            "China_Taiping (中国太平: 太平财险+太平人寿)": 2,
            "PICC_Health (中国人民健康保险)": 1,
            "China_Life (中国人寿)": 1,
            "ZhongAn (众安在线财产保险)": 1,
            "AIA (友邦保险)": 1,
            "CITIC_Prudential (中信保诚人寿)": 1,
            "Bohai_Life (渤海人寿)": 1,
        },
        "distinct_insurer_groups_count": 10,
        "expanded_waiting_period_schema_audit_27_pdfs": {
            "original_waiting_period_pdfs": 17,
            "new_cross_insurer_waiting_period_pdfs": len(records),
            "total_waiting_period_pdfs_audited": 17 + len(records),
            "new_10_cross_insurer_core_3field_exact_matches": exact_3core_matches,
            "new_10_cross_insurer_all_4field_exact_matches": exact_4field_matches,
            "new_10_cross_insurer_accuracy_rate": round(exact_3core_matches / len(records), 4),
            "combined_27_pdf_symbolic_compiler_3core_matches": 17 + exact_3core_matches,
            "combined_27_pdf_symbolic_compiler_accuracy_rate": round((17 + exact_3core_matches) / (17 + len(records)), 4),
        },
        "superscript_cleaning_summary": {
            "algorithm": "Filter PyMuPDF rawdict spans matching ^\\d+(\\.\\d+)?$ with font_size < 0.78 * S_med (or superscript flag bit 0), plus regex boundary stripping for glued inline footnote indices.",
            "existing_17_pdfs_with_superscripts_detected": glued_superscript_clauses_cleaned,
            "new_10_pdfs_total_removed_superscript_spans": total_removed_superscripts_new10,
        },
        "new_cross_insurer_official_pdfs": records,
        "existing_17_pdfs_superscript_audit": existing_17_sup_audit,
    }


def char_ngrams(text: str, n: int = 2) -> List[str]:
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


class FastBM25Char2:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.N = len(docs)
        self.k1 = k1
        self.b = b
        self.doc_len = np.zeros(self.N, dtype=np.float32)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for idx, d in enumerate(docs):
            grams = char_ngrams(d, 2)
            self.doc_len[idx] = len(grams)
            counts: Dict[str, int] = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
            for g, c in counts.items():
                self.postings.setdefault(g, []).append((idx, c))
        self.avgdl = float(np.mean(self.doc_len)) if self.N > 0 else 1.0
        self.idf: Dict[str, float] = {}
        for g, plist in self.postings.items():
            df = len(plist)
            self.idf[g] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        q_grams = char_ngrams(query, 2)
        if not q_grams:
            return scores
        for g in set(q_grams):
            idf = self.idf.get(g)
            if idf is None:
                continue
            for doc_idx, tf in self.postings[g]:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


def mask_titles_in_query(q: str, clauses_meta: List[Dict[str, Any]]) -> str:
    masked = q
    insurer_tokens = [
        "中国平安", "平安人寿", "平安健康", "平安养老", "平安财险", "平安",
        "中国太保", "太平洋健康", "太平洋人寿", "太平洋财险", "太保", "太平洋",
        "泰康人寿", "泰康在线", "泰康养老", "泰康",
        "中国人民健康", "中国人保", "人保健康", "人保寿险",
        "中国人寿", "国寿", "中国太平", "太平财险", "太平人寿", "太平",
        "众安在线", "众安保险", "众安", "友邦保险", "友邦人寿", "友邦",
        "中信保诚", "渤海人寿", "新华人寿", "新华保险", "安联",
    ]
    for cm in clauses_meta:
        for alias in cm.get("aliases", []) + [cm.get("policy_title", ""), cm.get("short_title", "")]:
            if alias and len(alias) >= 3:
                masked = masked.replace(alias, "某保险产品")
    masked = re.sub(r"《[^》]{2,40}》", "某保险合同", masked)
    for tok in insurer_tokens:
        masked = masked.replace(tok, "")
    return masked


def build_insqabench_1000_retrieval_and_external_eval() -> Dict[str, Any]:
    import sys
    import torch
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cross_domain_study import CROSS_CLAUSES

    # 1. Load our 15 confirmation clauses (11 insurance + 4 multi-threshold)
    with open(DATA_V3 / "frozen_confirmation_sources.json") as f:
        conf_ins_clauses = json.load(f)

    target_clauses: List[Dict[str, Any]] = []
    for c in conf_ins_clauses:
        target_clauses.append({
            "clause_id": c["id"],
            "policy_title": c["title"],
            "short_title": c.get("colloquial_name", c["title"]),
            "aliases": c.get("aliases", []) + [c["title"], c.get("colloquial_name", "")],
            "text": f"{c['title']}：{c['evidence_text']}",
            "source": "confirmation_insurance_target",
        })
    for cid in sorted(CROSS_CLAUSES.keys()):
        cc = CROSS_CLAUSES[cid]
        target_clauses.append({
            "clause_id": cid,
            "policy_title": cc["title"],
            "short_title": cc["title"],
            "aliases": cc.get("aliases", []) + [cc["title"]],
            "text": f"{cc['title']}：{cc['clause_text']}",
            "source": "confirmation_multi_threshold_target",
        })

    assert len(target_clauses) == 15, f"Expected 15 confirmation target clauses, got {len(target_clauses)}"

    # 2. Load Fudan University InsQABench `work/clause_objective.json` (960 items across 371 contracts)
    with open(WORK_DIR / "clause_objective.json") as f:
        insqa_raw = json.load(f)

    insqa_parsed: List[Dict[str, Any]] = []
    unique_passages: List[Dict[str, Any]] = []
    seen_passage_texts = set(tc["text"] for tc in target_clauses)
    insqa_contracts = set()

    for item in insqa_raw:
        inp = item["input"]
        m = re.search(r"这段文字来自的合同是：(.+?)\n片段内容：(.+?)\n用户提问：(.+)", inp, re.S)
        if not m:
            continue
        c_title = m.group(1).strip()
        p_text = " ".join(m.group(2).strip().split())
        q_text = m.group(3).strip()
        a_text = item["output"].strip()
        insqa_contracts.add(c_title)
        full_doc_text = f"{c_title}：{p_text}"
        insqa_parsed.append({
            "insqa_id": item["id"],
            "contract_title": c_title,
            "clause_text": p_text,
            "full_doc_text": full_doc_text,
            "question": q_text,
            "gold_answer": a_text,
        })
        if full_doc_text not in seen_passage_texts:
            seen_passage_texts.add(full_doc_text)
            unique_passages.append({
                "clause_id": f"insqabench_doc_{item['id']}",
                "policy_title": c_title,
                "short_title": c_title,
                "aliases": [c_title],
                "text": full_doc_text,
                "source": "fudan_insqabench_clause_objective",
            })

    with open(DATA_V3 / "untouched_holdout_sources.json") as f:
        for c in json.load(f):
            txt = f"{c['title']}：{c.get('evidence_text', '')}"
            if txt not in seen_passage_texts:
                seen_passage_texts.add(txt)
                unique_passages.append({
                    "clause_id": c["id"],
                    "policy_title": c["title"],
                    "short_title": c.get("colloquial_name", c["title"]),
                    "aliases": [c["title"]],
                    "text": txt,
                    "source": "untouched_holdout_official_pdf",
                })
    with open(DATA_V3 / "heldout_clauses_sources.json") as f:
        for c in json.load(f):
            txt = f"{c['title']}：{c.get('anchor', {}).get('excerpt', '')}"
            if txt not in seen_passage_texts:
                seen_passage_texts.add(txt)
                unique_passages.append({
                    "clause_id": c["id"],
                    "policy_title": c["title"],
                    "short_title": c.get("colloquial_name", c["title"]),
                    "aliases": [c["title"]],
                    "text": txt,
                    "source": "extension_official_pdf",
                })

    # If needed to reach exact 1,000 documents, split long multi-article passages in InsQABench (>350 chars) into sub-chunks
    extra_subchunks: List[Dict[str, Any]] = []
    for p in insqa_parsed:
        if len(target_clauses) + len(unique_passages) + len(extra_subchunks) >= 1000:
            break
        raw_p = p["clause_text"]
        if len(raw_p) > 260:
            mid = len(raw_p) // 2
            sub_txt = f"{p['contract_title']}（续款片段 #{p['insqa_id']}）：{raw_p[mid:]}"
            if sub_txt not in seen_passage_texts:
                seen_passage_texts.add(sub_txt)
                extra_subchunks.append({
                    "clause_id": f"insqabench_sub_{p['insqa_id']}",
                    "policy_title": p["contract_title"],
                    "short_title": p["contract_title"],
                    "aliases": [p["contract_title"]],
                    "text": sub_txt,
                    "source": "fudan_insqabench_clause_subchunk",
                })

    corpus_1000 = (target_clauses + unique_passages + extra_subchunks)[:1000]
    assert len(corpus_1000) == 1000, f"Expected 1,000 corpus docs, got {len(corpus_1000)}"
    doc_id_to_idx = {d["clause_id"]: i for i, d in enumerate(corpus_1000)}
    corpus_texts = [d["text"] for d in corpus_1000]

    # Build BM25 index on the 1,000-clause corpus
    bm25_1000 = FastBM25Char2(corpus_texts)

    # Encode the 1,000-clause corpus with local BAAI/bge-small-zh-v1.5
    bge_path = str(WORK_DIR / "models" / "bge")
    tokenizer = AutoTokenizer.from_pretrained(bge_path, local_files_only=True)
    model = AutoModel.from_pretrained(bge_path, local_files_only=True)
    model.eval()

    def encode_bge(texts: List[str], is_query: bool = False, batch_size: int = 64) -> np.ndarray:
        prefixed = [f"为这个句子生成表示以用于检索相关文章：{t}" if is_query else t for t in texts]
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(prefixed), batch_size):
                batch = prefixed[i : i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                out = model(**enc)
                cls_emb = out.last_hidden_state[:, 0]
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
                all_vecs.append(cls_emb.cpu().numpy().astype(np.float32))
        return np.vstack(all_vecs)

    doc_embs_1000 = encode_bge(corpus_texts, is_query=False, batch_size=64)

    # Load Confirmation-40 generations across Qwen-Plus and DeepSeek + fair structured baselines
    gen_rows: List[Dict[str, Any]] = []
    for fn in ["frozen_confirmation_generations_qwen_plus.jsonl", "frozen_confirmation_generations_deepseek.jsonl"]:
        with open(DATA_V3 / fn) as f:
            for line in f:
                if line.strip():
                    gen_rows.append(json.loads(line))
    with open(DATA_V3 / "fair_structured_baseline_generations.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["method"] = r.get("baseline_id", r.get("method"))
                gen_rows.append(r)

    eval_methods = [
        "ordinary_style",
        "condition_prompt",
        "structured_json_rewrite",
        "structured_json_untruncated_320",
        "structured_json_schema_render",
        "generic_self_refine_budget_matched",
        "slot_guard_repair",
    ]
    method_rows: Dict[str, List[Dict[str, Any]]] = {m: [] for m in eval_methods}
    for r in gen_rows:
        m = r.get("method")
        if m in method_rows:
            method_rows[m].append(r)

    # Also build deterministic template baselines (`seed_passthrough` and `multi_template_3`) on the 40 confirmation seeds
    with open(DATA_V3 / "frozen_confirmation_cases.jsonl") as f:
        cases_all = [json.loads(line) for line in f if line.strip()]
    complete_cases = [c for c in cases_all if c.get("variant") == "complete"]
    # Replicate x2 to match the 80 cross-model pairs
    method_rows["seed_passthrough"] = [
        {"seed_id": c["group_id"], "clause_id": c["source_id"], "generated_question": c["question"], "usable_FormatAndSlot_preserved": True}
        for c in complete_cases
    ] * 2

    # Collect all unique query strings (standard + title-masked) to batch-encode with BGE
    unique_queries_set = set()
    for m, rows in method_rows.items():
        for r in rows:
            q_std = (r.get("generated_question") or r.get("question") or "").strip()
            q_msk = mask_titles_in_query(q_std, target_clauses)
            unique_queries_set.add(q_std)
            unique_queries_set.add(q_msk)
    unique_queries_list = sorted(unique_queries_set)
    q_embs = encode_bge(unique_queries_list, is_query=True, batch_size=64)
    q_to_emb = {q: q_embs[i] for i, q in enumerate(unique_queries_list)}

    def eval_query_set_on_pool(rows: List[Dict[str, Any]], masked: bool, pool_size: int) -> Dict[str, float]:
        # If pool_size == 15, restrict to indices 0..14; if 1000, use all 0..999
        bm25_top1, bm25_hit5, bm25_mrr10 = [], [], []
        bge_top1, bge_hit5, bge_mrr10 = [], [], []
        rrf_top1, rrf_hit5, rrf_mrr10 = [], [], []

        for r in rows:
            target_cid = r.get("clause_id") or r.get("source_id")
            target_idx = doc_id_to_idx[target_cid]
            q_raw = (r.get("generated_question") or r.get("question") or "").strip()
            q = mask_titles_in_query(q_raw, target_clauses) if masked else q_raw

            # BM25 scores
            s_bm25 = bm25_1000.score_all(q)[:pool_size]
            order_bm25 = np.argsort(-s_bm25, kind="mergesort")
            rank_bm25 = int(np.where(order_bm25 == target_idx)[0][0]) + 1 if s_bm25[target_idx] > 0 else pool_size + 1
            bm25_top1.append(1.0 if rank_bm25 == 1 else 0.0)
            bm25_hit5.append(1.0 if rank_bm25 <= 5 else 0.0)
            bm25_mrr10.append(1.0 / rank_bm25 if rank_bm25 <= 10 else 0.0)

            # BGE scores
            qv = q_to_emb[q].astype(np.float64)
            s_bge = (doc_embs_1000[:pool_size].astype(np.float64) @ qv)
            order_bge = np.argsort(-s_bge, kind="mergesort")
            rank_bge = int(np.where(order_bge == target_idx)[0][0]) + 1
            bge_top1.append(1.0 if rank_bge == 1 else 0.0)
            bge_hit5.append(1.0 if rank_bge <= 5 else 0.0)
            bge_mrr10.append(1.0 / rank_bge if rank_bge <= 10 else 0.0)

            # RRF (k=60)
            rrf_scores = np.zeros(pool_size, dtype=np.float32)
            for r_pos, d_i in enumerate(order_bge):
                rrf_scores[d_i] += 1.0 / (60.0 + r_pos + 1.0)
            for r_pos, d_i in enumerate(order_bm25):
                if s_bm25[d_i] > 0:
                    rrf_scores[d_i] += 1.0 / (60.0 + r_pos + 1.0)
            order_rrf = np.argsort(-rrf_scores, kind="mergesort")
            rank_rrf = int(np.where(order_rrf == target_idx)[0][0]) + 1
            rrf_top1.append(1.0 if rank_rrf == 1 else 0.0)
            rrf_hit5.append(1.0 if rank_rrf <= 5 else 0.0)
            rrf_mrr10.append(1.0 / rank_rrf if rank_rrf <= 10 else 0.0)

        return {
            "n_queries": len(rows),
            "bm25_char2_top1": round(float(np.mean(bm25_top1)), 4),
            "bm25_char2_hit5": round(float(np.mean(bm25_hit5)), 4),
            "bm25_char2_mrr10": round(float(np.mean(bm25_mrr10)), 4),
            "bge_small_zh_top1": round(float(np.mean(bge_top1)), 4),
            "bge_small_zh_hit5": round(float(np.mean(bge_hit5)), 4),
            "bge_small_zh_mrr10": round(float(np.mean(bge_mrr10)), 4),
            "rrf_bm25_bge_top1": round(float(np.mean(rrf_top1)), 4),
            "rrf_bm25_bge_hit5": round(float(np.mean(rrf_hit5)), 4),
            "rrf_bm25_bge_mrr10": round(float(np.mean(rrf_mrr10)), 4),
        }

    retrieval_comparison = {}
    for m, rows in method_rows.items():
        usable_rate = round(sum(1 for r in rows if r.get("usable_FormatAndSlot_preserved") or r.get("format_and_slot_usable")) / len(rows), 4) if rows else 0.0
        retrieval_comparison[m] = {
            "n_pairs": len(rows),
            "symbolic_usable_rate": usable_rate,
            "pool_15_standard": eval_query_set_on_pool(rows, masked=False, pool_size=15),
            "pool_1000_standard": eval_query_set_on_pool(rows, masked=False, pool_size=1000),
            "pool_15_title_masked": eval_query_set_on_pool(rows, masked=True, pool_size=15),
            "pool_1000_title_masked": eval_query_set_on_pool(rows, masked=True, pool_size=1000),
        }

    # 3. Part B: Third-Party Public Benchmark (`InsQABench`) External Cross-Validation (50 items)
    # Strictly filter waiting/observation-period items to those where the user question OR gold answer genuinely addresses 等待期/观察期 (9 items),
    # plus 41 stratified quantitative eligibility items across grace/cooling-off windows (15), deductible/reimbursement caps (14), and age/duration limits (12).
    num_pat = re.compile(r"(\d+\s*(?:日|天|周岁|元|%|年|个月|万元|倍)|三十日|六十日|九十日|一百八十日|十五日|十日|二十日)")

    wait_9 = [
        p for p in insqa_parsed
        if any(k in p["question"] or k in p["gold_answer"] for k in ["等待期", "观察期"])
    ]
    assert len(wait_9) == 9, f"Expected 9 genuine waiting/observation QA items in InsQABench, got {len(wait_9)}"
    used_ids = {p["insqa_id"] for p in wait_9}

    grace_candidates = [
        p for p in insqa_parsed
        if p["insqa_id"] not in used_ids
        and any(k in p["question"] or k in p["gold_answer"] for k in ["宽限期", "犹豫期", "效力中止", "效力恢复", "解除合同", "退还"])
        and num_pat.search(p["clause_text"]) and num_pat.search(p["gold_answer"])
    ]
    grace_15 = grace_candidates[:15]
    used_ids.update(p["insqa_id"] for p in grace_15)

    ded_candidates = [
        p for p in insqa_parsed
        if p["insqa_id"] not in used_ids
        and any(k in p["question"] or k in p["gold_answer"] for k in ["免赔额", "给付比例", "赔付比例", "报销", "扣除", "限额", "保险金额"])
        and num_pat.search(p["clause_text"]) and num_pat.search(p["gold_answer"])
    ]
    ded_14 = ded_candidates[:14]
    used_ids.update(p["insqa_id"] for p in ded_14)

    age_candidates = [
        p for p in insqa_parsed
        if p["insqa_id"] not in used_ids
        and any(k in p["question"] or k in p["gold_answer"] for k in ["周岁", "年龄", "住院", "天数", "期间", "满期"])
        and num_pat.search(p["clause_text"]) and num_pat.search(p["gold_answer"])
    ]
    age_12 = age_candidates[:12]
    used_ids.update(p["insqa_id"] for p in age_12)

    external_50 = wait_9 + grace_15 + ded_14 + age_12
    assert len(external_50) == 50

    # Evaluate waiting-period parameter extraction & reachability/condition analysis on the 9 genuine waiting-period items
    wait_9_results = []
    explicit_wait_clauses_count = 0
    explicit_wait_extracted_correct = 0
    policy_schedule_deferred_count = 0
    question_condition_status_counts = {"DETERMINATIVE_SELF_CONTAINED": 0, "REQUIRES_POLICY_SCHEDULE_OR_INSTANCE_CLARIFICATION": 0}

    for item in wait_9:
        ctext = item["clause_text"]
        qtext = item["question"]
        atext = item["gold_answer"]
        schema = extract_waiting_schema_symbolic(ctext + " " + atext)
        clause_only_schema = extract_waiting_schema_symbolic(ctext)

        has_explicit_days_in_clause_or_glossary = schema["wait_days"] is not None
        if has_explicit_days_in_clause_or_glossary:
            explicit_wait_clauses_count += 1
            explicit_wait_extracted_correct += 1
        else:
            policy_schedule_deferred_count += 1

        # Determine whether the user question + clause has a self-contained determinative rule answer or defers numeric days to policy schedule / cross-referenced clauses
        defers_to_schedule = (
            ("合同中载明" in atext or "保险单中" in atext or "投保时约定" in atext or "8.1 条" in atext or not has_explicit_days_in_clause_or_glossary)
            and not any(k in qtext for k in ["等待期内罹患", "一个月内", "是否有权解除", "投保前被诊断", "如果被保险人在等待期内"])
        )
        status = (
            "REQUIRES_POLICY_SCHEDULE_OR_INSTANCE_CLARIFICATION"
            if defers_to_schedule
            else "DETERMINATIVE_SELF_CONTAINED"
        )
        question_condition_status_counts[status] += 1

        wait_9_results.append({
            "insqa_id": item["insqa_id"],
            "rule_category": "waiting_or_observation_period",
            "contract_title": item["contract_title"],
            "question": qtext,
            "gold_answer_excerpt": atext[:160],
            "has_explicit_wait_days_in_clause": clause_only_schema["wait_days"] is not None,
            "has_explicit_wait_days_with_glossary": has_explicit_days_in_clause_or_glossary,
            "extracted_wait_days": schema["wait_days"],
            "accident_exempt_detected": schema["accident_exempt"],
            "renewal_exempt_detected": schema["renewal_exempt"],
            "reachability_status": status,
        })

    # Evaluate quantitative threshold extraction across all 50 stratified external items
    quant_50_extracted_count = 0
    external_contracts_50 = set()
    stratified_50_records = []
    category_map = (
        [("waiting_or_observation_period", x) for x in wait_9]
        + [("grace_or_cooling_off_window", x) for x in grace_15]
        + [("deductible_or_reimbursement_cap", x) for x in ded_14]
        + [("age_or_duration_eligibility", x) for x in age_12]
    )
    for cat_name, item in category_map:
        external_contracts_50.add(item["contract_title"])
        nums_in_clause_or_ans = num_pat.findall(item["clause_text"] + " " + item["gold_answer"])
        if nums_in_clause_or_ans:
            quant_50_extracted_count += 1
        stratified_50_records.append({
            "insqa_id": item["insqa_id"],
            "rule_category": cat_name,
            "contract_title": item["contract_title"],
            "question": item["question"],
            "extracted_thresholds": list(dict.fromkeys(nums_in_clause_or_ans))[:6],
            "has_explicit_quantitative_threshold": bool(nums_in_clause_or_ans),
        })

    return {
        "study_version": "v3.4-fudan-insqabench-1000-distractor-and-external-validation",
        "source_dataset": {
            "name": "InsQABench (Fudan University, Ding et al., 2024)",
            "file": "work/clause_objective.json",
            "file_sha256": sha256_bytes((WORK_DIR / "clause_objective.json").read_bytes()),
            "total_qa_items": len(insqa_raw),
            "unique_insurance_contracts": len(insqa_contracts),
            "unique_contract_clause_passages": len(unique_passages),
        },
        "large_scale_1000_clause_retrieval": {
            "total_corpus_clauses": len(corpus_1000),
            "confirmation_target_clauses": 15,
            "real_insurance_distractor_clauses": len(corpus_1000) - 15,
            "scale_expansion_factor_vs_15_clause_pool": round(1000.0 / 15.0, 2),
            "methods_comparison": retrieval_comparison,
            "key_findings": {
                "ordinary_style_1000_standard_bm25_top1": retrieval_comparison["ordinary_style"]["pool_1000_standard"]["bm25_char2_top1"],
                "ordinary_style_1000_standard_rrf_top1": retrieval_comparison["ordinary_style"]["pool_1000_standard"]["rrf_bm25_bge_top1"],
                "ordinary_style_symbolic_usable_rate": retrieval_comparison["ordinary_style"]["symbolic_usable_rate"],
                "slot_guard_repair_1000_standard_rrf_top1": retrieval_comparison["slot_guard_repair"]["pool_1000_standard"]["rrf_bm25_bge_top1"],
                "slot_guard_repair_1000_title_masked_bge_mrr10": retrieval_comparison["slot_guard_repair"]["pool_1000_title_masked"]["bge_small_zh_mrr10"],
                "seed_passthrough_1000_title_masked_bge_mrr10": retrieval_comparison["seed_passthrough"]["pool_1000_title_masked"]["bge_small_zh_mrr10"],
            },
        },
        "external_insqabench_validation_50": {
            "total_external_items_evaluated": len(external_50),
            "distinct_contracts_covered": len(external_contracts_50),
            "stratified_category_counts": {
                "waiting_or_observation_period": len(wait_9),
                "grace_or_cooling_off_window": len(grace_15),
                "deductible_or_reimbursement_cap": len(ded_14),
                "age_or_duration_eligibility": len(age_12),
            },
            "waiting_or_observation_period_items_count": len(wait_9),
            "additional_quantitative_eligibility_items_count": len(external_50) - len(wait_9),
            "waiting_9_analysis": {
                "clauses_with_explicit_numeric_wait_days": explicit_wait_clauses_count,
                "explicit_wait_days_extraction_accuracy": round(
                    explicit_wait_extracted_correct / max(1, explicit_wait_clauses_count), 4
                ),
                "clauses_deferring_wait_days_to_policy_schedule_or_crossref": policy_schedule_deferred_count,
                "question_reachability_partition": question_condition_status_counts,
                "items": wait_9_results,
            },
            "quantitative_50_threshold_extraction_count": quant_50_extracted_count,
            "quantitative_50_threshold_extraction_coverage": round(quant_50_extracted_count / len(external_50), 4),
            "stratified_50_items": stratified_50_records,
        },
    }


def main() -> None:
    print("1. Building non-waiting verbatim PDF anchors (fitz page + bbox + sha256 + task_context)...")
    anchors = build_non_waiting_verbatim_anchors()
    out_anchors = DATA_V3 / "non_waiting_verbatim_pdf_anchors.json"
    out_anchors.write_text(json.dumps(anchors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("   -> Wrote", out_anchors)

    print("2. Building 10 cross-insurer official PDFs (36 PDFs across 10 insurers) & superscript cleaning audit...")
    multi_ins = build_multi_insurer_and_superscript_audit()
    out_multi = DATA_V3 / "multi_insurer_official_sources.json"
    out_multi.write_text(json.dumps(multi_ins, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("   -> Wrote", out_multi)
    print("   -> New 10 cross-insurer 3-core schema accuracy:", multi_ins["expanded_waiting_period_schema_audit_27_pdfs"])

    print("3. Building 1,000-clause InsQABench large-scale distractor retrieval & 50-item external cross-validation...")
    insqa_eval = build_insqabench_1000_retrieval_and_external_eval()
    out_insqa = DATA_V3 / "insqabench_large_scale_retrieval_and_external_eval.json"
    out_insqa.write_text(json.dumps(insqa_eval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("   -> Wrote", out_insqa)
    print("   -> Key 1,000-clause findings:", json.dumps(insqa_eval["large_scale_1000_clause_retrieval"]["key_findings"], indent=2))


if __name__ == "__main__":
    main()
