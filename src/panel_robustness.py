"""Panel-robustness and clean-subset sensitivity analyses for the insurance synthetic-query study.

Uses only the released logs in data_v3/ plus the repository's own rule functions
(no API calls). Five analyses:

  A. Reproduce the reported single-arm RAG scores (naive vs. panel truth) from raw logs.
  B. Panel accuracy against construction-derived gold labels (fully specified template
     questions, whose correct label is fixed by the compiled clause rules).
  C. Leave-one-judge-out and judge--reader self-preference audit: recompute panel truth
     with each judge removed and measure agreement between each judge and its matched reader.
  D. Robustness bounds: how much panel error is needed to overturn the ranking-reversal
     and zero-call-relabel conclusions.
  E. Clean-subset sensitivity (Clean-78 / Clean-75 / Clean-34): recompute pooled usability
     and single-arm evaluation after excluding template-product mismatch and cross-clause
     boundary seeds identified in Analysis B.

Usage: python3 -I src/panel_robustness.py <repo_root> <out_json>
"""
import collections, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stubload  # noqa: E402

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "data_v3" / "panel_robustness_results.json"
D = REPO / "data_v3"
m = stubload.load(str(REPO))
import frozen_confirmation_and_rag_study as fc  # noqa: E402

JUDGES = ["deepseek_v4_pro", "kimi_k3", "qwen3_8_max"]
PIPES = [p for p, _, _ in m.REAL_LLM_PIPELINES]
ABST6 = list(m.ABSTAINING_6_PIPELINES)
QWEN_READER_PIPES = ["L1_bge_qwen38max_abstaining", "L4_bm25_qwen38max_abstaining", "L9_bge_qwen38max_forced_binary"]


def jl(name):
    return [json.loads(l) for l in open(D / name, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------- seeds / gold
raw_dir = Path("/nonexistent")  # builders fall back to cached sources in data_v3
_, _, ins_seeds, cross_seeds, _ = fc.build_frozen_confirmation_sources_and_cases(raw_dir, D)
_, _, u_ins, u_cross = fc.build_untouched_holdout_sources_and_seeds(raw_dir, D)
nw_seeds, _ = m.build_non_waiting_seeds_list()
SEEDS = {}
for s in ins_seeds + u_ins:
    SEEDS[s["seed_id"]] = ("insurance", s)
for s in cross_seeds + u_cross:
    SEEDS[s["seed_id"]] = ("multi_threshold", s)
for s in nw_seeds:
    SEEDS[s["seed_id"]] = ("non_waiting", s)
GOLD = {sid: s["reference_decision"] for sid, (_, s) in SEEDS.items()}


def dom_of(sid):
    return "insurance" if SEEDS[sid][0] == "insurance" else "multi_threshold"


# ---------------------------------------------------------------- panel
def load_panel(fname):
    rec = {}
    for r in jl(fname):
        rec[(r["seed_id"], r["judge_id"])] = r["judgments_by_question"]
    return rec


PANEL = {"protocol_guided": load_panel("ccf_a_raw_clause_adjudications.jsonl"),
         "pure_verbatim": load_panel("ccf_a_pure_verbatim_adjudications.jsonl")}


def judge_label(rec, sid, q, j):
    ji = rec.get((sid, j), {}).get(q)
    if ji is None:
        return None
    if not ji["policy_identified"] or not ji["sentence_complete"] or ji["decision"] == "CLARIFY_REQUIRED":
        return "CLARIFY"
    return str(ji["decision"]).strip()


def panel_label(rec, sid, q, judges=JUDGES, tie="CLARIFY"):
    """Majority over the given judges; ties (1-1-1, or 1-1 with two judges) -> `tie`.
    With all three judges this reproduces resolve_panel_majority_label in the repository."""
    labs = [judge_label(rec, sid, q, j) for j in judges]
    if any(l is None for l in labs):
        raise KeyError((sid, q))
    c = collections.Counter(labs)
    need = len(judges) // 2 + 1
    for lbl, n in sorted(c.items(), key=lambda x: (-x[1], x[0])):
        if n >= need:
            return lbl
    return tie


# ---------------------------------------------------------------- reader cache
RAG = {}
for r in jl("ccf_a_real_llm_rag_responses.jsonl"):
    RAG[(r["reader_cfg_id"], r["retrieved_clause_id"], r["question"].strip())] = r["parsed_label"]
PROBE_CFG = {"bge_small_zh": "deepseek_v41_forced_binary", "bm25_char2": "qwenplus_forced_binary",
             "rrf_bm25_bge": "deepseek_v4pro_abstaining"}
CLAUSE_BY = collections.defaultdict(dict)
for (cfg, cid, q) in RAG:
    for ret, pc in PROBE_CFG.items():
        if cfg == pc:
            CLAUSE_BY[ret][q] = cid


def predict(pipe, q, dom):
    _, ret, cfg = next(x for x in m.REAL_LLM_PIPELINES if x[0] == pipe)
    q = q.strip()
    excl = "WAIT_EXCLUDED" if dom == "insurance" else "RULE_EXCLUDED"
    elig = "WAIT_NOT_EXCLUDED" if dom == "insurance" else "RULE_ELIGIBLE"
    cid = CLAUSE_BY[ret].get(q)
    if cid is None:
        return "CLARIFY" if "abstaining" in cfg else excl
    lab = RAG[(cfg, cid, q)]
    if "forced_binary" in cfg and lab not in (excl, elig):
        return excl
    return lab


# ---------------------------------------------------------------- generated rows
GEN = {"qwen_plus": jl("frozen_confirmation_generations_qwen_plus.jsonl"),
       "deepseek": jl("frozen_confirmation_generations_deepseek.jsonl")}
for r in jl("fair_structured_baseline_generations.jsonl"):
    GEN[r["model_family"]].append(r)
for fam in GEN:
    for r in GEN[fam]:
        r["format_and_slot_usable"] = bool(r.get("usable_FormatAndSlot_preserved", False))
METHODS = ["seed_passthrough", "ordinary_style", "condition_prompt", "structured_json_rewrite",
           "structured_json_untruncated_320", "verify_only_filter", "slot_guard_repair"]
CONF_SEEDS = {s["seed_id"] for s in ins_seeds + cross_seeds}


def rows_for(fam, method):
    if method == "mixed_ordinary_and_prompt_80":
        rows = [r for r in GEN[fam] if r["method"] in ("ordinary_style", "condition_prompt")]
    else:
        rows = [r for r in GEN[fam] if r["method"] == method]
    rows = [r for r in rows if r["seed_id"] in CONF_SEEDS]
    if method in ("verify_only_filter", "slot_guard_repair"):
        rows = [r for r in rows if r["format_and_slot_usable"]]
    return rows


def relabel(r):
    if r["format_and_slot_usable"]:
        return GOLD[r["seed_id"]]
    pu = (r.get("parse_uncertainty") or {}).get("is_parse_uncertain", False)
    if r.get("format_valid") and r.get("outcome") == "DEGRADED_TO_CLARIFY" and not pu:
        return "CLARIFY"
    return None


def tau(a, b):
    """Kendall tau-b (tie-corrected), identical to scipy.stats.kendalltau's default;
    returns 0.0 when undefined, matching the repository's NaN handling."""
    conc = disc = tx = ty = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = a[i] - a[j], b[i] - b[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    den = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return 0.0 if den == 0 else (conc - disc) / den


def evaluate(rows, truth_fn, label_fn=lambda r: GOLD[r["seed_id"]], pipes=PIPES):
    """Score every pipeline against benchmark labels (label_fn) and against panel truth
    (truth_fn), returning MAE and Kendall tau between the two score vectors."""
    rows = [r for r in rows if label_fn(r) is not None]
    bench = {p: 0 for p in pipes}
    true = {p: 0 for p in pipes}
    for r in rows:
        dom = dom_of(r["seed_id"])
        t = truth_fn(r)
        lab = label_fn(r)
        for p in pipes:
            pr = predict(p, r["question"], dom)
            bench[p] += pr == lab
            true[p] += pr == t
    n = max(1, len(rows))
    bv = [bench[p] / n for p in pipes]
    tv = [true[p] / n for p in pipes]
    return {"n": len(rows), "bench_scores": dict(zip(pipes, [round(x, 4) for x in bv])),
            "true_scores": dict(zip(pipes, [round(x, 4) for x in tv])),
            "mae": round(sum(abs(x - y) for x, y in zip(bv, tv)) / len(pipes), 4),
            "tau": round(tau(tv, bv), 4)}


results = {}

# ================================================================ A. reproduction
summ = json.load(open(D / "ccf_a_upgrade_summary.json", encoding="utf-8"))
rep = summ["experiment_3_real_llm_rag_downstream_selection"]["by_generator_family"]
repro = {}
rec3 = PANEL["protocol_guided"]
for fam in ("qwen_plus", "deepseek"):
    for meth in ("seed_passthrough", "ordinary_style", "condition_prompt", "slot_guard_repair",
                 "verify_only_filter", "structured_json_untruncated_320", "mixed_ordinary_and_prompt_80"):
        rows = rows_for(fam, meth)
        ev = evaluate(rows, lambda r: panel_label(rec3, r["seed_id"], r["question"]))
        reported = rep[fam][meth]
        ok_scores = all(abs(ev["bench_scores"][p] - reported["naive_pipeline_scores"][p]) < 1e-3 and
                        abs(ev["true_scores"][p] - reported["raw_clause_panel_true_pipeline_scores"][p]) < 1e-3
                        for p in PIPES)
        repro[f"{fam}/{meth}"] = {"mae": ev["mae"], "reported_mae": reported["mae_vs_raw_clause_panel_truth"],
                                  "tau": ev["tau"], "reported_tau": reported["kendall_tau_vs_panel_truth"],
                                  "all_18_scores_match": ok_scores, "n": ev["n"]}
results["A_reproduction"] = repro

# ================================================================ B. constructed gold
tmpl = {}
for fn in ("frozen_confirmation_generations_qwen_plus.jsonl", "frozen_confirmation_generations_deepseek.jsonl",
           "untouched_holdout_generations.jsonl", "ccf_a_non_waiting_multi_rule_generations.jsonl"):
    for r in jl(fn):
        if r["method"] == "seed_passthrough":
            tmpl[(r["seed_id"], r["question"])] = r
gold_res = {}
disagreements = []
for proto, rec in PANEL.items():
    items = [(sid, q) for (sid, q) in tmpl if all(judge_label(rec, sid, q, j) is not None for j in JUDGES)]
    per = {}
    for name, fn in [(j, (lambda jj: lambda s, q: judge_label(rec, s, q, jj))(j)) for j in JUDGES] + \
                    [("majority_3", lambda s, q: panel_label(rec, s, q))]:
        conf = collections.Counter()
        by_set = collections.defaultdict(lambda: [0, 0])
        for sid, q in items:
            pred = fn(sid, q)
            g = GOLD[sid]
            kind = "correct" if pred == g else ("said_clarify" if pred == "CLARIFY" else "flipped_decision")
            conf[kind] += 1
            st = SEEDS[sid][0] if not sid.startswith("uhold") else "post_hardening_" + SEEDS[sid][0]
            by_set[st][0] += pred == g
            by_set[st][1] += 1
            if name == "majority_3" and pred != g and proto == "protocol_guided":
                disagreements.append({"seed_id": sid, "gold": g, "panel": pred,
                                      "judges": {j: judge_label(rec, sid, q, j) for j in JUDGES}, "question": q})
        n = len(items)
        per[name] = {"n": n, "accuracy": round(conf["correct"] / n, 4), "said_clarify": conf["said_clarify"],
                     "flipped_decision": conf["flipped_decision"],
                     "by_set": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_set.items())}}
    gold_res[proto] = per
results["B_constructed_gold"] = gold_res
results["B_majority_disagreements_protocol_guided"] = disagreements

# ================================================================ C. leave-one-judge-out & self-preference
variants = {"all_3": (JUDGES, "CLARIFY")}
for j in JUDGES:
    rest = [x for x in JUDGES if x != j]
    variants[f"drop_{j}"] = (rest, "CLARIFY")
    variants[f"drop_{j}_tie_drop"] = (rest, None)
loo = {}
for vname, (judges, tie) in variants.items():
    out = {}
    for fam in ("qwen_plus", "deepseek"):
        for meth in ("ordinary_style", "condition_prompt", "slot_guard_repair", "mixed_ordinary_and_prompt_80"):
            rows = rows_for(fam, meth)
            truth = {}
            for r in rows:
                truth[(r["seed_id"], r["question"])] = panel_label(rec3, r["seed_id"], r["question"], judges, tie)
            rows_k = [r for r in rows if truth[(r["seed_id"], r["question"])] is not None]
            tf = lambda r: truth[(r["seed_id"], r["question"])]
            naive9 = evaluate(rows_k, tf)
            naive6 = evaluate(rows_k, tf, pipes=ABST6)
            rel9 = evaluate(rows_k, tf, label_fn=relabel)
            rel6 = evaluate(rows_k, tf, label_fn=relabel, pipes=ABST6)
            qwen_adv = round(sum(naive9["true_scores"][p] for p in QWEN_READER_PIPES) / 3, 4)
            out[f"{fam}/{meth}"] = {"n_eval": naive9["n"],
                                    "naive_all9": {"mae": naive9["mae"], "tau": naive9["tau"]},
                                    "naive_abst6": {"mae": naive6["mae"], "tau": naive6["tau"]},
                                    "relabel_all9": {"n": rel9["n"], "mae": rel9["mae"], "tau": rel9["tau"]},
                                    "relabel_abst6": {"mae": rel6["mae"], "tau": rel6["tau"]},
                                    "true_scores": naive9["true_scores"],
                                    "mean_true_score_qwen_reader_pipes": qwen_adv}
    loo[vname] = out
results["C_leave_one_judge_out"] = loo

# C2: Judge--reader self-preference audit across the 294 unique evaluated questions
seen_q = {}
for fam in ("qwen_plus", "deepseek"):
    for meth in METHODS:
        for r in rows_for(fam, meth):
            seen_q[(r["seed_id"], r["question"])] = dom_of(r["seed_id"])
self_pref = {}
for p in PIPES:
    agrees = {}
    for j in JUDGES:
        agrees[j] = round(sum(predict(p, q, dom) == judge_label(rec3, sid, q, j)
                              for (sid, q), dom in seen_q.items()) / len(seen_q), 4)
    self_pref[p] = agrees
results["C2_judge_reader_agreement_294q"] = self_pref

# ================================================================ D. robustness bounds
rng = random.Random(20261028)
REPS = 1000


def labels_for(dom):
    return (["WAIT_EXCLUDED", "WAIT_NOT_EXCLUDED", "CLARIFY"] if dom == "insurance"
            else ["RULE_EXCLUDED", "RULE_ELIGIBLE", "CLARIFY"])


def precompute(rows):
    return [(r, dom_of(r["seed_id"]), panel_label(rec3, r["seed_id"], r["question"]), relabel(r),
             {p: predict(p, r["question"], dom_of(r["seed_id"])) for p in PIPES}) for r in rows]


def fast_eval(pre, truth, use_relabel, pipes):
    bench = {p: 0 for p in pipes}
    true = {p: 0 for p in pipes}
    n = 0
    for (r, dom, _, rl, preds), t in zip(pre, truth):
        lab = rl if use_relabel else GOLD[r["seed_id"]]
        if lab is None:
            continue
        n += 1
        for p in pipes:
            bench[p] += preds[p] == lab
            true[p] += preds[p] == t
    bv = [bench[p] / n for p in pipes]
    tv = [true[p] / n for p in pipes]
    return sum(abs(x - y) for x, y in zip(bv, tv)) / len(pipes), tau(tv, bv)


rob = {}
for fam in ("qwen_plus", "deepseek"):
    pre = precompute(rows_for(fam, "ordinary_style"))
    base_truth = [x[2] for x in pre]
    res_fam = {"random_label_noise": {}, "adversarial_clarify_to_seed_label": {}}
    for e in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        stats = collections.defaultdict(list)
        for _ in range(REPS):
            truth = [rng.choice([l for l in labels_for(dom) if l != t]) if rng.random() < e else t
                     for (_, dom, _, _, _), t in zip(pre, base_truth)]
            for tag, rl, pipes in (("naive9", False, PIPES), ("naive6", False, ABST6),
                                   ("relabel9", True, PIPES), ("relabel6", True, ABST6)):
                mae, tv = fast_eval(pre, truth, rl, pipes)
                stats[tag + "_mae"].append(mae)
                stats[tag + "_tau"].append(tv)
        summ_e = {}
        for k, v in stats.items():
            v = sorted(v)
            summ_e[k] = {"median": round(v[len(v) // 2], 4), "p05": round(v[int(0.05 * len(v))], 4),
                         "p95": round(v[int(0.95 * len(v))], 4)}
        summ_e["P(naive9_tau<0)"] = round(sum(t < 0 for t in stats["naive9_tau"]) / REPS, 3)
        summ_e["P(relabel9_mae<naive9_mae)"] = round(
            sum(a < b for a, b in zip(stats["relabel9_mae"], stats["naive9_mae"])) / REPS, 3)
        res_fam["random_label_noise"][str(e)] = summ_e
    clar_idx = [i for i, t in enumerate(base_truth) if t == "CLARIFY"]
    res_fam["n_panel_clarify"] = len(clar_idx)
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k = round(frac * len(clar_idx))
        taus, maes, rmaes = [], [], []
        for _ in range(REPS if 0 < k < len(clar_idx) else 1):
            flip = set(rng.sample(clar_idx, k))
            truth = [GOLD[pre[i][0]["seed_id"]] if i in flip else t for i, t in enumerate(base_truth)]
            mae, tv = fast_eval(pre, truth, False, PIPES)
            rmae, _ = fast_eval(pre, truth, True, PIPES)
            taus.append(tv); maes.append(mae); rmaes.append(rmae)
        taus.sort(); maes.sort(); rmaes.sort()
        res_fam["adversarial_clarify_to_seed_label"][str(frac)] = {
            "k_flipped": k, "naive9_tau_median": round(taus[len(taus) // 2], 4),
            "P(naive9_tau<0)": round(sum(t < 0 for t in taus) / len(taus), 3),
            "naive9_mae_median": round(maes[len(maes) // 2], 4),
            "relabel9_mae_median": round(rmaes[len(rmaes) // 2], 4)}
    rob[fam] = res_fam
results["D_robustness"] = rob

# ================================================================ E. clean-subset sensitivity
all_pooled_rows = (
    jl("frozen_confirmation_generations_qwen_plus.jsonl")
    + jl("frozen_confirmation_generations_deepseek.jsonl")
    + jl("untouched_holdout_generations.jsonl")
    + jl("ccf_a_non_waiting_multi_rule_generations.jsonl")
)
mismatch_4 = {"fconf-ins-010", "uhold-ins-004", "uhold-ins-005", "uhold-ins-006"}
cross_6 = {"fconf-ins-017", "fconf-ins-018", "mrule-013", "mrule-018", "mrule-019", "mrule-024"}
waiver_3 = {"fconf-cross-001", "fconf-cross-007", "fconf-cross-011"}
excl_10 = mismatch_4 | cross_6
excl_13 = mismatch_4 | cross_6 | waiver_3

clean_pooled = {}
for subset_name, excl_set in [("all_88_seeds_176_pairs", set()),
                              ("clean_78_seeds_156_pairs_excl_mismatch_and_cross", excl_10),
                              ("clean_75_seeds_150_pairs_excl_all_13", excl_13)]:
    sub_res = {}
    for meth in ["seed_passthrough", "ordinary_style", "condition_prompt", "verify_only_filter", "slot_guard_repair"]:
        sub = [r for r in all_pooled_rows if r["method"] == meth and r["seed_id"] not in excl_set]
        u_strict = sum(1 for r in sub if r.get("usable_FormatAndSlot_preserved", False) or r.get("format_and_slot_usable", False))
        u_panel = sum(1 for r in sub if r.get("format_valid", True) and panel_label(rec3, r["seed_id"], r["question"]) == GOLD[r["seed_id"]])
        sub_res[meth] = {
            "n_pairs": len(sub),
            "strict_symbolic_usable": u_strict,
            "strict_symbolic_rate": round(u_strict / len(sub), 4),
            "panel_usable": u_panel,
            "panel_rate": round(u_panel / len(sub), 4),
        }
    clean_pooled[subset_name] = sub_res

excl_conf_6 = {"fconf-ins-010", "fconf-ins-017", "fconf-ins-018", "fconf-cross-001", "fconf-cross-007", "fconf-cross-011"}
clean_conf_34 = {}
for fam in ("qwen_plus", "deepseek"):
    for meth in ("seed_passthrough", "ordinary_style", "condition_prompt", "slot_guard_repair", "mixed_ordinary_and_prompt_80"):
        rows = [r for r in rows_for(fam, meth) if r["seed_id"] not in excl_conf_6]
        ev_n = evaluate(rows, lambda r: panel_label(rec3, r["seed_id"], r["question"]))
        ev_r = evaluate(rows, lambda r: panel_label(rec3, r["seed_id"], r["question"]), label_fn=relabel)
        clean_conf_34[f"{fam}/{meth}"] = {
            "n": ev_n["n"],
            "naive_mae": ev_n["mae"],
            "naive_tau": ev_n["tau"],
            "relabel_mae": ev_r["mae"],
            "relabel_tau": ev_r["tau"],
        }

results["E_clean_subset_sensitivity"] = {
    "pooled_usability_by_subset": clean_pooled,
    "confirmation_clean_34_seeds_rag_eval": clean_conf_34,
}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("written", OUT)
