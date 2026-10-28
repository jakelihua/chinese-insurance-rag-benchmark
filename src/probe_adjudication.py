"""Panel recall on construction-certified CLARIFY probes (missing_critical cases).

Closes the gap left by the panel-validity checks: the three-model panel had only judged
fully specified questions, so its ability to recognise an underspecified question was
never measured. For each of the 40 confirmation seeds, the missing_critical probe (one
pivotal slot removed, certified by two opposite-outcome completion witnesses) is judged

  A. "isolated": the probe alone in its batch (no contrast cue), and
  B. "with_controls": the probe shuffled with the seed's complete question (gold = seed
     decision) and counterfactual question (gold = flipped decision),

by all three judges under both protocols (protocol-guided and pure-verbatim), using the
repository's own prompt builder and API wrapper unchanged (temperature 0).
40 seeds x 3 judges x 2 protocols x 2 conditions = 480 calls.

Run (from the repository root):
    DASHSCOPE_API_KEY=<your key> python3 -I src/probe_adjudication.py
Re-running resumes from the cache and only issues missing calls.
    python3 -I src/probe_adjudication.py --score-only     # rescore cached results, no API calls
    python3 -I src/probe_adjudication.py --dry-run        # fake responses, for testing the pipeline

Outputs: data_v3/missing_probe_adjudications.jsonl, data_v3/missing_probe_summary.json
"""
import argparse, collections, hashlib, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import stubload  # noqa: E402  (stubs only PyMuPDF/torch/transformers/sklearn, unused here)

m = stubload.load(str(REPO))
import frozen_confirmation_and_rag_study as fc  # noqa: E402

D = REPO / "data_v3"
CACHE = D / "missing_probe_adjudications.jsonl"
SUMMARY = D / "missing_probe_summary.json"
JUDGES = m.ADJUDICATOR_MODELS            # [(judge_id, model_name), ...]
PROTOCOLS = {"protocol_guided": False, "pure_verbatim": True}
CONDITIONS = ("isolated", "with_controls")


def load_cases():
    _, _, ins, cross, _ = fc.build_frozen_confirmation_sources_and_cases(Path("/nonexistent"), D)
    seeds = {s["seed_id"]: ("insurance", s) for s in ins}
    seeds.update({s["seed_id"]: ("multi_threshold", s) for s in cross})
    sources = {s["id"]: s for s in json.loads((D / "frozen_confirmation_sources.json").read_text(encoding="utf-8"))}
    cases = collections.defaultdict(dict)
    for line in (D / "frozen_confirmation_cases.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            cases[c["group_id"]][c["variant"]] = c
    return seeds, sources, cases


def clause_for(dom, seed, sources):
    if dom == "insurance":
        src = sources[seed["policy_id"]]
        return src["title"], src["evidence_text"]
    cl = m.CROSS_CLAUSES[seed["clause_id"]]
    return cl["title"], cl["clause_text"]


def build_tasks(seeds, sources, cases):
    tasks = []
    for sid in sorted(seeds):
        dom, seed = seeds[sid]
        cv = cases[sid]
        title, text = clause_for(dom, seed, sources)
        items = {"isolated": [("missing_critical", cv["missing_critical"]["question"])],
                 "with_controls": [(v, cv[v]["question"]) for v in ("missing_critical", "complete", "counterfactual")]}
        for cond in CONDITIONS:
            lst = list(items[cond])
            rng = random.Random(int(hashlib.sha256(f"{sid}|{cond}".encode()).hexdigest()[:8], 16))
            rng.shuffle(lst)  # deterministic order that never leaks the variant
            anon = [(f"Q{i + 1}", q) for i, (_, q) in enumerate(lst)]
            variant_of = {f"Q{i + 1}": v for i, (v, _) in enumerate(lst)}
            for proto, pure in PROTOCOLS.items():
                for jid, model in JUDGES:
                    tasks.append({"key": f"{sid}|{cond}|{proto}|{jid}", "seed_id": sid, "domain_type": dom,
                                  "condition": cond, "protocol": proto, "judge_id": jid, "judge_model": model,
                                  "anon": anon, "variant_of": variant_of,
                                  "prompt": m._build_seed_adjudication_prompt(seed, dom, title, text, anon, pure_verbatim=pure)})
    return tasks


def fake_call(model_name, messages, api_key, **kw):
    import re
    ids = re.findall(r"^(Q\d+): ", messages[0]["content"], flags=re.M)
    arr = [{"id": q, "policy_identified": True, "sentence_complete": True, "decision": "CLARIFY_REQUIRED", "naturalness": 3} for q in ids]
    return {"content": json.dumps(arr), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "latency_ms": 0.0, "api_meta": {"provider": "dry_run", "requested_model": model_name}}


def run(tasks, api_key, dry):
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok") and r.get("dry_run", False) == dry:
                    done[r["key"]] = r
    todo = [t for t in tasks if t["key"] not in done]
    print(f"{len(tasks)} calls in total, {len(done)} cached, {len(todo)} to run", flush=True)
    call = fake_call if dry else m.call_dashscope_chat
    lock = threading.Lock()

    def work(t):
        try:
            res = call(model_name=t["judge_model"], messages=[{"role": "user", "content": t["prompt"]}],
                       api_key=api_key, max_tokens=1200, temperature=0.0, disable_thinking=True)
            by_id = {x.get("id"): x for x in m.parse_json_from_llm(res["content"]) if isinstance(x, dict)}
            judg = {}
            for qid, q in t["anon"]:
                j = by_id.get(qid) or {}
                judg[qid] = {"variant": t["variant_of"][qid], "question": q,
                             "policy_identified": bool(j.get("policy_identified", False)),
                             "sentence_complete": bool(j.get("sentence_complete", True)),
                             "decision": str(j.get("decision") or "CLARIFY_REQUIRED").strip(),
                             "parsed": qid in by_id}
            rec = {k: t[k] for k in ("key", "seed_id", "domain_type", "condition", "protocol", "judge_id", "judge_model")}
            rec.update({"ok": True, "dry_run": dry, "judgments": judg, "raw_content": res["content"],
                        "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                        "latency_ms": res["latency_ms"], "api_meta": res["api_meta"]})
        except Exception as e:  # keep going; a re-run retries failed calls
            rec = {"key": t["key"], "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        with lock:
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, t) for t in todo]), 1):
            r = fut.result()
            if r["ok"]:
                n_ok += 1; done[r["key"]] = r
            else:
                n_fail += 1; print("  failed:", r["key"], r["error"], flush=True)
            if i % 40 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done ({n_fail} failed)", flush=True)
    return done


def label(j):
    if not j["policy_identified"] or not j["sentence_complete"] or j["decision"] == "CLARIFY_REQUIRED":
        return "CLARIFY"
    return j["decision"]


def majority(labs):
    c = collections.Counter(labs)
    for lbl, n in sorted(c.items(), key=lambda x: (-x[1], x[0])):
        if n >= 2:
            return lbl
    return "CLARIFY"  # 1-1-1 split, as in the repository's resolver


def score(tasks, done, cases):
    gold = {}
    for sid, cv in cases.items():
        for v in ("missing_critical", "complete", "counterfactual"):
            if v in cv:
                gold[(sid, v)] = "CLARIFY" if v == "missing_critical" else cv[v]["reference_label"]
    out = {"n_calls_expected": len(tasks), "n_calls_completed": len(done),
           "missing_calls": sorted(t["key"] for t in tasks if t["key"] not in done)}
    for cond in CONDITIONS:
        for proto in PROTOCOLS:
            per = collections.defaultdict(lambda: collections.Counter())
            unparsed = 0
            by_item = collections.defaultdict(dict)
            for t in tasks:
                if t["condition"] != cond or t["protocol"] != proto or t["key"] not in done:
                    continue
                for qid, j in done[t["key"]]["judgments"].items():
                    unparsed += not j["parsed"]
                    by_item[(t["seed_id"], j["variant"])][t["judge_id"]] = label(j)
            res = {}
            judge_names = [jid for jid, _ in JUDGES] + ["majority_3"]
            for name in judge_names:
                stats = collections.defaultdict(lambda: [0, 0])
                for (sid, v), labs in by_item.items():
                    if name == "majority_3":
                        if len(labs) < len(JUDGES):
                            continue
                        pred = majority([labs[jid] for jid, _ in JUDGES])
                    elif name in labs:
                        pred = labs[name]
                    else:
                        continue
                    g = gold[(sid, v)]
                    stats[v][0] += pred == g
                    stats[v][1] += 1
                    if v != "missing_critical" and pred not in (g, "CLARIFY"):
                        stats[v + "_flipped"][0] += 1
                res[name] = {k: f"{a}/{b}" if not k.endswith("_flipped") else a for k, (a, b) in sorted(stats.items())}
            missed = sorted(sid for (sid, v), labs in by_item.items()
                            if v == "missing_critical" and len(labs) == len(JUDGES)
                            and majority([labs[jid] for jid, _ in JUDGES]) != "CLARIFY")
            out[f"{cond}/{proto}"] = {"by_judge": res, "probes_missed_by_majority": missed, "unparsed_items": unparsed}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global CACHE
    if a.dry_run:
        import tempfile
        CACHE = Path(tempfile.gettempdir()) / "missing_probe_dryrun.jsonl"
        CACHE.unlink(missing_ok=True)
    seeds, sources, cases = load_cases()
    tasks = build_tasks(seeds, sources, cases)
    if a.score_only:
        done = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("ok") and not r.get("dry_run", False):
                        done[r["key"]] = r
    else:
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not key and not a.dry_run:
            sys.exit("Set DASHSCOPE_API_KEY first (or use --dry-run / --score-only).")
        done = run(tasks, key, a.dry_run)
    summ = score(tasks, done, cases)
    if a.dry_run:
        print(json.dumps(summ, ensure_ascii=False, indent=1)[:1500])
        return
    SUMMARY.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {SUMMARY}")
    for k, v in summ.items():
        if isinstance(v, dict) and "by_judge" in v:
            mj = v["by_judge"].get("majority_3", {})
            print(f"{k}: majority probe recall {mj.get('missing_critical')}, complete {mj.get('complete', '-')}, "
                  f"counterfactual {mj.get('counterfactual', '-')}")


if __name__ == "__main__":
    main()
