import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from condition_data import decision, pivotal_change, seed_frames
from condition_guard import extract_fields, chinese_number
from analyze_v3 import (
    extract_fields_surface_normalized,
    evaluate_rule_outcome_under_observed,
    evaluate_slot_completeness,
)


class ConditionTests(unittest.TestCase):
    def test_different_policies_have_different_day45_result(self):
        frame=dict(enrollment='first',cause='illness',policy_day=45,approved_waiver=False)
        self.assertEqual(decision(dict(frame,policy_id='pa_p0781',plan_wait_days=90)),'WAIT_EXCLUDED')
        self.assertEqual(decision(dict(frame,policy_id='pa_h865',plan_wait_days=30)),'WAIT_NOT_EXCLUDED')

    def test_plan_and_approved_waiver_change_result(self):
        frame=dict(policy_id='cpic_86017',enrollment='new',cause='illness',policy_day=10,
                   approved_waiver=False,plan_wait_days=25)
        self.assertEqual(decision(frame),'WAIT_EXCLUDED')
        self.assertEqual(decision(dict(frame,plan_wait_days=7)),'WAIT_NOT_EXCLUDED')
        self.assertEqual(decision(dict(frame,approved_waiver=True)),'WAIT_NOT_EXCLUDED')

    def test_every_missing_case_has_two_disagreeing_completions(self):
        frames=seed_frames();self.assertEqual(len(frames),20)
        for frame in frames:
            field,change=pivotal_change(frame)
            self.assertNotEqual(frame[field],change[field])
            self.assertNotEqual(decision(frame),decision(change))
            self.assertEqual({k:v for k,v in frame.items() if k!=field},
                             {k:v for k,v in change.items() if k!=field})

    def test_non_pivotal_missing_cause_not_assumed_ambiguous(self):
        frame=dict(policy_id='pa_h865',enrollment='first',cause='illness',policy_day=45,
                   plan_wait_days=30,approved_waiver=False)
        self.assertEqual(decision(frame),decision(dict(frame,cause='accident')))
        self.assertEqual(pivotal_change(frame)[0],'policy_day')

    def test_first_diagnosis_is_not_first_enrollment(self):
        self.assertIsNone(extract_fields('保单生效第20天首次确诊疾病，是否受等待期影响？')['enrollment'])
        self.assertEqual(extract_fields('首次投保，生效第20天因病住院？')['enrollment'],'first')

    def test_cause_negation_and_time_units(self):
        self.assertEqual(extract_fields('并非意外造成，保单生效第45天因病治疗？')['cause'],'illness')
        self.assertEqual(extract_fields('因意外受伤，保单生效第20天？')['cause'],'accident')
        self.assertEqual(extract_fields('首次投保，保单生效第二十天因病治疗？')['policy_day'],20)
        self.assertEqual(chinese_number('一百零五'),105)
        self.assertIsNone(extract_fields('保单生效第十个月？')['policy_day'])

    def test_surface_normalized_diagnostic_vs_strict_v1(self):
        q='我首次投保了《平安关爱相伴团体医疗保险（B款）》，保单生效是第20天，因为不是意外疾病的原因发生了约定的医疗事项，是否因等待期排除？'
        self.assertIsNone(extract_fields(q)['policy_day'])
        self.assertEqual(extract_fields_surface_normalized(q)['policy_day'],20)
        q_waiver='我是在重新投保《太保互联网个人一年期住院E款医疗保险》，保单生效是第20天，因为不是意外导致的疾病，保险公司已经明确同意免去等待期，会被排除吗？'
        self.assertIsNone(extract_fields(q_waiver)['approved_waiver'])
        self.assertTrue(extract_fields_surface_normalized(q_waiver)['approved_waiver'])

    def test_rule_outcome_evaluation_detects_degradation_and_mutation(self):
        seed_frame=dict(policy_id='cpic_86017',enrollment='first',cause='illness',policy_day=10,
                        plan_wait_days=7,approved_waiver=False)
        req=['policy_id','enrollment','policy_day','cause','plan_wait_days']
        obs_missing=dict(policy_id=None,enrollment='first',cause='illness',policy_day=10,plan_wait_days=None,approved_waiver=None)
        status,mut,miss=evaluate_rule_outcome_under_observed(obs_missing,seed_frame,req,grant_source_policy=True)
        self.assertEqual(status,'DEGRADED_TO_CLARIFY')
        self.assertEqual(miss,['plan_wait_days'])
        obs_mutated=dict(policy_id='cpic_86017',enrollment='first',cause='illness',policy_day=2,plan_wait_days=7,approved_waiver=None)
        status2,mut2,_=evaluate_rule_outcome_under_observed(obs_mutated,seed_frame,req,grant_source_policy=True)
        self.assertEqual(status2,'COUNTERFACTUAL_FLIP')
        self.assertEqual(mut2,['policy_day'])

    def test_non_pivotal_missing_slot_separates_slot_completeness_from_decision_determinacy(self):
        frame=dict(policy_id='pa_h865',enrollment='first',cause='illness',policy_day=45,
                   plan_wait_days=30,approved_waiver=False)
        req=['policy_id','enrollment','policy_day','cause']
        obs=dict(frame,cause=None)
        slot_status,dec_status,mut,miss=evaluate_slot_completeness(obs,frame,req,grant_source_policy=False)
        self.assertEqual(slot_status,'NON_PIVOTAL_SLOT_OMITTED_DECISION_PRESERVED')
        self.assertEqual(dec_status,'PRESERVED_DECISION')
        self.assertEqual(mut,[])
        self.assertEqual(miss,['cause'])

    def test_cohens_kappa(self):
        from score_annotations import compute_task_metrics
        keys=['k1','k2','k3','k4']
        mapping={k:{'ref':'yes'} for k in keys}
        map_a={k:{'label':'yes' if i%2==0 else 'no','reason':''} for i,k in enumerate(keys)}
        res_perfect=compute_task_metrics(keys,map_a,map_a,mapping,reference_field='ref')
        self.assertAlmostEqual(res_perfect['cohen_kappa'],1.0)
        self.assertEqual(res_perfect['disagreement_count'],0)

    def test_cross_domain_double_witness_and_counterfactuals(self):
        from cross_domain_study import build_Controlled_Cases, cross_decision
        seeds, cases = build_Controlled_Cases()
        self.assertEqual(len(seeds), 16)
        self.assertEqual(len(cases), 64)
        missing_cases = [c for c in cases if c['variant'] == 'missing_critical']
        self.assertEqual(len(missing_cases), 16)
        for mc in missing_cases:
            self.assertEqual(mc['reference_decision'], 'CLARIFY')
            w = mc['counterexample_witnesses']
            self.assertEqual(len(w), 2)
            self.assertNotEqual(w[0]['decision'], w[1]['decision'])

    def test_cross_domain_slot_extractor(self):
        from cross_domain_study import build_seed_frames, render_question, extract_cross_slots_v2, evaluate_cross_outcome
        for s in build_seed_frames():
            q_complete = render_question(s, surface=False)
            obs = extract_cross_slots_v2(q_complete, s)
            outcome, missing, mutated = evaluate_cross_outcome(obs, s)
            self.assertEqual(outcome, 'PRESERVED_DECISION', f"Failed on {s['seed_id']}: {missing} / {mutated}")
            q_missing = render_question(s, drop_field=s['pivotal_field'])
            obs_m = extract_cross_slots_v2(q_missing, s)
            outcome_m, missing_m, _ = evaluate_cross_outcome(obs_m, s)
            self.assertEqual(outcome_m, 'DEGRADED_TO_CLARIFY')
            self.assertIn(s['pivotal_field'], missing_m)

    def test_cross_domain_polarity_and_seed_frame_independence(self):
        from cross_domain_study import extract_cross_slots_v2
        obs_neg_conf = extract_cross_slots_v2('普通行政文员，不负有保密义务')
        self.assertEqual(obs_neg_conf.get('condition'), 'disqualified')
        obs_pos_conf = extract_cross_slots_v2('员工属于负有保密义务的高级技术人员')
        self.assertEqual(obs_pos_conf.get('condition'), 'qualified')
        obs_stop_vte = extract_cross_slots_v2('目前已停用常规抗凝预防方案')
        self.assertEqual(obs_stop_vte.get('status'), 'secondary')
        obs_active_vte = extract_cross_slots_v2('目前处于路径内常规抗凝状态')
        self.assertEqual(obs_active_vte.get('status'), 'primary')
        obs_no_bleed = extract_cross_slots_v2('术后第20天没有活动性出血并发症')
        self.assertEqual(obs_no_bleed.get('condition'), 'qualified')
        obs_bleed = extract_cross_slots_v2('术后第7天出现了活动性出血并发症')
        self.assertEqual(obs_bleed.get('condition'), 'disqualified')
        obs_knee = extract_cross_slots_v2('患者接受膝关节置换，术后第7天')
        self.assertEqual(obs_knee.get('threshold_days'), 10)
        obs_hip = extract_cross_slots_v2('患者接受髋关节置换，术后第20天')
        self.assertEqual(obs_hip.get('threshold_days'), 35)

    def test_cross_domain_equivalence_regions_reference_independence_and_negated_waivers(self):
        from cross_domain_study import (
            build_seed_frames, cross_decision, enumerate_cross_decisions,
            evaluate_cross_outcome, extract_cross_slots_v2, render_question,
        )
        from analyze_v3 import extract_fields_surface_normalized
        seeds = {s['seed_id']: s for s in build_seed_frames()}

        # 1. Missing elapsed_day on 365-day threshold covers >365 equivalence region (e.g. 400 days)
        seed_365 = seeds['cross-010']
        obs_no_day = {k: seed_365[k] for k in seed_365['required_fields'] if k != 'elapsed_day'}
        found_decs = enumerate_cross_decisions(obs_no_day)
        self.assertEqual(found_decs, ['RULE_ELIGIBLE', 'RULE_EXCLUDED'])
        self.assertEqual(cross_decision(dict(seed_365, elapsed_day=400)), 'RULE_EXCLUDED')
        self.assertEqual(evaluate_cross_outcome(obs_no_day, seed_365)[0], 'DEGRADED_TO_CLARIFY')

        # 2. Admissible completion set is strictly independent of reference seed_frame
        a, b = seeds['cross-001'], seeds['cross-004']
        obs_shared = {k: a[k] for k in a['required_fields']}
        self.assertEqual(enumerate_cross_decisions(obs_shared, a), enumerate_cross_decisions(obs_shared, b))
        self.assertEqual(evaluate_cross_outcome(obs_shared, a)[0], 'PRESERVED_DECISION')
        self.assertNotEqual(evaluate_cross_outcome(obs_shared, b)[0], 'PRESERVED_DECISION')

        # 3. Negated waivers are parsed as False and caught as CONDITION_MUTATED against positive waiver seeds
        neg_waiver_q = render_question(b).replace('已获肿瘤专家组', '尚未获肿瘤专家组')
        parsed_neg = extract_cross_slots_v2(neg_waiver_q)
        self.assertIs(parsed_neg.get('approved_waiver'), False)
        self.assertEqual(evaluate_cross_outcome(parsed_neg, b)[0], 'CONDITION_MUTATED')

        self.assertIs(extract_cross_slots_v2('且未批准延长申请时限').get('approved_waiver'), False)
        ins_neg = extract_fields_surface_normalized(
            '按《平安员工及家属住院医疗保险》，我是重新投保，已经生效第20天，但未获保险公司批准免除等待期'
        )
        self.assertIs(ins_neg.get('approved_waiver'), False)

    def test_reachability_invariants_and_repair_trigger(self):
        import json
        from cross_domain_study import (
            build_seed_frames, cross_decision, enumerate_cross_decisions,
            evaluate_cross_outcome, extract_cross_slots_v2, render_question,
            should_trigger_slot_guard_repair,
        )
        seeds = {s['seed_id']: s for s in build_seed_frames()}

        # 1. Counterexample A: adding positive waiver to cross-001 (not in required_fields) flips decision
        s1 = seeds['cross-001']
        q_added_waiver = render_question(s1).replace('，请问', '，且已获肿瘤专家组急诊特批豁免洗脱期，请问')
        obs1 = extract_cross_slots_v2(q_added_waiver)
        self.assertIs(obs1.get('approved_waiver'), True)
        self.assertEqual(enumerate_cross_decisions(obs1), ['RULE_ELIGIBLE'])
        out1, miss1, mut1 = evaluate_cross_outcome(obs1, s1)
        self.assertEqual(out1, 'CONDITION_MUTATED')
        self.assertIn('approved_waiver', mut1)

        # 2. Counterexample B: deleting positive waiver from cross-004 yields singleton ['RULE_EXCLUDED'] under default-False rule
        s4 = seeds['cross-004']
        q_deleted_waiver = render_question(s4).replace('，且已获肿瘤专家组急诊特批豁免洗脱期', '')
        obs4 = extract_cross_slots_v2(q_deleted_waiver)
        self.assertIsNone(obs4.get('approved_waiver'))
        self.assertEqual(enumerate_cross_decisions(obs4), ['RULE_EXCLUDED'])
        out4, miss4, mut4 = evaluate_cross_outcome(obs4, s4)
        self.assertEqual(out4, 'CONDITION_MUTATED')
        self.assertIn('approved_waiver', miss4)
        self.assertIn('approved_waiver', mut4)

        # 3. Insurance evaluator also catches non-required slot mutation (adding approved_waiver=True on first enrollment)
        ins_seed = dict(
            policy_id='pa_h865', enrollment='first', cause='illness',
            policy_day=20, plan_wait_days=30, approved_waiver=False,
        )
        ins_req = ['policy_id', 'enrollment', 'policy_day', 'cause']
        ins_obs = dict(ins_seed, approved_waiver=True)
        ins_out, ins_mut, _ = evaluate_rule_outcome_under_observed(ins_obs, ins_seed, ins_req, grant_source_policy=False)
        self.assertEqual(ins_out, 'COUNTERFACTUAL_FLIP')
        self.assertIn('approved_waiver', ins_mut)

        # 4. Unified repair trigger across all 16 condition_prompt outputs and zero reachability invariant violations across all 48 outputs
        gen_path = Path(__file__).parents[1] / 'data_v3' / 'cross_domain_generations.jsonl'
        rows = [json.loads(line) for line in gen_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertEqual(len(rows), 48)
        cp_triggers = []
        for r in rows:
            s = seeds[r['seed_id']]
            ref_dec = cross_decision(s)
            obs = extract_cross_slots_v2(r['question'])
            reach = enumerate_cross_decisions(obs)
            out, miss, mut = evaluate_cross_outcome(obs, s)
            if out == 'PRESERVED_DECISION':
                self.assertEqual(reach, [ref_dec])
                self.assertEqual(mut, [])
            elif out == 'DEGRADED_TO_CLARIFY':
                self.assertNotEqual(len(reach), 1)
            elif out == 'CONDITION_MUTATED':
                self.assertTrue(bool(mut) or (len(reach) == 1 and reach[0] != ref_dec))
            if r['method'] == 'condition_prompt':
                if should_trigger_slot_guard_repair(miss, mut):
                    cp_triggers.append(r['seed_id'])
        self.assertEqual(len(cp_triggers), 10)
        self.assertIn('cross-001', cp_triggers)

    def test_heldout_official_clauses_and_equal_budget_baselines(self):
        import json
        from equal_budget_and_heldout_study import (
            HELDOUT_POLICIES,
            evaluate_insurance_outcome_frozen,
            extract_insurance_fields_frozen,
            insurance_decision,
        )
        root = Path(__file__).parents[1]
        held_cases = [
            json.loads(line)
            for line in (root / 'data_v3' / 'heldout_controlled_cases.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        self.assertEqual(len(held_cases), 64)
        by_group = {}
        for c in held_cases:
            by_group.setdefault(c['group_id'], {})[c['variant']] = c
        self.assertEqual(len(by_group), 16)
        for gid, grp in by_group.items():
            mc = grp['missing_critical']
            self.assertEqual(mc['reference_label'], 'CLARIFY')
            self.assertEqual(len(mc['counterexample_witnesses']), 2)
            w0, w1 = mc['counterexample_witnesses']
            self.assertEqual(insurance_decision(w0['completion'], HELDOUT_POLICIES), w0['label'])
            self.assertEqual(insurance_decision(w1['completion'], HELDOUT_POLICIES), w1['label'])
            self.assertNotEqual(w0['label'], w1['label'])
            seed_frame = w0['completion']
            req = ['policy_id', 'enrollment', 'policy_day', 'cause']
            if len(HELDOUT_POLICIES[seed_frame['policy_id']]['days']) > 1:
                req.append('plan_wait_days')
            if seed_frame['enrollment'] == 'new':
                req.append('approved_waiver')
            for var_name in ['complete', 'surface']:
                obs = extract_insurance_fields_frozen(grp[var_name]['question'], HELDOUT_POLICIES)
                out, _, _ = evaluate_insurance_outcome_frozen(obs, seed_frame, req, HELDOUT_POLICIES)
                self.assertEqual(out, 'PRESERVED_DECISION')
            obs_mc = extract_insurance_fields_frozen(mc['question'], HELDOUT_POLICIES)
            out_mc, _, _ = evaluate_insurance_outcome_frozen(obs_mc, seed_frame, req, HELDOUT_POLICIES)
            self.assertEqual(out_mc, 'DEGRADED_TO_CLARIFY')
            obs_cf = extract_insurance_fields_frozen(grp['counterfactual']['question'], HELDOUT_POLICIES)
            out_cf, _, _ = evaluate_insurance_outcome_frozen(obs_cf, seed_frame, req, HELDOUT_POLICIES)
            self.assertEqual(out_cf, 'CONDITION_MUTATED')

        summary = json.loads((root / 'data_v3' / 'equal_budget_and_heldout_summary.json').read_text(encoding='utf-8'))
        cross_methods = summary['cross_domain_equal_budget_comparison']['methods']
        self.assertEqual(cross_methods['slot_guard_repair']['total_llm_calls'], 26)
        self.assertEqual(cross_methods['generic_self_refine_budget_matched']['total_llm_calls'], 26)
        self.assertEqual(cross_methods['slot_guard_repair']['preserved_decision'], 12)
        self.assertEqual(cross_methods['slot_guard_repair']['condition_mutated'], 0)
        self.assertEqual(cross_methods['generic_self_refine_budget_matched']['preserved_decision'], 8)
        self.assertEqual(cross_methods['generic_self_refine_budget_matched']['condition_mutated'], 4)
        self.assertEqual(cross_methods['generic_self_refine_all']['preserved_decision'], 8)
        self.assertEqual(cross_methods['generic_self_refine_all']['condition_mutated'], 4)

    def test_heldout_benefit_disambiguation_and_parser_parity(self):
        import json
        from condition_data import POLICIES
        from equal_budget_and_heldout_study import (
            HELDOUT_POLICIES,
            evaluate_insurance_outcome_frozen,
            extract_insurance_fields_frozen,
        )
        root = Path(__file__).parents[1]

        # 1. Parser parity on all 40 development generations and boundary probes
        dev_rows = [
            json.loads(line)
            for line in (root / 'data_v3' / 'generated_questions.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        for r in dev_rows:
            self.assertEqual(
                extract_fields_surface_normalized(r['question']),
                extract_insurance_fields_frozen(r['question'], POLICIES),
            )
        for probe, expected_waiver in [
            ('未获保险公司批准免除了等待期', False),
            ('是否已获保险公司批准免等待期还不清楚', None),
        ]:
            dev_p = extract_fields_surface_normalized(probe)
            held_p = extract_insurance_fields_frozen(probe, POLICIES)
            self.assertEqual(dev_p, held_p)
            self.assertIs(dev_p['approved_waiver'], expected_waiver)

        # 2. Benefit-choice ambiguity on pa_2011_36 is rejected as DEGRADED_TO_CLARIFY
        seed_f = dict(
            policy_id='pa_2011_36', benefit_id='general_hospital',
            enrollment='first', cause='illness', policy_day=45,
            plan_wait_days=30, approved_waiver=False,
        )
        req = ['policy_id', 'benefit_id', 'enrollment', 'policy_day', 'cause', 'plan_wait_days']
        ambig_q = (
            '按《平安住院安心医疗保险》条款，我是首次投保，保单生效45天后因非意外疾病发生约定医疗事项，'
            '是按一般住院医疗津贴（30天等待期）还是按癌症住院医疗津贴（90天等待期）承担责任？'
        )
        obs_ambig = extract_insurance_fields_frozen(ambig_q, HELDOUT_POLICIES)
        self.assertIsNone(obs_ambig['benefit_id'])
        out_ambig, miss_ambig, _ = evaluate_insurance_outcome_frozen(obs_ambig, seed_f, req, HELDOUT_POLICIES)
        self.assertEqual(out_ambig, 'DEGRADED_TO_CLARIFY')
        self.assertIn('benefit_id', miss_ambig)

        # 3. Mismatched benefit vs waiting period (general_hospital with 90-day wait) is caught as CONDITION_MUTATED
        contradict_q = (
            '按《平安住院安心医疗保险》条款，我是首次投保，保单生效第45天，因非意外疾病住院治疗，'
            '想申请一般住院医疗津贴（约定等待期为90天），这种情况会因为等待期限制而不能理赔吗？'
        )
        obs_con = extract_insurance_fields_frozen(contradict_q, HELDOUT_POLICIES)
        out_con, _, mut_con = evaluate_insurance_outcome_frozen(obs_con, seed_f, req, HELDOUT_POLICIES)
        self.assertEqual(out_con, 'CONDITION_MUTATED')
        self.assertIn('plan_wait_days', mut_con)

    def test_rubric_source_separation_and_watchlist_integrity(self):
        import json
        root = Path(__file__).parents[1]

        # 1. Check source_type separation in heldout_controlled_cases.jsonl
        held_cases = [
            json.loads(line)
            for line in (root / 'data_v3' / 'heldout_controlled_cases.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        self.assertEqual(len(held_cases), 64)
        for c in held_cases:
            by_slot = {r['slot']: r for r in c['atomic_rubrics']}
            self.assertEqual(by_slot['policy_and_clause_rule']['source_type'], 'pdf_contract_clause')
            self.assertIsNotNone(by_slot['policy_and_clause_rule']['pdf_file'])
            self.assertIn('bbox_pdf_points', by_slot['policy_and_clause_rule'])
            self.assertEqual(by_slot['scenario_case_parameters']['source_type'], 'synthetic_seed_frame')
            self.assertNotIn('bbox_pdf_points', by_slot['scenario_case_parameters'])

        # 2. Check deterministic watchlist schema and cross-009 accuracy
        audit_rows = [
            json.loads(line)
            for line in (root / 'data_v3' / 'item_by_item_semantic_audit.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        self.assertEqual(len(audit_rows), 264)
        audit_summary = json.loads((root / 'data_v3' / 'item_by_item_semantic_audit_summary.json').read_text(encoding='utf-8'))
        self.assertIn('NOT_external_semantic_validation_or_human_annotation', audit_summary['record_provenance'])
        self.assertNotIn('agreement_rate', json.dumps(audit_summary))
        by_aid = {r['audit_id']: r for r in audit_rows}
        for m in ['condition_prompt', 'generic_self_refine_budget_matched', 'generic_self_refine_all']:
            r = by_aid[f'cross-cross-009-{m}']
            self.assertEqual(r['missing_slots'], ['elapsed_day'])
            self.assertIn('当事人实际经过天数', r['automated_verifier_explanation'])
            self.assertNotIn('第45天', r['question'])
        r_guard = by_aid['cross-cross-009-slot_guard_repair']
        self.assertEqual(r_guard['standalone_outcome'], 'PRESERVED_DECISION')
        self.assertEqual(r_guard['missing_slots'], [])
        r_ho8 = by_aid['heldout-heldout-008-structured_json_rewrite']
        self.assertEqual(r_ho8['watchlist_category'], 'json_truncated_at_token_cap')

        # 3. Check draft_zh.md policy table alignment against sources.json + heldout_clauses_sources.json
        dev_sources = json.loads((root / 'data_v3' / 'sources.json').read_text(encoding='utf-8'))
        held_sources = json.loads((root / 'data_v3' / 'heldout_clauses_sources.json').read_text(encoding='utf-8'))
        all_sources = {s['id']: s for s in dev_sources + held_sources if 'sha256' in s}
        zh_text = (root / 'paper' / 'draft_zh.md').read_text(encoding='utf-8')
        matched_pids = set()
        for line in zh_text.splitlines():
            if '|' in line and ('开发集' in line or '扩展集' in line):
                cells = [c.strip().strip('`') for c in line.strip('|').split('|')]
                if len(cells) >= 9 and cells[1] in all_sources:
                    pid = cells[1]
                    src = all_sources[pid]
                    self.assertTrue((src['title'] in cells[2]) or (cells[2] in src['title']), f"Title mismatch on {pid}")
                    self.assertEqual(cells[-1], src['sha256'][:12], f"SHA12 mismatch on {pid}")
                    matched_pids.add(pid)
        self.assertEqual(matched_pids, set(all_sources.keys()))

        # 4. Check word-for-word abstract alignment between draft_en.md and main.tex
        import re
        md_abs = re.search(
            r'## Abstract\s+(.+?)\s+---',
            (root / 'paper' / 'draft_en.md').read_text(encoding='utf-8'),
            re.S,
        ).group(1).strip()
        tex_abs = re.search(
            r'\\begin\{abstract\}\s+(.+?)\s+\\end\{abstract\}',
            (root / 'paper' / 'overleaf_package' / 'main.tex').read_text(encoding='utf-8'),
            re.S,
        ).group(1).strip()
        norm_tex = (
            tex_abs.replace(r'\%', '%')
            .replace(r'\text{--}', '–')
            .replace('{,}', ',')
            .replace('$', '')
            .replace('p=0.125', 'p = 0.125')
        )
        self.assertEqual(md_abs, norm_tex)
        self.assertLessEqual(len(md_abs.split()), 200)

    def test_contrastive_negation_and_multiday_conflict_handling(self):
        from condition_data import POLICIES
        from cross_domain_study import (
            build_seed_frames,
            evaluate_cross_outcome,
            extract_cross_slots_v2,
            should_trigger_slot_guard_repair,
        )
        from equal_budget_and_heldout_study import (
            evaluate_insurance_outcome_frozen,
            extract_insurance_fields_frozen,
        )

        # 1. Insurance development seed (pa_h865, first, illness, day=20, wait=30)
        ins_seed = {
            'id': 'probe-ins-001',
            'policy_id': 'pa_h865',
            'enrollment': 'first',
            'cause': 'illness',
            'policy_day': 20,
            'plan_wait_days': 30,
            'approved_waiver': False,
            'required_fields': ['policy_id', 'enrollment', 'policy_day', 'cause'],
        }
        q_contrast_ins = (
            '我第一次买《平安员工及家属住院医疗保险》，不是保单生效第20天，而是第45天，'
            '因为非意外生病住院治疗，想问这种情况能报销吗？'
        )
        obs_dev = extract_fields_surface_normalized(q_contrast_ins)
        self.assertEqual(obs_dev['policy_day'], 45)
        out_dev, _, _ = evaluate_rule_outcome_under_observed(obs_dev, ins_seed, ins_seed['required_fields'])
        self.assertEqual(out_dev, 'COUNTERFACTUAL_FLIP')
        obs_frozen = extract_insurance_fields_frozen(q_contrast_ins, POLICIES)
        self.assertEqual(obs_frozen['policy_day'], 45)
        out_frozen, _, _ = evaluate_insurance_outcome_frozen(
            obs_frozen, ins_seed, ins_seed['required_fields'], POLICIES
        )
        self.assertEqual(out_frozen, 'CONDITION_MUTATED')

        for q_ambig_ins in [
            '我第一次买《平安员工及家属住院医疗保险》，保单生效不到20天，因为非意外生病住院治疗，想问这种情况能报销吗？',
            '我第一次买《平安员工及家属住院医疗保险》，保单生效第20天或第45天，因为非意外生病住院治疗，想问这种情况能报销吗？',
        ]:
            obs_ambig = extract_fields_surface_normalized(q_ambig_ins)
            self.assertIsNone(obs_ambig['policy_day'])
            out_ambig, _, _ = evaluate_rule_outcome_under_observed(obs_ambig, ins_seed, ins_seed['required_fields'])
            self.assertEqual(out_ambig, 'DEGRADED_TO_CLARIFY')

        # 2. Multi-threshold seed cross-001 (med_egfr_tki, primary, elapsed_day=7, threshold_days=14, qualified)
        from cross_domain_study import render_question
        c_seed = build_seed_frames()[0]
        base_q = render_question(c_seed)
        q_contrast_cross = base_q.replace('停用前序药物第7天', '并不是停用前序药物第7天，而是第20天')
        obs_c = extract_cross_slots_v2(q_contrast_cross)
        self.assertEqual(obs_c['elapsed_day'], 20)
        out_c, miss_c, mut_c = evaluate_cross_outcome(obs_c, c_seed)
        self.assertEqual(out_c, 'CONDITION_MUTATED')
        self.assertEqual(mut_c, ['elapsed_day'])
        self.assertTrue(should_trigger_slot_guard_repair(miss_c, mut_c, out_c))

        for q_ambig_cross in [
            base_q.replace('停用前序药物第7天', '停用前序药物不到7天'),
            base_q.replace('停用前序药物第7天', '停用前序药物第7天或第20天'),
        ]:
            obs_ac = extract_cross_slots_v2(q_ambig_cross)
            self.assertIsNone(obs_ac.get('elapsed_day'))
            out_ac, miss_ac, mut_ac = evaluate_cross_outcome(obs_ac, c_seed)
            self.assertEqual(out_ac, 'DEGRADED_TO_CLARIFY')
            self.assertIn('elapsed_day', miss_ac)
            self.assertTrue(should_trigger_slot_guard_repair(miss_ac, mut_ac, out_ac))

    def test_optional_unknown_waiver_triggers_repair_and_granted_source_sensitivity(self):
        import json
        from cross_domain_study import (
            build_seed_frames,
            evaluate_cross_outcome,
            extract_cross_slots_v2,
            render_question,
            should_trigger_slot_guard_repair,
        )
        from score_annotations import compute_task_metrics
        root = Path(__file__).parents[1]

        # 1. Optional field set to 'unknown' must populate missing and trigger repair
        c_seed = build_seed_frames()[0]
        q_unk_waiver = render_question(c_seed, drop_field='approved_waiver')
        obs_uw = extract_cross_slots_v2(q_unk_waiver)
        self.assertEqual(obs_uw['approved_waiver'], 'unknown')
        out_uw, miss_uw, mut_uw = evaluate_cross_outcome(obs_uw, c_seed)
        self.assertEqual(out_uw, 'DEGRADED_TO_CLARIFY')
        self.assertIn('approved_waiver', miss_uw)
        self.assertTrue(should_trigger_slot_guard_repair(miss_uw, mut_uw, out_uw))

        # 2. Granted-source-clause sensitivity counts on multi-threshold set
        cd_sum = json.loads((root / 'data_v3' / 'cross_domain_summary.json').read_text(encoding='utf-8'))
        cd_methods = cd_sum['methods']
        self.assertEqual(cd_methods['ordinary_style']['preserved_decision'], 1)
        self.assertEqual(cd_methods['ordinary_style']['granted_source_clause_preserved_decision'], 7)
        self.assertEqual(cd_methods['slot_guard_repair']['preserved_decision'], 12)
        self.assertEqual(cd_methods['slot_guard_repair']['granted_source_clause_preserved_decision'], 15)

        eb_sum = json.loads((root / 'data_v3' / 'equal_budget_and_heldout_summary.json').read_text(encoding='utf-8'))
        eb_methods = eb_sum['cross_domain_equal_budget_comparison']['methods']
        self.assertEqual(eb_methods['ordinary_style']['preserved_decision'], 1)
        self.assertEqual(eb_methods['ordinary_style']['granted_source_policy_preserved_decision'], 7)
        self.assertEqual(eb_methods['slot_guard_repair']['preserved_decision'], 12)
        self.assertEqual(eb_methods['slot_guard_repair']['granted_source_policy_preserved_decision'], 15)
        self.assertEqual(eb_methods['generic_self_refine_budget_matched']['granted_source_policy_preserved_decision'], 8)
        paired_granted = eb_sum['paired_mcnemar_tests'][
            'cross_guard_vs_generic_budget_matched_granted_clause_sensitivity'
        ]
        self.assertEqual(len(paired_granted['left_only_wins']), 7)
        self.assertEqual(len(paired_granted['right_only_wins']), 0)
        self.assertAlmostEqual(paired_granted['exact_mcnemar_two_sided_p'], 0.015625)

        # 3. Single-category Cohen's kappa returns None + undefined_single_category
        keys = [f'item-{i}' for i in range(5)]
        map_a = {k: {'label': 'PRESERVED_DECISION', 'reason': 'ok'} for k in keys}
        map_b = {k: {'label': 'PRESERVED_DECISION', 'reason': 'ok'} for k in keys}
        mapping = {k: {'original_id': k} for k in keys}
        m_single = compute_task_metrics(keys, map_a, map_b, mapping)
        self.assertEqual(m_single['raw_agreement'], 1.0)
        self.assertIsNone(m_single['cohen_kappa'])
        self.assertEqual(m_single['cohen_kappa_status'], 'undefined_single_category')

        # 4. Reference list in draft_en.md matches corrected citations in references.bib
        en_text = (root / 'paper' / 'draft_en.md').read_text(encoding='utf-8')
        bib_text = (root / 'paper' / 'overleaf_package' / 'references.bib').read_text(encoding='utf-8')
        self.assertNotIn('Feng et al., 2016', en_text)
        self.assertIn('Feng et al., 2015', en_text)
        for token in [
            'Feng, M., Xiang, B., Glass, M. R., Wang, L., & Zhou, B. (2015). Applying Deep Learning to Answer Selection: A Study and An Open Task.',
            'Divekar, A. (2026). Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference.',
            'Chen, L., Zhang, R., Guo, J., Fan, Y., & Cheng, X. (2024). Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework.',
            'Filice, S., Horowitz, G., Carmel, D., Karnin, Z., Lewin-Eytan, L., & Maarek, Y. (2025). Generating Q&A Benchmarks for RAG Evaluation with DataMorgana.',
            'Fu, W., Wei, B., Hu, J., Cai, Z., & Liu, J. (2024). QGEval: Benchmarking Multi-dimensional Evaluation for Question Generation.',
            'Kucia, F. J., & Gawlik, B. M. (2026). Beyond Benchmark Scores: How Synthetic and Authentic Query Distributions Diverge in RAG Evaluation.',
            'van Elburg, J., van der Putten, P., & Marx, M. (2025). Can we Evaluate RAGs with Synthetic Data?',
            'Zhang, T., Qin, P., Deng, Y., Huang, C., Lei, W., Liu, J., Jin, D., Liang, H., & Chua, T.-S. (2024). CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models.',
            'Zhao, J., Fang, K., & Cheng, L. (2026). When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification.',
            'Zhu, K., Luo, Y., Xu, D., Yan, Y., Liu, Z., Yu, S., Wang, R., Wang, S., Li, Y., Zhang, N., Han, X., Liu, Z., & Sun, M. (2025). RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework.',
        ]:
            self.assertIn(token, en_text)
        self.assertIn('year={2015}', bib_text)
        self.assertIn('author={Divekar, Abhishek}', bib_text)

    def test_masters_thesis_frozen_confirmation_and_rag_selection(self):
        """Verify all Master's Thesis deliverables:
        - PARSE_UNCERTAIN vs CLARIFY_REQUIRED separation
        - 24-document Clause-to-Rule Mapping Audit (20 official insurance PDFs + 4 multi-threshold rules)
        - 120-case Gold Diagnostic & Adversarial Suite (0.0% False Accept Rate)
        - 40-seed Strictly Frozen Confirmation Set (v3.0-frozen, aliases=[] on all 11 new PDFs, 2 cloud model families)
        - Downstream RAG System Evaluation & Selection impact
        """
        import json
        from frozen_confirmation_and_rag_study import (
            FROZEN_CONFIRM_POLICIES,
            VERIFIER_VERSION,
            detect_parse_uncertainty,
        )

        root = Path(__file__).resolve().parents[1]
        self.assertEqual(VERIFIER_VERSION, 'v3.0-frozen')
        self.assertEqual(len(FROZEN_CONFIRM_POLICIES), 11)
        fc_sources = json.loads((root / 'data_v3' / 'frozen_confirmation_sources.json').read_text(encoding='utf-8'))
        fc_src_map = {s['id']: s for s in fc_sources}
        for pid, meta in FROZEN_CONFIRM_POLICIES.items():
            self.assertEqual(meta['aliases'], [], f'Frozen confirmation policy {pid} must have empty aliases')
            self.assertIn(pid, fc_src_map)
            self.assertEqual(len(fc_src_map[pid]['sha256']), 64)
            self.assertGreaterEqual(fc_src_map[pid]['anchor']['pdf_page_1based'], 1)

        # 1. PARSE_UNCERTAIN detection on month/week units, calendar spans, negated/conflicting durations
        p_month = detect_parse_uncertainty('按《平安安康医疗保险（2018）》条款，我是首次投保，保单生效满两个月后因非意外疾病住院治疗，请问是否受等待期限制？')
        self.assertTrue(p_month['is_parse_uncertain'])
        self.assertIn('unsupported_time_unit_or_calendar_date', p_month['reasons'])

        p_cal = detect_parse_uncertainty('按《平安安康医疗保险（2018）》条款，我是首次投保，3月1日保单正式生效，同年5月10日因非意外疾病住院治疗，请问是否受等待期限制？')
        self.assertTrue(p_cal['is_parse_uncertain'])
        self.assertIn('unsupported_time_unit_or_calendar_date', p_cal['reasons'])

        p_clean_omit = detect_parse_uncertainty('按《平安安康医疗保险（2018）》条款，我是首次投保，因非意外疾病住院治疗，请问是否受等待期限制？')
        self.assertFalse(p_clean_omit['is_parse_uncertain'])

        # 2. Clause-to-Rule Mapping Audit (20 official insurance PDFs + 4 multi-threshold rules = 24)
        mapping_audit = json.loads((root / 'data_v3' / 'clause_rule_mapping_audit.json').read_text(encoding='utf-8'))
        self.assertEqual(mapping_audit['total_rule_documents'], 24)
        self.assertEqual(mapping_audit['total_official_insurance_pdfs'], 20)
        self.assertEqual(mapping_audit['total_multi_threshold_rules'], 4)
        self.assertEqual(len(mapping_audit['insurance_contracts']), 20)
        self.assertEqual(len(mapping_audit['multi_threshold_rules']), 4)

        # 3. 120-case Gold Diagnostic Suite (0.0% False Accept Rate, 100% precision on accepted)
        gold_sum = json.loads((root / 'data_v3' / 'gold_diagnostic_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(gold_sum['total_cases'], 120)
        self.assertEqual(gold_sum['gold_positive_preserved_cases'], 60)
        self.assertEqual(gold_sum['gold_negative_changed_or_ambiguous_cases'], 60)
        self.assertEqual(gold_sum['false_accept_fp'], 0)
        self.assertEqual(gold_sum['false_accept_rate_among_gold_negatives'], 0.0)
        self.assertEqual(gold_sum['false_reject_rate_on_supported_day_syntax_categories_1_to_7'], 0.0)
        self.assertEqual(gold_sum['precision_on_accepted'], 1.0)
        self.assertEqual(gold_sum['parse_uncertain_flagged_total'], 45)

        # 4. 40-seed Frozen Confirmation Set & Dual-Implementation Three-Layer Audit across Qwen-Plus and DeepSeek
        fc_sum = json.loads((root / 'data_v3' / 'frozen_confirmation_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(fc_sum['verifier_version'], 'v3.0-frozen')
        tl_sum = fc_sum['three_layer_frozen_confirmation_audit_summary']
        self.assertEqual(tl_sum['total_audited_generations'], 400)
        self.assertEqual(tl_sum['true_accept_tp'], 290)
        self.assertEqual(tl_sum['false_accept_fp'], 6)
        self.assertEqual(tl_sum['true_reject_tn'], 88)
        self.assertEqual(tl_sum['false_reject_fn'], 16)
        self.assertAlmostEqual(tl_sum['false_accept_rate'], 0.0638, places=4)
        self.assertAlmostEqual(tl_sum['false_reject_rate'], 0.0523, places=4)
        self.assertAlmostEqual(tl_sum['precision_on_accepted'], 0.9797, places=4)
        self.assertAlmostEqual(tl_sum['recall_on_preserved'], 0.9477, places=4)
        self.assertEqual(tl_sum['by_method']['slot_guard_repair']['fp'], 0)
        self.assertEqual(tl_sum['by_method']['structured_json_rewrite']['fp'], 6)

        qwen_40 = fc_sum['model_family_results']['qwen_plus']['combined_40_seeds']
        ds_40 = fc_sum['model_family_results']['deepseek']['combined_40_seeds']
        self.assertEqual(qwen_40['ordinary_style']['preserved_decision'], 0)
        self.assertEqual(qwen_40['condition_prompt']['preserved_decision'], 37)
        self.assertEqual(qwen_40['generic_self_refine_budget_matched']['preserved_decision'], 36)
        self.assertEqual(qwen_40['generic_self_refine_all']['preserved_decision'], 34)
        self.assertEqual(qwen_40['slot_guard_repair']['preserved_decision'], 39)

        self.assertEqual(ds_40['ordinary_style']['preserved_decision'], 1)
        self.assertEqual(ds_40['condition_prompt']['preserved_decision'], 33)
        self.assertEqual(ds_40['generic_self_refine_budget_matched']['preserved_decision'], 34)
        self.assertEqual(ds_40['generic_self_refine_all']['preserved_decision'], 34)
        self.assertEqual(ds_40['slot_guard_repair']['preserved_decision'], 39)

        pooled = fc_sum['pooled_cross_model_paired_mcnemar_80_seeds']
        self.assertEqual(pooled['slot_guard_vs_condition_prompt']['left_success'], 78)
        self.assertEqual(pooled['slot_guard_vs_condition_prompt']['right_success'], 70)
        self.assertAlmostEqual(pooled['slot_guard_vs_condition_prompt']['exact_mcnemar_two_sided_p'], 0.007812, places=5)
        self.assertEqual(pooled['slot_guard_vs_generic_self_refine_budget_matched']['left_success'], 78)
        self.assertEqual(pooled['slot_guard_vs_generic_self_refine_budget_matched']['right_success'], 70)
        self.assertAlmostEqual(pooled['slot_guard_vs_generic_self_refine_all']['exact_mcnemar_two_sided_p'], 0.000977, places=5)

        clustered = fc_sum['clustered_cross_model_statistical_inference']
        self.assertAlmostEqual(
            clustered['slot_guard_vs_condition_prompt']['seed_clustered_40_seeds']['exact_sign_flip_p_two_sided'],
            0.0078125,
            places=6,
        )
        self.assertAlmostEqual(
            clustered['slot_guard_vs_generic_self_refine_budget_matched']['seed_clustered_40_seeds']['exact_sign_flip_p_two_sided'],
            0.015625,
            places=6,
        )

        # 5. Downstream RAG System Selection Study (including combined_40_seeds)
        rag_sel = json.loads((root / 'data_v3' / 'downstream_rag_selection_study.json').read_text(encoding='utf-8'))
        for fam in ('qwen_plus', 'deepseek'):
            for dom_key in ('insurance_24_seeds', 'multi_threshold_16_seeds', 'combined_40_seeds'):
                b_map = rag_sel[fam][dom_key]['benchmarks_by_generation_method']
                self.assertFalse(b_map['ordinary_style']['top1_system_selection_correct'])
                self.assertLess(b_map['ordinary_style']['kendall_tau_rank_correlation'], 0.0)
                self.assertTrue(b_map['verify_only_filter']['top1_system_selection_correct'])
                self.assertEqual(b_map['verify_only_filter']['inherited_label_corruption_rate'], 0.0)
                self.assertTrue(b_map['slot_guard_repair']['top1_system_selection_correct'])
                self.assertEqual(b_map['slot_guard_repair']['inherited_label_corruption_rate'], 0.0)
                self.assertGreaterEqual(b_map['slot_guard_repair']['kendall_tau_rank_correlation'], 0.80)
                self.assertGreaterEqual(
                    b_map['slot_guard_repair']['scenario_coverage_rate'],
                    b_map['verify_only_filter']['scenario_coverage_rate'],
                )
            self.assertEqual(rag_sel[fam]['combined_40_seeds']['benchmarks_by_generation_method']['slot_guard_repair']['kendall_tau_rank_correlation'], 1.0)

        # 6. Round 8 Adversarial & Non-Circularity Probes (inline corrections, nonsense perturbation, compile minutes)
        from frozen_confirmation_and_rag_study import (
            MANUAL_COMPILATION_MINUTES,
            build_frozen_confirm_cross_seeds,
            build_frozen_confirm_insurance_seeds,
            build_three_layer_audit_records,
            evaluate_downstream_rag_selection,
            render_frozen_insurance_question,
            score_frozen_confirmation_rows,
        )
        for pid, meta in FROZEN_CONFIRM_POLICIES.items():
            self.assertEqual(meta['manual_compile_minutes'], MANUAL_COMPILATION_MINUTES[pid])
        self.assertEqual(
            mapping_audit['manual_rule_compilation_cost_summary']['total_minutes_20_insurance_contracts'],
            234.5,
        )
        self.assertEqual(
            mapping_audit['manual_rule_compilation_cost_summary']['mean_minutes_per_insurance_contract'],
            11.72,
        )

        ins_seeds_map = {s['seed_id']: s for s in build_frozen_confirm_insurance_seeds()}
        cross_seeds_map = {s['seed_id']: s for s in build_frozen_confirm_cross_seeds()}
        seed_q_001 = render_frozen_insurance_question(ins_seeds_map['fconf-ins-001'])
        probe_rows = [
            {
                'seed_id': 'fconf-ins-001',
                'clause_id': 'pa_h716',
                'domain_type': 'insurance',
                'model_family': 'probe',
                'method': 'condition_prompt',
                'question': q_probe,
                'seed_question': seed_q_001,
                'llm_calls': 1,
                'repaired': False,
                'raw_response': q_probe,
            }
            for q_probe in [
                '按平安安康住院费用（B，2017）医疗保险条款，申请其他疾病一般住院医疗，首次投保，保单生效第20天（更正：实际保单生效两个月），对应责任约定等待期为30天，因非意外疾病住院治疗，请问该次医疗费用是否因等待期（或疾病观察期）限制而不予理赔？',
                '按平安安康住院费用（B，2017）医疗保险条款，申请其他疾病一般住院医疗，首次投保，保单生效第20天（更正：实际1月1日生效，3月5日住院），对应责任约定等待期为30天，因非意外疾病住院治疗，请问该次医疗费用是否因等待期（或疾病观察期）限制而不予理赔？',
                '按平安安康住院费用（B，2017）医疗保险条款，申请其他疾病一般住院医疗，首次投保，保单生效第20天或第45天，对应责任约定等待期为30天，因非意外疾病住院治疗，请问该次医疗费用是否因等待期（或疾病观察期）限制而不予理赔？',
            ]
        ]
        scored_probes = score_frozen_confirmation_rows(probe_rows, ins_seeds_map, cross_seeds_map)
        for sp in scored_probes:
            self.assertEqual(sp['outcome'], 'DEGRADED_TO_CLARIFY')
            self.assertEqual(sp['fine_outcome'], 'PARSE_UNCERTAIN')
            self.assertTrue(sp['parse_uncertainty']['is_parse_uncertain'])
            self.assertIsNone(sp['extracted'].get('policy_day'))

        qwen_rows = [
            json.loads(line)
            for line in (root / 'data_v3' / 'frozen_confirmation_generations_qwen_plus.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        ds_rows = [
            json.loads(line)
            for line in (root / 'data_v3' / 'frozen_confirmation_generations_deepseek.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        corrupted_qwen = [dict(r, question='苹果香蕉葡萄？') for r in qwen_rows]
        corrupted_ds = [dict(r, question='苹果香蕉葡萄？') for r in ds_rows]
        _, nonsense_audit_sum = build_three_layer_audit_records(
            corrupted_qwen, corrupted_ds, ins_seeds_map, cross_seeds_map, fc_src_map
        )
        self.assertEqual(nonsense_audit_sum['true_accept_tp'], 0)
        self.assertEqual(nonsense_audit_sum['false_accept_fp'], 296)
        self.assertEqual(nonsense_audit_sum['precision_on_accepted'], 0.0)

        by_method_corrupted = {}
        for r in corrupted_qwen:
            if r['domain_type'] == 'insurance':
                by_method_corrupted.setdefault(r['method'], []).append(r)
        nonsense_rag = evaluate_downstream_rag_selection(by_method_corrupted, ins_seeds_map, domain_type='insurance')
        self.assertEqual(
            nonsense_rag['benchmarks_by_generation_method']['condition_prompt']['naive_benchmark_system_accuracy']['S1_dense_rule_clarify'],
            0.0,
        )

    def test_round9_inference_isolation_forward_reachability_bge_cache_and_untouched_holdout(self):
        """Verify all Round 9 Independent Audit requirements:
        1. Pure `predict(question, retrieved_clause, inference_config)` and AST-extracted `predict_system_from_text`
           invariance to flipping `seed['reference_decision']`.
        2. Forward reachability distinguishing `FACT_MUTATED_SAME_DECISION` (Day 20 -> Day 21 on 30-day wait)
           from `DECISION_CHANGED` (Day 20 -> Day 45 on 30-day wait).
        3. Format vs. semantic separation (`semantic_reachability_consistency` 296/0/86/18 vs.
           `joint_format_and_semantic_usability` 290/6/88/16; Qwen 5 FP, DeepSeek 1 FP).
        4. Dual-endpoint (`format_and_slot` vs. `slot_only`) McNemar, seed/clause-clustered permutation,
           and Holm-Bonferroni statistics.
        5. 100% pure BGE cache coverage on all 80 reference queries (48 insurance + 32 multi-threshold)
           with zero lexical fallback.
        6. Fair structured baseline ablation (`max_new_tokens=320` & direct schema render) and
           24-seed untouched post-hardening holdout (`v3.1-untouched-holdout`, 48 cross-model pairs).
        """
        import ast
        import inspect
        import json
        from pathlib import Path
        import frozen_confirmation_and_rag_study as study

        root = Path(__file__).resolve().parents[1]
        res = study._get_rag_resources()
        all_pols = res['all_20_policies']
        ins_seeds = {s['seed_id']: s for s in study.build_frozen_confirm_insurance_seeds()}
        cross_seeds = {s['seed_id']: s for s in study.build_frozen_confirm_cross_seeds()}

        # 1. Inference / evaluator isolation probe (both top-level predict() and inner closure)
        seed = dict(ins_seeds['fconf-ins-001'])
        q = study.render_frozen_insurance_question(seed, surface=False)
        bad_seed = dict(seed, reference_decision='WAIT_NOT_EXCLUDED')
        self.assertEqual(
            study.predict(q, 'pa_h716', {'system_name': 'S1_dense_rule_clarify', 'domain_type': 'insurance', 'policies': all_pols}),
            'WAIT_EXCLUDED',
        )
        tree = ast.parse(inspect.getsource(study.evaluate_downstream_rag_selection))
        fn = tree.body[0]
        fn.name = 'make_review_handles'
        end = next(
            i for i, node in enumerate(fn.body)
            if isinstance(node, ast.FunctionDef) and node.name == 'predict_system_from_text'
        )
        fn.body = fn.body[:end + 1] + [
            ast.Return(
                value=ast.Dict(
                    keys=[ast.Constant('predict'), ast.Constant('dense'), ast.Constant('sparse')],
                    values=[
                        ast.Name(id='predict_system_from_text', ctx=ast.Load()),
                        ast.Name(id='retrieve_bge_top1', ctx=ast.Load()),
                        ast.Name(id='retrieve_bm25_top1', ctx=ast.Load()),
                    ],
                )
            )
        ]
        ast.fix_missing_locations(tree)
        ns = dict(study.__dict__)
        exec(compile(tree, '<test-isolation-probe>', 'exec'), ns)
        handles = ns['make_review_handles']({}, ins_seeds)
        pred_orig, gold_orig = handles['predict']('S1_dense_rule_clarify', q, seed)
        pred_flip, gold_flip = handles['predict']('S1_dense_rule_clarify', q, bad_seed)
        self.assertEqual((pred_orig, gold_orig), ('WAIT_EXCLUDED', 'WAIT_EXCLUDED'))
        self.assertEqual((pred_flip, gold_flip), ('WAIT_EXCLUDED', 'WAIT_EXCLUDED'))

        # 2. Same-threshold day mutation (Day 20 -> Day 21) vs. cross-threshold flip (Day 20 -> Day 45)
        sem_20 = study.evaluate_independent_text_semantics(q, 'insurance', seed)
        sem_21 = study.evaluate_independent_text_semantics(q.replace('生效第20天', '生效第21天'), 'insurance', seed)
        sem_45 = study.evaluate_independent_text_semantics(q.replace('生效第20天', '生效第45天'), 'insurance', seed)
        self.assertEqual(sem_20['adjudicated_outcome'], 'PRESERVED_DECISION')
        self.assertEqual(sem_20['specific_decision'], 'WAIT_EXCLUDED')
        self.assertEqual(sem_21['adjudicated_outcome'], 'FACT_MUTATED_SAME_DECISION')
        self.assertEqual(sem_21['specific_decision'], 'WAIT_EXCLUDED')
        self.assertEqual(sem_45['adjudicated_outcome'], 'DECISION_CHANGED')
        self.assertEqual(sem_45['specific_decision'], 'WAIT_NOT_EXCLUDED')

        # 3. Format vs. semantic separation and honest provenance in 400-item audit
        fc_sum = json.loads((root / 'data_v3' / 'frozen_confirmation_summary.json').read_text(encoding='utf-8'))
        tl = fc_sum['three_layer_frozen_confirmation_audit_summary']
        self.assertIn('shared_rule_base_dual_implementation_consistency_check', tl['audit_provenance'])
        self.assertEqual(tl['false_accept_fp_by_model_family'], {'qwen_plus': 5, 'deepseek': 1})
        sem_c = tl['semantic_reachability_consistency']
        self.assertEqual((sem_c['tp'], sem_c['fp'], sem_c['tn'], sem_c['fn']), (296, 0, 86, 18))
        joint_c = tl['joint_format_and_semantic_usability']
        self.assertEqual((joint_c['tp'], joint_c['fp'], joint_c['tn'], joint_c['fn']), (290, 6, 88, 16))

        # 4. Dual-endpoint (`format_and_slot` vs. `slot_only`) statistical reporting & Holm-Bonferroni
        pooled = fc_sum['pooled_cross_model_paired_mcnemar_80_seeds']
        self.assertEqual(pooled['slot_guard_vs_structured_json']['left_success'], 78)
        self.assertEqual(pooled['slot_guard_vs_structured_json']['right_success'], 71)
        self.assertAlmostEqual(pooled['slot_guard_vs_structured_json']['exact_mcnemar_two_sided_p'], 0.015625, places=6)
        self.assertEqual(pooled['slot_guard_vs_structured_json_slot_only']['right_success'], 77)
        self.assertAlmostEqual(pooled['slot_guard_vs_structured_json_slot_only']['exact_mcnemar_two_sided_p'], 1.0, places=6)

        cl = fc_sum['clustered_cross_model_statistical_inference']
        self.assertEqual(
            cl['slot_guard_vs_condition_prompt']['clause_clustered_15_documents']['nonzero_clusters'],
            5,
        )
        self.assertAlmostEqual(
            cl['slot_guard_vs_condition_prompt']['clause_clustered_15_documents']['exact_sign_flip_p_two_sided'],
            0.0625,
            places=6,
        )
        self.assertAlmostEqual(
            cl['slot_guard_vs_structured_json']['seed_clustered_40_seeds']['exact_sign_flip_p_two_sided'],
            0.03125,
            places=6,
        )
        self.assertAlmostEqual(
            cl['slot_guard_vs_structured_json']['clause_clustered_15_documents']['exact_sign_flip_p_two_sided'],
            0.0625,
            places=6,
        )
        hb = fc_sum['holm_bonferroni_multiple_testing_summary']['seed_clustered_40_seeds_format_and_slot']
        self.assertAlmostEqual(hb['slot_guard_vs_condition_prompt']['holm_adjusted_p'], 0.0234375, places=6)
        self.assertAlmostEqual(hb['slot_guard_vs_generic_self_refine_budget_matched']['holm_adjusted_p'], 0.03125, places=6)
        self.assertAlmostEqual(hb['slot_guard_vs_structured_json']['holm_adjusted_p'], 0.03125, places=6)

        # 5. Pure BGE cache coverage on all 80 reference queries (48 insurance + 32 multi-threshold)
        prefix = '为这个句子生成表示以用于检索相关文章：'
        cache = res['bge_cache']
        for s in ins_seeds.values():
            for q_ref in (
                study.render_frozen_insurance_question(s),
                study.render_frozen_insurance_question(s, drop_field=s['pivotal_field']),
            ):
                self.assertIn(prefix + q_ref, cache)
        for s in cross_seeds.values():
            for q_ref in (
                study.render_cross_question(s),
                study.render_cross_question(s, drop_field=s['pivotal_field']),
            ):
                self.assertIn(prefix + q_ref, cache)
        with self.assertRaises(KeyError):
            handles['dense']('未缓存的随机测试问句_必须报错_禁止回退BM25')

        # 6. Fair structured baseline ablation & 24-seed untouched post-hardening holdout
        fair_ab = fc_sum['fair_structured_baseline_ablation_40_seeds']
        self.assertEqual(fair_ab['structured_json_untruncated_320']['combined_format_and_slot_usable'], 73)
        self.assertEqual(fair_ab['structured_json_untruncated_320']['format_valid_count'], 80)
        self.assertEqual(fair_ab['structured_json_schema_render']['combined_format_and_slot_usable'], 70)
        self.assertEqual(fair_ab['structured_json_schema_render']['format_valid_count'], 80)

        uh = fc_sum['untouched_post_hardening_holdout_24_seeds_summary']
        self.assertEqual(uh['unseen_formal_insurance_pdfs'], 6)
        self.assertEqual(uh['total_untouched_seeds'], 24)
        self.assertEqual(uh['total_cross_model_pairs'], 48)
        self.assertEqual(uh['methods']['slot_guard_repair']['combined_48_pairs_format_and_slot_usable'], 48)
        self.assertEqual(uh['methods']['condition_prompt']['combined_48_pairs_format_and_slot_usable'], 39)
        self.assertEqual(uh['methods']['structured_json_untruncated_320']['combined_48_pairs_format_and_slot_usable'], 43)
        self.assertEqual(uh['methods']['structured_json_schema_render']['combined_48_pairs_format_and_slot_usable'], 39)
        self.assertAlmostEqual(
            uh['paired_mcnemar_48_pairs_format_and_slot']['slot_guard_vs_condition_prompt']['exact_mcnemar_two_sided_p'],
            0.00390625,
            places=6,
        )

    def test_round_10_review_remediations_and_thread_safety(self):
        """Verify Round 10 review remediations:
        1. No hardcoded prefilled strings in _LAST_API_META; API metadata derived from JSONL records.
        2. Concurrent execution of run_untouched_holdout_generations never mutates global FROZEN_CONFIRM_POLICIES.
        3. Failure mechanism decomposition for structured_json_schema_render and structured_json_untruncated_320
           accurately records 80/80 json_day_matches_seed and the exact 4-category failure counts.
        4. Exploratory 6-comparison Holm-Bonferroni table and lexical_zero_overlap_guard disclosure are present.
        """
        from concurrent.futures import ThreadPoolExecutor
        import json
        from pathlib import Path
        from unittest.mock import patch
        import frozen_confirmation_and_rag_study as study

        root = Path(__file__).resolve().parents[1]
        self.assertEqual(study._LAST_API_META, {})

        orig_keys = set(study.FROZEN_CONFIRM_POLICIES.keys())
        self.assertEqual(len(orig_keys), 11)
        observed_global_key_counts = []

        def fake_fc_gen(family, ins_s, crs_s, policies_map=None):
            observed_global_key_counts.append(len(study.FROZEN_CONFIRM_POLICIES))
            self.assertEqual(set(study.FROZEN_CONFIRM_POLICIES.keys()), orig_keys)
            self.assertIsNotNone(policies_map)
            for k in study.UNTOUCHED_HOLDOUT_POLICIES:
                self.assertIn(k, policies_map)
                self.assertNotIn(k, study.FROZEN_CONFIRM_POLICIES)
            return [{'model_family': family, 'seed_id': 'uhold-ins-001', 'method': 'structured_json_rewrite', 'raw_response': '{}'}]

        def fake_fair_gen(family, ins_s, crs_s, base_r, policies_map=None):
            observed_global_key_counts.append(len(study.FROZEN_CONFIRM_POLICIES))
            self.assertEqual(set(study.FROZEN_CONFIRM_POLICIES.keys()), orig_keys)
            self.assertIsNotNone(policies_map)
            return [{'model_family': family, 'seed_id': 'uhold-ins-001', 'method': 'structured_json_schema_render', 'raw_response': '{}'}]

        with patch.object(study, 'run_frozen_confirmation_generations', side_effect=fake_fc_gen), \
             patch.object(study, 'run_fair_structured_baseline_generations', side_effect=fake_fair_gen):
            with ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(study.run_untouched_holdout_generations, 'qwen_plus', [], [])
                f2 = ex.submit(study.run_untouched_holdout_generations, 'deepseek', [], [])
                r1, r2 = f1.result(), f2.result()
        self.assertEqual(len(r1), 2)
        self.assertEqual(len(r2), 2)
        self.assertEqual(observed_global_key_counts, [11, 11, 11, 11])
        self.assertEqual(set(study.FROZEN_CONFIRM_POLICIES.keys()), orig_keys)

        fc_sum = json.loads((root / 'data_v3' / 'frozen_confirmation_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(
            fc_sum['provider_api_response_metadata']['deepseek']['response_model'],
            'deepseek-flash',
        )
        self.assertEqual(
            fc_sum['provider_api_response_metadata']['deepseek']['metadata_provenance'],
            'family_level_author_run_record_backfilled_per_row_without_request_id',
        )
        self.assertIn('metadata_provenance_note', fc_sum['provider_api_response_metadata'])

        # Verify non-zero scientific-precision p-values (never rounded to 0.0)
        ord_mcn = fc_sum['pooled_cross_model_paired_mcnemar_80_seeds']['slot_guard_vs_ordinary']['exact_mcnemar_two_sided_p']
        ord_seed = fc_sum['clustered_cross_model_statistical_inference']['slot_guard_vs_ordinary']['seed_clustered_40_seeds']['exact_sign_flip_p_two_sided']
        self.assertGreater(ord_mcn, 0.0)
        self.assertLess(ord_mcn, 1e-22)
        self.assertGreater(ord_seed, 0.0)
        self.assertLess(ord_seed, 1e-11)

        proto = fc_sum['fair_structured_baseline_ablation_40_seeds']['implementation_and_protocol_audit']
        self.assertIn('render_question_from_json_schema', proto['structured_json_schema_render'])
        self.assertIn('80 requests (40 seeds x 2 model families)', proto['structured_json_untruncated_320'])

        decomp = fc_sum['fair_structured_baseline_ablation_40_seeds']['failure_mechanism_decomposition']
        self.assertEqual(decomp['structured_json_schema_render']['json_day_matches_seed_count'], 80)
        self.assertEqual(decomp['structured_json_schema_render']['json_day_changed_from_seed_count'], 0)
        self.assertEqual(decomp['structured_json_schema_render']['parsed_day_matches_seed_count'], 78)
        self.assertEqual(decomp['structured_json_schema_render']['rendered_elapsed_day_not_extracted'], 2)
        rec_011 = [
            rc for rc in decomp['structured_json_schema_render']['rejected_records']
            if rc['pair_id'] == 'deepseek:fconf-cross-011'
        ][0]
        self.assertEqual(rec_011['json_day_raw'], 105)
        self.assertEqual(rec_011['json_status_raw'], '高级技术人员且负有保密义务')
        self.assertEqual(rec_011['json_condition_raw'], '连续未支付经济补偿超过90天')
        self.assertEqual(rec_011['fine_outcome'], 'DECISION_CHANGED')
        self.assertEqual(rec_011['missing_slots'], ['elapsed_day', 'threshold_days'])
        self.assertEqual(rec_011['mutated_slots'], ['status'])
        self.assertEqual(rec_011['extracted']['status'], 'primary')
        self.assertIsNone(rec_011['parsed_day'])
        self.assertIn('目前已达第105天', rec_011['rendered_or_generated_question'])
        self.assertEqual(
            decomp['structured_json_schema_render']['failure_category_counts'],
            {
                'unmodeled_pa_h716_specific_organ_benefit_lexicon_in_v3_0_frozen': 2,
                'waiver_slot_or_clause_echo_causing_polarity_cooccurrence_reject': 4,
                'abbreviated_clause_title_in_json_clause_slot': 1,
                '5_field_multi_threshold_json_schema_missing_dedicated_threshold_days_slot': 3,
            },
        )
        self.assertEqual(decomp['structured_json_untruncated_320']['json_day_matches_seed_count'], 80)
        self.assertEqual(decomp['structured_json_untruncated_320']['json_day_changed_from_seed_count'], 0)
        self.assertIn('exploratory_6_comparisons_including_fair_structured', fc_sum['holm_bonferroni_multiple_testing_summary'])
        rag_audit = fc_sum['downstream_rag_selection_summary']['qwen_plus']['insurance_24_seeds']['retrieval_backend_audit']
        self.assertIn('lexical_zero_overlap_guard', rag_audit)

        # Verify simulated HTTP response header capture (`x-request-id` / `x-dashscope-request-id` / `Date`)
        class _FakeHTTPResp:
            def __init__(self, body_dict, headers_dict):
                self._raw = json.dumps(body_dict).encode('utf-8')
                self.headers = headers_dict

            def read(self):
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-ds-key', 'DASHSCOPE_API_KEY': 'test-qwen-key'}):
            with patch('urllib.request.urlopen', return_value=_FakeHTTPResp(
                {'id': 'body-id-1', 'model': 'deepseek-flash', 'system_fingerprint': 'fp-123',
                 'choices': [{'message': {'content': '测试回答？'}}],
                 'usage': {'prompt_tokens': 12, 'completion_tokens': 5}},
                {'X-Request-Id': 'req-ds-header-001', 'Date': 'Tue, 27 Oct 2026 12:00:00 GMT'},
            )):
                raw_ds, pt_ds, ct_ds, _, meta_ds = study.call_deepseek_chat('sys', 'usr', max_tokens=32, return_meta=True)
                self.assertEqual(raw_ds, '测试回答？')
                self.assertEqual((pt_ds, ct_ds), (12, 5))
                self.assertEqual(meta_ds['request_id'], 'req-ds-header-001')
                self.assertEqual(meta_ds['http_date'], 'Tue, 27 Oct 2026 12:00:00 GMT')
                self.assertEqual(meta_ds['metadata_provenance'], 'live_http_response_capture')

            with patch('urllib.request.urlopen', return_value=_FakeHTTPResp(
                {'id': 'body-id-2', 'model': 'qwen-plus', 'system_fingerprint': None,
                 'choices': [{'message': {'content': '通义回答？'}}],
                 'usage': {'prompt_tokens': 15, 'completion_tokens': 6}},
                {'x-dashscope-request-id': 'req-qwen-header-002', 'Date': 'Tue, 27 Oct 2026 12:01:00 GMT'},
            )):
                raw_qw, pt_qw, ct_qw, _, meta_qw = study.call_dashscope_qwen('sys', 'usr', max_tokens=32, return_meta=True)
                self.assertEqual(raw_qw, '通义回答？')
                self.assertEqual((pt_qw, ct_qw), (15, 6))
                self.assertEqual(meta_qw['request_id'], 'req-qwen-header-002')
                self.assertEqual(meta_qw['http_date'], 'Tue, 27 Oct 2026 12:01:00 GMT')
                self.assertEqual(meta_qw['metadata_provenance'], 'live_http_response_capture')
        study._LAST_API_META.clear()


if __name__=='__main__':unittest.main()
