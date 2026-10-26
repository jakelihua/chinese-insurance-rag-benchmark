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


if __name__=='__main__':unittest.main()
