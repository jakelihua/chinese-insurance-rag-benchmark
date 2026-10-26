import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from condition_data import decision, pivotal_change, seed_frames
from condition_guard import extract_fields, chinese_number
from analyze_v3 import extract_fields_surface_normalized, evaluate_rule_outcome_under_observed


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

    def test_cohens_kappa(self):
        from score_annotations import compute_task_metrics
        keys=['k1','k2','k3','k4']
        mapping={k:{'ref':'yes'} for k in keys}
        map_a={k:{'label':'yes' if i%2==0 else 'no','reason':''} for i,k in enumerate(keys)}
        res_perfect=compute_task_metrics(keys,map_a,map_a,mapping,reference_field='ref')
        self.assertAlmostEqual(res_perfect['cohen_kappa'],1.0)
        self.assertEqual(res_perfect['disagreement_count'],0)


if __name__=='__main__':unittest.main()
