import json
import unittest

from scripts.experiments.teacher_team import run as team


class TeacherTeamTest(unittest.TestCase):
    def test_evidence_ids_only_reference_visible_history_records(self):
        raw = {'diagnosis': 'Recall Failure', 'action': 'C', 'evidence': [1, 2]}
        self.assertEqual(team.decision(json.dumps(raw), 2)['evidence'], [1, 2])
        for invalid in ([True], [1, 1], [0], [3], '1'):
            with self.subTest(invalid=invalid):
                raw['evidence'] = invalid
                result = team.decision(json.dumps(raw), 2)
                self.assertEqual(result['evidence'], [])
                self.assertEqual(result['diagnosis'], 'Recall Failure')

    def test_both_arms_share_context_and_final_role_receives_prior_proposals(self):
        row = {'history': ['only public history'], 'current': 'current task',
               'solution': {'answer': ['5'], 'analysis': 'method'}, 'strategy': 'PRIVATE_GOLD'}
        done = {'diagnoser': {'raw': '{"diagnosis":"Recall Failure","reason":"first"}'},
                'pedagogue': {'raw': '{"action":"C","feedback":"second"}'}}
        messages = {stage: team.messages(row, stage, done) for stage in team.STAGES}
        single = messages['single'][1]['content']
        prefix = single.removesuffix('\n\n' + team.INSTRUCTIONS['single'])
        for stage, prompt in messages.items():
            self.assertEqual(prompt[0], messages['single'][0])
            self.assertTrue(prompt[1]['content'].startswith(prefix))
            self.assertNotIn('PRIVATE_GOLD', prompt[1]['content'])
            if stage == 'single':
                self.assertNotIn('## diagnoser建议', prompt[1]['content'])
        self.assertIn('first', messages['verifier'][1]['content'])
        self.assertIn('second', messages['verifier'][1]['content'])
