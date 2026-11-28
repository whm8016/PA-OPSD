import unittest

from scripts.experiments.teaching_skills.run import route, teacher_pair


def record(qid, day, result='正确', tags='小数, 小数加减'):
    return f'[{qid}]\n练习\n知识点: {tags}\n时间: 2026-11-{day:02d}T08:00:00\n结果: {result}'


class TeachingSkillsTest(unittest.TestCase):
    def test_recall_preserves_original_history_position_and_excludes_current_duplicate(self):
        current = record(1, 15, '错误')
        history = [record(2, 1), record(3, 14, tags='小数'), current]
        decision = route(current, history)
        self.assertEqual(decision['card_id'], 'recall')
        self.assertEqual([item['position'] for item in decision['evidence']], [1])
        self.assertEqual(decision['evidence'][0]['gap_days'], 14)

    def test_transfer_requires_distinct_prior_questions_and_abstains_after_wrong_attempt(self):
        current = record(1, 15, '错误')
        self.assertIsNone(route(current, [record(2, 12), record(2, 13)])['card_id'])
        history = [record(2, 12), record(3, 13)]
        self.assertEqual(route(current, history)['card_id'], 'transfer')
        self.assertIsNone(route(current, history + [record(4, 14, '错误')])['card_id'])
        with_error = route(current, history + [record(4, 14, '错误'), record(3, 14)])
        self.assertEqual(with_error['card_id'], 'transfer')
        self.assertEqual([item['position'] for item in with_error['evidence']], [4, 1, 3])

    def test_future_or_unverifiable_history_cannot_support_a_card(self):
        current = record(1, 15, '错误')
        with self.assertRaisesRegex(ValueError, 'position 2'):
            route(current, [record(2, 1), record(3, 16, tags='无关')])
        unknown = record(3, 13).replace('2026-11-13T08:00:00', '未知')
        self.assertIsNone(route(current, [record(2, 1), unknown])['card_id'])

    def test_probe_arms_share_history_and_exclude_references_and_solution(self):
        row = {'sample_id': 'fixture', 'current': record(1, 15, '错误'),
               'history': [record(2, 12), record(3, 13)],
               'target': {'diagnosis': 'GOLD_CANARY'}, 'solution': {'analysis': 'SOLUTION_CANARY'}}
        decision = route(row['current'], row['history'])
        pair = teacher_pair(row, decision)
        control, treatment = pair['without_card'], pair['with_card']
        self.assertEqual(control[0], treatment[0])
        self.assertEqual(treatment[1]['content'].split('\n\n候选教学卡')[0], control[1]['content'])
        self.assertEqual(pair['history_positions'], [1, 2])
        self.assertIn('请核对 第 2 条距当前 2.00 天', control[1]['content'])
        self.assertNotIn('CANARY', str(pair))


if __name__ == '__main__':
    unittest.main()
