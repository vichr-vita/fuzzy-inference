import unittest

from fuzzy_inference.rule import Rule


class RuleTestCase(unittest.TestCase):

    def test_rule(self):
        rule = Rule().IF(
            (
                ('age', 'young'),
                ('wealth', 'rich')
            )
        ).THEN(
            ('risk', 'low')
        )
