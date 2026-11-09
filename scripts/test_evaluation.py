"""Focused checks for held-out answer normalization."""

from types import SimpleNamespace
import unittest

import torch

from evaluate_model import normalize_final_answer, score_language_model


class NormalizeFinalAnswerTests(unittest.TestCase):
    def test_uses_final_answer_label(self) -> None:
        self.assertEqual(
            normalize_final_answer("Work: 2 + 3 = 5\n\nFinal answer: 5."),
            "5",
        )

    def test_normalizes_commas_and_decimals(self) -> None:
        self.assertEqual(normalize_final_answer("Final answer: 370,000"), "3.7E+5")
        self.assertEqual(normalize_final_answer("Answer: 4.0"), "4")

    def test_normalizes_equivalent_fractions(self) -> None:
        self.assertEqual(normalize_final_answer("Final answer: 3/4"), "0.75")

    def test_uses_last_final_answer_if_repeated(self) -> None:
        self.assertEqual(
            normalize_final_answer("Final answer: 8\nActually, Final answer: 9"),
            "9",
        )

    def test_empty_generation_is_not_an_error(self) -> None:
        self.assertEqual(normalize_final_answer(""), "")


class ScoreLanguageModelTests(unittest.TestCase):
    def test_loss_is_weighted_by_target_token_count(self) -> None:
        class FixedLossModel:
            def eval(self) -> None:
                return None

            def __call__(self, input_ids, **_kwargs):
                loss = 2.0 if int(input_ids[0, 0]) == 1 else 4.0
                return SimpleNamespace(loss=torch.tensor(loss))

        examples = [
            ("short", {"input_ids": [1, 2], "labels": [-100, 2]}),
            ("long", {"input_ids": [3, 4, 5, 6], "labels": [-100, 4, 5, 6]}),
        ]
        report = score_language_model(FixedLossModel(), examples, pad_token_id=0)
        self.assertEqual(report["target_tokens"], 4)
        self.assertEqual(report["mean_token_nll"], 3.5)


if __name__ == "__main__":
    unittest.main()
