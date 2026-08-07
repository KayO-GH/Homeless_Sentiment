"""Tests for the word-cloud workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "04_word_clouds.py"
SPEC = importlib.util.spec_from_file_location("word_clouds", MODULE_PATH)
assert SPEC and SPEC.loader
word_clouds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(word_clouds)


STOPWORDS = {"the", "and", "le", "et", "dans", "la"}


class WordCloudTests(unittest.TestCase):
    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_cleaning_removes_stopwords_and_keywords(self, _mock_stopwords: object) -> None:
        frequencies = word_clouds.word_frequencies(
            "The homelessness shelters encampments were sleeping rough dans le logements et Sans-Abri. "
            "Homes remain. https://example.com/path www.example.org/page"
        )
        self.assertEqual(frequencies, {"were": 1, "homes": 1, "remain": 1})

    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_create_wordcloud_accepts_string_and_list(self, _mock_stopwords: object) -> None:
        with self.subTest("string"):
            with self._temporary_directory() as directory:
                output = word_clouds.create_wordcloud("alpha beta", Path(directory) / "string.png", "String")
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 0)
        with self.subTest("list"):
            with self._temporary_directory() as directory:
                output = word_clouds.create_wordcloud(["alpha", "beta"], Path(directory) / "list.png", "List")
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 0)

    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_generate_expected_groups(self, _mock_stopwords: object) -> None:
        rows = [
            {"text": "alpha", "narrative": "Housing Crisis"},
            {"text": "beta", "narrative": "Public Life Crisis"},
            {"text": "gamma", "narrative": "(Society) Moral Crisis"},
            {"text": "delta", "narrative": "Governance and Policy Challenge"},
            {"text": "excluded", "narrative": "Unclear or mixed"},
            {"text": "irrelevant", "narrative": "Not relevant to homelessness"},
        ]
        with self._temporary_directory() as directory:
            input_path = Path(directory) / "input.csv"
            output_dir = Path(directory) / "clouds"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            outputs = word_clouds.generate_wordclouds(input_path, output_dir)
            self.assertEqual(len(outputs), 5)
            self.assertEqual({path.name for path in outputs}, {"all_narratives.png", "housing_crisis.png", "public_life_crisis.png", "society_moral_crisis.png", "governance_and_policy_challenge.png"})
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))

    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_empty_words_and_missing_columns_fail_clearly(self, _mock_stopwords: object) -> None:
        with self.assertRaisesRegex(ValueError, "No usable words"):
            word_clouds.create_wordcloud("homeless sleeping rough", Path("unused.png"), "Empty")
        with self._temporary_directory() as directory:
            input_path = Path(directory) / "missing.csv"
            pd.DataFrame({"text": ["alpha"]}).to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "narrative"):
                word_clouds.generate_wordclouds(input_path, Path(directory) / "clouds")

    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_missing_narrative_group_fails_clearly(self, _mock_stopwords: object) -> None:
        with self._temporary_directory() as directory:
            input_path = Path(directory) / "partial.csv"
            pd.DataFrame(
                [{"text": "alpha", "narrative": "Housing Crisis"}]
            ).to_csv(input_path, index=False)
            with self.assertRaisesRegex(ValueError, "Public Life Crisis"):
                word_clouds.generate_wordclouds(input_path, Path(directory) / "clouds")

    def _temporary_directory(self):
        from tempfile import TemporaryDirectory

        return TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
