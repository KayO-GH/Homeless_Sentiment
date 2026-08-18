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


class FakeToken:
    def __init__(self, text: str, lemma: str, is_alpha: bool = True) -> None:
        self.text = text
        self.lemma_ = lemma
        self.is_alpha = is_alpha
        self.is_space = False


class FakePipeline:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def __call__(self, text: str) -> list[FakeToken]:
        return [
            FakeToken(token, self.mapping.get(token, token))
            for token in text.split()
        ]


class FakeDetection:
    def __init__(self, language: str, start_index: int, end_index: int) -> None:
        self.language = language
        self.start_index = start_index
        self.end_index = end_index


class FakeConfidence:
    def __init__(self, language: object, value: float) -> None:
        self.language = language
        self.value = value


class FakeDetector:
    def __init__(
        self,
        detections: list[FakeDetection],
        confidence_values: dict[str, list[FakeConfidence]] | None = None,
    ) -> None:
        self.detections = detections
        self.confidence_values = confidence_values or {}

    def detect_multiple_languages_of(self, _text: str) -> list[FakeDetection]:
        return self.detections

    def compute_language_confidence_values(self, text: str) -> list[FakeConfidence]:
        return self.confidence_values.get(text, [])


class WordCloudTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lemma_patcher = patch.object(
            word_clouds,
            "lemmatized_tokens",
            side_effect=lambda text: word_clouds.TOKEN_PATTERN.findall(text.casefold()),
        )
        self.lemma_patcher.start()

    def tearDown(self) -> None:
        self.lemma_patcher.stop()

    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_cleaning_removes_stopwords_and_keywords(self, _mock_stopwords: object) -> None:
        frequencies = word_clouds.word_frequencies(
            "The homelessness shelters encampments were sleeping rough dans le logements et Sans-Abri. "
            "Homes remain. https://example.com/path www.example.org/page"
        )
        self.assertEqual(frequencies, {"were": 1, "homes": 1, "remain": 1})

    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_cleaning_removes_words_from_config(self, _mock_stopwords: object) -> None:
        frequencies = word_clouds.word_frequencies(
            "People person persons useful words être avoir aller okay ok"
        )
        self.assertEqual(frequencies, {"useful": 1, "words": 1})

    @patch.object(word_clouds, "get_stopwords", return_value=STOPWORDS)
    def test_cleaning_removes_geography_words_from_config(self, _mock_stopwords: object) -> None:
        frequencies = word_clouds.word_frequencies(
            "Canada Canadian Toronto Montréal Montreal Vancouver Hamilton Ottawa Calgary "
            "Victoria Winnipeg Edmonton Halifax London Quebec Québec Sudbury Ontario city cities useful"
        )
        self.assertEqual(frequencies, {"useful": 1})

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

    def test_lemmatizes_english_and_french_spans(self) -> None:
        self.lemma_patcher.stop()
        from lingua import Language

        english = "English words"
        french = "mots français"
        text = english + " " + french
        detector = FakeDetector(
            [
                FakeDetection(Language.ENGLISH, 0, len(english)),
                FakeDetection(Language.FRENCH, len(english) + 1, len(text)),
            ]
        )
        pipelines = {
            Language.ENGLISH: FakePipeline({"words": "word"}),
            Language.FRENCH: FakePipeline({"mots": "mot", "français": "français"}),
        }
        with patch.object(
            word_clouds, "get_language_detector", return_value=detector
        ), patch.object(
            word_clouds,
            "_get_language_pipeline",
            side_effect=lambda language: pipelines[language],
        ):
            frequencies = word_clouds.lemmatized_tokens(text)

        self.assertEqual(frequencies, ["english", "word", "mot", "français"])

    def test_excludes_confident_non_english_french_span(self) -> None:
        self.lemma_patcher.stop()
        from lingua import Language

        english = "English words"
        spanish = "hola mundo"
        french = "mots français"
        text = f"{english} {spanish} {french}"
        detector = FakeDetector(
            [
                FakeDetection(Language.ENGLISH, 0, len(english)),
                FakeDetection(
                    Language.SPANISH,
                    len(english) + 1,
                    len(english) + 1 + len(spanish),
                ),
                FakeDetection(Language.FRENCH, len(english) + len(spanish) + 2, len(text)),
            ],
            {
                spanish: [
                    FakeConfidence(Language.SPANISH, 0.90),
                    FakeConfidence(Language.ENGLISH, 0.05),
                    FakeConfidence(Language.FRENCH, 0.05),
                ]
            },
        )
        pipelines = {
            Language.ENGLISH: FakePipeline({"words": "word"}),
            Language.FRENCH: FakePipeline({"mots": "mot", "français": "français"}),
        }
        with patch.object(
            word_clouds, "get_language_detector", return_value=detector
        ), patch.object(
            word_clouds,
            "_get_language_pipeline",
            side_effect=lambda language: pipelines[language],
        ):
            tokens = word_clouds.lemmatized_tokens(text)

        self.assertEqual(tokens, ["english", "word", "mot", "français"])

    def test_retains_ambiguous_non_english_french_span(self) -> None:
        self.lemma_patcher.stop()
        from lingua import Language

        spanish = "hola mundo"
        detector = FakeDetector(
            [FakeDetection(Language.SPANISH, 0, len(spanish))],
            {
                spanish: [
                    FakeConfidence(Language.SPANISH, 0.75),
                    FakeConfidence(Language.ENGLISH, 0.15),
                    FakeConfidence(Language.FRENCH, 0.10),
                ]
            },
        )
        with patch.object(word_clouds, "get_language_detector", return_value=detector):
            tokens = word_clouds.lemmatized_tokens(spanish)

        self.assertEqual(tokens, ["hola", "mundo"])

    def test_unidentified_span_falls_back_to_normalized_tokens(self) -> None:
        self.lemma_patcher.stop()
        detector = FakeDetector([])
        with patch.object(word_clouds, "get_language_detector", return_value=detector):
            tokens = word_clouds.lemmatized_tokens("Unclear TEXT")
        self.assertEqual(tokens, ["unclear", "text"])

    def _temporary_directory(self):
        from tempfile import TemporaryDirectory

        return TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
