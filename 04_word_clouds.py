#!/usr/bin/env python3
"""Generate narrative-specific word clouds for homelessness-related Reddit posts."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm
from wordcloud import WordCloud


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("homelessness_narrative_topic_classification.csv")
DEFAULT_OUTPUT_DIR = Path("wordclouds")
EXCLUDED_WORDS_PATH = Path(__file__).resolve().parent / "config" / "wordcloud_excluded_words.txt"
GEOGRAPHY_EXCLUDED_WORDS_PATH = (
    Path(__file__).resolve().parent / "config" / "wordcloud_geography_excluded_words.txt"
)

EXCLUDED_NARRATIVES = {
    "Unclear or mixed",
    "Not relevant to homelessness",
}

NARRATIVE_OUTPUTS = (
    ("Housing Crisis", "housing_crisis.png"),
    ("Public Life Crisis", "public_life_crisis.png"),
    ("(Society) Moral Crisis", "society_moral_crisis.png"),
    ("Governance and Policy Challenge", "governance_and_policy_challenge.png"),
)

EXCLUDED_KEYWORDS = (
    "homeless",
    "homelessness",
    "tent",
    "tents",
    "shelter",
    "shelters",
    "sleeping rough",
    "unhoused",
    "encampment",
    "encampments",
    "sans-abri",
    "itinerance",
    "itinerant",
    "itinerants",
    "campement",
    "campements",
    "tente",
    "tentes",
    "abri",
    "abris",
    "refuge",
    "refuges",
    "dormir dehors",
    "vivre dans la rue",
    "crise du logement",
    "logement",
    "logements",
)


def get_stopwords() -> set[str]:
    """Return English and French NLTK stopwords, downloading the corpus if needed."""
    try:
        return set(stopwords.words("english")) | set(stopwords.words("french"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        try:
            return set(stopwords.words("english")) | set(stopwords.words("french"))
        except LookupError as error:
            raise RuntimeError(
                "NLTK stopwords could not be downloaded. Run "
                "`uv run python -m nltk.downloader stopwords` and try again."
            ) from error


def _read_excluded_words(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as config_file:
        return {
            line.strip().casefold()
            for line in config_file
            if line.strip() and not line.lstrip().startswith("#")
        }


def get_excluded_words() -> set[str]:
    """Read general word-cloud exclusions from the committed config file."""
    return _read_excluded_words(EXCLUDED_WORDS_PATH)


def get_geography_excluded_words() -> set[str]:
    """Read geography-specific exclusions from the committed config file."""
    return _read_excluded_words(GEOGRAPHY_EXCLUDED_WORDS_PATH)


def _keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    alternatives = []
    for keyword in sorted(keywords, key=len, reverse=True):
        escaped = re.escape(keyword).replace(r"\ ", r"\s+")
        alternatives.append(escaped)
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


KEYWORD_PATTERN = _keyword_pattern(EXCLUDED_KEYWORDS)
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.UNICODE)
NON_TARGET_LANGUAGE_CONFIDENCE = 0.80
NON_TARGET_LANGUAGE_MARGIN = 0.20
_LANGUAGE_DETECTOR: Any | None = None
_LANGUAGE_PIPELINES: dict[str, Any] = {}


def get_language_detector() -> Any:
    """Return a reusable Lingua detector covering every supported language."""
    global _LANGUAGE_DETECTOR
    if _LANGUAGE_DETECTOR is None:
        from lingua import LanguageDetectorBuilder

        _LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
    return _LANGUAGE_DETECTOR


def _target_languages() -> frozenset[Any]:
    """Return the languages retained for word-cloud processing."""
    from lingua import Language

    return frozenset({Language.ENGLISH, Language.FRENCH})


def _is_confident_non_target_language(
    detector: Any, language: Any, text: str
) -> bool:
    """Return whether a span is confidently neither English nor French."""
    target_languages = _target_languages()
    if language in target_languages:
        return False

    confidence_values = detector.compute_language_confidence_values(text)
    if not confidence_values:
        return False

    highest_confidence = max(confidence_values, key=lambda value: value.value)
    if highest_confidence.language in target_languages:
        return False

    confidence_by_language = {
        confidence.language: confidence.value for confidence in confidence_values
    }
    target_confidence = max(
        (confidence_by_language.get(target_language, 0.0) for target_language in target_languages),
        default=0.0,
    )
    return (
        highest_confidence.value >= NON_TARGET_LANGUAGE_CONFIDENCE
        and highest_confidence.value - target_confidence >= NON_TARGET_LANGUAGE_MARGIN
    )


def _language_spans(text: str) -> list[tuple[Any | None, str]]:
    """Split text into retained Lingua-detected spans and unidentified gaps.

    Lingua's mixed-language API is applied per sentence because short
    code-switched sentences can otherwise be overwhelmed by the neighboring
    language in a whole-post detection. Confidently detected non-English/French
    spans are excluded; uncertain spans are retained for fallback tokenization.
    """
    detector = get_language_detector()
    sentences = list(SENTENCE_PATTERN.finditer(text)) or [None]

    spans: list[tuple[Any | None, str]] = []
    for sentence_match in sentences:
        sentence_start = sentence_match.start() if sentence_match else 0
        sentence_end = sentence_match.end() if sentence_match else len(text)
        sentence = text[sentence_start:sentence_end]
        detections = detector.detect_multiple_languages_of(sentence)
        if not detections:
            spans.append((None, sentence))
            continue

        cursor = 0
        for detection in detections:
            start = detection.start_index
            end = detection.end_index
            if start > cursor:
                spans.append((None, sentence[cursor:start]))
            span = sentence[start:end]
            if not _is_confident_non_target_language(detector, detection.language, span):
                language = (
                    detection.language
                    if detection.language in _target_languages()
                    else None
                )
                spans.append((language, span))
            cursor = end
        if cursor < len(sentence):
            spans.append((None, sentence[cursor:]))
    return spans


def _get_language_pipeline(language: Any) -> Any | None:
    """Load the matching spaCy pipeline once, or return None for fallback text."""
    from lingua import Language

    model_by_language = {
        Language.ENGLISH: "en_core_web_sm",
        Language.FRENCH: "fr_core_news_sm",
    }
    model_name = model_by_language.get(language)
    if model_name is None:
        return None
    if model_name not in _LANGUAGE_PIPELINES:
        import spacy

        try:
            _LANGUAGE_PIPELINES[model_name] = spacy.load(model_name)
        except OSError as error:
            raise RuntimeError(
                f"spaCy model '{model_name}' is required. Install it with "
                f"`python -m spacy download {model_name}`."
            ) from error
    return _LANGUAGE_PIPELINES[model_name]


def _fallback_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def _lemmatize_span(language: Any | None, text: str) -> list[str]:
    """Lemmatize a detected span, falling back to normalized tokens if needed."""
    pipeline = _get_language_pipeline(language) if language is not None else None
    if pipeline is None:
        return _fallback_tokens(text)

    lemmas: list[str] = []
    for token in pipeline(text):
        if not token.is_alpha or token.is_space:
            continue
        lemma = token.lemma_.strip().casefold()
        if not lemma or lemma == "-pron-":
            lemma = token.text.casefold()
        lemmas.extend(TOKEN_PATTERN.findall(lemma))
    return lemmas


def lemmatized_tokens(text: str) -> list[str]:
    """Return language-aware lemmas after excluding confident foreign spans."""
    return [
        lemma
        for language, span in _language_spans(text)
        for lemma in _lemmatize_span(language, span)
    ]


def word_frequencies(texts: str | Iterable[str]) -> Counter[str]:
    """Return frequencies after applying the shared word-cloud text cleaning rules."""
    if isinstance(texts, str):
        text_values = [texts]
    else:
        text_values = [str(text) for text in texts if text is not None]

    ignored_words = (
        {word.casefold() for word in get_stopwords()}
        | get_excluded_words()
        | get_geography_excluded_words()
    )
    frequencies: Counter[str] = Counter()
    for text in tqdm(text_values, desc="Processing posts", unit="post"):
        without_urls = URL_PATTERN.sub(" ", text)
        without_keywords = KEYWORD_PATTERN.sub(" ", without_urls)
        frequencies.update(
            token
            for token in lemmatized_tokens(without_keywords)
            if token not in ignored_words
        )
    return frequencies


def create_wordcloud(
    texts: str | Iterable[str], output_path: Path, title: str
) -> Path:
    """Create and save one deterministic word cloud from a string or strings."""
    frequencies = word_frequencies(texts)
    if not frequencies:
        raise ValueError(f"No usable words remain for '{title}'.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = WordCloud(
        background_color="white",
        width=1600,
        height=900,
        max_words=200,
        collocations=False,
        random_state=42,
    ).generate_from_frequencies(frequencies)

    figure, axis = plt.subplots(figsize=(16, 9))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title(title)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_wordclouds(input_path: Path, output_dir: Path) -> list[Path]:
    """Generate aggregate and per-narrative word clouds from a classification CSV."""
    data = pd.read_csv(input_path)
    required_columns = {"text", "narrative"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV must contain columns: {missing}.")

    usable = data[data["text"].notna() & data["text"].astype(str).str.strip().ne("")]
    groups = [("All included narratives", usable[~usable["narrative"].isin(EXCLUDED_NARRATIVES)], "all_narratives.png")]
    groups.extend(
        (narrative, usable[usable["narrative"].eq(narrative)], filename)
        for narrative, filename in NARRATIVE_OUTPUTS
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for title, group, filename in tqdm(groups, desc="Generating clouds", unit="cloud"):
        output_path = create_wordcloud(group["text"].tolist(), output_dir / filename, title)
        saved_paths.append(output_path)
        print(f"Saved {output_path} from {len(group)} posts.")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input CSV (default: {DEFAULT_INPUT})")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    generate_wordclouds(args.input, args.output_dir)


if __name__ == "__main__":
    main()
