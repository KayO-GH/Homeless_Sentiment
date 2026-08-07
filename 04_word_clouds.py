#!/usr/bin/env python3
"""Generate narrative-specific word clouds for homelessness-related Reddit posts."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("homelessness_narrative_topic_classification.csv")
DEFAULT_OUTPUT_DIR = Path("wordclouds")

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


def _keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    alternatives = []
    for keyword in sorted(keywords, key=len, reverse=True):
        escaped = re.escape(keyword).replace(r"\ ", r"\s+")
        alternatives.append(escaped)
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


KEYWORD_PATTERN = _keyword_pattern(EXCLUDED_KEYWORDS)
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def word_frequencies(texts: str | Iterable[str]) -> Counter[str]:
    """Return frequencies after applying the shared word-cloud text cleaning rules."""
    if isinstance(texts, str):
        source_text = texts
    else:
        source_text = " ".join(str(text) for text in texts if text is not None)

    without_urls = URL_PATTERN.sub(" ", source_text)
    without_keywords = KEYWORD_PATTERN.sub(" ", without_urls.casefold())
    ignored_words = {word.casefold() for word in get_stopwords()}
    return Counter(
        token
        for token in TOKEN_PATTERN.findall(without_keywords)
        if token not in ignored_words
    )


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
    for title, group, filename in groups:
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
