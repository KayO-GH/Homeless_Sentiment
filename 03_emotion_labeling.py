#!/usr/bin/env python3
"""Add the top multi-label emotion and score to a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline


MODEL = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
DEFAULT_INPUT = Path("homelessness_narrative_topic_classification_with_created_utc.csv")


def select_device() -> int | torch.device:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        type=Path,
        help=f"Input CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    output = args.output or args.input.with_name(
        f"{args.input.stem}_with_emotion{args.input.suffix}"
    )
    data = pd.read_csv(args.input)
    if "text" not in data.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    texts = data["text"].fillna("").astype(str).tolist()
    emotions = ["neutral"] * len(texts)
    scores = [0.0] * len(texts)

    classifier = pipeline(
        "text-classification",
        model=MODEL,
        device=select_device(),
        top_k=None,
    )

    nonempty = [(i, text) for i, text in enumerate(texts) if text.strip()]
    for start in range(0, len(nonempty), args.batch_size):
        batch = nonempty[start : start + args.batch_size]
        predictions = classifier(
            [text for _, text in batch],
            truncation=True,
            max_length=512,
        )
        for (row_index, _), prediction in zip(batch, predictions):
            top = max(prediction, key=lambda item: item["score"])
            emotions[row_index] = top["label"]
            scores[row_index] = float(top["score"])

        print(f"Processed {min(start + args.batch_size, len(nonempty))}/{len(nonempty)} posts", flush=True)

    data["emotion"] = emotions
    data["emotion_score"] = scores
    data.to_csv(output, index=False)
    print(f"Saved {output} with {len(data)} rows.")


if __name__ == "__main__":
    main()
