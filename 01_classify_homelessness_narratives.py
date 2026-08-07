#!/usr/bin/env python3
"""Classify Reddit posts with a pinned OpenAI model and controlled taxonomy.

Usage:
  export OPENAI_API_KEY='...'
  python 01_classify_homelessness_narratives.py --input "input.csv"

The script saves a resumable JSONL checkpoint and validates the final CSV.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

import pandas as pd

load_dotenv()

MODEL = "gpt-5.4-mini-2026-03-17"
SEED = 20260720 # Date of processing
REQUIRED_COLUMNS = ["post_id", "text", "city", "url"]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["narrative", "specific_topic"]

TOPICS = {
    "Housing Crisis": [
        "Housing affordability and rising rents",
        "Eviction, displacement, and insecure tenancy (i.e., across ownership, rent, co-op etc. setups)",
        "Poverty, unemployment, and income insecurity",
        "Non-market and supportive housing solutions",
    ],
    "Public Life Crisis": [
        "Encampments in public space",
        "Encampment clearance, displacement",
        "Perceived lack of public order, safety",
        "Reported harassment, property damage",
        "Visible substance use, drug-related activity",
        "Neighbourhood change and public-space contestation",
    ],
    "(Society) Moral Crisis": [
        "First-person accounts of homelessness",
        "Requests for assistance due to imminent homelessness",
        "Access to food, hygiene, employment, and other basic needs",
        "Emergency shelter access, shelter conditions, and shelter health and safety",
        "Community aid, outreach, volunteering and direct assistance",
        "Human dignity, compassion, and stigmatization",
        "Health, trauma, and vulnerability",
    ],
    "Governance and Policy Challenge": [
        "Government responsibility, accountability, policy failure, and intergovernmental coordination",
        "Debates over homelessness policy solutions",
        "Public funding and resource allocation",
        "Shelter-system capacity and service delivery",
        "Policing, bylaws, and enforcement",
        "Mental-health, addiction, harm-reduction and rehabilitation responses",
    ],
    "Unclear or mixed": [
        "Multiple narratives with no dominant frame",
        "General discussion of homelessness",
        "Insufficient text",
        "Ambiguous or unclear context",
    ],
    "Not relevant to homelessness": [
        "Animal shelter or animal rescue",
        "Political or campus protest encampment",
        "Camping or recreational tent",
        "Transit or bus shelter",
        "Shelter-in-place or emergency instruction",
        "Physical shelter structure unrelated to homelessness",
        "Festival, event, commercial, or temporary tent",
        "Generic discussion thread with no homelessness content",
        "Keyword used incidentally",
        "Other context unrelated to homelessness",
    ],
}

SYSTEM_PROMPT = """You are a meticulous research classifier for Canadian Reddit posts.
Classify the post's dominant substantive meaning into exactly one permitted specific topic. The narrative will be derived from the selected topic after classification. Use text as primary evidence and city only as contextual evidence. Never use URLs or infer facts absent from text.

Classify framing, not keyword occurrence. Choose the central issue, argument, experience, request, or policy response; use Unclear or mixed sparingly.

Use these theme definitions:
- Housing Crisis: primarily explains homelessness through housing markets, poverty, inequality, income, employment, eviction, inadequate housing supply, or other structural conditions.
- Public Life Crisis: primarily discusses encampments, public-space use, safety, cleanliness, crime, neighbourhood impacts, downtown conditions, or conflicts over shared urban spaces.
- (Society) Moral Crisis: centers shared lived experiences, immediate needs, survival, help-seeking, humanitarian assistance, shelter experiences, empathy, stigma, dignity, or direct community support.
- Governance and Policy Challenge: focuses on what governments, police, health systems, service providers, or public institutions are doing or should do.

When both could apply, classify visible substance use or drug-related activity as Public Life Crisis when the focus is a public-space impact; classify mental-health, addiction, harm-reduction, or rehabilitation responses as Governance and Policy Challenge when the focus is an institutional or service response. Exclude posts that are not about homelessness, even if they contain relevant keywords. Use Not relevant to homelessness for these.

Allowed narrative-to-topic mapping:
""" + json.dumps(TOPICS, ensure_ascii=False, indent=2)

TOPIC_TO_NARRATIVE = {
    topic: narrative
    for narrative, topics in TOPICS.items()
    for topic in topics
}

SCHEMA = {
    "name": "homelessness_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "specific_topic": {
                "type": "string",
                "enum": list(TOPIC_TO_NARRATIVE),
            },
        },
        "required": ["specific_topic"],
    },
}


def text_for_prompt(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def valid(result: dict[str, str]) -> bool:
    return (
        result.get("narrative") in TOPICS
        and result.get("specific_topic") in TOPICS[result["narrative"]]
    )


def classify(text: str, city: str) -> dict[str, str]:
    if not text.strip():
        return {"narrative": "Unclear or mixed", "specific_topic": "Insufficient text"}
    user_content = f"city: {city}\n\ntext:\n{text}"
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "seed": SEED,
                "response_format": {"type": "json_schema", "json_schema": SCHEMA},
            }
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as api_response:
                response = json.load(api_response)
            result = json.loads(response["choices"][0]["message"]["content"])
            specific_topic = result.get("specific_topic")
            if specific_topic in TOPIC_TO_NARRATIVE:
                return {
                    "narrative": TOPIC_TO_NARRATIVE[specific_topic],
                    "specific_topic": specific_topic,
                }
            raise ValueError(f"Incompatible model output: {result}")
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 2**attempt + random.uniform(0, 0.5)))
    raise RuntimeError(f"Classification failed after five attempts: {last_error}")


def load_checkpoint(path: Path) -> dict[int, dict[str, str]]:
    completed: dict[int, dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if valid(item["result"]):
                    completed[item["row_index"]] = item["result"]
    return completed


def validate(source: pd.DataFrame, output: pd.DataFrame) -> None:
    assert list(output.columns) == OUTPUT_COLUMNS, "Output columns or order are wrong."
    assert len(source) == len(output), "Output row count differs from input."
    for column in REQUIRED_COLUMNS:
        assert source[column].equals(output[column]), f"Source field changed: {column}"
    assert not output[["narrative", "specific_topic"]].isna().any().any(), "Empty label found."
    assert output.apply(lambda r: valid(r[["narrative", "specific_topic"]].to_dict()), axis=1).all(), "Invalid topic pair found."
    assert source["post_id"].value_counts(dropna=False).sort_index().equals(output["post_id"].value_counts(dropna=False).sort_index()), "post_id occurrences changed."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="homeless_reddit_04-05-2026_clean_posts.csv",
        type=Path,
        help="Source CSV (default: homeless_reddit_04-05-2026_clean_posts.csv).",
    )
    parser.add_argument("--output", default="homelessness_narrative_topic_classification.csv", type=Path)
    parser.add_argument("--checkpoint", default="homelessness_classification_checkpoint.jsonl", type=Path)
    parser.add_argument(
        "--workers",
        "--threads",
        dest="workers",
        default=12,
        type=int,
        help="Number of concurrent classification requests (default: 12).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    random.seed(SEED)
    source = pd.read_csv(args.input, dtype=object)
    missing = [c for c in REQUIRED_COLUMNS if c not in source.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    carried = source[REQUIRED_COLUMNS].copy(deep=True)
    done = load_checkpoint(args.checkpoint)
    labels: list[dict[str, str] | None] = [done.get(i) for i in range(len(carried))]

    pending = [
        (i, text_for_prompt(row["text"]), text_for_prompt(row["city"]))
        for i, row in carried.iterrows()
        if labels[i] is None
    ]
    completed = len(carried) - len(pending)
    failures: list[tuple[int, Exception]] = []

    with args.checkpoint.open("a", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(classify, text, city): i
                for i, text, city in pending
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    labels[i] = future.result()
                except Exception as exc:
                    failures.append((i, exc))
                    print(f"ERROR row {i + 1}: {exc}", file=sys.stderr, flush=True)
                    continue

                checkpoint.write(
                    json.dumps(
                        {"row_index": i, "result": labels[i]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                checkpoint.flush()
                completed += 1
                print(f"Classified {completed}/{len(carried)}", flush=True)

    if failures:
        rows = ", ".join(str(i + 1) for i, _ in failures)
        raise RuntimeError(f"Classification failed for row(s): {rows}")

    output = pd.concat([carried.reset_index(drop=True), pd.DataFrame(labels)], axis=1)
    validate(carried.reset_index(drop=True), output)
    output.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Saved {args.output} with {len(output)} rows.")
    print("\nNarrative counts:\n" + output["narrative"].value_counts().to_string())
    print("\nTen most common topics:\n" + output["specific_topic"].value_counts().head(10).to_string())
    for narrative, sample in output.groupby("narrative", sort=False):
        print(f"\nQC sample: {narrative}")
        for _, row in sample.head(5).iterrows():
            excerpt = str(row["text"]).replace("\n", " ")[:200]
            print(f"- {row['specific_topic']}: {excerpt}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
