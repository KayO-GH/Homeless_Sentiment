"""Regression checks for the revised controlled homelessness taxonomy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "01_classify_homelessness_narratives.py"
SPEC = importlib.util.spec_from_file_location("homelessness_classifier", MODULE_PATH)
assert SPEC and SPEC.loader
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


REVISED_TOPICS = {
    "Housing Crisis": {
        "Housing affordability and rising rents",
        "Eviction, displacement, and insecure tenancy (i.e., across ownership, rent, co-op etc. setups)",
        "Poverty, unemployment, and income insecurity",
        "Non-market and supportive housing solutions",
    },
    "Public Life Crisis": {
        "Encampments in public space",
        "Encampment clearance, displacement",
        "Perceived lack of public order, safety",
        "Reported harassment, property damage",
        "Visible substance use, drug-related activity",
        "Neighbourhood change and public-space contestation",
    },
    "(Society) Moral Crisis": {
        "First-person accounts of homelessness",
        "Requests for assistance due to imminent homelessness",
        "Access to food, hygiene, employment, and other basic needs",
        "Emergency shelter access, shelter conditions, and shelter health and safety",
        "Community aid, outreach, volunteering and direct assistance",
        "Human dignity, compassion, and stigmatization",
        "Health, trauma, and vulnerability",
    },
    "Governance and Policy Challenge": {
        "Government responsibility, accountability, policy failure, and intergovernmental coordination",
        "Debates over homelessness policy solutions",
        "Public funding and resource allocation",
        "Shelter-system capacity and service delivery",
        "Policing, bylaws, and enforcement",
        "Mental-health, addiction, harm-reduction and rehabilitation responses",
    },
}

SAFETY_THEMES = {"Unclear or mixed", "Not relevant to homelessness"}


class ClassificationTaxonomyTests(unittest.TestCase):
    def test_revised_codes_map_to_exactly_one_theme(self) -> None:
        self.assertEqual(set(classifier.TOPICS), set(REVISED_TOPICS) | SAFETY_THEMES)
        self.assertEqual(
            {theme: set(classifier.TOPICS[theme]) for theme in REVISED_TOPICS},
            REVISED_TOPICS,
        )
        revised_codes = set().union(*REVISED_TOPICS.values())
        self.assertEqual(set(classifier.TOPIC_TO_NARRATIVE), revised_codes | {
            topic
            for theme, topics in classifier.TOPICS.items()
            if theme in SAFETY_THEMES
            for topic in topics
        })
        self.assertEqual(len(classifier.TOPIC_TO_NARRATIVE), len(set(classifier.TOPIC_TO_NARRATIVE)))

    def test_schema_contains_only_active_topics(self) -> None:
        active_topics = set(classifier.TOPIC_TO_NARRATIVE)
        schema_topics = set(classifier.SCHEMA["schema"]["properties"]["specific_topic"]["enum"])
        self.assertEqual(schema_topics, active_topics)
        self.assertNotIn("Housing First and long-term policy solutions", schema_topics)
        self.assertNotIn("Open drug use and visible substance use", schema_topics)

    def test_blank_text_uses_retained_safeguard(self) -> None:
        self.assertEqual(
            classifier.classify("", "Toronto"),
            {"narrative": "Unclear or mixed", "specific_topic": "Insufficient text"},
        )

    def test_non_homelessness_safeguard_is_valid(self) -> None:
        result = {
            "narrative": "Not relevant to homelessness",
            "specific_topic": "Animal shelter or animal rescue",
        }
        self.assertTrue(classifier.valid(result))

    def test_validation_accepts_each_theme_and_safeguard(self) -> None:
        labels = [
            ("Housing Crisis", "Housing affordability and rising rents"),
            ("Public Life Crisis", "Encampments in public space"),
            ("(Society) Moral Crisis", "First-person accounts of homelessness"),
            ("Governance and Policy Challenge", "Policing, bylaws, and enforcement"),
            ("Unclear or mixed", "General discussion of homelessness"),
            ("Not relevant to homelessness", "Animal shelter or animal rescue"),
        ]
        source = pd.DataFrame(
            [
                {"post_id": str(index), "text": f"post {index}", "city": "Toronto", "url": f"https://example.com/{index}"}
                for index in range(len(labels))
            ]
        )
        output = source.copy()
        output["narrative"] = [narrative for narrative, _ in labels]
        output["specific_topic"] = [topic for _, topic in labels]
        classifier.validate(source, output)


if __name__ == "__main__":
    unittest.main()
