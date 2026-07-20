from pathlib import Path

import pandas as pd


source_file = Path("homeless_reddit_04-05-2026_clean_posts.csv")
target_file = Path("homelessness_narrative_topic_classification.csv")
output_file = Path("homelessness_narrative_topic_classification_with_created_utc.csv")

source = pd.read_csv(source_file, usecols=["post_id", "created_utc"])
target = pd.read_csv(target_file)

if source["post_id"].duplicated().any():
    raise ValueError("Source contains duplicate post_id values.")

source["created_utc"] = pd.to_datetime(
    source["created_utc"],
    format="%m/%d/%Y %H:%M",
    errors="raise",
).dt.strftime("%Y-%m-%d %H:%M:%S")

enriched = target.merge(source, on="post_id", how="left", validate="many_to_one")
enriched.to_csv(output_file, index=False)

print(f"Saved {output_file} with {len(enriched)} rows.")
