"""
Preprocess real review datasets → unified Parquet file.

Datasets:
  Amazon  — All_Beauty.jsonl.gz, Grocery.jsonl.gz, Electronics.jsonl.gz
             (Amazon Reviews 2023 JSONL format, from McAuley Lab / UCSD)
  Yelp    — yelp_academic_dataset_review.json (JSON lines)
  Twitter — Tweets.csv.zip (US Airline Sentiment, Kaggle)

Output:
  data/processed/reviews_unified.parquet

Usage:
    python scripts/preprocess.py

Place your raw files following this layout:
    data/raw/amazon/All_Beauty.jsonl.gz
    data/raw/amazon/Grocery.jsonl.gz
    data/raw/amazon/Electronics.jsonl.gz
    data/raw/Yelp/yelp_academic_dataset_review.json
    data/raw/Twitter/Tweets.csv.zip
"""
# Future imports
from __future__ import annotations

#standard library imports
import os
import re
import sys
import json
import gzip
from typing import Optional

import pandas as pd
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths to the raw datasets
AMAZON_BEAUTY_PATH      = "data/raw/amazon/All_Beauty.jsonl.gz"
AMAZON_GROCERY_PATH     = "data/raw/amazon/Grocery.jsonl.gz"
AMAZON_ELECTRONICS_PATH = "data/raw/amazon/Electronics.jsonl.gz"
YELP_PATH               = "data/raw/Yelp/yelp_academic_dataset_review.json"
TWITTER_PATH = "data/raw/Twitter/Tweets.csv.zip"



# How many reviews to sample per sentiment class, per Amazon category
AMAZON_PER_CLASS = 20000

# How many reviews to sample per sentiment class from Yelp
YELP_PER_CLASS = 150000

# Path to the output file
OUTPUT_PATH = "data/processed/reviews_unified.parquet"

# Create the output directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw/amazon", exist_ok=True)
os.makedirs("data/raw/Yelp",   exist_ok=True)
os.makedirs("data/raw/Twitter", exist_ok=True)

# Function to clean text
def clean_text(text:str) -> str:
    """
    Clean and normalise review text.
    Steps: URL removal → @mentions → hashtags → HTML → emoji→text
           → non-ASCII removal → lowercase.
    """
    text = str(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove user mentions
    text = re.sub(r"@\w+", " ", text)
    # Remove the '#' symbol but keep the hashtag word
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Convert emojis into text (requires emoji package)
    try:
        import emoji
        text = emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        pass   # skip emoji conversion if package not installed
    # Replace underscores with spaces (from emoji names like :thumbs_up:)
    text = text.replace("_", " ")
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Remove extra spaces and convert to lowercase
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# Function to map rating to sentiment class
def rating_to_sentiment(rating:int) -> str:
    """ Map rating to sentiment class.
    Rating 1-2: Negative
    3: Neutral
    4-5: Positive
    """
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

# Function to sample reviews from each sentiment class
def balanced_sample(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample per_class reviews from each sentiment class.
    If a class has fewer rows than per_class, takes all rows from that class.
    """
    sampled = []
    for _, group in df.groupby("sentiment_label"):
        sampled.append(group.sample(n=min(len(group), per_class), random_state=seed))
    # Concatenate the sampled reviews and return the result
    return pd.concat(sampled, ignore_index=True)


def standardise_amazon(df:pd.DataFrame, category:str) -> pd.DataFrame:
    """
    Standardise a raw Amazon JSONL DataFrame into the unified schema
    Steps:
    - Add platform column
    - Add product category column
    - Add text column
    - Add rating column
    - Add date column
    - Add review_id column
    - Create the sentiment label
    - Return the standardized DataFrame
    """
    output = pd.DataFrame({
        "platform": "amazon",
        "product_category": category,
        # Combine title and text columns, strip trailing dots, and convert to lowercase
        "text": (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip(". ").str.strip(),
        "rating": df["rating"].astype(float),
        "date": pd.to_datetime(
            df["timestamp"],
            unit="ms",
            errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "review_id": df["asin"].astype(str) + "_" + df.index.astype(str),
    })

    # Create the sentiment label   
    output["sentiment_label"] = output["rating"].apply(rating_to_sentiment)

    # Return the standardized DataFrame
    return output

# Function to load an Amazon Reviews 2023 JSONL.gz file in chunks, standardise, and return a class-balanced sample
def load_amazon_jsonl(path: str, category: str, per_class: int,
                      chunk_size: int = 100000) -> Optional[pd.DataFrame]:
    """
    Load an Amazon Reviews 2023 JSONL.gz file in chunks, standardise,
    and return a class-balanced sample.
    """
    # Check whether the dataset file exists
    if not os.path.exists(path):
        print(f" Not found: {path}  (skipping)")
        return None

    # Display the dataset being loaded
    print(f"   Loading {category} from {path}...")

    # Store processed chunks
    chunks: list[pd.DataFrame] = []
    total = 0

    # Open compressed or uncompressed file
    opener = gzip.open if path.endswith(".gz") else open

    # Store JSON records before converting them into a DataFrame
    buffer: list[dict] = []

    # Read the file line by line
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Remove extra spaces
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Convert each JSON record into a dictionary
            try:
                buffer.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            # Process one chunk when the buffer reaches the chunk size
            if len(buffer) >= chunk_size:
                # Convert the buffer into a DataFrame
                chunk_df = pd.DataFrame(buffer)
                # Ensure required columns exist
                for col in ["title", "text", "rating", "timestamp", "asin"]:
                    if col not in chunk_df.columns:
                        chunk_df[col] = None
                # Remove reviews with missing or very short text
                chunk_df = chunk_df[
                    chunk_df["text"].notna() &
                    (chunk_df["text"].astype(str).str.len() > 10)
                ]
                # Standardise the chunk
                if not chunk_df.empty:
                    chunks.append(standardise_amazon(chunk_df, category))
                # Update the number of processed records
                total += len(buffer)
                # Clear the buffer for the next chunk
                buffer = []
                # Display the loading progress
                print(f"      ...{total:,} rows read", end="\r")
    # Process any remaining records
    if buffer:
        # Convert the remaining records into a DataFrame
        chunk_df = pd.DataFrame(buffer)
        # Ensure required columns exist
        for col in ["title", "text", "rating", "timestamp", "asin"]:
            if col not in chunk_df.columns:
                chunk_df[col] = None
        # Remove reviews with missing or very short text
        chunk_df = chunk_df[
            chunk_df["text"].notna() &
            (chunk_df["text"].astype(str).str.len() > 10)
        ]
        # Standardise the remaining chunk
        if not chunk_df.empty:
            chunks.append(standardise_amazon(chunk_df, category))

        # Update the total number of processed records
        total += len(buffer)

    # Return None if no valid reviews were found
    if not chunks:
        print(f"   ⚠  No valid rows in {path}")
        return None

    # Combine all processed chunks into a single DataFrame
    raw_df = pd.concat(chunks, ignore_index=True)

    # Display the total number of loaded reviews
    print(f"   {category}: {len(raw_df):,} reviews loaded → balancing to {per_class}/class")

    # Create a balanced dataset
    balanced = balanced_sample(raw_df, per_class)

    # Display the balanced dataset information
    print(f"   {category}: {len(balanced):,} reviews after balancing")
    print(f"   {category}: class dist: {balanced['sentiment_label'].value_counts().to_dict()}")

    # Return the balanced dataset
    return balanced


# Loading Yelp data
def load_yelp(path: str, per_class: int) -> Optional[pd.DataFrame]:
    """
    Load Yelp academic dataset (JSON lines format).
    stars→rating, date[:10], review_id, balanced_sample.
    """
    # Check whether the dataset file exists
    if not os.path.exists(path):
        print(f"   ⚠  Not found: {path}  (skipping)")
        return None

    # Display the dataset being loaded
    print(f"   Loading Yelp from {path}...")

    # Store the processed Yelp reviews
    rows: list[dict] = []

    # Read the dataset line by line
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Remove extra spaces
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Convert each JSON record into a dictionary
            try:
                d = json.loads(line)

                # Extract the review text
                text = d.get("text", "")

                # Skip reviews with very short text
                if len(str(text)) < 10:
                    continue

                # Store the required fields
                rows.append({
                    "platform":         "yelp",
                    "product_category": "",
                    "text":             str(text),
                    "rating":           float(d.get("stars", 3)),
                    "date":             str(d.get("date", ""))[:10],
                    "review_id":        str(d.get("review_id", f"YLP_{len(rows)}")),
                })

            # Skip invalid JSON records
            except Exception:
                continue

    # Return None if no valid reviews were found
    if not rows:
        print("  No valid Yelp rows")
        return None

    # Convert the collected reviews into a DataFrame
    yelp_std = pd.DataFrame(rows)

    # Create the sentiment label from the rating
    yelp_std["sentiment_label"] = yelp_std["rating"].apply(rating_to_sentiment)

    # Display the total number of loaded reviews
    print(f"   Yelp: {len(yelp_std):,} reviews → balancing to {per_class}/class")

    # Create a balanced dataset
    yelp_balanced = balanced_sample(yelp_std, per_class)

    # Display the balanced dataset information
    print(f"   Yelp: {len(yelp_balanced):,} reviews after balancing")
    print(f"   Yelp: class dist: {yelp_balanced['sentiment_label'].value_counts().to_dict()}")

    # Return the balanced dataset
    return yelp_balanced


#Load Twitter Data
def load_twitter(path: str) -> Optional[pd.DataFrame]:
    """
    Load Twitter US Airline Sentiment CSV (Kaggle, optionally zipped).
    Matches notebook: airline_sentiment, tweet_created→date, tweet_id→review_id.
    """

    # Check whether the dataset file exists
    if not os.path.exists(path):
        print(f"   ⚠  Not found: {path}  (skipping)")
        return None

    # Display the dataset being loaded
    print(f"   Loading Twitter from {path}...")

    # Detect whether the dataset is compressed
    compression = "zip" if path.endswith(".zip") else "infer"

    # Load the Twitter dataset
    tweets = pd.read_csv(path, compression=compression)

    # Standardise the Twitter dataset
    twt_std = pd.DataFrame({
        "platform":         "twitter_airline",
        "product_category": "",
        "text":             tweets["text"].astype(str),
        "sentiment_label":  tweets["airline_sentiment"],
        "rating":           tweets["airline_sentiment"].map({
                                "positive": 5,
                                "neutral": 3,
                                "negative": 1
                            }),
        "date": pd.to_datetime(
            tweets["tweet_created"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "review_id": tweets["tweet_id"].astype(str),
    })

    # Display the total number of loaded reviews
    print(f"   Twitter: {len(twt_std):,} reviews")

    # Display the class distribution
    print(f"   Twitter: class dist: {twt_std['sentiment_label'].value_counts().to_dict()}")

    # Return the standardised dataset
    return twt_std



# MAIN PIPLEINE
def main() -> None:
    """
    Pipeline Steps:
    1. Load Amazon review datasets.
    2. Load the Yelp review dataset.
    3. Load the Twitter review dataset.
    4. Merge all datasets into one DataFrame.
    5. Clean the review text.
    6. Perform data quality checks and filtering.
    7. Save the final dataset.
    8. Display the preprocessing summary.
    """
    #store all three Datasets
    dfs : list[pd.Dataframe] = []

    #Amazon(3 categories)
    print("\n Amazon Reviews 2023 (3 categories)...")

    #Define Amazon Dataset
    amazon_paths = {
        "All_Beauty": AMAZON_BEAUTY_PATH,
        "Grocery" : AMAZON_GROCERY_PATH,
        "Electronics": AMAZON_ELECTRONICS_PATH,
    }
    for category,path in amazon_paths.items():
        #load each amazon category
        amz_cat = load_amazon_jsonl(path,category,AMAZON_PER_CLASS)
        if amz_cat is not None:
            dfs.append(amz_cat)
    
    #Load Yelp Dataset
    yelp_df = load_yelp(YELP_PATH,YELP_PER_CLASS)
    if yelp_df is not None:
        dfs.append(yelp_df)
    
    #Load Twitter Data
    twt_df = load_twitter(TWITTER_PATH)
    if twt_df is not None:
        dfs.append(twt_df)
    
    # Stop the program if no datasets were loaded
    if not dfs:
        print("\nNo datasets loaded. Check file paths at the top of this script.")
        print("Expected locations:")
        for p in [AMAZON_BEAUTY_PATH, AMAZON_GROCERY_PATH, AMAZON_ELECTRONICS_PATH,
                  YELP_PATH, TWITTER_PATH]:
            print(f"  {p}")
        sys.exit(1)


    #Merge all datasets
    print("\n Merging all datasets...")
    df = pd.concat(dfs,ignore_index=True)

    #Apply text cleaning on merge dataset
    print("Cleaning text...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    #Final quality filtering (exact notebook steps)
    df["product_category"] = df["product_category"].fillna("")
    df = df.dropna(subset=["text", "sentiment_label"])
    df = df[df["cleaned_text"].str.len() > 15]
    df = df.drop_duplicates(subset="cleaned_text")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3)
    df = df.reset_index(drop=True)

    # Ensure all required columns exist
    required_cols = ["review_id", "platform", "product_category","text", "cleaned_text", "rating", "sentiment_label", "date"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[required_cols]

    # Save to Parquet 
    df.to_parquet(OUTPUT_PATH, index=False)

    # Display the preprocessing summary
    print(f" Preprocessing complete!")
    print(f"  Output    : {OUTPUT_PATH}")
    print(f"  Total rows: {len(df):,}")

    # Display the platform distribution
    print(f"\n  Platform distribution:")
    for plat, cnt in df["platform"].value_counts().items():
        print(f"    {plat:<25} {cnt:>8,}")

    # Display the sentiment distribution
    print(f"\n  Sentiment distribution:")
    for sent, cnt in df["sentiment_label"].value_counts().items():
        print(f"    {sent:<25} {cnt:>8,}")

    # Display the Amazon category distribution
    if "product_category" in df.columns:
        cats = df[df["product_category"] != ""]["product_category"].value_counts()
        if not cats.empty:
            print(f"\n  Amazon categories:")
            for cat, cnt in cats.items():
                print(f"    {cat:<25} {cnt:>8,}")


if __name__ == "__main__":
    main()