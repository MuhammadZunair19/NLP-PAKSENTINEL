"""
Build a self-constructed Pakistan dataset using BeautifulSoup and Tweepy.

Output schema:
    text,label,source,date,url,collection_method,label_reason

Classes:
    Real   -> Dawn / Geo / ARY article headlines and summaries
    Fake   -> X/Twitter posts collected from misinformation-oriented queries
    Satire -> X/Twitter posts collected from satire/parody handles or satire queries

This script is designed to be configurable because site structures and X access
can change. Feed URLs and Twitter credentials can be provided through `.env`
or environment variables.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
import tweepy
from bs4 import BeautifulSoup
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedSource:
    source: str
    feed_url: str
    label: str = "Real"
    label_reason: str = "Published by a mainstream Pakistani news outlet."


DEFAULT_FEEDS = [
    FeedSource("Dawn", "https://www.dawn.com/feeds/home"),
    FeedSource("Geo", "https://www.geo.tv/rss/1/1"),
    FeedSource("ARY", "https://arynews.tv/feed/"),
]

DEFAULT_FAKE_QUERIES = [
    '"miracle cure" Pakistan lang:en -is:retweet',
    '"viral claim" Pakistan lang:en -is:retweet',
    '"unverified" Pakistan politics lang:en -is:retweet',
    '"forward this" Pakistan lang:en -is:retweet',
]

DEFAULT_SATIRE_QUERIES = [
    '"satire" Pakistan politics lang:en -is:retweet',
    '"parody" Pakistan government lang:en -is:retweet',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape a Pakistan dataset for PakSentinel.")
    parser.add_argument("--output", default="./data/raw/pakistan/pakistan_dataset.csv")
    parser.add_argument("--real-per-source", type=int, default=500)
    parser.add_argument("--fake-limit", type=int, default=800)
    parser.add_argument("--satire-limit", type=int, default=800)
    parser.add_argument("--request-timeout", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config() -> Dict[str, object]:
    load_dotenv()
    satire_handles = [
        handle.strip().lstrip("@")
        for handle in os.getenv("TWITTER_SATIRE_HANDLES", "").split(",")
        if handle.strip()
    ]
    fake_queries = [
        query.strip()
        for query in os.getenv("TWITTER_FAKE_QUERIES", "").split("||")
        if query.strip()
    ] or DEFAULT_FAKE_QUERIES
    satire_queries = [
        query.strip()
        for query in os.getenv("TWITTER_SATIRE_QUERIES", "").split("||")
        if query.strip()
    ] or DEFAULT_SATIRE_QUERIES

    feeds = [
        FeedSource("Dawn", os.getenv("DAWN_FEED_URL", DEFAULT_FEEDS[0].feed_url)),
        FeedSource("Geo", os.getenv("GEO_FEED_URL", DEFAULT_FEEDS[1].feed_url)),
        FeedSource("ARY", os.getenv("ARY_FEED_URL", DEFAULT_FEEDS[2].feed_url)),
    ]
    return {
        "feeds": feeds,
        "twitter_bearer_token": os.getenv("TWITTER_BEARER_TOKEN", ""),
        "twitter_fake_queries": fake_queries,
        "twitter_satire_queries": satire_queries,
        "twitter_satire_handles": satire_handles,
    }


def _clean_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _request(url: str, timeout: int) -> requests.Response:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def scrape_rss_feed(feed: FeedSource, limit: int, timeout: int) -> List[Dict[str, str]]:
    logger.info("Scraping %s feed from %s", feed.source, feed.feed_url)
    response = _request(feed.feed_url, timeout=timeout)
    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all(["item", "entry"])

    records: List[Dict[str, str]] = []
    for item in items[:limit]:
        title = item.find("title")
        description = item.find("description") or item.find("summary")
        link = item.find("link")
        pub_date = item.find("pubDate") or item.find("published") or item.find("updated")

        title_text = _clean_text(title.get_text(" ", strip=True)) if title else ""
        description_text = _clean_text(description.get_text(" ", strip=True)) if description else ""
        if not title_text and not description_text:
            continue

        text = _clean_text(f"{title_text}. {description_text}".strip(". "))
        records.append(
            {
                "text": text,
                "label": feed.label,
                "source": feed.source,
                "date": pub_date.get_text(strip=True) if pub_date else "",
                "url": link.get_text(strip=True) if link else feed.feed_url,
                "collection_method": "BeautifulSoup RSS scrape",
                "label_reason": feed.label_reason,
            }
        )
    return records


def build_twitter_client(bearer_token: str) -> Optional[tweepy.Client]:
    if not bearer_token:
        logger.warning("TWITTER_BEARER_TOKEN not configured. Skipping X/Twitter collection.")
        return None
    return tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)


def scrape_twitter_search(
    client: tweepy.Client,
    queries: Iterable[str],
    limit: int,
    label: str,
    label_reason: str,
) -> List[Dict[str, str]]:
    queries = list(queries)
    records: List[Dict[str, str]] = []
    per_query_limit = max(10, limit // max(1, len(queries)))
    for query in queries:
        logger.info("Searching X/Twitter query: %s", query)
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            tweet_fields=["created_at", "lang"],
            max_results=min(100, per_query_limit),
        )
        collected = 0
        for page in paginator:
            if not page.data:
                continue
            for tweet in page.data:
                text = _clean_text(tweet.text)
                if not text:
                    continue
                records.append(
                    {
                        "text": text,
                        "label": label,
                        "source": "X",
                        "date": tweet.created_at.isoformat() if tweet.created_at else "",
                        "url": f"https://twitter.com/i/web/status/{tweet.id}",
                        "collection_method": "Tweepy recent-search",
                        "label_reason": label_reason,
                    }
                )
                collected += 1
                if collected >= per_query_limit:
                    break
            if collected >= per_query_limit:
                break
    return records[:limit]


def scrape_twitter_handles(
    client: tweepy.Client,
    handles: Iterable[str],
    limit: int,
    label: str,
    label_reason: str,
) -> List[Dict[str, str]]:
    handles = list(handles)
    if not handles:
        return []

    records: List[Dict[str, str]] = []
    per_handle_limit = max(10, limit // len(handles))
    for handle in handles:
        logger.info("Collecting tweets from satire/parody handle @%s", handle)
        user = client.get_user(username=handle)
        if not user.data:
            logger.warning("Could not resolve handle @%s", handle)
            continue

        paginator = tweepy.Paginator(
            client.get_users_tweets,
            id=user.data.id,
            tweet_fields=["created_at", "lang"],
            exclude=["retweets", "replies"],
            max_results=min(100, per_handle_limit),
        )
        collected = 0
        for page in paginator:
            if not page.data:
                continue
            for tweet in page.data:
                text = _clean_text(tweet.text)
                if not text:
                    continue
                records.append(
                    {
                        "text": text,
                        "label": label,
                        "source": f"X:@{handle}",
                        "date": tweet.created_at.isoformat() if tweet.created_at else "",
                        "url": f"https://twitter.com/{handle}/status/{tweet.id}",
                        "collection_method": "Tweepy user-timeline",
                        "label_reason": label_reason,
                    }
                )
                collected += 1
                if collected >= per_handle_limit:
                    break
            if collected >= per_handle_limit:
                break
    return records[:limit]


def deduplicate_records(records: List[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["text"] = df["text"].astype(str).map(_clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return df


def build_dataset(
    output_path: Path,
    real_per_source: int,
    fake_limit: int,
    satire_limit: int,
    timeout: int,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to rebuild it.")

    config = load_config()
    records: List[Dict[str, str]] = []

    for feed in config["feeds"]:
        records.extend(scrape_rss_feed(feed, limit=real_per_source, timeout=timeout))

    twitter_client = build_twitter_client(config["twitter_bearer_token"])
    if twitter_client is not None:
        records.extend(
            scrape_twitter_search(
                twitter_client,
                config["twitter_fake_queries"],
                limit=fake_limit,
                label="Fake",
                label_reason="Collected from misinformation-oriented X/Twitter search queries for manual review and fake-news class curation.",
            )
        )

        satire_records = scrape_twitter_handles(
            twitter_client,
            config["twitter_satire_handles"],
            limit=satire_limit,
            label="Satire",
            label_reason="Collected from satire/parody X/Twitter handles.",
        )
        if not satire_records:
            satire_records = scrape_twitter_search(
                twitter_client,
                config["twitter_satire_queries"],
                limit=satire_limit,
                label="Satire",
                label_reason="Collected from satire/parody X/Twitter search queries and intended for manual validation.",
            )
        records.extend(satire_records)

    dataset = deduplicate_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %d Pakistan records to %s", len(dataset), output_path)
    if not dataset.empty:
        logger.info("Class distribution: %s", dataset["label"].value_counts().to_dict())
        logger.info("Source distribution: %s", dataset["source"].value_counts().head(10).to_dict())
    return output_path


def main():
    args = parse_args()
    output_path = Path(args.output)
    build_dataset(
        output_path=output_path,
        real_per_source=args.real_per_source,
        fake_limit=args.fake_limit,
        satire_limit=args.satire_limit,
        timeout=args.request_timeout,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
