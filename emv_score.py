"""
EMV (Emotional Marketing Value) headline scorer.
Scrapes the AMI Headline Analyzer to score titles in bulk.

Usage:
    python emv_score.py "Your Headline Here"
    python emv_score.py --channel rank_recon        # score titles from topic backlog
    python emv_score.py --file titles.txt           # one title per line
"""

import re
import sys
import time
import argparse
import requests
from bs4 import BeautifulSoup

URL = "https://www.aminstitute.com/process/headline.cgi"


def get_emv_score(headline: str) -> dict:
    """Return {'headline': str, 'score': float, 'classification': str}."""
    resp = requests.post(URL, data={"text": headline, "category": "Uncategorized"}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Score is in <span class="step-number ...">50.00%</span>
    score_el = soup.select_one("span.step-number")
    score = float(re.search(r"([\d.]+)%", score_el.text).group(1)) if score_el else 0.0

    # Classification is in <h4 class="card-title ..."><strong>Spiritual</strong></h4>
    classification_el = soup.select_one("section.content15 h4.card-title strong")
    classification = classification_el.text.strip() if classification_el else "Unknown"

    return {"headline": headline, "score": score, "classification": classification}


def score_titles(titles: list[str], delay: float = 1.0) -> list[dict]:
    """Score a list of titles with a polite delay between requests."""
    results = []
    for i, title in enumerate(titles):
        result = get_emv_score(title)
        results.append(result)
        print(f"  {result['score']:5.1f}% [{result['classification']:12s}]  {title}")
        if i < len(titles) - 1:
            time.sleep(delay)
    return results


def main():
    parser = argparse.ArgumentParser(description="Score headlines using AMI EMV Analyzer")
    parser.add_argument("headline", nargs="*", help="Headline(s) to score")
    parser.add_argument("--file", "-f", help="File with one headline per line")
    args = parser.parse_args()

    titles = []

    if args.file:
        with open(args.file) as f:
            titles = [line.strip() for line in f if line.strip()]
    elif args.headline:
        titles = [" ".join(args.headline)]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nScoring {len(titles)} headline(s)...\n")
    results = score_titles(titles)

    # Summary
    if len(results) > 1:
        avg = sum(r["score"] for r in results) / len(results)
        best = max(results, key=lambda r: r["score"])
        print(f"\n--- Summary ---")
        print(f"Average EMV: {avg:.1f}%")
        print(f"Best:        {best['score']:.1f}% [{best['classification']}] {best['headline']}")


if __name__ == "__main__":
    main()
