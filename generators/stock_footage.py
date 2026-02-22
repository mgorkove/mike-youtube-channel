"""Stock footage sourcing via Pexels API.

Downloads free-for-commercial-use video clips matching scene descriptions.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from config_loader import Config

logger = logging.getLogger(__name__)

PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"


def _get_api_key() -> str:
    """Return Pexels API key from environment."""
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "PEXELS_API_KEY environment variable is not set. "
            "Get a free key at https://www.pexels.com/api/"
        )
    return key


def search_videos(query: str, per_page: int = 5, min_duration: int = 5) -> list[dict]:
    """Search Pexels for videos matching a query.

    Returns a list of video result dicts with keys: id, url, duration,
    and video_files (list of available file downloads).
    """
    api_key = _get_api_key()
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape",
        "size": "medium",
    }

    resp = requests.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for video in data.get("videos", []):
        if video.get("duration", 0) >= min_duration:
            results.append({
                "id": video["id"],
                "url": video["url"],
                "duration": video["duration"],
                "video_files": video.get("video_files", []),
            })
    return results


def _pick_best_file(video_files: list[dict], target_width: int = 1920) -> str | None:
    """Pick the best video file URL — prefer HD, closest to target width."""
    candidates = [
        f for f in video_files
        if f.get("width") and f.get("link") and f.get("quality") == "hd"
    ]
    if not candidates:
        # Fallback to any file with a link
        candidates = [f for f in video_files if f.get("link")]
    if not candidates:
        return None

    # Sort by closeness to target width (prefer >= target)
    candidates.sort(key=lambda f: abs(f.get("width", 0) - target_width))
    return candidates[0]["link"]


def download_video(video_url: str, output_path: Path) -> Path:
    """Download a video file to disk."""
    resp = requests.get(video_url, stream=True, timeout=120)
    resp.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded clip: {output_path}")
    return output_path


def _fetch_single_clip(
    index: int,
    query: str,
    clips_dir: Path,
    total: int,
) -> Path:
    """Search and download one clip for a given search query."""
    logger.info(f"Searching Pexels ({index + 1}/{total}): '{query}'")
    results = search_videos(query, per_page=5)

    if not results:
        # Broaden search with simpler query (first two words)
        simple_query = " ".join(query.split()[:2])
        logger.warning(f"No results for '{query}', trying '{simple_query}'")
        results = search_videos(simple_query, per_page=5)

    if not results:
        raise RuntimeError(f"No Pexels videos found for query: '{query}'")

    # Try each result until we find a downloadable file
    for video in results:
        file_url = _pick_best_file(video["video_files"])
        if file_url:
            clip_path = clips_dir / f"clip_{index:03d}.mp4"
            return download_video(file_url, clip_path)

    raise RuntimeError(f"No downloadable video files for query: '{query}'")


def fetch_stock_clips(
    search_queries: list[str],
    output_dir: Path,
    config: Config,
) -> list[Path]:
    """Fetch stock footage clips from Pexels for each search query.

    Downloads clips in parallel (up to 5 workers) and returns an ordered
    list of clip file paths.
    """
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    total = len(search_queries)
    logger.info(f"Fetching {total} stock footage clips from Pexels...")

    clip_paths: dict[int, Path] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_single_clip, i, query, clips_dir, total): i
            for i, query in enumerate(search_queries)
        }

        for future in as_completed(futures):
            idx = futures[future]
            clip_paths[idx] = future.result()

    # Return in order
    return [clip_paths[i] for i in range(total)]
