"""Pull channel analytics from YouTube Data API v3."""

import json
import sys
from pathlib import Path

from config_loader import load_config
from upload.youtube import get_youtube_service


def main():
    config = load_config()
    service = get_youtube_service(config)

    # 1. Get channel info
    ch_resp = service.channels().list(
        part="snippet,statistics,contentDetails",
        mine=True,
    ).execute()

    if not ch_resp.get("items"):
        print("ERROR: No channel found for these credentials.")
        return

    channel = ch_resp["items"][0]
    stats = channel["statistics"]
    print("=" * 70)
    print(f"CHANNEL: {channel['snippet']['title']}")
    print(f"Subscribers: {stats.get('subscriberCount', 'hidden')}")
    print(f"Total views: {stats.get('viewCount', 0)}")
    print(f"Total videos: {stats.get('videoCount', 0)}")
    print("=" * 70)

    # 2. Get all uploaded videos
    uploads_playlist = channel["contentDetails"]["relatedPlaylists"]["uploads"]

    video_ids = []
    page_token = None
    while True:
        pl_resp = service.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=page_token,
        ).execute()

        for item in pl_resp.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        page_token = pl_resp.get("nextPageToken")
        if not page_token:
            break

    if not video_ids:
        print("No videos found on channel.")
        return

    # 3. Get detailed stats for each video (batch 50 at a time)
    videos = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        v_resp = service.videos().list(
            part="snippet,statistics,contentDetails,status",
            id=",".join(batch),
        ).execute()
        videos.extend(v_resp.get("items", []))

    # 4. Display results
    print(f"\n{'#':<4} {'Title':<60} {'Views':>8} {'Likes':>6} {'Comments':>8} {'Duration':>10} {'Status':<12}")
    print("-" * 120)

    total_views = 0
    total_likes = 0
    total_comments = 0
    video_data = []

    for i, v in enumerate(videos, 1):
        title = v["snippet"]["title"][:58]
        views = int(v["statistics"].get("viewCount", 0))
        likes = int(v["statistics"].get("likeCount", 0))
        comments = int(v["statistics"].get("commentCount", 0))
        duration = v["contentDetails"]["duration"]  # ISO 8601
        privacy = v["status"]["privacyStatus"]
        publish_at = v["status"].get("publishAt", "")

        total_views += views
        total_likes += likes
        total_comments += comments

        status_str = privacy
        if publish_at:
            status_str = f"sched {publish_at[:10]}"

        print(f"{i:<4} {title:<60} {views:>8} {likes:>6} {comments:>8} {duration:>10} {status_str:<12}")

        video_data.append({
            "title": v["snippet"]["title"],
            "video_id": v["id"],
            "views": views,
            "likes": likes,
            "comments": comments,
            "duration": duration,
            "privacy": privacy,
            "publishAt": publish_at,
            "publishedAt": v["snippet"].get("publishedAt", ""),
            "tags": v["snippet"].get("tags", []),
            "categoryId": v["snippet"].get("categoryId", ""),
            "description_length": len(v["snippet"].get("description", "")),
        })

    print("-" * 120)
    print(f"{'TOTALS':<64} {total_views:>8} {total_likes:>6} {total_comments:>8}")

    # 5. Summary stats
    public_videos = [v for v in video_data if v["privacy"] == "public"]
    private_videos = [v for v in video_data if v["privacy"] == "private"]
    scheduled_videos = [v for v in video_data if v["publishAt"]]

    print(f"\n{'=' * 70}")
    print("BREAKDOWN:")
    print(f"  Public videos:    {len(public_videos)}")
    print(f"  Private videos:   {len(private_videos)}")
    print(f"  Scheduled:        {len(scheduled_videos)}")

    if public_videos:
        avg_views = sum(v["views"] for v in public_videos) / len(public_videos)
        avg_likes = sum(v["likes"] for v in public_videos) / len(public_videos)
        avg_comments = sum(v["comments"] for v in public_videos) / len(public_videos)
        print(f"\n  Avg views/video (public):    {avg_views:.1f}")
        print(f"  Avg likes/video (public):    {avg_likes:.1f}")
        print(f"  Avg comments/video (public): {avg_comments:.1f}")

        if avg_views > 0:
            print(f"  Like rate (likes/views):     {avg_likes/avg_views*100:.2f}%")

        # Top 5 by views
        top = sorted(public_videos, key=lambda v: v["views"], reverse=True)[:5]
        print(f"\n  TOP 5 BY VIEWS:")
        for v in top:
            print(f"    {v['views']:>6} views | {v['title'][:55]}")

        # Bottom 5 by views
        bottom = sorted(public_videos, key=lambda v: v["views"])[:5]
        print(f"\n  BOTTOM 5 BY VIEWS:")
        for v in bottom:
            print(f"    {v['views']:>6} views | {v['title'][:55]}")

        # Videos with 0 views
        zero_view = [v for v in public_videos if v["views"] == 0]
        if zero_view:
            print(f"\n  WARNING: {len(zero_view)} public video(s) with 0 views!")
            for v in zero_view:
                print(f"    - {v['title'][:60]}")

    # Dump raw data for further analysis
    raw_path = Path("channel_data.json")
    raw_path.write_text(json.dumps(video_data, indent=2))
    print(f"\nRaw data saved to {raw_path}")


if __name__ == "__main__":
    main()
