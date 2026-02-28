"""Pull YouTube Analytics data: impressions, CTR, retention, traffic sources."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from googleapiclient.discovery import build

from config_loader import load_config
from upload.youtube import get_youtube_service, SCOPES


def main():
    config = load_config()

    # Reuse credentials from the Data API auth
    youtube = get_youtube_service(config)

    # Build Analytics API service using same credentials
    token_path = Path(config.youtube_token_file)
    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    analytics = build("youtubeAnalytics", "v2", credentials=creds)

    # Load video data from previous channel_analytics.py run
    video_data = json.loads(Path("channel_data.json").read_text())
    public_videos = [v for v in video_data if v["privacy"] == "public"]

    # Date range: from first video to today
    start_date = "2026-02-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 80)
    print("YOUTUBE ANALYTICS REPORT")
    print("=" * 80)

    # 1. Channel-level: impressions, CTR, views, watch time
    print("\n--- CHANNEL OVERVIEW ---")
    try:
        resp = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched,averageViewDuration,subscribersGained",
        ).execute()
        if resp.get("rows"):
            row = resp["rows"][0]
            print(f"  Views:                  {row[0]}")
            print(f"  Watch time (minutes):   {row[1]:.1f}")
            print(f"  Avg view duration (s):  {row[2]:.0f}")
            print(f"  Subscribers gained:     {row[3]}")
    except Exception as e:
        print(f"  Error fetching channel overview: {e}")

    # 2. Channel-level impressions & CTR
    print("\n--- IMPRESSIONS & CTR ---")
    try:
        resp = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,impressions,impressionClickThroughRate",
            dimensions="day",
            sort="day",
        ).execute()
        if resp.get("rows"):
            total_impressions = 0
            total_views_from_imp = 0
            print(f"  {'Date':<14} {'Impressions':>12} {'Views':>8} {'CTR':>8}")
            print(f"  {'-'*44}")
            for row in resp["rows"]:
                day, views, impressions, ctr = row
                total_impressions += impressions
                total_views_from_imp += views
                print(f"  {day:<14} {impressions:>12.0f} {views:>8.0f} {ctr:>7.2%}")
            print(f"  {'-'*44}")
            avg_ctr = total_views_from_imp / total_impressions if total_impressions > 0 else 0
            print(f"  {'TOTAL':<14} {total_impressions:>12.0f} {total_views_from_imp:>8.0f} {avg_ctr:>7.2%}")
        else:
            print("  No impression data available yet.")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Per-video stats
    print("\n--- PER-VIDEO PERFORMANCE ---")
    try:
        resp = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched,averageViewDuration,impressions,impressionClickThroughRate",
            dimensions="video",
            sort="-views",
        ).execute()
        if resp.get("rows"):
            # Map video IDs to titles
            id_to_title = {v["video_id"]: v["title"] for v in video_data}
            print(f"  {'Title':<45} {'Views':>6} {'WatchMin':>9} {'AvgDur':>7} {'Impr':>7} {'CTR':>7}")
            print(f"  {'-'*85}")
            for row in resp["rows"]:
                vid, views, watch_min, avg_dur, impressions, ctr = row
                title = id_to_title.get(vid, vid)[:43]
                print(f"  {title:<45} {views:>6.0f} {watch_min:>9.1f} {avg_dur:>6.0f}s {impressions:>7.0f} {ctr:>6.2%}")
        else:
            print("  No per-video data available yet.")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. Traffic sources
    print("\n--- TRAFFIC SOURCES ---")
    try:
        resp = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched",
            dimensions="insightTrafficSourceType",
            sort="-views",
        ).execute()
        if resp.get("rows"):
            print(f"  {'Source':<35} {'Views':>8} {'Watch Min':>10}")
            print(f"  {'-'*55}")
            for row in resp["rows"]:
                source, views, watch_min = row
                print(f"  {source:<35} {views:>8.0f} {watch_min:>10.1f}")
        else:
            print("  No traffic source data available yet.")
    except Exception as e:
        print(f"  Error: {e}")

    # 5. Audience retention for top video
    top_video = max(public_videos, key=lambda v: v["views"])
    if top_video["views"] > 0:
        print(f"\n--- AUDIENCE RETENTION: {top_video['title'][:50]} ---")
        try:
            resp = analytics.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="audienceWatchRatio",
                dimensions="elapsedVideoTimeRatio",
                filters=f"video=={top_video['video_id']}",
                sort="elapsedVideoTimeRatio",
            ).execute()
            if resp.get("rows"):
                print(f"  {'Time %':>8} {'Retention':>10}")
                print(f"  {'-'*20}")
                for row in resp["rows"]:
                    time_pct, retention = row
                    bar = "#" * int(retention * 50)
                    print(f"  {time_pct:>7.0%} {retention:>9.1%}  {bar}")
            else:
                print("  No retention data available yet (may take 48-72hrs).")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
