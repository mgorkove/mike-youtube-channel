"""Compute YouTube publish schedule for a batch of videos.

Given a video count and publish slots (e.g. 8 AM and 6 PM ET),
returns a list of ISO 8601 UTC datetime strings for the YouTube
``publishAt`` field.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def compute_publish_schedule(
    video_count: int,
    timezone: str = "America/New_York",
    publish_times: list[list[int]] | None = None,
) -> list[str]:
    """Return *video_count* ISO 8601 UTC publish datetimes.

    Videos are assigned round-robin across *publish_times* slots,
    starting from the next Monday after today.

    Parameters
    ----------
    video_count:
        Number of publish slots to generate.
    timezone:
        IANA timezone name for the publish times.
    publish_times:
        List of ``[hour, minute]`` pairs (24-hour clock) in *timezone*.
        Defaults to ``[[8, 0], [18, 0]]`` (8 AM and 6 PM).
    """
    if publish_times is None:
        publish_times = [[8, 0], [18, 0]]

    tz = ZoneInfo(timezone)
    now = datetime.now(tz)

    # Find next Monday
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # always next week
    start_date = (now + timedelta(days=days_until_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    schedule: list[str] = []
    day_offset = 0
    for i in range(video_count):
        slot = i % len(publish_times)
        if i > 0 and slot == 0:
            day_offset += 1
        hour, minute = publish_times[slot]
        publish_dt = start_date.replace(hour=hour, minute=minute) + timedelta(
            days=day_offset
        )
        utc_dt = publish_dt.astimezone(ZoneInfo("UTC"))
        schedule.append(utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

    return schedule
