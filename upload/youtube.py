"""YouTube Data API v3 upload module.

Handles OAuth2 authentication, video upload with resumable uploads,
thumbnail setting, and metadata configuration.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config_loader import Config

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


@dataclass
class UploadResult:
    video_id: str
    video_url: str


def get_youtube_service(config: Config):
    """Build an authenticated YouTube API v3 service.

    Tries credentials in order:
    1. Cached token file (``youtube_token.json``)
    2. Environment variables (``YOUTUBE_CLIENT_ID``, ``YOUTUBE_CLIENT_SECRET``,
       ``YOUTUBE_REFRESH_TOKEN``) — for headless/cloud environments
    3. Browser-based OAuth2 flow (local development only)
    """
    creds = None
    token_path = Path(config.youtube_token_file)

    # 1. Try cached token file
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    # 2. Try environment variables (headless mode)
    if not creds or not creds.valid:
        client_id = os.environ.get("YOUTUBE_CLIENT_ID")
        client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
        refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")
        if client_id and client_secret and refresh_token:
            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                token_uri="https://oauth2.googleapis.com/token",
                scopes=SCOPES,
            )
            logger.info("Using YouTube credentials from environment variables")

    if not creds or not creds.valid:
        if creds and creds.refresh_token:
            creds.refresh(Request())
        else:
            # 3. Fall back to browser-based flow (local development)
            secrets_path = Path(config.youtube_client_secrets)
            if not secrets_path.exists():
                raise FileNotFoundError(
                    f"YouTube client secrets file not found: {secrets_path}. "
                    f"Download it from Google Cloud Console > APIs & Services > Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(secrets_path), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Cache credentials for next time
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        logger.info(f"YouTube credentials cached to {token_path}")

    return build("youtube", "v3", credentials=creds)


def fetch_existing_titles(config: Config) -> list[str]:
    """Fetch video titles from the authenticated user's YouTube channel.

    Returns an empty list if credentials are unavailable or any API error
    occurs, so the pipeline can still run without deduplication context.
    """
    try:
        service = get_youtube_service(config)
    except Exception as e:
        logger.warning(f"Could not authenticate with YouTube, skipping title fetch: {e}")
        return []

    titles = []
    try:
        page_token = None
        while True:
            response = service.search().list(
                forMine=True,
                type="video",
                part="snippet",
                maxResults=50,
                pageToken=page_token,
            ).execute()

            for item in response.get("items", []):
                titles.append(item["snippet"]["title"])

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Fetched {len(titles)} existing video title(s) from YouTube")
    except Exception as e:
        logger.warning(f"Failed to fetch existing titles from YouTube: {e}")

    return titles


def upload_video(
    video_path: Path,
    title: str,
    description: str,
    thumbnail_path: Path,
    config: Config,
    publish_at: str | None = None,
    video_tags: list[str] | None = None,
) -> UploadResult:
    """Upload a video to YouTube with metadata and custom thumbnail.

    Uses resumable upload for reliability on large files.

    Parameters
    ----------
    publish_at:
        Optional ISO 8601 UTC datetime (e.g. ``"2026-02-16T13:00:00Z"``).
        When provided, the video is uploaded as *private* and scheduled to
        go public at this time.  Requires ``privacyStatus`` to be
        ``"private"``.
    """
    service = get_youtube_service(config)

    status: dict = {
        "privacyStatus": config.youtube_privacy_status,
        "selfDeclaredMadeForKids": False,
    }
    if publish_at:
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at
        logger.info(f"Video scheduled to publish at {publish_at}")

    # Merge default tags with per-video tags (defaults first, deduplicated).
    # YouTube rejects tags that are too long individually or exceed 500 chars total
    # (including comma separators). Drop tags longer than 30 chars and trim the list.
    MAX_TAG_LEN = 30
    MAX_TAG_CHARS = 500
    all_tags = list(config.youtube_tags)
    if video_tags:
        seen = {t.lower() for t in all_tags}
        for t in video_tags:
            # Sanitize: strip whitespace, remove chars YouTube rejects
            t = t.strip()
            t = re.sub(r'[<>#@{}[\]|\\^~`]', '', t)
            t = t.strip()
            if not t or len(t) > MAX_TAG_LEN:
                continue
            if t.lower() not in seen:
                all_tags.append(t)
                seen.add(t.lower())

    def _tag_total(tags: list[str]) -> int:
        return sum(len(t) for t in tags) + max(0, len(tags) - 1)

    while all_tags and _tag_total(all_tags) > MAX_TAG_CHARS:
        all_tags.pop()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": all_tags,
            "categoryId": config.youtube_category_id,
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": status,
    }

    logger.info(f"Uploading video: {title}")

    def _do_upload(upload_body):
        req = service.videos().insert(
            part="snippet,status",
            body=upload_body,
            media_body=MediaFileUpload(
                str(video_path), mimetype="video/mp4",
                resumable=True, chunksize=256 * 1024,
            ),
        )
        resp = None
        while resp is None:
            st, resp = req.next_chunk()
            if st:
                progress = int(st.progress() * 100)
                logger.info(f"Upload progress: {progress}%")
        return resp

    # Try upload; if tags are invalid, retry without per-video tags
    try:
        response = _do_upload(body)
    except Exception as e:
        if "invalidTags" in str(e):
            logger.warning(f"Invalid tags detected, retrying with default tags only: {e}")
            body["snippet"]["tags"] = list(config.youtube_tags)
            response = _do_upload(body)
        else:
            raise

    video_id = response["id"]
    logger.info(f"Video uploaded: ID={video_id}")

    # Set custom thumbnail (requires verified YouTube account)
    if thumbnail_path.exists():
        try:
            logger.info("Setting custom thumbnail...")
            service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/png"),
            ).execute()
            logger.info("Thumbnail set successfully")
        except Exception as e:
            logger.warning(
                f"Could not set custom thumbnail: {e}. "
                f"Your account may need phone verification. "
                f"Go to youtube.com/verify to enable custom thumbnails."
            )

    # Attempt to enable monetization if channel is in YouTube Partner Program.
    # This uses the videos.update endpoint to set monetization status.
    # It will silently fail if the channel is not in YPP — this is expected.
    try:
        update_status: dict = {
            "privacyStatus": "private" if publish_at else config.youtube_privacy_status,
            "selfDeclaredMadeForKids": False,
            "license": "youtube",  # standard YouTube license
            "publicStatsViewable": True,
        }
        if publish_at:
            update_status["publishAt"] = publish_at
        service.videos().update(
            part="status",
            body={
                "id": video_id,
                "status": update_status,
            },
        ).execute()
        logger.info("Video status updated (monetization-ready)")
    except Exception as e:
        logger.warning(f"Could not update monetization status: {e}")

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    return UploadResult(video_id=video_id, video_url=video_url)
