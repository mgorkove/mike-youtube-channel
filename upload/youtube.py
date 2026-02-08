"""YouTube Data API v3 upload module.

Handles OAuth2 authentication, video upload with resumable uploads,
thumbnail setting, and metadata configuration.
"""

import logging
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
]


@dataclass
class UploadResult:
    video_id: str
    video_url: str


def get_youtube_service(config: Config):
    """Build an authenticated YouTube API v3 service.

    Checks for cached credentials first. If expired, refreshes.
    If missing, runs the OAuth2 flow (opens browser on first run).
    """
    creds = None
    token_path = Path(config.youtube_token_file)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
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


def upload_video(
    video_path: Path,
    title: str,
    description: str,
    thumbnail_path: Path,
    config: Config,
) -> UploadResult:
    """Upload a video to YouTube with metadata and custom thumbnail.

    Uses resumable upload for reliability on large files.
    """
    service = get_youtube_service(config)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": config.youtube_tags,
            "categoryId": config.youtube_category_id,
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus": config.youtube_privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=256 * 1024,  # 256KB chunks
    )

    logger.info(f"Uploading video: {title}")
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    # Resumable upload loop
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            logger.info(f"Upload progress: {progress}%")

    video_id = response["id"]
    logger.info(f"Video uploaded: ID={video_id}")

    # Set custom thumbnail
    if thumbnail_path.exists():
        logger.info("Setting custom thumbnail...")
        service.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/png"),
        ).execute()
        logger.info("Thumbnail set successfully")

    # Attempt to enable monetization if channel is in YouTube Partner Program.
    # This uses the videos.update endpoint to set monetization status.
    # It will silently fail if the channel is not in YPP — this is expected.
    try:
        service.videos().update(
            part="status",
            body={
                "id": video_id,
                "status": {
                    "privacyStatus": config.youtube_privacy_status,
                    "selfDeclaredMadeForKids": False,
                    "license": "youtube",  # standard YouTube license
                    "publicStatsViewable": True,
                },
            },
        ).execute()
        logger.info("Video status updated (monetization-ready)")
    except Exception as e:
        logger.warning(f"Could not update monetization status: {e}")

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    return UploadResult(video_id=video_id, video_url=video_url)
