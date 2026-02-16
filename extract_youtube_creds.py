#!/usr/bin/env python3
"""Extract YouTube OAuth credentials for cloud deployment.

Reads the local youtube_token.json and client_secrets.json files and
prints the three values needed as environment variables / GCP secrets:

    YOUTUBE_CLIENT_ID
    YOUTUBE_CLIENT_SECRET
    YOUTUBE_REFRESH_TOKEN

Usage:
    python extract_youtube_creds.py
"""

import json
import sys
from pathlib import Path


def main() -> None:
    token_path = Path("youtube_token.json")
    secrets_path = Path("client_secrets.json")

    if not token_path.exists():
        print(f"Error: {token_path} not found. Run the pipeline locally first to create it.")
        sys.exit(1)
    if not secrets_path.exists():
        print(f"Error: {secrets_path} not found.")
        sys.exit(1)

    token_data = json.loads(token_path.read_text())
    secrets_data = json.loads(secrets_path.read_text())

    # client_secrets.json can have keys under "installed" or "web"
    app_data = secrets_data.get("installed") or secrets_data.get("web")
    if not app_data:
        print("Error: Could not find 'installed' or 'web' key in client_secrets.json")
        sys.exit(1)

    client_id = app_data["client_id"]
    client_secret = app_data["client_secret"]
    refresh_token = token_data.get("refresh_token")

    if not refresh_token:
        print("Error: No refresh_token found in youtube_token.json. "
              "Re-run the OAuth flow locally.")
        sys.exit(1)

    print("Add these to GCP Secret Manager (or your .env file):\n")
    print(f"YOUTUBE_CLIENT_ID={client_id}")
    print(f"YOUTUBE_CLIENT_SECRET={client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN={refresh_token}")


if __name__ == "__main__":
    main()
