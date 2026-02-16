"""Send an email notification with the pipeline results summary.

Usage (standalone):
    python notify.py output/results.json

Environment variables:
    NOTIFY_EMAIL_FROM     Sender Gmail address
    NOTIFY_EMAIL_TO       Recipient email address
    NOTIFY_EMAIL_APP_PASSWORD  Gmail App Password (NOT your regular password)
"""

import json
import logging
import os
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)


def send_notification(results_path: str | Path) -> None:
    """Read results.json and send an email summary."""
    email_from = os.environ.get("NOTIFY_EMAIL_FROM")
    email_to = os.environ.get("NOTIFY_EMAIL_TO")
    app_password = os.environ.get("NOTIFY_EMAIL_APP_PASSWORD")

    if not all([email_from, email_to, app_password]):
        logger.warning(
            "Email notification skipped: set NOTIFY_EMAIL_FROM, "
            "NOTIFY_EMAIL_TO, and NOTIFY_EMAIL_APP_PASSWORD env vars"
        )
        return

    results_path = Path(results_path)
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))
    succeeded = sum(1 for r in results if r["success"])
    total = len(results)

    subject = f"Video Pipeline: {succeeded}/{total} succeeded"
    if succeeded == total:
        subject = f"Video Pipeline: All {total} videos uploaded"

    lines = [f"{succeeded}/{total} videos uploaded successfully.\n"]
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        detail = r.get("video_url") or r.get("error") or "no details"
        lines.append(f"  [{status}] {r['title'] or r['topic']}")
        lines.append(f"           {detail}")
    body = "\n".join(lines)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(email_from, app_password)
        smtp.send_message(msg)

    logger.info(f"Notification email sent to {email_to}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <results.json>")
        sys.exit(1)
    send_notification(sys.argv[1])
