import json
import hashlib
import logging

from pywebpush import WebPushException, webpush

from ai_hub.config import VAPID_PRIVATE_KEY_PATH, VAPID_SUBJECT
from ai_hub.logging_config import log_event, setup_logging
from ai_hub.memory.sqlite_db import list_push_subscriptions

logger = logging.getLogger(__name__)
setup_logging()


def _endpoint_fingerprint(endpoint: str) -> str:
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]


def send_test_push() -> None:
    send_notification(
        title="AI Hub",
        body="Test-Push erfolgreich. Dein Setup funktioniert.",
        url="/",
    )


def send_notification(title: str, body: str, url: str = "/") -> None:
    subscriptions = list_push_subscriptions()

    if not subscriptions:
        log_event(logger, "push_skipped", subscriptions=0, url=url)
        return

    payload = {
        "title": title,
        "body": body,
        "url": url,
    }

    for item in subscriptions:
        subscription = item["subscription"]
        endpoint_hash = _endpoint_fingerprint(subscription["endpoint"])

        try:
            webpush(
                subscription_info=subscription,
                data=json.dumps(payload),
                vapid_private_key=VAPID_PRIVATE_KEY_PATH,
                vapid_claims={"sub": VAPID_SUBJECT},
            )
            log_event(logger, "push_sent", endpoint_hash=endpoint_hash, url=url)
        except WebPushException as ex:
            log_event(logger, "push_failed", endpoint_hash=endpoint_hash, error=repr(ex))

            if ex.response is not None:
                log_event(logger, "push_failed_http", endpoint_hash=endpoint_hash, status_code=ex.response.status_code)
                try:
                    log_event(logger, "push_failed_body", endpoint_hash=endpoint_hash, response_text=ex.response.text)
                except Exception:
                    pass
