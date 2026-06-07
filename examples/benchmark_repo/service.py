import logging

logger = logging.getLogger(__name__)
RETRY_BACKOFF_MS = 250
REQUEST_TIMEOUT_MS = 3000


class Metrics:
    def emit_latency(self, name: str, value: float) -> None:
        pass


metrics = Metrics()


def retry_delay(attempt: int) -> int:
    logger.info("retry configured")
    return RETRY_BACKOFF_MS * attempt


def load_user(user_id: str) -> dict:
    logger.info("loading user")
    metrics.emit_latency("user.load.latency", 12.5)
    return {"id": user_id, "timeout": REQUEST_TIMEOUT_MS}


def api_handler(request: dict) -> dict:
    return load_user(request["user_id"])

