from celery import shared_task


@shared_task
def ping() -> str:
    """Trivial task used to verify the Celery worker + Redis wiring."""
    return "pong"
