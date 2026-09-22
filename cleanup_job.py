import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from config import get_settings
from document_store import document_store
import structlog

logger = structlog.get_logger()


def cleanup_job():
    settings = get_settings()
    output_dir = Path(settings.output_dir)
    if not output_dir.exists():
        return

    now = datetime.utcnow()
    ttl = timedelta(days=settings.cleanup_ttl_days)

    for img_path in output_dir.glob("*.png"):
        # Expect naming: {doc_id}_page_{num}.png
        stem = img_path.stem
        if "_page_" not in stem:
            continue
        doc_id = stem.rsplit("_page_", 1)[0]

        meta = document_store.get_meta(doc_id)
        should_delete = False
        reason = ""
        if not meta:
            should_delete = True
            reason = "no_metadata"
        elif (now - datetime.fromisoformat(meta.created_at)) > ttl:
            should_delete = True
            reason = "ttl_expired"

        if should_delete:
            try:
                img_path.unlink()
                logger.info("cleanup.deleted", file=img_path.name, doc_id=doc_id, reason=reason)
            except Exception as e:
                logger.warning("cleanup.failed", file=img_path.name, error=str(e))


def start_scheduler():
    settings = get_settings()
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        cleanup_job,
        trigger=CronTrigger(hour=settings.cleanup_schedule_hour, minute=0),
        id="daily_cleanup",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("cleanup.scheduler_started", hour=settings.cleanup_schedule_hour)
    return scheduler