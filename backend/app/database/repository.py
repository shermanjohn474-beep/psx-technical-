"""Engine/session management and small repository helpers."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings
from app.database.models import AIAudit, AuditLog, Base


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    eng = create_engine(url, future=True, **kw)
    Base.metadata.create_all(eng)
    return eng


def session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def record_audit(actor: str, method: str, path: str, status: int, detail: str | None = None) -> None:
    with session_scope() as s:
        s.add(AuditLog(actor=actor, method=method, path=path[:300], status=status, detail=detail))


def record_ai_audit(record: dict) -> None:
    with session_scope() as s:
        s.add(AIAudit(report_id=record.get("report_id"), provider=record.get("provider", "?"), model=record.get("model"),
                      status=record.get("status", "?"), record=record))


def recent_audit(limit: int = 100) -> list[AuditLog]:
    with session_scope() as s:
        return list(s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)))
