from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from album_memory.config import MemoryConfig


class Database:
    """Lazy SQLAlchemy boundary.

    Engine construction happens only when AlbumMemory first needs storage.
    SQLAlchemy does not open a connection until session() is entered.
    """

    def __init__(self, config: MemoryConfig):
        self.engine = create_engine(
            config.database.resolved_url(),
            echo=config.database.echo,
            pool_pre_ping=True,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
        )
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
