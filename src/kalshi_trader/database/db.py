"""SQLite engine/session management (WAL mode so the dashboard can read while we write)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from kalshi_trader.database.tables import Base


def _set_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.close()


def add_missing_columns(engine: Engine) -> list[str]:
    """Additive schema migration: ``create_all`` creates new tables but never alters
    existing ones, so columns added to the ORM since a database was created are appended
    with ``ALTER TABLE ... ADD COLUMN`` (nullable, with the ORM's scalar default if any).
    Columns are never dropped or changed, so older code can still read the database."""
    added = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {
                row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")')
            }
            if not existing:
                continue
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" '
                ddl += column.type.compile(dialect=engine.dialect)
                default = column.default.arg if column.default is not None else None
                if isinstance(default, bool):
                    ddl += f" DEFAULT {int(default)}"
                elif isinstance(default, int | float):
                    ddl += f" DEFAULT {default}"
                elif isinstance(default, str):
                    ddl += " DEFAULT '" + default.replace("'", "''") + "'"
                conn.exec_driver_sql(ddl)
                added.append(f"{table.name}.{column.name}")
    return added


class Database:
    def __init__(self, path: Path | str) -> None:
        if str(path) == ":memory:":
            self.engine: Engine = create_engine(
                "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        else:
            db_path = Path(path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False, "timeout": 30},
            )
        event.listen(self.engine, "connect", _set_sqlite_pragmas)
        Base.metadata.create_all(self.engine)
        add_missing_columns(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
