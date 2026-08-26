from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url


def secure_database_url(database_url: str) -> URL:
    """Return a psycopg URL with encrypted PostgreSQL connections enforced."""
    try:
        url = make_url(database_url)
    except Exception as exc:
        raise ValueError("DATABASE_URL is not a valid database URL.") from exc

    if url.get_backend_name() != "postgresql":
        raise ValueError("DATABASE_URL must use PostgreSQL.")

    host = (url.host or "").lower()
    # Neon pooler hosts include "-pooler" before the regional hostname.
    if host.endswith(".neon.tech") and "-pooler." not in host:
        raise ValueError("DATABASE_URL must use Neon's pooled connection host.")

    query = dict(url.query)
    sslmode = query.get("sslmode")
    # Upgrade missing or unsafe SSL modes while preserving stricter verification.
    if not isinstance(sslmode, str) or sslmode.lower() not in {
        "require",
        "verify-ca",
        "verify-full",
    }:
        query["sslmode"] = "require"
    query.setdefault("connect_timeout", "5")
    return url.set(drivername="postgresql+psycopg", query=query)


def create_database_engine(database_url: str) -> Engine:
    # Creating an engine does not create tables; Alembic owns the schema.
    return create_engine(
        secure_database_url(database_url),
        pool_pre_ping=True,
        pool_recycle=300,
        hide_parameters=True,
    )
