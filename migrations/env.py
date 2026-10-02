"""Alembic environment.

The app talks to Postgres with psycopg directly rather than through an ORM, so
there is no SQLAlchemy metadata to autogenerate against and `--autogenerate` is
not available. Migrations here are hand-written revisions, which is the normal
pattern for a non-ORM project: each file states the change it makes and you
review it before it touches a database.

The URL comes from `app.db.dsn()`, the same source the application uses, so a
migration can never be pointed at a different database than the app by accident.
An explicit `sqlalchemy.url` in alembic.ini still overrides it, which is what you
want for a one-off.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from app import db

config_alembic = context.config

if config_alembic.config_file_name is not None:
    fileConfig(config_alembic.config_file_name)

target_metadata = None

url = config_alembic.get_main_option("sqlalchemy.url") or db.dsn()


def _to_sqlalchemy_url(dsn: str):
    """Turn any DSN the app accepts into a SQLAlchemy URL on the psycopg 3 driver.

    Two things to handle:

    - SQLAlchemy reads `postgresql://` as psycopg2, which is not a dependency
      here. Only the driver name is changed; host, credentials and query
      parameters are left alone.
    - `DATABASE_URL` may be given in libpq keyword form
      (`host=... dbname=... user=...`), which SQLAlchemy cannot parse at all.
      That form is rebuilt into a real URL rather than passed through.
    """
    if "://" not in dsn:
        from psycopg.conninfo import conninfo_to_dict
        from sqlalchemy.engine import URL

        parts = conninfo_to_dict(dsn)
        return URL.create(
            "postgresql+psycopg",
            host=parts.get("host", "localhost"),
            port=parts.get("port"),
            username=parts.get("user"),
            password=parts.get("password"),
            database=parts.get("dbname"),
        )

    scheme, _, rest = dsn.partition("://")
    if "+" in scheme:
        return dsn
    return f"{scheme}+psycopg://{rest}"


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it, for review."""
    context.configure(
        url=_to_sqlalchemy_url(url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_to_sqlalchemy_url(url), poolclass=None)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()