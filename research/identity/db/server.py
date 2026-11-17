"""Initializes the pgvector extension and creates tables against the configured database.

Run only with an explicit research opt-in:
    PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 python -m research.identity.db.server
"""

import os

if os.getenv("PGOGGLES_IDENTITY_RESEARCH_ENABLED") != "1":
    raise SystemExit(
        "Identity research is disabled. Set "
        "PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 to run this module."
    )

from sqlalchemy import text

from .database import engine
from .tables import Base


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    init_db()
