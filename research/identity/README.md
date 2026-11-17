# Identity Research

This optional package preserves the legacy face-vector database experiments for
consented local datasets. It is not part of the default pgoggles SDK install.

Install its dependencies explicitly:

```bash
PGOGGLES_INSTALL_IDENTITY_RESEARCH=1 ./setup.sh
```

## Layout

```
research/identity/
| - db/
|     | - database.py   # SQLAlchemy engine/session setup (reads PGOGGLES_DATABASE_URL)
|     | - schema.py      # shared schema constants (EMBEDDING_DIM = 512)
|     | - tables.py      # ORM models (Base, Person)
|     | - queries.py     # query interface: add_person, find_nearest, get_person_by_id, delete_person(s)
|     | - server.py      # one-shot init script: creates pgvector extension + tables
| - embeddings/
|     | - face_embedder.py  # FaceEmbedder: numpy ndarray frame -> 512-dim embedding (insightface)
```

This is the fetch/lookup half of the identity pipeline (frame -> embedding -> nearest
`Person` record), ported from the `pgoggles-face-module` prototype so it lives next to
the SDK it plugs into. The scraper that populates this same database (name + photo ->
embedding -> row) remains isolated in `pgoggles-face-module`; both projects talk to the
same Postgres + pgvector database via `PGOGGLES_DATABASE_URL`.

## Database

- Postgres + the `pgvector` extension.
- Single table so far: `persons(id: Integer PK, name: String, embedding: Vector(512))`.
- `db/schema.py` holds `EMBEDDING_DIM = 512` (matches insightface's default output
  size, e.g. buffalo_l) — table definitions and the embedder both import this
  constant rather than hardcoding 512.
- Connection string comes from the `PGOGGLES_DATABASE_URL` env var, loaded from a
  `.env` file at the repo root via `python-dotenv`. There is no in-code fallback —
  `db/database.py` raises `KeyError` if the var isn't set.
- Nearest-neighbor lookup uses pgvector's `l2_distance` operator via SQLAlchemy
  (`db/queries.py::find_nearest`).

Run only with an explicit research opt-in:

```bash
PGOGGLES_IDENTITY_RESEARCH_ENABLED=1 python -m research.identity.db.server
```

## Demo

See `packages/python-sdk/examples/face_fetch_demo.py` for the full pipeline wired
into the pgoggles SDK: onDetect("person") -> embed the frame on a background
worker thread -> look up the embedding in this package's database -> draw the
matched name as a bounding-box label for as long as the person stays detected.
