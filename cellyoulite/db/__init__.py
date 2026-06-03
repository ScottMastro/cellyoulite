"""SQLite persistence layer. See DB.md for the schema and migration plan.

Public entry points:
- connection.connect() / connection.db_path()
- migrate.migrate()  (run on app startup)
- repo.*             (typed read/write helpers; no SQL in route handlers)
"""
