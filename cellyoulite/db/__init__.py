"""SQLite persistence layer. schema.sql is the v1 snapshot; migrate.py holds
the ordered steps that bring it to the current shape.

Public entry points:
- connection.connect() / connection.db_path()
- migrate.migrate()  (run on app startup)
- repo.*             (typed read/write helpers; no SQL in route handlers)
"""
