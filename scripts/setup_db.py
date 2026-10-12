"""Create the application PostgreSQL role and database (idempotent, non-destructive).

Reads .env: POSTGRES_ADMIN_USER / POSTGRES_ADMIN_PASSWORD (superuser, used only here) and
DB_NAME / DB_USER / DB_PASSWORD / DB_HOST / DB_PORT. Never drops anything.
The role gets CREATEDB so the test runner can create its temporary test database.

    python scripts/setup_db.py
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> int:
    admin_pw = os.environ.get("POSTGRES_ADMIN_PASSWORD", "")
    if not admin_pw:
        print("POSTGRES_ADMIN_PASSWORD is empty in .env; cannot create role/database.", file=sys.stderr)
        return 2
    host, port = os.environ.get("DB_HOST", "localhost"), os.environ.get("DB_PORT", "5432")
    name, user, pw = os.environ["DB_NAME"], os.environ["DB_USER"], os.environ["DB_PASSWORD"]
    with psycopg.connect(host=host, port=port, dbname="postgres", autocommit=True,
                         user=os.environ.get("POSTGRES_ADMIN_USER", "postgres"), password=admin_pw) as conn:
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,)).fetchone()
        if exists:
            conn.execute(sql.SQL("ALTER ROLE {} WITH LOGIN CREATEDB PASSWORD {}").format(
                sql.Identifier(user), sql.Literal(pw)))
            print(f"role {user}: updated password/privileges")
        else:
            conn.execute(sql.SQL("CREATE ROLE {} WITH LOGIN CREATEDB PASSWORD {}").format(
                sql.Identifier(user), sql.Literal(pw)))
            print(f"role {user}: created")
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            print(f"database {name}: already exists (left untouched)")
        else:
            conn.execute(sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8'").format(
                sql.Identifier(name), sql.Identifier(user)))
            print(f"database {name}: created")
    with psycopg.connect(host=host, port=port, dbname=name, user=user, password=pw) as conn:
        print("connected as app user:", conn.execute("SELECT version()").fetchone()[0][:40])
    return 0


if __name__ == "__main__":
    sys.exit(main())
