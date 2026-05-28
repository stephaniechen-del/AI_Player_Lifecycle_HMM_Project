import argparse
import json
import os
import socket
from contextlib import closing
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sshtunnel import SSHTunnelForwarder


ROOT = Path(__file__).resolve().parent


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env(name, default=None, required=False):
    value = os.environ.get(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resolve_path(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def query_public_bullet(user_id, limit=None):
    load_env_file(ROOT / ".env")

    redshift_host = env("REDSHIFT_HOST", required=True)
    redshift_port = int(env("REDSHIFT_PORT", "5439"))
    redshift_database = env("REDSHIFT_DATABASE", required=True)
    redshift_user = env("REDSHIFT_USER", required=True)
    redshift_password = env("REDSHIFT_PASSWORD", required=True)
    redshift_ssl = parse_bool(env("REDSHIFT_SSL", "true"))

    ssh_host = env("SSH_TUNNEL_HOST", required=True)
    ssh_port = int(env("SSH_TUNNEL_PORT", "22"))
    ssh_user = env("SSH_TUNNEL_USER", required=True)
    ssh_key_path = resolve_path(env("SSH_PRIVATE_KEY_PATH", required=True))
    local_port = find_free_port()

    sql = """
        SELECT *
        FROM "transform-agfish-game".public.bullet
        WHERE CAST(user_id AS VARCHAR) = %s
    """
    params = [str(user_id)]
    if limit is not None:
        sql += "\nLIMIT %s"
        params.append(int(limit))

    with SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_pkey=str(ssh_key_path),
        remote_bind_address=(redshift_host, redshift_port),
        local_bind_address=("127.0.0.1", local_port),
    ) as tunnel:
        connection = psycopg2.connect(
            host="127.0.0.1",
            port=tunnel.local_bind_port,
            dbname=redshift_database,
            user=redshift_user,
            password=redshift_password,
            sslmode="require" if redshift_ssl else "prefer",
            connect_timeout=20,
        )
        try:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                columns = [desc.name for desc in cursor.description]
        finally:
            connection.close()

    return {"user_id": str(user_id), "row_count": len(rows), "columns": columns, "rows": rows}


def write_csv(result, output_csv):
    output_path = Path(output_csv)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["rows"], columns=result["columns"]).to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Query Redshift public.bullet records for one user_id.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    result = query_public_bullet(args.user_id, args.limit)
    if args.output_csv:
        output_path = write_csv(result, args.output_csv)
        result["output_csv"] = str(output_path)
        result["rows"] = []

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

