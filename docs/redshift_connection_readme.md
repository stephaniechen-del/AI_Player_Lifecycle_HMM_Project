# Redshift Connection Kit

This folder is a portable connection package for querying Redshift through the bastion SSH tunnel.

## Setup

1. Copy `.env.example` to `.env`.
2. Fill in `REDSHIFT_USER` and `REDSHIFT_PASSWORD`.
3. Put the bastion PEM file at:

```text
secrets/oceanhunter-prod-bastion-ec2.pem
```

4. Lock down the key permissions:

```bash
chmod 600 secrets/oceanhunter-prod-bastion-ec2.pem
```

5. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Query One User

```bash
python3 query_public_bullet_user.py --user-id USER_ID --output-csv outputs/user_bullet.csv
```

The script connects through the SSH tunnel and runs:

```sql
SELECT *
FROM "transform-agfish-game".public.bullet
WHERE CAST(user_id AS VARCHAR) = %s
```

## Notes

- `.env`, `secrets/`, and `outputs/` are ignored by git.
- Keep credentials out of chat and documents. Share only this folder with local `.env` and `secrets/` when the AI/runtime needs database access.
- If credentials were pasted into chat, rotate the Redshift password and bastion key.

