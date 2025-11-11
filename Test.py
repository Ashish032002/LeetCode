# create_state_abbrev_schema.py
# Creates schema + table for Indian State Abbreviations

from sqlalchemy import text
from db_config import get_engine

def create_schema_and_table():
    engine = get_engine()

    ddl_statements = [
        # Create schema if not exists
        """
        CREATE SCHEMA IF NOT EXISTS ref;
        """,

        # Create table
        """
        CREATE TABLE IF NOT EXISTS ref.indian_state_abbrev (
            id SERIAL PRIMARY KEY,
            state TEXT NOT NULL,
            abbreviation TEXT NOT NULL,
            alias TEXT NULL
        );
        """,

        # Ensure uniqueness (case-insensitive)
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_state_abbrev_unique
        ON ref.indian_state_abbrev (upper(state), upper(abbreviation));
        """
    ]

    with engine.begin() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))

    print("✅ Schema + table created successfully: ref.indian_state_abbrev")

B
if __name__ == "__main__":
    create_schema_and_table()

# upload_state_abbrev.py
# Upload abbreviations from CSV/Excel and normalize them

import re
import argparse
import pandas as pd
from sqlalchemy import text
from db_config import get_engine

# Seed extra known alternates (based on your screenshots & Indian usage patterns)
SEED_ALTERNATES = {
    "Telangana": ["TS", "TG", "TE"],
    "Uttarakhand": ["UK", "UA", "UC"],
    "Odisha": ["OD", "OR", "ORS"],
    "Delhi": ["DL", "DEL", "ND"],
    "Jammu and Kashmir": ["JK", "J&K"],
    "Jharkhand": ["JH", "JHD", "JD"],
    "Chhattisgarh": ["CG", "CHG"]
}

def normalize_abbrev(raw):
    # split on non-letters and uppercase each
    parts = re.split(r"[^A-Za-z]+", str(raw))
    return [p.strip().upper() for p in parts if p.strip()]

def upload_abbreviations(path, sheet=None, replace=False):
    engine = get_engine()

    # Load file
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)

    # Normalize expected column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Expected: state, abbreviation (can have multiple)
    if "state" not in df.columns or "abbreviation" not in df.columns:
        raise ValueError("Input must contain columns: state, abbreviation")

    rows = []
    for _, row in df.iterrows():
        state = str(row["state"]).strip().title()
        ab_list = normalize_abbrev(row["abbreviation"])
        for ab in ab_list:
            rows.append((state, ab, None))

    # Add seeds
    for state, abbrevs in SEED_ALTERNATES.items():
        for ab in abbrevs:
            rows.append((state.title(), ab.upper(), None))

    # Sort and deduplicate
    rows = list(set(rows))

    with engine.begin() as conn:
        if replace:
            conn.execute(text("TRUNCATE TABLE ref.indian_state_abbrev;"))
            print("🧹 Table truncated before reload.")

        for state, ab, alias in rows:
            conn.execute(text("""
                INSERT INTO ref.indian_state_abbrev (state, abbreviation, alias)
                VALUES (:state, :abbreviation, :alias)
                ON CONFLICT (upper(state), upper(abbreviation)) DO NOTHING;
            """), {"state": state, "abbreviation": ab, "alias": alias})

    print(f"✅ Uploaded {len(rows)} normalized abbreviation entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to CSV/Excel abbreviations file")
    parser.add_argument("--sheet", required=False, help="Excel sheet name (if using Excel)")
    parser.add_argument("--replace", action="store_true", help="Truncate before inserting")
    args = parser.parse_args()

    upload_abbreviations(args.path, sheet=args.sheet, replace=args.replace)
