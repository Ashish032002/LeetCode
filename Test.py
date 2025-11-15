import os
import pandas as pd
from sqlalchemy import text
from db_config import get_db_connection

def main():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "datasets", "world_cities.xlsx")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Read Excel
    df = pd.read_excel(path)

    # Normalize column names
    rename = {}
    for c in df.columns:
        lc = c.strip().lower().replace(" ", "")

        if lc in ["city"]:
            rename[c] = "city"
        elif lc in ["country", "nation"]:
            rename[c] = "country"
        elif lc in ["iso2"]:
            rename[c] = "iso2"
        elif lc in ["iso3"]:
            rename[c] = "iso3"

    df = df.rename(columns=rename)

    # Ensure required columns exist
    for col in ["city", "country", "iso2", "iso3"]:
        if col not in df.columns:
            df[col] = None

    # Clean values
    df["city"] = df["city"].astype(str).str.strip().str.title()
    df["country"] = df["country"].astype(str).str.strip().str.title()
    df["iso2"] = df["iso2"].astype(str).str.strip().str.upper()
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()

    # Insert into PostgreSQL
    eng = get_db_connection()

    with eng.begin() as con:
        # Empty old table
        con.execute(text("DELETE FROM ref.world_cities"))

        # Insert new records
        df.to_sql("world_cities", 
                  con, 
                  schema="ref", 
                  if_exists="append", 
                  index=False)

    print("world_cities uploaded successfully!")

if __name__ == "__main__":
    main()

with eng.begin() as con:
    # Empty the table but keep structure
    con.execute(text("TRUNCATE TABLE ref.world_cities RESTART IDENTITY"))

    # Insert all rows again
    df.to_sql(
        "world_cities",
        con,
        schema="ref",
        if_exists="append",
        index=False
    )

