import os
import pandas as pd
from sqlalchemy import text
from db_config import get_db_connection


def main():
    here = os.path.dirname(__file__)
    
    # Locate the Excel file
    for path in [os.path.join(here, 'input_addresses.xlsx'), r'C:\Users\ashish.singh\Desktop\address_validator_v18_next_countries\datasets\input_addresses.xlsx']:
        if os.path.exists(path):
            df = pd.read_excel(path)
            break
    else:
        raise FileNotFoundError('Input_addresses.xlsx not found.')

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure all required columns exist
    required_cols = ['address1', 'address2', 'address3', 'city', 'state', 'pincode', 'country']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ''

    # Select only the required columns, fill missing values, and take first 10,000 rows
    df = df[required_cols].fillna('').head(10_000)

    # Connect to the database
    eng = get_db_connection()
    with eng.begin() as con:
        # Truncate the existing table
        con.execute(text('TRUNCATE av.input_addresses RESTART IDENTITY CASCADE'))

        # Prepare SQL insert statement
        insert_sql = text("""
            INSERT INTO av.input_addresses(address1, address2, address3, city, state, pincode, country)
            VALUES(:a1, :a2, :a3, :c, :s, :p, :country)
        """)

        # Bulk insert using executemany style (faster than row-by-row)
        records = [
            {
                'a1': r['address1'],
                'a2': r['address2'],
                'a3': r['address3'],
                'c': r['city'],
                's': r['state'],
                'p': str(r['pincode']),
                'country': r['country']
            }
            for _, r in df.iterrows()
        ]
        con.execute(insert_sql, records)

    print(f'Loaded {len(df)} rows into av.input_addresses')


if __name__ == '__main__':
    main()
