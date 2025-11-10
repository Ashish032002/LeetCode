import os
import pandas as pd
from sqlalchemy import text
from db_config import get_db_connection

def pick(cols, cands):
    """Find the first matching column name from a list of candidates."""
    look = {c.lower().replace(' ', ''): c for c in cols}
    for cand in cands:
        k = cand.lower().replace(' ', '')
        if k in look:
            return look[k]
    return None

def main():
    here = os.path.dirname(__file__)
    for path in [os.path.join(here, 'Countries.xlsx'), 'datasets/Countries.xlsx']:
        if os.path.exists(path):
            df = pd.read_excel(path)
            break
    else:
        raise FileNotFoundError('Countries.xlsx not found.')
    
    country_col = pick(df.columns, ['Country', 'country_name', 'name'])
    if not country_col:
        raise ValueError('Countries.xlsx is missing a column for country names (e.g., "Country", "name").')
    
    out = pd.DataFrame({'name': df[country_col].astype(str).str.strip().str.title()}).dropna().drop_duplicates()
    eng = get_db_connection()
    with eng.begin() as con:
        out.to_sql('countries', con, schema='av', if_exists='replace', index=False, method='multi')
    print(f'Loaded {len(out)} rows into av.countries')

if __name__ == '__main__':
    main()