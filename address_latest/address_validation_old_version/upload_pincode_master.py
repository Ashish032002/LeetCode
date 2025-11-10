import os, re, pandas as pd
from sqlalchemy import text
from db_config import get_db_connection

def norm_pin(s):
    if pd.isna(s): return None
    m=re.search(r'(\d{6})', str(s)); return m.group(1) if m else None

def clean_div_to_city(x:str)->str:
    s=str(x or '').strip()
    s=re.sub(r'\bdivision(?: office)?\b','',s,flags=re.I).strip()
    return re.sub(r'\s+',' ',s).title()

def pick(cols,cands):
    look={c.lower().replace(' ',''):c for c in cols}
    for cand in cands:
        k=cand.lower().replace(' ','')
        if k in look: return look[k]
    return None

def main():
    here=os.path.dirname(__file__)
    for path in [os.path.join(here,'pincode_master.xlsx'),'datasets/pincode_master.xlsx']:
        if os.path.exists(path):
            df=pd.read_excel(path); break
    else:
        raise FileNotFoundError('pincode_master.xlsx not found.')
    div=pick(df.columns,['DivisionName','Division','City','CityName','OfficeName'])
    st =pick(df.columns,['StateName','State','State_Name'])
    pin=pick(df.columns,['Pincode','Pin','Zip'])
    if not all([div,st,pin]): raise ValueError('pincode_master.xlsx missing needed columns')
    out=pd.DataFrame({'city':df[div].apply(clean_div_to_city),'state':df[st].astype(str).str.strip().str.title(),'pincode':df[pin].apply(norm_pin)})
    out=out.dropna(subset=['pincode']).drop_duplicates(subset=['pincode','city','state'])
    eng=get_db_connection()
    with eng.begin() as con:
        out.to_sql('pincode_ref', con, schema='av', if_exists='append', index=False)
    print(f'Loaded {len(out)} rows into av.pincode_ref')

if __name__=='__main__': main()
