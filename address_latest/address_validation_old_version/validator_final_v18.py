import os, re, json, pandas as pd, unicodedata
from sqlalchemy import text
from db_config import get_db_connection

try:
    from rapidfuzz import fuzz
    def sim(a,b): return fuzz.token_set_ratio(a or '', b or '')/100
except:
    import difflib
    def sim(a,b): return difflib.SequenceMatcher(None,(a or '').lower(),(b or '').lower()).ratio()

BATCH=10000
CITY_THR=0.80
STATE_THR=0.80
STOP_WORDS={'division','city','district','region','zone','south','north','east','west','office','so','bo','po','post','branch','rms','ho','do','co','nodal','sub','urban','rural'}
ALIASES={'bombay':'mumbai','calcutta':'kolkata','bengaluru':'bangalore','gurgaon':'gurugram','gurugram':'gurgaon','trivandrum':'thiruvananthapuram','madras':'chennai','nasik':'nashik','vadodara':'baroda','baroda':'vadodara'}

def strip_accents(s):
    return ''.join(ch for ch in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(ch))
def clean_text(s):
    s=strip_accents(str(s or '')).lower()
    s=re.sub(r'[^\w\s]',' ',' '.join(s.split()))
    s=re.sub(r'\s+',' ',s).strip()
    return s
def normalize_city(s):
    s=clean_text(s)
    toks=[ALIASES.get(t,t) for t in s.split() if t not in STOP_WORDS]
    return ' '.join(toks).strip().title()
def extract_pin(txt):
    if not txt: return None
    m=re.search(r'(?<!\d)(\d{6})(?!\d)', str(txt)); return m.group(1) if m else None
def load_state_map():
    here=os.path.dirname(__file__)
    import pandas as _pd
    df=_pd.read_csv(os.path.join(here,'datasets','abbreviation_list 1.csv'))
    df.columns=df.columns.str.strip().str.title()
    df['State']=df['State'].astype(str).str.strip().str.title()
    df['Abbreviation']=df['Abbreviation'].astype(str).str.strip().str.upper()
    s2a=dict(zip(df['State'],df['Abbreviation'])); a2s=dict(zip(df['Abbreviation'],df['State']))
    return s2a,a2s
def normalize_state(s, s2a, a2s):
    if s is None or str(s).strip()=='': return None
    t=clean_text(s).replace('.','').upper()
    if t in a2s: return a2s[t]
    return t.title()
def overall(csim, ssim):
    o=0.55*csim+0.45*ssim
    lvl='High' if o>=0.80 else ('Medium' if o>=0.65 else 'Low')
    return o,lvl
def evaluate(dfp, city_n, state_n, full_addr, s2a, a2s):
    cands=[]; addr_clean=clean_text(full_addr).title()
    for _,m in dfp.iterrows():
        mc=normalize_city(m['city']); ms=normalize_state(m['state'], s2a, a2s)
        cs=sim(mc, city_n)
        if sim(mc, addr_clean) >= 0.80: cs=max(cs,0.90)
        ss=sim(ms, state_n) if state_n else 0.0
        cands.append({'master_city':mc,'master_state':ms,'cs':cs,'ss':ss})
    cands.sort(key=lambda x:(x['cs'],x['ss']), reverse=True)
    return cands

def main():
    print('Starting validator v18 ...')
    eng=get_db_connection()
    s2a,a2s=load_state_map()
    import pandas as _pd
    with eng.begin() as con:
        m1=_pd.read_sql('SELECT city,state,pincode FROM av.master_ref', con)
        m2=_pd.read_sql('SELECT city,state,pincode FROM av.pincode_ref', con)
        inp=_pd.read_sql('SELECT * FROM av.input_addresses', con)
        countries = _pd.read_sql('SELECT name FROM av.countries', con)['name'].tolist()
    by_pin_m={k:v for k,v in m1.groupby('pincode')}
    by_pin_p={k:v for k,v in m2.groupby('pincode')}
    with eng.begin() as con:
        con.execute(text('TRUNCATE av.validation_result_final RESTART IDENTITY CASCADE'))
    total=len(inp); parts=(total+BATCH-1)//BATCH
    for bi in range(parts):
        s,e=bi*BATCH, min((bi+1)*BATCH,total)
        sub=inp.iloc[s:e].copy(); rows=[]
        for _,r in sub.iterrows():
            input_id=int(r['id'])
            a1,a2,a3 = r.get('address1',''), r.get('address2',''), r.get('address3','')
            city_in, state_in, pin_in, country_in = r.get('city',''), r.get('state',''), str(r.get('pincode','')), r.get('country','')
            full_addr=' '.join([str(x) for x in [a1,a2,a3,city_in,state_in,pin_in,country_in] if x and str(x).lower()!='nan'])
            pin=extract_pin(full_addr) or extract_pin(pin_in)
            city_n=normalize_city(city_in); state_n=normalize_state(state_in, s2a, a2s) or ''
            
            # Country detection logic
            all_found_countries = set()
            # Search in full address
            addr_clean_for_country = clean_text(full_addr) # Includes country_in
            found_countries = []
            for country in countries:
                score = sim(clean_text(country), addr_clean_for_country)
                if score > 0.8: # A threshold to consider a country as potentially mentioned
                    found_countries.append({'country': country, 'score': score})
                    all_found_countries.add(country)
            # Search in input country column
            country_in_clean = clean_text(country_in)
            for country in countries:
                if sim(clean_text(country), country_in_clean) > 0.8:
                    all_found_countries.add(country)
            found_countries.sort(key=lambda x: x['score'], reverse=True)
            
            flag='No'; reason=''; amb='none'; source='none'
            city_conf=state_conf=country_conf=0.0; best_city=city_n; best_state=state_n; best_country=None
            master_city=master_state=pincode_city=pincode_state=None; poss=[]; all_found_countries_list = []
            import pandas as _pd
            if pin and re.fullmatch(r'\d{6}', pin):
                c_m = evaluate(by_pin_m.get(pin, _pd.DataFrame()), city_n, state_n, full_addr, s2a, a2s) if pin in by_pin_m else []
                c_p = evaluate(by_pin_p.get(pin, _pd.DataFrame()), city_n, state_n, full_addr, s2a, a2s) if pin in by_pin_p else []
                match_m = any((c['cs']>=CITY_THR and (not state_n or c['ss']>=STATE_THR)) for c in c_m)
                match_p = any((c['cs']>=CITY_THR and (not state_n or c['ss']>=STATE_THR)) for c in c_p)
                if match_m or match_p:
                    if c_m: master_city,master_state = c_m[0]['master_city'],c_m[0]['master_state']
                    if c_p: pincode_city,pincode_state = c_p[0]['master_city'],c_p[0]['master_state']
                    all_c=(c_m+c_p) or [{'master_city':best_city,'master_state':best_state,'cs':0,'ss':0}]
                    all_c.sort(key=lambda x:(x['cs'],x['ss']), reverse=True)
                    top=all_c[0]
                    best_city,best_state=top['master_city'],top['master_state']
                    city_conf,state_conf=top['cs'],top['ss']
                    source='master_ref' if top in c_m else 'pincode_ref'
                    flag='No'; amb='none'; reason=''
                else:
                    flag='Yes'; amb='city/state mismatch'; reason='No strong match in either source'                    
                    if c_m: master_city,master_state = c_m[0]['master_city'],c_m[0]['master_state']
                    if c_p: pincode_city,pincode_state = c_p[0]['master_city'],c_p[0]['master_state']
                    poss = []
                    # dict1
                    poss.append({'master_city': master_city, 'master_state': master_state, 'cs': c_m[0]['cs'] if c_m else 0.0, 'ss': c_m[0]['ss'] if c_m else 0.0})
                    # dict2
                    poss.append({'in_country': list(all_found_countries)})
                    
                    # dict3 
                    in_cs = sim(city_n, master_city) if master_city else (sim(city_n, pincode_city) if pincode_city else 0.0)
                    in_ss = sim(state_n, master_state) if master_state else (sim(state_n, pincode_state) if pincode_state else 0.0)
                    poss.append({'in_city': city_n, 'in_state': state_n, 'cs': in_cs, 'ss': in_ss})
                    # dict3
                    poss.append({'pincode_city': pincode_city, 'pincode_state': pincode_state, 'cs': c_p[0]['cs'] if c_p else 0.0, 'ss': c_p[0]['ss'] if c_p else 0.0})

                    if c_m or c_p:
                        all_c=(c_m+c_p); all_c.sort(key=lambda x:(x['cs'],x['ss']), reverse=True)
                        top=all_c[0]
                        best_city,best_state=top['master_city'],top['master_state']
                        city_conf,state_conf=top['cs'],top['ss']
                        source='master_ref' if top in c_m else 'pincode_ref'
            else:
                flag='Yes'; amb='pincode not found'; reason='No/invalid pincode'
                poss = []
                # dict1
                poss.append({'master_city': None, 'master_state': None, 'cs': 0.0, 'ss': 0.0}) # master ref
                # dict2
                poss.append({'in_country': list(all_found_countries)})
                # dict3
                poss.append({'in_city': city_n, 'in_state': state_n, 'cs': 0.0, 'ss': 0.0}) # input city/state
                # dict4
                poss.append({'pincode_city': None, 'pincode_state': None, 'cs': 0.0, 'ss': 0.0}) # pincode ref

            if len(all_found_countries) > 1:
                flag = 'Yes'
                reason = 'Multiple countries found'
            best_country = ', '.join(all_found_countries) if all_found_countries else None
            overall_conf, level = overall( city_conf, state_conf )
            loc_txt=clean_text(' '.join([a1,a2,a3])).title()
            for w in set((best_city or '').split()):
                loc_txt=re.sub(rf'\b{re.escape(w)}\b','',loc_txt,flags=re.I)
            loc_txt=re.sub(r'\s+',' ',loc_txt).strip().title() or None
            rows.append({
                'input_id':input_id, 'address1':' '.join([str(x) for x in [a1, a2, a3] if x]).strip(), 'city':best_city,
                'state':best_state,'pincode':pin,'country':best_country, 'country_confidence': round(float(country_conf or 0), 3),
                'city_confidence':round(float(city_conf or 0), 3), 'state_confidence':round(float(state_conf or 0), 3),
                'overall_confidence':round(float(overall_conf or 0), 3), 'confidence_level':level,
                'flag':flag,'reason':reason,'ambiguity_type':amb,'source_used':source,
                'master_city':master_city,'master_state':master_state,'pincode_city':pincode_city,'pincode_state':pincode_state,
                'locality':loc_txt,'possible_addresses':json.dumps(poss),
                'in_address1':a1,'in_address2':a2,'in_address3':a3,'in_city':city_in,'in_state':state_in,'in_pincode':pin_in,'in_country':country_in
            })
        df=pd.DataFrame(rows)
        cols=['in_address1','in_address2','in_address3','in_city','in_state','in_pincode','in_country','input_id','address1','city','state','pincode','country','city_confidence','state_confidence','overall_confidence','confidence_level','flag','reason','ambiguity_type','source_used','master_city','master_state','pincode_city','pincode_state','locality','possible_addresses']
        df[cols].to_excel(f'datasets/validated_output_part{bi+1}_v18.xlsx', index=False)
        with eng.begin() as con:
            for _,x in df.iterrows():
                con.execute(text(""" 
                    INSERT INTO av.validation_result_final(
                        input_id,address1,city,state,pincode,country,country_confidence,
                        city_confidence,state_confidence,overall_confidence,confidence_level,
                        flag,reason,ambiguity_type,source_used,master_city,master_state,
                        pincode_city,pincode_state,locality,possible_addresses,in_address1,
                        in_address2,in_address3,in_city,in_state,in_pincode,in_country
                    ) VALUES(
                        :iid,:a,:c,:s,:p,:co,:cco,:cc,:sc,:oc,:cl,:f,:r,:amb,:src,:mc,:ms,:pc,:ps,:loc,:pa::jsonb,
                        :ia1,:ia2,:ia3,:ic,:is,:ip,:ico
                    ) 
                """), {
                    'iid':int(x['input_id']), 'a':x['address1'], 'c':x['city'], 's':x['state'], 'p':x['pincode'], 'co':x['country'], 'cco':x['country_confidence'],
                    'cc':x['city_confidence'], 'sc':x['state_confidence'], 'oc':x['overall_confidence'], 'cl':x['confidence_level'],
                    'f':x['flag'], 'r':x['reason'], 'amb':x['ambiguity_type'], 'src':x['source_used'],
                    'mc':x['master_city'], 'ms':x['master_state'], 'pc':x['pincode_city'], 'ps':x['pincode_state'], 'loc':x['locality'],
                    'pa':x['possible_addresses'], 'ia1':x['in_address1'], 'ia2':x['in_address2'], 'ia3':x['in_address3'],
                    'ic':x['in_city'], 'is':x['in_state'], 'ip':x['in_pincode'], 'ico':x['in_country']
                })
        print(f'Batch {bi+1}/{parts} -> datasets/validated_output_part{bi+1}_v18.xlsx ({len(df)} rows)')
    print('All done v18.')

if __name__=='__main__': main()
