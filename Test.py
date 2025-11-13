import os, re, json
import pandas as pd
from db_config import get_engine

# ---------------------- BASIC UTILS ----------------------

try:
    from rapidfuzz import fuzz
    def sim(a,b): 
        return fuzz.token_set_ratio(str(a or ""), str(b or "")) / 100.0
except:
    import difflib
    def sim(a,b):
        return difflib.SequenceMatcher(None, str(a or "").lower(), str(b or "").lower()).ratio()

THRESH = 0.80
PIN_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')

CITY_DIRECTIONS = {
    "north","south","east","west","n","s","e","w",
    "northwest","northeast","southwest","southeast","nw","ne","sw","se",
    "city","moffusil","division","district","zone","sector","block","phase","central","suburban"
}

def Title(s): return str(s or "").strip().title()
def Upper(s): return str(s or "").strip().upper()

def norm_text(s):
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def tokens_alpha(s):
    return [t for t in re.split(r"[^A-Za-z]+", str(s or "")) if t]

def ngrams(words, sizes):
    out=set()
    L=len(words)
    for n in sizes:
        if n<=0: continue
        for i in range(L-n+1):
            out.add(Title(" ".join(words[i:i+n])))
    return out

def clean_city_tokens(words):
    return [w for w in words if w.lower() not in CITY_DIRECTIONS]

def clean_output_city(s):
    w = clean_city_tokens(tokens_alpha(s))
    return Title(" ".join(w)) if w else Title(s)

def remove_terms(text, terms):
    s = " " + (text or "") + " "
    for t in terms:
        if not t: continue
        t = re.escape(str(t))
        s = re.sub(rf"(?i)\b{t}\b"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return s

# ---------- SCORING ----------
def score_value_match(chosen, input_value, best_ngram):
    s1 = sim(chosen, input_value) if input_value else 0
    s2 = sim(chosen, best_ngram) if best_ngram else 0
    return round(100*max(s1,s2),2)

def score_consistency(chosen, val):
    return round(100*sim(chosen,val),2) if val else 0

def score_ambiguity(cands):
    k=len({Title(x) for x in cands if x})
    if k<=1: return 100
    return max(40, 100 - 20*(k-1))

def bundle(v,c,a):
    return round(0.5*v + 0.4*c + 0.1*a,2)

# --------------------- MAIN -----------------------------

def main(limit=1000, excel=None, batch=200):

    eng=get_engine()
    with eng.begin() as con:
        postal = pd.read_sql("SELECT city,state,pincode,country FROM ref.postal_pincode",con)
        rta    = pd.read_sql("SELECT city,state,pincode,country FROM ref.rta_pincode",con)
        world  = pd.read_sql("SELECT city,country FROM ref.world_cities",con)
        t30    = pd.read_sql("SELECT city FROM ref.t30_cities",con)["city"].astype(str).str.title().tolist()
        inputs = pd.read_sql(f"SELECT * FROM input.addresses ORDER BY id LIMIT {limit}",con)

    # normalize
    for df,cols in [(postal,["city","state","country"]), (rta,["city","state","country"]), (world,["city","country"])]:
        for c in cols: df[c]=df[c].astype(str).map(Title)

    city_master=set(postal["city"])|set(rta["city"])|set(world["city"])
    country_master=set(postal["country"])|set(world["country"])|{"India"}

    # pin index
    pin_index={}
    for src,df in [("postal",postal),("rta",rta)]:
        for _,rr in df.iterrows():
            pin_index.setdefault(str(rr["pincode"]),[]).append(rr)

    results=[]

    for start in range(0,len(inputs),batch):
        chunk=inputs.iloc[start:start+batch].copy()

        for _,rec in chunk.iterrows():
            rid=int(rec["id"])
            a1,a2,a3 = rec["address1"], rec["address2"], rec["address3"]
            in_city = Title(rec["city"])
            in_state_raw = Title(rec["state"])
            in_country = Title(rec["country"])
            in_pin = str(rec["pincode"] or "").strip()

            whole = norm_text(" ".join([str(x or "") for x in [a1,a2,a3,in_city,in_state_raw,in_country,in_pin]]))
            words = tokens_alpha(whole)

            # ngrams
            city_ng = ngrams(clean_city_tokens(words), {1,2})
            state_ng = ngrams(words, {1,2})
            country_ng = ngrams(words, {1,2,3,4})

            # city candidates
            city_cand=set()
            for g in city_ng|({in_city} if in_city else set()):
                if g in city_master:
                    city_cand.add(g)
                else:
                    for cm in city_master:
                        if cm[0]==g[0] and sim(g,cm)>=THRESH:
                            city_cand.add(cm)

            # state candidates (simple fuzzy)
            state_cand=set()
            for g in state_ng|({in_state_raw} if in_state_raw else set()):
                if not g: continue
                state_cand.add(Title(g))

            # country candidates
            country_cand=set()
            for g in country_ng|({in_country} if in_country else set()):
                if g in country_master:
                    country_cand.add(g); continue
                best=None;best_s=0
                for cm in country_master:
                    s=sim(g,cm)
                    if s>best_s: best,best_s=cm,s
                if best_s>=THRESH: country_cand.add(best)

            # PINCODE ABSOLUTE PRIORITY
            pins_text=set(PIN_RE.findall(whole))
            chosen_pin=None
            chosen_row=None

            if pins_text:
                # take exact pincode from text first
                for p in pins_text:
                    chosen_pin=p
                    if p in pin_index:
                        chosen_row=pin_index[p][0]
                    break
            else:
                # fallback city→pin
                pins_from_city=set()
                for c in city_cand:
                    df=postal[postal["city"]==c]
                    df2=rta[rta["city"]==c]
                    for _,r0 in pd.concat([df,df2]).iterrows():
                        pins_from_city.add(str(r0["pincode"]))
                if pins_from_city:
                    chosen_pin=list(pins_from_city)[0]
                    if chosen_pin in pin_index:
                        chosen_row=pin_index[chosen_pin][0]

            # choose city/state/country
            def best_match(cands,inp):
                if not cands: return inp
                return max(cands, key=lambda x: sim(x,inp))

            if chosen_row is not None:
                # consistency with DB row
                city_final = chosen_row["city"]
                state_final = chosen_row["state"]
                country_final = chosen_row["country"]
            else:
                city_final = best_match(city_cand, in_city)
                state_final = best_match(state_cand, in_state_raw)
                country_final = best_match(country_cand, in_country)

            city_final_clean = clean_output_city(city_final)

            # possible lists
            poss_city = sorted({clean_output_city(c) for c in city_cand if c}|{city_final_clean})
            poss_state = sorted({Title(s) for s in state_cand if s}|{Title(state_final)})
            poss_country = sorted({Title(k) for k in country_cand if k}|{Title(country_final)})

            # possible pincodes
            poss_pin = sorted(set(pins_text) |
                              ({chosen_pin} if chosen_pin else set()))

            # local_address
            remove_list = poss_pin + [city_final_clean, state_final, country_final]
            local_address = remove_terms(whole, remove_list)

            # scoring
            best_city_ng = max(city_ng or [""], key=lambda x: sim(x,city_final_clean))
            best_state_ng = max(state_ng or [""], key=lambda x: sim(x,state_final))
            best_country_ng = max(country_ng or [""], key=lambda x: sim(x,country_final))

            c_val = score_value_match(city_final_clean, in_city, best_city_ng)
            s_val = score_value_match(state_final, in_state_raw, best_state_ng)
            k_val = score_value_match(country_final, in_country, best_country_ng)

            c_con = score_consistency(city_final_clean, chosen_row["city"] if chosen_row is not None else None)
            s_con = score_consistency(state_final, chosen_row["state"] if chosen_row is not None else None)
            k_con = score_consistency(country_final, chosen_row["country"] if chosen_row is not None else None)

            c_amb = score_ambiguity(poss_city)
            s_amb = score_ambiguity(poss_state)
            k_amb = score_ambiguity(poss_country)

            overall = round((bundle(c_val,c_con,c_amb) +
                             bundle(s_val,s_con,s_amb) +
                             bundle(k_val,k_con,k_amb))/3,2)

            # reasons
            reasons=[]
            if chosen_row is None and chosen_pin: reasons.append("pincode_not_in_master")
            if chosen_row is not None:
                if sim(city_final_clean, chosen_row["city"]) < THRESH: reasons.append("mismatch_city_vs_pincode")
                if sim(state_final, chosen_row["state"]) < THRESH: reasons.append("mismatch_state_vs_pincode")
                if sim(country_final, chosen_row["country"]) < THRESH: reasons.append("mismatch_country_vs_pincode")
            if c_val < 80: reasons.append("low_city_value_match")
            if s_val < 80: reasons.append("low_state_value_match")
            if k_val < 80: reasons.append("low_country_value_match")
            if c_amb < 100: reasons.append("ambiguous_city_candidates")
            if s_amb < 100: reasons.append("ambiguous_state_candidates")
            if k_amb < 100: reasons.append("ambiguous_country_candidates")

            ambiguous_flag = 1 if overall < 85 else 0

            results.append({
                "input_id": rid,
                "address1": a1, "address2": a2, "address3": a3,
                "input_city": in_city, "input_state_raw": in_state_raw,
                "input_state": in_state_raw, "input_country": in_country,
                "input_pincode": in_pin,
                "concatenated_address": whole,

                "output_pincode": chosen_pin,
                "output_city": city_final_clean,
                "output_state": state_final,
                "output_country": country_final,

                "t30_city_possible": 1 if city_final_clean in t30 else 0,
                "foreign_country_possible": 1 if Title(country_final)!="India" else 0,
                "pincode_found": 1 if chosen_pin else 0,
                "ambiguous_address_flag": ambiguous_flag,

                "all_possible_countries": json.dumps(poss_country),
                "all_possible_states": json.dumps(poss_state),
                "all_possible_cities": json.dumps(poss_city),
                "all_possible_pincodes_text": json.dumps(sorted(pins_text)),
                "all_possible_pincodes_db": "[]",
                "all_possible_pincodes": json.dumps(poss_pin),

                "city_value_match": c_val,
                "city_consistency_with_pincode": c_con,
                "city_ambiguity_penalty": c_amb,

                "state_value_match": s_val,
                "state_consistency_with_pincode": s_con,
                "state_ambiguity_penalty": s_amb,

                "country_value_match": k_val,
                "country_consistency_with_pincode": k_con,
                "country_ambiguity_penalty": k_amb,

                "overall_score": overall,
                "reason": "; ".join(reasons)
            })

    df=pd.DataFrame(results)
    out = excel or "validation_results.xlsx"
    with pd.ExcelWriter(out) as x:
        df.to_excel(x,index=False,sheet_name="results")
    print("DONE:",len(df),"rows →",out)


if __name__=="__main__":
    main()
