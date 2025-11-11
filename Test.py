# validate_addresses_phase2.py
# Phase-2 validator: exhaustive matching with 1-gram/2-gram n-gram search over
# concatenated address; strict pincode-first; robust state short-form expansion;
# flags (T30/foreign/pincode); ONLY the 6 pincode binary scores; audit trail;
# local address remainder; strips directional/division terms from OUTPUT city.
#
# Requires: db_config.get_engine()

import os, re, json
import pandas as pd
from sqlalchemy import text
from db_config import get_engine

# ---------- Similarity ----------
THRESH = 0.80
try:
    from rapidfuzz import fuzz
    def sim(a,b): return fuzz.token_set_ratio(str(a or ""), str(b or ""))/100.0
except Exception:
    import difflib
    def sim(a,b):
        return difflib.SequenceMatcher(None, str(a or "").lower(), str(b or "").lower()).ratio()

# ---------- Basic helpers ----------
PIN_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')

CITY_STOPWORDS = {
    # generic directions & layout words
    "west","east","north","south","n","s","e","w","nw","ne","sw","se",
    "southwest","southeast","northwest","northeast",
    "sector","zone","block","phase","ward","dist","district","taluka","taluk","mandal",
    # India-specific noisy suffixes
    "city","moffusil","division","gpo","mumbai","pune"  # “city/moffusil/division” removed from output
}

# when cleaning FINAL output_city we also remove compound direction phrases
CITY_PHRASE_STOP = [
    " city", " north", " south", " east", " west",
    " north east", " north west", " south east", " south west",
    " moffusil", " division"
]

def norm_text(s):
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def Title(s): return str(s or "").strip().title()
def Upper(s): return str(s or "").strip().upper()

def extract_pincodes(*fields):
    pins=set()
    for f in fields:
        if not f: continue
        for m in PIN_RE.findall(str(f)):
            pins.add(m)
    return sorted(pins)

def _tokens_alpha(s: str):
    return [t for t in re.split(r"[^A-Za-z]+", str(s or "")) if len(t) > 0]

def make_ngrams_for_match(concat_text: str):
    toks = _tokens_alpha(concat_text)
    grams = set()
    # unigrams (≥2 chars)
    for t in toks:
        if len(t) >= 2:
            grams.add(Title(t))
    # bigrams
    for i in range(len(toks)-1):
        a, b = toks[i], toks[i+1]
        if len(a) >= 2 and len(b) >= 2:
            grams.add(Title(f"{a} {b}"))
    return grams

def best_sim_from_concat(candidate_value: str, concat_text: str, extra: list = None):
    grams = make_ngrams_for_match(concat_text)
    if extra:
        grams.update([Title(x) for x in extra if x])
    if not grams:
        return 0.0
    return max(sim(candidate_value, g) for g in grams)

def base_city(city: str)->str:
    """Remove directional & structure stopwords for matching; title-case result."""
    if not city: return None
    tokens = [t for t in re.split(r"[^A-Za-z]+", str(city)) if t]
    keep = [t for t in tokens if t.lower() not in CITY_STOPWORDS]
    return Title(" ".join(keep)) if keep else Title(city)

def clean_city_output(city: str)->str:
    """Aggressive cleaner for FINAL output city: drop 'City/Moffusil/Division'
       and any directional phrases like 'South East', 'North West'."""
    if not city: return city
    s = " " + Title(city) + " "
    s = s.lower()
    # remove compound phrases first
    for ph in CITY_PHRASE_STOP:
        s = s.replace(ph, " ")
    # strip leftover single tokens
    tokens = [t for t in re.split(r"[^a-z]+", s) if t]
    tokens = [t for t in tokens if t and t not in CITY_STOPWORDS]
    cleaned = Title(" ".join(tokens))
    return cleaned

def remove_terms(text: str, terms: list)->str:
    if not text: return text
    s = " " + text + " "
    for t in terms:
        if not t: continue
        t = re.escape(str(t).strip())
        s = re.sub(rf"(?i)\b{t}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- State short-forms (DB + static fallback) ----------
def _norm_token(x: str) -> str:
    x = str(x or "").strip()
    x = re.sub(r"[^A-Za-z]", "", x)
    return x.upper()

def _initials_of(state_name: str) -> str:
    toks = re.split(r"[^A-Za-z]+", str(state_name or ""))
    toks = [t for t in toks if t]
    if not toks: return ""
    return "".join(t[0] for t in toks).upper()

STATE_ALIAS_STATIC = {
    "Andhra Pradesh": {"AP","AD","ANDHRAPRADESH"},
    "Arunachal Pradesh": {"AR","ARN"},
    "Assam": {"AS"},
    "Bihar": {"BH","BR"},
    "Chhattisgarh": {"CG","CHH","CT"},
    "Goa": {"GO","GA"},
    "Gujarat": {"GU","GJ"},
    "Haryana": {"HA","HR"},
    "Himachal Pradesh": {"HP","HIM"},
    "Jammu And Kashmir": {"JK","J&K","JNK","JAMMUANDKASHMIR"},
    "Jharkhand": {"JH","JHD","JD"},
    "Karnataka": {"KA","KAR"},
    "Kerala": {"KE","KL"},
    "Madhya Pradesh": {"MP","MD","MADHYAPRADESH"},
    "Maharashtra": {"MA","MH"},
    "Manipur": {"MN"},
    "Meghalaya": {"ML","ME"},
    "Mizoram": {"MZ"},
    "Nagaland": {"NL"},
    "Odisha": {"OD","OR","ORISSA","ODISHA"},
    "Punjab": {"PU","PB"},
    "Rajasthan": {"RA","RJ"},
    "Sikkim": {"SK"},
    "Tamil Nadu": {"TN","TM"},
    "Telangana": {"TG","TS","TE"},
    "Tripura": {"TR"},
    "Uttar Pradesh": {"UP","UTP"},
    "Uttarakhand": {"UK","UA","UC"},
    "West Bengal": {"WB","W.B","WBG"},
    "Delhi": {"DL","ND","DELHI","NCT"},
    "Chandigarh": {"CH","CHD"},
    "Puducherry": {"PO","PY","PONDICHERRY"},
    "Ladakh": {"LA"},
    "Lakshadweep": {"LD"}
}

def expand_state_abbrev(state_in, states_df):
    if not state_in: return state_in
    s_raw = str(state_in).strip()
    s_norm = re.sub(r"[’'`“”]", "", s_raw)
    s_title = Title(s_norm)
    tok = _norm_token(s_norm)

    states_df = states_df.copy()
    if "state" not in states_df.columns:
        states_df["state"] = []
    states_df["state"] = states_df["state"].astype(str).map(Title)
    if "abbreviation" not in states_df.columns:
        states_df["abbreviation"] = ""
    states_df["abbreviation"] = states_df["abbreviation"].astype(str).fillna("")

    alias = {}
    for _, row in states_df.iterrows():
        canon = Title(row["state"])
        alias.setdefault(canon, set())
        for t in re.split(r"[^A-Za-z]+", str(row["abbreviation"])):
            if t: alias[canon].add(_norm_token(t))
        alias[canon].add(_initials_of(canon))
        alias[canon].add(_norm_token(canon))
        if canon in STATE_ALIAS_STATIC:
            alias[canon].update({_norm_token(x) for x in STATE_ALIAS_STATIC[canon]})

    # direct alias hit
    for canon, toks in alias.items():
        if tok in toks:
            return canon
    if s_title in alias:
        return s_title

    # fuzzy to canonical
    try:
        from rapidfuzz import fuzz
        best_c, best_s = None, -1.0
        for canon in alias.keys():
            sc = fuzz.token_set_ratio(canon, s_title)/100.0
            if sc > best_s:
                best_c, best_s = canon, sc
        if best_s >= THRESH:
            return best_c
    except Exception:
        import difflib
        cands = list(alias.keys())
        best = difflib.get_close_matches(s_title, cands, n=1, cutoff=THRESH)
        if best:
            return best[0]
    return s_title

def bin1(x): 
    try: return 1 if (x is not None and float(x) >= THRESH) else 0
    except: return 0

# ---------- Main ----------
def main(limit=1000, excel=None):
    eng = get_engine()
    with eng.begin() as con:
        postal = pd.read_sql("SELECT city,state,pincode,country FROM ref.postal_pincode", con)
        rta    = pd.read_sql("SELECT city,state,pincode,country FROM ref.rta_pincode", con)
        # optional state abbrev table
        try:
            states = pd.read_sql("SELECT state,abbreviation FROM ref.indian_state_abbrev", con)
        except Exception:
            states = pd.DataFrame({"state":[],"abbreviation":[]})
        world  = pd.read_sql("SELECT city,country FROM ref.world_cities", con)
        t30    = pd.read_sql("SELECT city FROM ref.t30_cities", con)["city"].astype(str).str.title().unique().tolist()
        inputs = pd.read_sql(f"SELECT * FROM input.addresses ORDER BY id LIMIT {int(limit)}", con)

    for df, cols in [(postal,["city","state","country"]), (rta,["city","state","country"]), (world,["city","country"])]:
        for c in cols: df[c] = df[c].astype(str).map(Title)
    states["state"] = states.get("state", pd.Series([], dtype=str)).astype(str).map(Title)
    if "abbreviation" in states.columns:
        states["abbreviation"] = states["abbreviation"].astype(str).map(Upper)
    else:
        states["abbreviation"] = ""

    # pin index
    pin_index = {}
    for src, df in [("postal", postal), ("rta", rta)]:
        for _, rr in df.iterrows():
            pin_index.setdefault(str(rr["pincode"]), []).append((src, rr))

    t30_set = {Title(x) for x in t30}

    out_rows, audit = [], []

    for _, r in inputs.iterrows():
        input_id = int(r["id"])
        in_addr1 = r.get("address1"); in_addr2=r.get("address2"); in_addr3=r.get("address3")
        in_city  = Title(r.get("city"))
        in_state_raw = Title(r.get("state"))
        in_state = expand_state_abbrev(in_state_raw, states)
        in_country = Title(r.get("country"))
        in_pin   = str(r.get("pincode") or "").strip()

        concat = norm_text(" ".join([str(x or "") for x in [in_addr1,in_addr2,in_addr3,in_city,in_state,in_country,in_pin]]))
        in_city_base = base_city(in_city)

        all_countries = set([in_country]) if in_country else set()
        all_states = set([in_state]) if in_state else set()
        all_cities = set([in_city_base if in_city_base else in_city]) if in_city else set()

        candidate_pins = extract_pincodes(concat, in_pin)
        all_pincodes_text = set(candidate_pins)
        all_pincodes_db = set()

        P_input_city=P_city_db=P_input_state=P_state_db=P_input_country=P_country_db=0

        out_pin=out_city=out_state=out_country=None
        source_used="fallback"

        # --- PINCODE-FIRST ---
        if candidate_pins:
            best = None; best_score=-1
            for p in candidate_pins:
                rows = pin_index.get(p) or []
                for src, rr in rows:
                    c=Title(rr["city"]); s=Title(rr["state"]); k=Title(rr["country"])
                    all_countries.add(k); all_states.add(s); all_cities.add(base_city(c))
                    all_pincodes_db.add(str(p))
                    sc_city = max(sim(c, in_city), sim(c, in_city_base), best_sim_from_concat(c, concat, [in_city, in_city_base]))
                    sc_state= max(sim(s, in_state), best_sim_from_concat(s, concat, [in_state]))
                    sc_ctry = max(sim(k, in_country), best_sim_from_concat(k, concat, [in_country]))
                    score = 1.0 + 0.35*sc_city + 0.35*sc_state + 0.10*sc_ctry
                    if score > best_score:
                        best = {"pincode":p,"city":c,"state":s,"country":k,"src":src,
                                "sc_city":sc_city,"sc_state":sc_state,"sc_ctry":sc_ctry}
                        best_score = score
                    audit.append({"input_id":input_id,"match_type":"pincode","candidate":f"{p}|{c}|{s}|{k}","score":score,"source":src})
            if best:
                out_pin = best["pincode"]
                out_city = best["city"]; out_state = best["state"]; out_country = best["country"]
                source_used="pincode"
                P_input_city   = bin1(best["sc_city"]);   P_city_db    = 1
                P_input_state  = bin1(best["sc_state"]);  P_state_db   = 1
                P_input_country= bin1(best["sc_ctry"]);   P_country_db = 1

        # --- CITY→PIN fallback (tight, includes n-grams) ---
        if out_pin is None:
            for src, df in [("postal", postal), ("rta", rta)]:
                df_tmp = df.copy()
                df_tmp["cscore"] = df_tmp["city"].apply(lambda x: max(sim(x, in_city), sim(x, in_city_base), best_sim_from_concat(x, concat, [in_city, in_city_base])) if in_city or in_city_base else 0)
                df_tmp["sscore"] = df_tmp["state"].apply(lambda x: max(sim(x, in_state), best_sim_from_concat(x, concat, [in_state])) if in_state else 0)
                df_tmp["kscore"] = df_tmp["country"].apply(lambda x: max(sim(x, in_country), best_sim_from_concat(x, concat, [in_country])) if in_country else 0)
                mask = ((df_tmp["cscore"]>=THRESH) & (df_tmp["sscore"]>=0.60)) | ((df_tmp["sscore"]>=THRESH) & (df_tmp["cscore"]>=0.60))
                if in_country:
                    mask = mask & (df_tmp["kscore"]>=0.60)
                df_tmp = df_tmp[mask]
                for _, rr in df_tmp.iterrows():
                    c=Title(rr["city"]); s=Title(rr["state"]); k=Title(rr["country"])
                    all_cities.add(base_city(c)); all_states.add(s); all_countries.add(k); all_pincodes_db.add(str(rr["pincode"]))
                    score = 0.45*rr["cscore"] + 0.45*rr["sscore"] + 0.10*rr["kscore"]
                    audit.append({"input_id":input_id,"match_type":"city→pin","candidate":f"{rr['pincode']}|{c}|{s}|{k}","score":score,"source":src})
            cands = [a for a in audit if a["input_id"]==input_id and a["match_type"]=="city→pin"]
            if cands:
                best = max(cands, key=lambda x: x["score"])
                parts = best["candidate"].split("|")
                out_pin = parts[0] if len(parts)>0 else None
                out_city= parts[1] if len(parts)>1 else None
                out_state=parts[2] if len(parts)>2 else None
                out_country=parts[3] if len(parts)>3 else None
                source_used="city→pin"

        # --- WORLD CITIES fallback ---
        if out_city is None and in_city:
            bc = base_city(in_city)
            wc = world[world["city"]==bc]
            if not wc.empty:
                wc["kscore"] = wc["country"].apply(lambda k: max(sim(k, in_country) if in_country else 0,
                                                                 best_sim_from_concat(k, concat, [in_country])))
                idx = wc["kscore"].astype(float).idxmax()
                out_city = bc
                out_country = Title(wc.loc[idx,"country"])
                if out_state is None: out_state = in_state
                source_used="world_cities"
                all_countries.update(wc["country"].astype(str).map(Title).tolist())

        # --- T30 / Foreign flags ---
        t30_hit = False
        if out_city and Title(base_city(out_city)) in t30_set:
            t30_hit = True
        else:
            for c in list(all_cities)+([out_city] if out_city else []):
                if c and Title(base_city(c)) in t30_set:
                    t30_hit = True; break
        countries_all = set([Title(x) for x in list(all_countries)+([out_country] if out_country else []) if x])
        foreign_flag = 1 if any(k!="India" for k in countries_all) else 0

        # lists
        all_countries_list = sorted([x for x in countries_all])
        all_states_list = sorted({x for x in list(all_states)+([out_state] if out_state else []) if x})
        all_cities_list = sorted({Title(base_city(x)) for x in list(all_cities)+([out_city] if out_city else []) if x})

        all_pincodes_text_list = sorted(all_pincodes_text)
        all_pincodes_db_list = sorted(all_pincodes_db)[:100]
        all_pincodes_union = sorted(set(all_pincodes_text_list) | set(all_pincodes_db_list))

        # local address remainder
        remove_list = []
        if out_pin: remove_list.append(out_pin)
        if out_country: remove_list.append(out_country)
        if out_state: remove_list.append(out_state)
        for c in list(all_cities_list)+([out_city] if out_city else []):
            if c: remove_list.append(c)
        local_address = remove_terms(concat, remove_list)

        # FINAL: strip direction/division tokens from OUTPUT city
        if out_city:
            out_city = clean_city_output(out_city)

        out_rows.append({
            # Inputs
            "input_id": input_id,
            "address1": r.get("address1"), "address2": r.get("address2"), "address3": r.get("address3"),
            "input_city": in_city, "input_state_raw": in_state_raw, "input_state": in_state,
            "input_country": in_country, "input_pincode": in_pin,
            "concatenated_address": concat,

            # Outputs
            "output_pincode": out_pin,
            "output_city": out_city,
            "output_state": out_state,
            "output_country": out_country,

            # ONLY the 6 required binary scores
            "Pincode-input_city_match": P_input_city,
            "Pincode-city_db_match": P_city_db,
            "Pincode-input_state_match": P_input_state,
            "Pincode-state_db_match": P_state_db,
            "Pincode-input_country_match": P_input_country,
            "Pincode-country_db_match": P_country_db,

            # Flags
            "t30_city_possible": 1 if t30_hit else 0,
            "foreign_country_possible": foreign_flag,
            "pincode_found": 1 if out_pin else 0,
            "source_used": source_used,

            # All possible (for transparency)
            "all_possible_countries": json.dumps(all_countries_list, ensure_ascii=False),
            "all_possible_states": json.dumps(all_states_list, ensure_ascii=False),
            "all_possible_cities": json.dumps(all_cities_list, ensure_ascii=False),
            "all_possible_pincodes_text": json.dumps(all_pincodes_text_list, ensure_ascii=False),
            "all_possible_pincodes_db": json.dumps(all_pincodes_db_list, ensure_ascii=False),
            "all_possible_pincodes": json.dumps(all_pincodes_union, ensure_ascii=False),

            # Remainder
            "local_address": local_address
        })

    result_df = pd.DataFrame(out_rows)
    audit_df = pd.DataFrame(audit) if audit else pd.DataFrame(columns=["input_id","match_type","candidate","score","source"])

    # Excel
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    xls_path = excel or os.path.join(out_dir, "validation_results.xlsx")
    with pd.ExcelWriter(xls_path) as xl:
        result_df.to_excel(xl, index=False, sheet_name="results")
        audit_df.to_excel(xl, index=False, sheet_name="audit")

    # DB persists
    eng = get_engine()
    with eng.begin() as con:
        minimal = result_df.rename(columns={
            "output_pincode":"chosen_pincode",
            "output_city":"chosen_city",
            "output_state":"chosen_state",
            "output_country":"chosen_country",
        })[[
            "input_id","chosen_pincode","chosen_city","chosen_state","chosen_country",
            "t30_city_possible","foreign_country_possible","pincode_found","source_used"
        ]]
        minimal.to_sql("validation_result", con, schema="output", if_exists="append", index=False, method="multi")
        result_df.to_sql("validation_result_full", con, schema="output", if_exists="append", index=False, method="multi")
        audit_df.to_sql("audit_matches", con, schema="output", if_exists="append", index=False, method="multi")

    print(f" Validation done: {len(result_df)} rows. Excel → {xls_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000, help="Process first N rows (default 1000).")
    ap.add_argument("--excel", type=str, default=None, help="Excel output path.")
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel)
