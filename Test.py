
import os, re, json
import pandas as pd
from sqlalchemy import text
from db_config import get_engine

# ----- Fuzzy similarity (RapidFuzz preferred) ---------------------------------
try:
    from rapidfuzz import fuzz
    def sim(a,b): return fuzz.token_set_ratio(str(a or ""), str(b or ""))/100.0
except Exception:
    import difflib
    def sim(a,b):
        return difflib.SequenceMatcher(None, str(a or "").lower(), str(b or "").lower()).ratio()

THRESH = 0.80
PIN_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')

CITY_STOPWORDS = {
    "west","east","north","south","n","s","e","w","nw","ne","sw","se",
    "sector","zone","block","phase","ward","dist","district","taluka","taluk","mandal"
}

def norm_text(s):
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def Title(s): return str(s or "").strip().title()
def Upper(s): return str(s or "").strip().upper()

def base_city(city: str)->str:
    if not city: return None
    tokens = [t for t in re.split(r"[^A-Za-z]+", str(city)) if t]
    keep = [t for t in tokens if t.lower() not in CITY_STOPWORDS]
    return Title(" ".join(keep)) if keep else Title(city)

def extract_pincodes(*fields):
    pins=set()
    for f in fields:
        if not f: continue
        for m in PIN_RE.findall(str(f)):
            pins.add(m)
    return sorted(pins)

def _norm_token(x: str) -> str:
    x = str(x or "").strip()
    x = re.sub(r"[^A-Za-z]", "", x)
    return x.upper()

def _initials_of(state_name: str) -> str:
    toks = re.split(r"[^A-Za-z]+", str(state_name or ""))
    toks = [t for t in toks if t]
    if not toks: return ""
    return "".join(t[0] for t in toks).upper()

def expand_state_abbrev(state_in, states_df):
    if not state_in: return state_in
    s_raw = str(state_in).strip()
    s_title = Title(s_raw)
    tok = _norm_token(s_raw)

    states_df = states_df.copy()
    states_df["state"] = states_df["state"].astype(str).map(Title)
    if "abbreviation" not in states_df.columns:
        states_df["abbreviation"] = ""
    states_df["abbreviation"] = states_df["abbreviation"].astype(str).map(lambda x: x.strip().upper())

    alias_cols = [c for c in states_df.columns if c.lower() in {"alias","aliases","alt","alternate","short","shortcode","code2","code"}]

    hits = states_df[states_df["abbreviation"] == tok]
    if not hits.empty: return Title(hits.iloc[0]["state"])

    hits = states_df[states_df["state"] == s_title]
    if not hits.empty: return Title(hits.iloc[0]["state"])

    states_df["_state_token"] = states_df["state"].map(_norm_token)
    hits = states_df[states_df["_state_token"] == tok]
    if not hits.empty: return Title(hits.iloc[0]["state"])

    for c in alias_cols:
        s = states_df[c].astype(str)
        alias_match = states_df[s.str.split(",").apply(lambda arr: any(_norm_token(a)==tok for a in arr))]
        if not alias_match.empty: return Title(alias_match.iloc[0]["state"])

    states_df["_initials"] = states_df["state"].map(_initials_of)
    hits = states_df[states_df["_initials"] == tok]
    if not hits.empty: return Title(hits.iloc[0]["state"])

    try:
        from rapidfuzz import fuzz
        states_df["_f_state"] = states_df["state"].apply(lambda x: fuzz.token_set_ratio(x, s_title)/100.0)
        states_df["_f_abbr"]  = states_df["abbreviation"].apply(lambda x: fuzz.token_set_ratio(x, tok)/100.0)
        states_df["_f"] = states_df[["_f_state","_f_abbr"]].max(axis=1)
        best = states_df.sort_values("_f", ascending=False).iloc[0]
        if float(best["_f"]) >= THRESH:
            return Title(best["state"])
    except Exception:
        import difflib
        candidates = list(states_df["state"].astype(str))
        best = difflib.get_close_matches(s_title, candidates, n=1, cutoff=THRESH)
        if best:
            return Title(best[0])

    return s_title

def bin_score(val: float, thr: float=THRESH)->int:
    try:
        return 1 if (val is not None and float(val) >= thr) else 0
    except Exception:
        return 0

def remove_terms(text: str, terms: list)->str:
    if not text: return text
    s = " " + text + " "
    for t in terms:
        if not t: continue
        t = re.escape(str(t).strip())
        s = re.sub(rf"(?i)\b{t}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(limit=1000, excel=None):
    eng = get_engine()
    with eng.begin() as con:
        postal = pd.read_sql("SELECT city,state,pincode,country FROM ref.postal_pincode", con)
        rta    = pd.read_sql("SELECT city,state,pincode,country FROM ref.rta_pincode", con)
        states = pd.read_sql("SELECT state,abbreviation FROM ref.indian_state_abbrev", con)
        world  = pd.read_sql("SELECT city,country FROM ref.world_cities", con)
        countries = pd.read_sql("SELECT name FROM ref.countries", con)["name"].astype(str).str.title().unique().tolist()
        t30    = pd.read_sql("SELECT city FROM ref.t30_cities", con)["city"].astype(str).str.title().unique().tolist()
        inputs = pd.read_sql(f"SELECT * FROM input.addresses ORDER BY id LIMIT {int(limit)}", con)

    for df, cols in [(postal,["city","state","country"]), (rta,["city","state","country"]), (world,["city","country"])]:
        for c in cols: df[c] = df[c].astype(str).map(Title)

    states["state"] = states["state"].astype(str).map(Title)
    states["abbreviation"] = states["abbreviation"].astype(str).map(Upper)

    pin_index = {}
    for src, df in [("postal", postal), ("rta", rta)]:
        for _, r in df.iterrows():
            pin_index.setdefault(r["pincode"], []).append((src, r))

    out_rows = []
    audit = []

    for _, r in inputs.iterrows():
        input_id = int(r["id"])
        in_addr1 = r.get("address1"); in_addr2=r.get("address2"); in_addr3=r.get("address3")
        in_city  = Title(r.get("city"))
        in_state_raw = Title(r.get("state"))
        in_state = expand_state_abbrev(in_state_raw, states)
        in_country = Title(r.get("country"))
        in_pin   = str(r.get("pincode") or "").strip()

        concat = norm_text(" ".join([str(x or "") for x in [in_addr1,in_addr2,in_addr3,in_city,in_state,in_country,in_pin]]))

        all_countries = set([in_country]) if in_country else set()
        all_states = set([in_state]) if in_state else set()
        all_cities = set([in_city]) if in_city else set()

        in_city_base = base_city(in_city)

        candidate_pins = extract_pincodes(concat, in_pin)
        all_pincodes_text = set(candidate_pins)
        all_pincodes_db = set()

        pincode_found = 1 if any(pin_index.get(p) for p in candidate_pins) else 0

        # 6 required pincode-related binary scores from docx
        P_input_city=0; P_city_db=0; P_input_state=0; P_state_db=0; P_input_country=0; P_country_db=0

        out_pin=None; out_city=None; out_state=None; out_country=None
        source_used="fallback"

        if candidate_pins:
            best = None; best_score=-1
            for p in candidate_pins:
                rows = pin_index.get(p) or []
                for src, rr in rows:
                    c=rr["city"]; s=rr["state"]; k=rr["country"]
                    all_countries.add(k); all_states.add(s); all_cities.add(c)
                    all_pincodes_db.add(str(p))

                    sc_city_in  = sim(c, in_city) if in_city else 0
                    sc_city_in2 = sim(c, in_city_base) if in_city_base else 0
                    sc_city = max(sc_city_in, sc_city_in2)
                    sc_state = sim(s, in_state) if in_state else 0
                    sc_ctry = sim(k, in_country) if in_country else 0

                    score = 1.0 + 0.35*sc_city + 0.35*sc_state + 0.10*sc_ctry
                    if score > best_score:
                        best = {"pincode":p,"city":c,"state":s,"country":k,"src":src,
                                "sc_city":sc_city,"sc_state":sc_state,"sc_ctry":sc_ctry}
                        best_score = score
                    audit.append({"input_id":input_id,"match_type":"pincode","candidate":f"{p}|{c}|{s}|{k}","score":score,"source":src})

            if best:
                out_pin = best["pincode"]; out_city=best["city"]; out_state=best["state"]; out_country=best["country"]
                source_used="pincode"
                # Binary flags only
                P_input_city   = 1 if best["sc_city"]  >= THRESH else 0
                P_city_db      = 1  # DB vs DB is always 1 for the chosen row
                P_input_state  = 1 if best["sc_state"] >= THRESH else 0
                P_state_db     = 1
                P_input_country= 1 if best["sc_ctry"]  >= THRESH else 0
                P_country_db   = 1

        if out_pin is None:
            for src, df in [("postal", postal), ("rta", rta)]:
                df_tmp = df.copy()
                df_tmp["c_raw"]  = df_tmp["city"].apply(lambda x: sim(x, in_city) if in_city else 0)
                df_tmp["c_base"] = df_tmp["city"].apply(lambda x: sim(x, in_city_base) if in_city_base else 0)
                df_tmp["cscore"] = df_tmp[["c_raw","c_base"]].max(axis=1)
                df_tmp["sscore"]= df_tmp["state"].apply(lambda x: sim(x, in_state) if in_state else 0)
                df_tmp["kscore"]= df_tmp["country"].apply(lambda x: sim(x, in_country) if in_country else 0)

                mask = (df_tmp["cscore"]>=THRESH) | (df_tmp["sscore"]>=THRESH) | ((df_tmp["cscore"]>=0.70)&(df_tmp["sscore"]>=0.70))
                df_tmp = df_tmp[mask]
                for _, rr in df_tmp.iterrows():
                    all_cities.add(rr["city"]); all_states.add(rr["state"]); all_countries.add(rr["country"])
                    all_pincodes_db.add(str(rr["pincode"]))
                    score = 0.45*rr["cscore"] + 0.45*rr["sscore"] + 0.10*rr["kscore"]
                    audit.append({"input_id":input_id,"match_type":"city→pin","candidate":f"{rr['pincode']}|{rr['city']}|{rr['state']}|{rr['country']}","score":score,"source":src})

            candidates = [a for a in audit if a["input_id"]==input_id and a["match_type"]=="city→pin"]
            if candidates:
                best = max(candidates, key=lambda x: x["score"])
                parts = best["candidate"].split("|")
                out_pin = parts[0] if len(parts)>0 else None
                out_city= parts[1] if len(parts)>1 else None
                out_state=parts[2] if len(parts)>2 else None
                out_country=parts[3] if len(parts)>3 else None
                source_used="city→pin"

        if out_city is None and in_city:
            base_c = base_city(in_city)
            wc = world[world["city"]==base_c]
            if not wc.empty:
                wc["kscore"] = wc["country"].apply(lambda k: sim(k, in_country) if in_country else 0)
                idx = wc["kscore"].astype(float).idxmax()
                out_city = base_c
                out_country = wc.loc[idx,"country"]
                if out_state is None: out_state = in_state
                source_used="world_cities"
                all_countries.update(wc["country"].tolist())

        t30_set = set([Title(x) for x in t30])
        t30_hit = False
        if out_pin:
            t30_hit = True if out_city and base_city(out_city) in t30_set else False
        else:
            for c in list(all_cities)+([out_city] if out_city else []):
                if c and base_city(c) in t30_set:
                    t30_hit = True; break

        countries_all = set([Title(x) for x in list(all_countries)+([out_country] if out_country else []) if x])
        foreign_flag = 1 if any(k!="India" for k in countries_all) else 0

        all_countries_list = sorted([x for x in countries_all])
        all_states_list = sorted({x for x in list(all_states)+([out_state] if out_state else []) if x})
        all_cities_list = sorted({Title(base_city(x)) for x in list(all_cities)+([out_city] if out_city else []) if x})

        # pincode lists
        all_pincodes_text_list = sorted(all_pincodes_text)
        all_pincodes_db_list = sorted(all_pincodes_db)
        all_pincodes_union = sorted(set(all_pincodes_text_list) | set(all_pincodes_db_list))

        remove_list = []
        if out_pin: remove_list.append(out_pin)
        if out_country: remove_list.append(out_country)
        if out_state: remove_list.append(out_state)
        for c in list(all_cities_list)+([out_city] if out_city else []):
            if c: remove_list.append(c)
        local_address = remove_terms(concat, remove_list)

        out_rows.append({
            # Inputs
            "input_id": input_id,
            "address1": r.get("address1"), "address2": r.get("address2"), "address3": r.get("address3"),
            "input_city": in_city, "input_state_raw": in_state_raw, "input_state": in_state,
            "input_country": in_country, "input_pincode": in_pin,
            "concatenated_address": concat,

            # Outputs (names as requested)
            "output_pincode": out_pin,
            "output_city": out_city,
            "output_state": out_state,
            "output_country": out_country,

            # ONLY the 6 Docx Scores (binary)
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

            # All possible
            "all_possible_countries": json.dumps(all_countries_list, ensure_ascii=False),
            "all_possible_states": json.dumps(all_states_list, ensure_ascii=False),
            "all_possible_cities": json.dumps(all_cities_list, ensure_ascii=False),
            "all_possible_pincodes_text": json.dumps(all_pincodes_text_list, ensure_ascii=False),
            "all_possible_pincodes_db": json.dumps(all_pincodes_db_list, ensure_ascii=False),
            "all_possible_pincodes": json.dumps(all_pincodes_union, ensure_ascii=False),

            # Local address remainder
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

    # Persist to DB
    eng = get_engine()
    with eng.begin() as con:
        # Minimal compatibility table
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

        # Rich tables
        result_df.to_sql("validation_result_full", con, schema="output", if_exists="append", index=False, method="multi")
        audit_df.to_sql("audit_matches", con, schema="output", if_exists="append", index=False, method="multi")

    print(f"✅ Validation done: {len(result_df)} rows. Excel → {xls_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000, help="Process this many rows first (default 1000).")
    ap.add_argument("--excel", type=str, default=None, help="Path to Excel output file.")
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel)
