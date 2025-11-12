# validate_addresses_phase2.py
import os, re, json
import pandas as pd
from sqlalchemy import text
from db_config import get_engine

THRESH = 0.80
PIN_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')  # India PIN
DIR_TOKENS = {
    "north","south","east","west","n","s","e","w",
    "north-east","north-west","south-east","south-west","ne","nw","se","sw"
}
CITY_NOISE = {"sector","zone","block","phase","ward","dist","district","taluka","taluk","mandal","city","moffusil","gpo"}

# ---------- similarity ----------
try:
    from rapidfuzz import fuzz
    def sim(a,b): return fuzz.token_set_ratio(str(a or ""), str(b or ""))/100.0
except Exception:
    import difflib
    def sim(a,b):
        return difflib.SequenceMatcher(None, str(a or "").lower(), str(b or "").lower()).ratio()

def Title(s): return str(s or "").strip().title()
def Upper(s): return str(s or "").strip().upper()

def norm_text(s):
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _alpha_tokens(s):
    return [t for t in re.split(r"[^A-Za-z]+", str(s or "")) if t]

def _ngrams_12(s):
    toks = _alpha_tokens(s)
    grams=set()
    for t in toks:
        if len(t) >= 2:
            grams.add(Title(t))
    for i in range(len(toks)-1):
        a,b=toks[i], toks[i+1]
        if len(a)>=2 and len(b)>=2:
            grams.add(Title(f"{a} {b}"))
    return grams

def base_city(city):
    if not city: return None
    toks=[t for t in _alpha_tokens(city) if t.lower() not in DIR_TOKENS|CITY_NOISE]
    return Title(" ".join(toks)) if toks else Title(city)

def strip_city_directions(city):
    if not city: return city
    toks=[t for t in _alpha_tokens(city) if t.lower() not in DIR_TOKENS|CITY_NOISE]
    return Title(" ".join(toks)) if toks else Title(city)

def extract_pins(*fields):
    pins=set()
    for f in fields:
        if not f: continue
        pins.update(PIN_RE.findall(str(f)))
    return sorted(pins)

# ---------- state abbrev expansion ----------
def _norm_token(x: str) -> str:
    x = str(x or "").strip()
    x = re.sub(r"[^A-Za-z]", "", x)
    return x.upper()

def _initials_of(name: str) -> str:
    toks = _alpha_tokens(name)
    return "".join(t[0] for t in toks).upper() if toks else ""

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
    states_df["state"] = states_df["state"].astype(str).map(Title)
    if "abbreviation" not in states_df.columns: states_df["abbreviation"] = ""
    states_df["abbreviation"] = states_df["abbreviation"].astype(str).fillna("")

    alias={}
    for _, row in states_df.iterrows():
        canon = Title(row["state"])
        alias.setdefault(canon, set())
        for t in re.split(r"[^A-Za-z]+", str(row["abbreviation"])):
            if t: alias[canon].add(_norm_token(t))
        alias[canon].add(_initials_of(canon))
        alias[canon].add(_norm_token(canon))
        if canon in STATE_ALIAS_STATIC:
            alias[canon].update({_norm_token(x) for x in STATE_ALIAS_STATIC[canon]})

    for canon, toks in alias.items():
        if tok in toks: return canon
    if s_title in alias: return s_title

    try:
        from rapidfuzz import fuzz
        best_c, best_s = None, -1.0
        for canon in alias.keys():
            sc = fuzz.token_set_ratio(canon, s_title)/100.0
            if sc > best_s: best_c, best_s = canon, sc
        if best_s >= THRESH: return best_c
    except Exception:
        import difflib
        cands = list(alias.keys())
        best = difflib.get_close_matches(s_title, cands, n=1, cutoff=THRESH)
        if best: return best[0]
    return s_title

# ---------- discovery limited to input/concat ----------
def discover_cities(concat, in_city, master_cities):
    grams = _ngrams_12(concat)
    if in_city: grams.add(Title(in_city)); grams.add(base_city(in_city))
    discovered=set()
    for cand in grams:
        for mc in master_cities:
            if sim(cand, mc) >= THRESH:
                discovered.add(Title(mc))
    return discovered

def main(limit=1000, excel=None):
    eng = get_engine()
    with eng.begin() as con:
        postal = pd.read_sql("SELECT city,state,pincode,country FROM ref.postal_pincode", con)
        rta    = pd.read_sql("SELECT city,state,pincode,country FROM ref.rta_pincode", con)
        world  = pd.read_sql("SELECT city,country FROM ref.world_cities", con)
        t30_df = pd.read_sql("SELECT city FROM ref.t30_cities", con)
        inputs = pd.read_sql(f"SELECT * FROM input.addresses ORDER BY id LIMIT {int(limit)}", con)
        try:
            states = pd.read_sql("SELECT state,abbreviation FROM ref.indian_state_abbrev", con)
        except Exception:
            states = pd.DataFrame({"state":[],"abbreviation":[]})

    # normalize refs
    for df, cols in [(postal,["city","state","country"]), (rta,["city","state","country"]), (world,["city","country"])]:
        for c in cols: df[c] = df[c].astype(str).map(Title)
    master_cities = sorted(set(postal["city"]).union(set(rta["city"])).union(set(world["city"])))
    t30_set = {Title(x) for x in t30_df["city"].astype(str).map(Title).unique().tolist()}

    # reverse index: city -> rows
    city_index={}
    for src, df in [("postal", postal), ("rta", rta)]:
        for _, row in df.iterrows():
            city_index.setdefault(row["city"], []).append((src, row))

    out_rows=[]; audit=[]

    for _, rec in inputs.iterrows():
        input_id = int(rec["id"])
        addr1, addr2, addr3 = rec.get("address1"), rec.get("address2"), rec.get("address3")
        in_city  = Title(rec.get("city"))
        in_state_raw = Title(rec.get("state"))
        in_country = Title(rec.get("country"))
        in_pin   = str(rec.get("pincode") or "").strip()

        concat = norm_text(" ".join([str(x or "") for x in [addr1,addr2,addr3,in_city,in_state_raw,in_country,in_pin]]))
        in_state = expand_state_abbrev(in_state_raw, states)

        # discover cities only from input/concat
        discovered_cities = discover_cities(concat, in_city, master_cities)

        # candidate pins: text + pins attached to those discovered cities
        candidate_pins = set(extract_pins(concat, in_pin))
        for c in discovered_cities:
            for src, row in city_index.get(c, []):
                candidate_pins.add(str(row["pincode"]))

        best=None; best_score=-1
        for p in sorted(candidate_pins):
            priority = 1 if p in extract_pins(concat) else 0
            rows=[]
            for c in discovered_cities:
                rows += [it for it in city_index.get(c, []) if str(it[1]["pincode"]) == p]
            if not rows: 
                continue
            for src, rr in rows:
                c=rr["city"]; s=rr["state"]; k=rr["country"]
                sc_city = max(sim(c, in_city), sim(c, base_city(in_city)),
                              max((sim(c, g) for g in _ngrams_12(concat)), default=0))
                sc_state= max(sim(s, in_state),
                              max((sim(s, g) for g in _ngrams_12(concat)), default=0))
                sc_ctry = max(sim(k, in_country),
                              max((sim(k, g) for g in _ngrams_12(concat)), default=0))
                score = priority + 0.40*sc_city + 0.40*sc_state + 0.10*sc_ctry
                if score > best_score:
                    best = {"pincode":p,"city":c,"state":s,"country":k,"src":src,
                            "sc_city":sc_city,"sc_state":sc_state,"sc_ctry":sc_ctry}
                    best_score = score
                audit.append({"input_id":input_id,"match_type":"pin-eval",
                              "candidate":f"{p}|{c}|{s}|{k}","score":score,"source":src})

        out_pin=None; out_city=None; out_state=None; out_country=None; source_used="none"
        P_input_city=0; P_city_db=0; P_input_state=0; P_state_db=0; P_input_country=0; P_country_db=0

        if best:
            out_pin = best["pincode"]
            out_city = best["city"]
            out_state = best["state"]
            out_country = best["country"]
            source_used = "pincode/discovered"
            P_input_city   = 1 if best["sc_city"]  >= THRESH else 0
            P_city_db      = 1
            P_input_state  = 1 if best["sc_state"] >= THRESH else 0
            P_state_db     = 1
            P_input_country= 1 if best["sc_ctry"]  >= THRESH else 0
            P_country_db   = 1
        else:
            # fallback: best row under discovered city list
            candidates=[]
            for c in discovered_cities:
                for src, rr in city_index.get(c, []):
                    sc_state = max(sim(rr["state"], in_state),
                                   max((sim(rr["state"], g) for g in _ngrams_12(concat)), default=0))
                    sc_ctry  = max(sim(rr["country"], in_country),
                                   max((sim(rr["country"], g) for g in _ngrams_12(concat)), default=0))
                    sc = 0.60*sc_state + 0.20*sc_ctry
                    candidates.append((sc, src, rr))
                    audit.append({"input_id":input_id,"match_type":"city-fallback",
                                  "candidate":f"{rr['pincode']}|{rr['city']}|{rr['state']}|{rr['country']}",
                                  "score":sc,"source":src})
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                sc, src, rr = candidates[0]
                out_pin=str(rr["pincode"]); out_city=rr["city"]; out_state=rr["state"]; out_country=rr["country"]
                source_used="city-fallback"

        # clean directions from output city
        if out_city:
            out_city = strip_city_directions(out_city)

        # flags
        t30_flag = 1 if (out_city and base_city(out_city) in t30_set) else 0
        foreign_flag = 1 if (out_country and Title(out_country) != "India") else 0
        pin_found_flag = 1 if out_pin else 0

        # possibles (restricted)
        all_possible_cities = sorted({strip_city_directions(c) for c in discovered_cities})
        all_possible_pins_text = extract_pins(concat, in_pin)
        all_possible_pins_db = sorted({str(it[1]["pincode"]) for c in discovered_cities for it in city_index.get(c, [])})
        all_possible_pins = sorted(set(all_possible_pins_text) | set(all_possible_pins_db))

        # local address remainder
        remove_terms=set()
        if out_pin: remove_terms.add(out_pin)
        if out_country: remove_terms.add(out_country)
        if out_state: remove_terms.add(out_state)
        for c in all_possible_cities + ([out_city] if out_city else []):
            remove_terms.add(c)
        s=" "+concat+" "
        for t in remove_terms:
            t = re.escape(str(t))
            s = re.sub(rf"(?i)\b{t}\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        local_address = s

        out_rows.append({
            # inputs
            "input_id": input_id,
            "address1": addr1, "address2": addr2, "address3": addr3,
            "input_city": in_city, "input_state_raw": in_state_raw, "input_state": in_state,
            "input_country": in_country, "input_pincode": in_pin,
            "concatenated_address": concat,

            # outputs
            "output_pincode": out_pin,
            "output_city": out_city,
            "output_state": out_state,
            "output_country": out_country,

            # six binary scores
            "Pincode-input_city_match": P_input_city,
            "Pincode-city_db_match": P_city_db,
            "Pincode-input_state_match": P_input_state,
            "Pincode-state_db_match": P_state_db,
            "Pincode-input_country_match": P_input_country,
            "Pincode-country_db_match": P_country_db,

            # flags
            "t30_city_possible": t30_flag,
            "foreign_country_possible": foreign_flag,
            "pincode_found": pin_found_flag,
            "source_used": source_used,

            # possibles (restricted)
            "all_possible_cities": json.dumps(all_possible_cities, ensure_ascii=False),
            "all_possible_pincodes_text": json.dumps(all_possible_pins_text, ensure_ascii=False),
            "all_possible_pincodes_db": json.dumps(all_possible_pins_db[:100], ensure_ascii=False),
            "all_possible_pincodes": json.dumps(all_possible_pins, ensure_ascii=False),

            # remainder
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

    # DB
    eng = get_engine()
    with eng.begin() as con:
        result_df.to_sql("validation_result_full", con, schema="output", if_exists="append", index=False, method="multi")
        minimal = result_df.rename(columns={
            "output_pincode":"chosen_pincode",
            "output_city":"chosen_city",
            "output_state":"chosen_state",
            "output_country":"chosen_country",
        })[["input_id","chosen_pincode","chosen_city","chosen_state","chosen_country",
            "t30_city_possible","foreign_country_possible","pincode_found","source_used"]]
        minimal.to_sql("validation_result", con, schema="output", if_exists="append", index=False, method="multi")

    print(f"✅ Validation done for {len(result_df)} rows. Excel → {xls_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--excel", type=str, default=None)
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel)
