import os, re, json, math
import pandas as pd
from sqlalchemy import text
from db_config import get_engine

try:
    from rapidfuzz import fuzz
    def sim(a,b): return fuzz.token_set_ratio(str(a or ""), str(b or ""))/100.0
except Exception:
    import difflib
    def sim(a,b): return difflib.SequenceMatcher(None, str(a or "").lower(), str(b or "").lower()).ratio()

THR = 0.80
PIN_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')

def norm(s):
    s = str(s or "").strip()
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def title(s): 
    return str(s or "").strip().title()

def extract_all_pins(*fields):
    pins = set()
    for f in fields:
        if not f: continue
        for m in PIN_RE.findall(str(f)):
            pins.add(m)
    return sorted(pins)

def expand_state_from_abbrev(state_in, abbrev_df):
    if not state_in: return state_in
    s = title(state_in)
    row = abbrev_df[abbrev_df["abbreviation"].str.upper()==s.upper()]
    if not row.empty:
        return title(row.iloc[0]["state"])
    return s

def best_k(items, key, k=5):
    return sorted(items, key=key, reverse=True)[:k]

def main(limit=1000, to_excel=True):
    eng = get_engine()
    with eng.begin() as con:
        postal = pd.read_sql("SELECT city,state,pincode,country FROM ref.postal_pincode", con)
        rta    = pd.read_sql("SELECT city,state,pincode,country FROM ref.rta_pincode", con)
        states = pd.read_sql("SELECT state, abbreviation FROM ref.indian_state_abbrev", con)
        world  = pd.read_sql("SELECT city,country FROM ref.world_cities", con)
        countries = pd.read_sql("SELECT name FROM ref.countries", con)["name"].str.title().unique().tolist()
        t30    = pd.read_sql("SELECT city FROM ref.t30_cities", con)["city"].str.title().unique().tolist()
        inputs = pd.read_sql(f"SELECT * FROM input.addresses ORDER BY id LIMIT {int(limit)}", con)

    postal["city_t"] = postal["city"].apply(title)
    postal["state_t"]= postal["state"].apply(title)
    postal["country_t"]= postal["country"].apply(title)
    rta["city_t"] = rta["city"].apply(title)
    rta["state_t"]= rta["state"].apply(title)
    rta["country_t"]= rta["country"].apply(title)
    world["city_t"]= world["city"].apply(title)
    world["country_t"]= world["country"].apply(title)
    states["state_t"]= states["state"].apply(title)
    states["abbreviation"]= states["abbreviation"].astype(str).str.upper()

    pin_map = {}
    for src, df in [("postal", postal), ("rta", rta)]:
        for _, r in df.iterrows():
            pin_map.setdefault(r["pincode"], []).append((src, r))

    results = []
    audit_rows = []

    for _, row in inputs.iterrows():
        input_id = int(row["id"])
        # Concatenated text uses: address1, address2, address3, city, state, country, pincode
        addr_fields = [row.get(c) for c in ["address1","address2","address3","city","state","country","pincode"]]
        addr_text = norm(" ".join([str(x or "") for x in addr_fields]))

        city_in = title(row.get("city"))
        state_in= title(row.get("state"))
        country_in = title(row.get("country"))
        pin_in = str(row.get("pincode") or "").strip()
        state_in = expand_state_from_abbrev(state_in, states)

        # 1) Extract/confirm pins, then score against city/state/country
        candidate_pins = set(extract_all_pins(addr_text, pin_in))
        pin_candidates = []
        for p in candidate_pins:
            if p in pin_map:
                src_rows = pin_map[p]
                for src, r in src_rows:
                    sc_city = sim(title(r["city"]), city_in) if city_in else 0
                    sc_state= sim(title(r["state"]), state_in) if state_in else 0
                    sc_ctry = sim(title(r["country"]), country_in) if country_in else 0
                    # Weighted score: strong pincode presence, plus field alignment checks
                    score = 1.0*1.0 + 0.30*sc_city + 0.30*sc_state + 0.10*sc_ctry
                    pin_candidates.append({
                        "pincode": p, "city": title(r["city"]), "state": title(r["state"]), "country": title(r["country"]),
                        "score": score, "src": src,
                        "sc_city": sc_city, "sc_state": sc_state, "sc_country": sc_ctry
                    })
                    audit_rows.append({"input_id": input_id, "match_type":"pincode", "candidate": f"{p}|{r['city']}|{r['state']}", "score": score, "source": src})

        # 2) If no strong pin path, try city/state → pin fallback across postal+rta
        city_to_pin_candidates = []
        if not pin_candidates:
            for src, df in [("postal", postal), ("rta", rta)]:
                df_tmp = df.copy()
                df_tmp["cscore"] = df_tmp["city_t"].apply(lambda c: sim(c, city_in) if city_in else 0)
                df_tmp["sscore"]= df_tmp["state_t"].apply(lambda s: sim(s, state_in) if state_in else 0)
                df_tmp["kscore"]= df_tmp["country_t"].apply(lambda k: sim(k, country_in) if country_in else 0)
                # Keep candidates where (city>=0.80) or (state>=0.80) or (city>=0.70 and state>=0.70)
                filt = (df_tmp["cscore"]>=THR) | (df_tmp["sscore"]>=THR) | ((df_tmp["cscore"]>=0.70)&(df_tmp["sscore"]>=0.70))
                for _, r in df_tmp[filt].iterrows():
                    score = 0.45*r["cscore"] + 0.45*r["sscore"] + 0.10*r["kscore"]
                    city_to_pin_candidates.append({
                        "pincode": r["pincode"], "city": r["city_t"], "state": r["state_t"], "country": r["country_t"],
                        "score": score, "src": src, "sc_city": r["cscore"], "sc_state": r["sscore"], "sc_country": r["kscore"]
                    })
            for c in city_to_pin_candidates:
                audit_rows.append({"input_id": input_id, "match_type":"city→pin", "candidate": f"{c['pincode']}|{c['city']}|{c['state']}", "score": c["score"], "source": c["src"]})

        # 3) World cities/countries reconciliation (no pincode path)
        world_candidates = []
        if city_in:
            wc = world[world["city_t"]==city_in]
            for _, r in wc.iterrows():
                sc_city = sim(city_in, r["city_t"])
                sc_ctry = sim(country_in, r["country_t"]) if country_in else 0
                score = 0.7*sc_city + 0.3*sc_ctry
                world_candidates.append({"city": r["city_t"], "country": r["country_t"], "score": score})
                audit_rows.append({"input_id": input_id, "match_type":"world_city", "candidate": f"{r['city_t']}|{r['country_t']}", "score": score, "source": "world_cities"})

        # 4) Final choice: prefer pincode-based; else city→pin; else world; else fallback to input
        chosen = None
        candidate_pool = pin_candidates or city_to_pin_candidates
        if candidate_pool:
            top = max(candidate_pool, key=lambda x: x["score"])
            chosen = {
                "pincode": top["pincode"],
                "city": top["city"],
                "state": top["state"],
                "country": top["country"],
                "source": "pincode" if pin_candidates else "city→pin"
            }
        elif world_candidates:
            top = max(world_candidates, key=lambda x: x["score"])
            chosen = {"pincode": None, "city": top["city"], "state": state_in or None, "country": top["country"], "source":"world_cities"}
        else:
            chosen = {"pincode": pin_in or None, "city": city_in or None, "state": state_in or None, "country": country_in or None, "source":"fallback"}

        # 5) Flags
        flag_t30 = 1 if (chosen["city"] and title(chosen["city"]) in set(t30)) else 0
        flag_foreign = 1 if (chosen["country"] and title(chosen["country"]) != "India") else 0
        flag_pin_found = 1 if chosen["pincode"] else 0

        # 6) Collect audit candidates (top 5 for each dimension)
        all_pins = [c["pincode"] for c in sorted(pin_candidates or city_to_pin_candidates, key=lambda x: x["score"], reverse=True)[:5]]
        all_cities = list({c["city"] for c in sorted((pin_candidates or []) + world_candidates, key=lambda x: x["score"], reverse=True)[:5]})
        all_states = list({c["state"] for c in sorted(pin_candidates or city_to_pin_candidates, key=lambda x: x["score"], reverse=True)[:5] if c.get("state")})
        all_countries = list({(c.get("country") or country_in) for c in sorted((pin_candidates or []) + world_candidates, key=lambda x: x["score"], reverse=True)[:5] if (c.get("country") or country_in)})

        results.append({
            "input_id": input_id,
            "chosen_pincode": chosen["pincode"],
            "chosen_city": chosen["city"],
            "chosen_state": chosen["state"],
            "chosen_country": chosen["country"],
            "score_pincode_input_city": None,
            "score_pincode_city_db": None,
            "score_pincode_input_state": None,
            "score_pincode_state_db": None,
            "score_pincode_input_country": None,
            "score_pincode_country_db": None,
            "flag_t30_possible": flag_t30,
            "flag_foreign_country_possible": flag_foreign,
            "flag_pincode_found": flag_pin_found,
            "source_used": chosen["source"],
            "all_pincodes": json.dumps(all_pins, ensure_ascii=False),
            "all_cities": json.dumps(all_cities, ensure_ascii=False),
            "all_states": json.dumps(all_states, ensure_ascii=False),
            "all_countries": json.dumps(all_countries, ensure_ascii=False),
        })

    out_df = pd.DataFrame(results)
    audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame(columns=["input_id","match_type","candidate","score","source"])

    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    excel_path = os.path.join(outputs_dir, "validation_results.xlsx")
    if to_excel:
        with pd.ExcelWriter(excel_path) as xl:
            out_df.to_excel(xl, index=False, sheet_name="results")
            audit_df.to_excel(xl, index=False, sheet_name="audit")

    eng = get_engine()
    with eng.begin() as con:
        out_df.to_sql("validation_result", con, schema="output", if_exists="append", index=False, method="multi")
        audit_df.to_sql("audit_matches", con, schema="output", if_exists="append", index=False, method="multi")

    print(f"Wrote {len(out_df)} rows to output.validation_result; Excel at: {excel_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000, help="Process this many rows first (default 1000).")
    args = ap.parse_args()
    main(limit=args.limit)
