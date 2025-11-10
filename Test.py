
import os, re, json
import pandas as pd
from sqlalchemy import text
from db_config import get_engine

try:
    from rapidfuzz import fuzz
    def sim(a,b): return fuzz.token_set_ratio(str(a or ""), str(b or ""))/100.0
except Exception:
    import difflib
    def sim(a,b): 
        return difflib.SequenceMatcher(None, str(a or "").lower(), str(b or "").lower()).ratio()

THR = 0.80
PIN_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')

DIRECTION_TOKENS = {
    "west","east","north","south","n","s","e","w","nw","ne","sw","se",
    "sector","zone","block","phase","ward","dist","district","taluka","taluk","mandal"
}

def norm_text(s):
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def title(s): 
    return str(s or "").strip().title()

def base_city(city_in: str) -> str:
    if not city_in: return None
    tokens = [t for t in re.split(r"[^A-Za-z]+", str(city_in)) if t]
    kept = [t for t in tokens if t.lower() not in DIRECTION_TOKENS]
    if not kept:
        return title(city_in)
    return title(" ".join(kept))

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

def choose_best(cands, key):
    return max(cands, key=key) if cands else None

def main(limit=1000, excel_path=None):
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

    rows = []
    audits = []

    for _, row in inputs.iterrows():
        input_id = int(row["id"])
        in_addr1 = row.get("address1")
        in_addr2 = row.get("address2")
        in_addr3 = row.get("address3")
        in_city  = title(row.get("city"))
        in_state = title(row.get("state"))
        in_country = title(row.get("country"))
        in_pin   = str(row.get("pincode") or "").strip()

        concatenated = norm_text(" ".join([str(x or "") for x in [in_addr1,in_addr2,in_addr3,in_city,in_state,in_country,in_pin]]))

        in_state_expanded = expand_state_from_abbrev(in_state, states)
        in_city_base = base_city(in_city)

        candidate_pins = set(extract_all_pins(concatenated, in_pin))
        pin_candidates = []
        for p in candidate_pins:
            if p in pin_map:
                for src, r in pin_map[p]:
                    c = title(r["city"]); s = title(r["state"]); k = title(r["country"])
                    sc_city_raw  = sim(c, in_city) if in_city else 0
                    sc_city_base = sim(c, in_city_base) if in_city_base else 0
                    sc_city = max(sc_city_raw, sc_city_base)
                    sc_state = sim(s, in_state_expanded) if in_state_expanded else 0
                    sc_ctry  = sim(k, in_country) if in_country else 0
                    score = 1.0 + 0.35*sc_city + 0.35*sc_state + 0.10*sc_ctry
                    pin_candidates.append({
                        "src": src, "pincode": p, "city": c, "state": s, "country": k,
                        "sc_city_raw": sc_city_raw, "sc_city_base": sc_city_base, "sc_city": sc_city,
                        "sc_state": sc_state, "sc_country": sc_ctry, "score": score
                    })
                    audits.append({"input_id": input_id, "match_type":"pincode", "candidate": f"{p}|{c}|{s}|{k}", "score": score, "source": src})

        city_to_pin_candidates = []
        if not pin_candidates:
            for src, df in [("postal", postal), ("rta", rta)]:
                df_tmp = df.copy()
                df_tmp["city_score_raw"]  = df_tmp["city_t"].apply(lambda c: sim(c, in_city) if in_city else 0)
                df_tmp["city_score_base"] = df_tmp["city_t"].apply(lambda c: sim(c, in_city_base) if in_city_base else 0)
                df_tmp["city_score"] = df_tmp[["city_score_raw","city_score_base"]].max(axis=1)
                df_tmp["state_score"]= df_tmp["state_t"].apply(lambda s: sim(s, in_state_expanded) if in_state_expanded else 0)
                df_tmp["country_score"]= df_tmp["country_t"].apply(lambda k: sim(k, in_country) if in_country else 0)

                filt = (df_tmp["city_score"]>=THR) | (df_tmp["state_score"]>=THR) | ((df_tmp["city_score"]>=0.70)&(df_tmp["state_score"]>=0.70))
                for _, r in df_tmp[filt].iterrows():
                    score = 0.45*r["city_score"] + 0.45*r["state_score"] + 0.10*r["country_score"]
                    city_to_pin_candidates.append({
                        "src": src, "pincode": r["pincode"], "city": r["city_t"], "state": r["state_t"], "country": r["country_t"],
                        "sc_city_raw": r["city_score_raw"], "sc_city_base": r["city_score_base"], "sc_city": r["city_score"],
                        "sc_state": r["state_score"], "sc_country": r["country_score"], "score": score
                    })
            for c in city_to_pin_candidates:
                audits.append({"input_id": input_id, "match_type":"city→pin", "candidate": f"{c['pincode']}|{c['city']}|{c['state']}|{c['country']}", "score": c["score"], "source": c["src"]})

        world_candidates = []
        if in_city_base:
            wc = world[world["city_t"]==in_city_base]
            for _, r in wc.iterrows():
                sc_city = sim(in_city_base, r["city_t"])
                sc_ctry = sim(in_country, r["country_t"]) if in_country else 0
                score = 0.75*sc_city + 0.25*sc_ctry
                world_candidates.append({"city": r["city_t"], "country": r["country_t"], "score": score})
                audits.append({"input_id": input_id, "match_type":"world_city", "candidate": f"{r['city_t']}|{r['country_t']}", "score": score, "source": "world_cities"})

        if pin_candidates:
            top = choose_best(pin_candidates, key=lambda x: x["score"])
            chosen = top.copy(); chosen_src = "pincode"
        elif city_to_pin_candidates:
            top = choose_best(city_to_pin_candidates, key=lambda x: x["score"])
            chosen = top.copy(); chosen_src = "city→pin"
        elif world_candidates:
            top = choose_best(world_candidates, key=lambda x: x["score"])
            chosen = {"pincode": None, "city": top["city"], "state": in_state_expanded or None, "country": top["country"],
                      "sc_city_raw": None, "sc_city_base": None, "sc_city": None, "sc_state": None, "sc_country": None,
                      "score": top["score"]}
            chosen_src = "world_cities"
        else:
            chosen = {"pincode": in_pin or None, "city": in_city_base or in_city or None, "state": in_state_expanded or None, "country": in_country or None,
                      "sc_city_raw": None, "sc_city_base": None, "sc_city": None, "sc_state": None, "sc_country": None,
                      "score": 0.0}
            chosen_src = "fallback"

        t30_set = set(t30)
        flag_t30 = 1 if (chosen.get("city") and base_city(chosen["city"]) in t30_set) else 0
        flag_foreign = 1 if (chosen.get("country") and title(chosen["country"]) != "India") else 0
        flag_pin_present = 1 if chosen.get("pincode") else 0

        rows.append({
            "input_id": input_id,
            "input_address1": in_addr1, "input_address2": in_addr2, "input_address3": in_addr3,
            "input_city": in_city, "input_city_base": in_city_base,
            "input_state_raw": in_state, "input_state": in_state_expanded,
            "input_country": in_country, "input_pincode": in_pin,
            "concatenated_address": concatenated,

            "output_pincode": chosen.get("pincode"),
            "output_city": chosen.get("city"),
            "output_state": chosen.get("state"),
            "output_country": chosen.get("country"),
            "source_used": chosen_src,
            "total_score": chosen.get("score"),

            "score_city_raw": chosen.get("sc_city_raw"),
            "score_city_base": chosen.get("sc_city_base"),
            "score_city_final": chosen.get("sc_city"),
            "score_state": chosen.get("sc_state"),
            "score_country": chosen.get("sc_country"),

            "flag_t30_possible": flag_t30,
            "flag_foreign_country_possible": flag_foreign,
            "flag_pincode_found": flag_pin_present
        })

    result_df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audits) if audits else pd.DataFrame(columns=["input_id","match_type","candidate","score","source"])

    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    if excel_path is None:
        excel_path = os.path.join(out_dir, "validation_results.xlsx")
    with pd.ExcelWriter(excel_path) as xl:
        result_df.to_excel(xl, index=False, sheet_name="results")
        audit_df.to_excel(xl, index=False, sheet_name="audit")

    eng = get_engine()
    with eng.begin() as con:
        minimal = result_df.rename(columns={
            "output_pincode":"chosen_pincode",
            "output_city":"chosen_city",
            "output_state":"chosen_state",
            "output_country":"chosen_country",
            "score_city_final":"score_pincode_input_city",
            "score_state":"score_pincode_input_state",
            "score_country":"score_pincode_input_country"
        })[
            ["input_id","chosen_pincode","chosen_city","chosen_state","chosen_country",
             "score_pincode_input_city","score_pincode_input_state","score_pincode_input_country",
             "flag_t30_possible","flag_foreign_country_possible","flag_pincode_found","source_used"]
        ]
        minimal.to_sql("validation_result", con, schema="output", if_exists="append", index=False, method="multi")
        result_df.to_sql("validation_result_full", con, schema="output", if_exists="append", index=False, method="multi")
        audit_df.to_sql("audit_matches", con, schema="output", if_exists="append", index=False, method="multi")

    print(f"Validation complete: {len(result_df)} rows. Excel -> {excel_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000, help="Process this many rows first (default 1000).")
    ap.add_argument("--excel", type=str, default=None, help="Path to Excel output file.")
    args = ap.parse_args()
    main(limit=args.limit, excel_path=args.excel)
