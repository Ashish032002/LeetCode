import os
import re
import json
import math
import pandas as pd
from sqlalchemy import text
from db_config import get_engine

# ---------------- Similarity -----------------
try:
    from rapidfuzz import fuzz

    def sim(a, b) -> float:
        return fuzz.token_set_ratio(str(a or ""), str(b or "")) / 100.0

except Exception:
    import difflib

    def sim(a, b) -> float:
        return difflib.SequenceMatcher(
            None, str(a or "").lower(), str(b or "").lower()
        ).ratio()


THRESH = 0.80
PIN_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

# Words to strip from city for clean output
CITY_DIRECTIONS = {
    "north",
    "south",
    "east",
    "west",
    "n",
    "s",
    "e",
    "w",
    "northwest",
    "northeast",
    "southwest",
    "southeast",
    "nw",
    "ne",
    "sw",
    "se",
    "city",
    "moffusil",
    "division",
    "district",
    "zone",
    "sector",
    "block",
    "phase",
}


def Title(s):
    return str(s or "").strip().title()


def Upper(s):
    return str(s or "").strip().upper()


def norm_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_pincodes(*fields):
    pins = set()
    for f in fields:
        if not f:
            continue
        for m in PIN_RE.findall(str(f)):
            pins.add(m)
    return sorted(pins)


def tokens_alpha(s: str):
    return [t for t in re.split(r"[^A-Za-z]+", str(s or "")) if len(t) > 0]


def ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def make_city_state_ngrams(concat_text: str):
    """
    1-gram + 2-gram ngrams used for CITY / STATE fuzzy.
    """
    toks = tokens_alpha(concat_text)
    grams = set()
    # unigrams
    for t in toks:
        if len(t) >= 2:
            grams.add(Title(t))
    # bigrams
    for bg in ngrams(toks, 2):
        a, b = bg.split(" ", 1)
        if len(a) >= 2 and len(b) >= 2:
            grams.add(Title(bg))
    return grams


def make_country_ngrams(concat_text: str):
    """
    1-gram to 4-gram ngrams for COUNTRY fuzzy (for things like 'United States Of America').
    """
    toks = tokens_alpha(concat_text)
    grams = set()
    for n in range(1, 5):  # 1,2,3,4-grams
        for g in ngrams(toks, n):
            grams.add(Title(g))
    return grams


def clean_output_city(city: str):
    """Drop direction / division words from output city like 'Bengaluru South' -> 'Bengaluru'"""
    if not city:
        return city
    parts = [p for p in tokens_alpha(city)]
    parts = [p for p in parts if p.lower() not in CITY_DIRECTIONS]
    return Title(" ".join(parts)) if parts else Title(city)


def remove_terms(text: str, terms: list) -> str:
    if not text:
        return text
    s = " " + text + " "
    for t in terms:
        if not t:
            continue
        t = re.escape(str(t).strip())
        s = re.sub(rf"(?i)\b{t}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- State abbreviation expansion (DB + static) ----------
def _norm_token(x: str) -> str:
    x = str(x or "").strip()
    x = re.sub(r"[^A-Za-z]", "", x)
    return x.upper()


def _initials_of(name: str) -> str:
    toks = [t for t in re.split(r"[^A-Za-z]+", str(name or "")) if t]
    return "".join(t[0] for t in toks).upper() if toks else ""


STATE_ALIAS_STATIC = {
    "Andhra Pradesh": {"AP", "AD", "ANDHRAPRADESH"},
    "Arunachal Pradesh": {"AR", "ARN"},
    "Assam": {"AS"},
    "Bihar": {"BH", "BR"},
    "Chhattisgarh": {"CG", "CHH", "CT"},
    "Goa": {"GO", "GA"},
    "Gujarat": {"GU", "GJ"},
    "Haryana": {"HA", "HR"},
    "Himachal Pradesh": {"HP", "HIM"},
    "Jammu And Kashmir": {"JK", "J&K", "JNK", "JAMMUANDKASHMIR"},
    "Jharkhand": {"JH", "JHD", "JD"},
    "Karnataka": {"KA", "KAR"},
    "Kerala": {"KE", "KL"},
    "Madhya Pradesh": {"MP", "MD", "MADHYAPRADESH"},
    "Maharashtra": {"MA", "MH"},
    "Manipur": {"MN"},
    "Meghalaya": {"ML", "ME"},
    "Mizoram": {"MZ"},
    "Nagaland": {"NL"},
    "Odisha": {"OD", "OR", "ORISSA", "ODISHA"},
    "Punjab": {"PU", "PB"},
    "Rajasthan": {"RA", "RJ"},
    "Sikkim": {"SK"},
    "Tamil Nadu": {"TN", "TM"},
    "Telangana": {"TG", "TS", "TE"},
    "Tripura": {"TR"},
    "Uttar Pradesh": {"UP", "UTP"},
    "Uttarakhand": {"UK", "UA", "UC"},
    "West Bengal": {"WB", "W.B", "WBG"},
    "Delhi": {"DL", "ND", "DELHI", "NCT"},
    "Chandigarh": {"CH", "CHD"},
    "Puducherry": {"PO", "PY", "PONDICHERRY"},
    "Ladakh": {"LA"},
    "Lakshadweep": {"LD"},
}


def expand_state_abbrev(state_in, states_df):
    if not state_in:
        return state_in
    s_raw = str(state_in).strip()
    s_norm = re.sub(r"[’'`“”]", "", s_raw)
    s_title = Title(s_norm)
    tok = _norm_token(s_norm)

    states_df = states_df.copy()
    states_df["state"] = states_df["state"].astype(str).map(Title)
    if "abbreviation" not in states_df.columns:
        states_df["abbreviation"] = ""
    states_df["abbreviation"] = states_df["abbreviation"].astype(str).fillna("")

    alias = {}
    for _, row in states_df.iterrows():
        canon = Title(row["state"])
        alias.setdefault(canon, set())
        for t in re.split(r"[^A-Za-z]+", str(row["abbreviation"])):
            if not t:
                continue
            alias[canon].add(_norm_token(t))
        alias[canon].add(_initials_of(canon))
        alias[canon].add(_norm_token(canon))
        if canon in STATE_ALIAS_STATIC:
            alias[canon].update({_norm_token(x) for x in STATE_ALIAS_STATIC[canon]})

    # direct match on alias tokens
    for canon, toks in alias.items():
        if tok in toks:
            return canon

    # direct match on full title
    if s_title in alias:
        return s_title

    # fuzzy fallback
    try:
        from rapidfuzz import fuzz

        best_c, best_s = None, -1.0
        for canon in alias.keys():
            sc = fuzz.token_set_ratio(canon, s_title) / 100.0
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


# ---------- Main validator ----------
def main(limit=1000, excel=None, batch_size=200):
    eng = get_engine()
    with eng.begin() as con:
        postal = pd.read_sql(
            "SELECT city,state,pincode,country FROM ref.postal_pincode", con
        )
        rta = pd.read_sql(
            "SELECT city,state,pincode,country FROM ref.rta_pincode", con
        )
        try:
            states = pd.read_sql(
                "SELECT state,abbreviation FROM ref.indian_state_abbrev", con
            )
        except Exception:
            states = pd.DataFrame({"state": [], "abbreviation": []})
        world = pd.read_sql("SELECT city,country FROM ref.world_cities", con)
        t30 = (
            pd.read_sql("SELECT city FROM ref.t30_cities", con)["city"]
            .astype(str)
            .str.title()
            .unique()
            .tolist()
        )
        inputs = pd.read_sql(
            f"SELECT * FROM input.addresses ORDER BY id LIMIT {int(limit)}", con
        )

    # Normalize refs
    for df, cols in [
        (postal, ["city", "state", "country"]),
        (rta, ["city", "state", "country"]),
        (world, ["city", "country"]),
    ]:
        for c in cols:
            df[c] = df[c].astype(str).map(Title)

    # Master city & country sets
    city_master_set = (
        set(postal["city"].tolist())
        | set(rta["city"].tolist())
        | set(world["city"].tolist())
    )
    country_master_set = (
        set(postal["country"].tolist())
        | set(rta["country"].tolist())
        | set(world["country"].tolist())
    )

    # Pin index
    pin_index = {}
    for src, df in [("postal", postal), ("rta", rta)]:
        for _, rr in df.iterrows():
            pin_index.setdefault(str(rr["pincode"]), []).append((src, rr))

    t30_set = set(Title(x) for x in t30)

    results = []
    audits = []

    for start in range(0, len(inputs), batch_size):
        chunk = inputs.iloc[start : start + batch_size].copy()
        for _, r in chunk.iterrows():
            input_id = int(r["id"])
            in_addr1 = r.get("address1")
            in_addr2 = r.get("address2")
            in_addr3 = r.get("address3")
            in_city = Title(r.get("city"))
            in_state_raw = Title(r.get("state"))
            in_country = Title(r.get("country"))
            in_pin = str(r.get("pincode") or "").strip()

            # expand state abbreviation
            in_state = expand_state_abbrev(in_state_raw, states)

            concat = norm_text(
                " ".join(
                    [
                        str(x or "")
                        for x in [
                            in_addr1,
                            in_addr2,
                            in_addr3,
                            in_city,
                            in_state,
                            in_country,
                            in_pin,
                        ]
                    ]
                )
            )

            # n-grams
            grams_city_state = make_city_state_ngrams(concat)
            grams_country = make_country_ngrams(concat)

            # ----- CITY candidates (from input + concat) -----
            city_candidates = set()
            city_seed = set()
            if in_city:
                city_seed.add(in_city)
            city_seed |= grams_city_state

            for g in city_seed:
                if not g:
                    continue
                if g in city_master_set:
                    city_candidates.add(g)
                    continue
                # fuzzy on first-letter filtered
                for cm in (x for x in city_master_set if x[0] == g[0]):
                    if sim(g, cm) >= THRESH:
                        city_candidates.add(cm)

            # ----- COUNTRY candidates (1..4-gram) -----
            country_candidates = set()
            if in_country:
                country_candidates.add(in_country)
            for g in grams_country:
                if g in country_master_set:
                    country_candidates.add(g)
                    continue
                for cm in (x for x in country_master_set if x[0] == g[0]):
                    if sim(g, cm) >= THRESH:
                        country_candidates.add(cm)

            # ----- Pincodes from text -----
            pins_text = set(extract_pincodes(concat, in_pin))

            out_pin = None
            out_city = None
            out_state = None
            out_country = None
            source_used = "fallback"

            # store best fuzzy scores related to pincode path
            sc_city_pc = 0.0
            sc_state_pc = 0.0
            sc_country_pc = 0.0

            # 1) pincode-first
            if pins_text:
                best = None
                best_score = -1.0
                for p in pins_text:
                    rows = pin_index.get(p) or []
                    for src, rr in rows:
                        c = rr["city"]
                        s = rr["state"]
                        k = rr["country"]

                        sc_city = max(
                            sim(c, in_city) if in_city else 0.0,
                            max(
                                (sim(c, g) for g in city_candidates), default=0.0
                            ),
                        )
                        sc_state = sim(s, in_state) if in_state else 0.0
                        sc_country = max(
                            sim(k, in_country) if in_country else 0.0,
                            max(
                                (sim(k, g) for g in country_candidates),
                                default=0.0,
                            ),
                        )
                        score = (
                            1.0
                            + 0.45 * sc_city
                            + 0.35 * sc_state
                            + 0.10 * sc_country
                        )

                        audits.append(
                            {
                                "input_id": input_id,
                                "match_type": "pincode",
                                "candidate": f"{p}|{c}|{s}|{k}",
                                "score": score,
                                "source": src,
                            }
                        )
                        if score > best_score:
                            best = (p, c, s, k, src, sc_city, sc_state, sc_country)
                            best_score = score

                if best:
                    (
                        out_pin,
                        out_city,
                        out_state,
                        out_country,
                        source_used,
                        sc_city_pc,
                        sc_state_pc,
                        sc_country_pc,
                    ) = best

            # 2) city→pin only from discovered city candidates
            all_possible_pincodes_db = set()
            if out_pin is None and city_candidates:
                for src, df in [("postal", postal), ("rta", rta)]:
                    sub = df[df["city"].isin(city_candidates)].copy()
                    for _, rr in sub.iterrows():
                        all_possible_pincodes_db.add(str(rr["pincode"]))
                        audits.append(
                            {
                                "input_id": input_id,
                                "match_type": "city→pin",
                                "candidate": f"{rr['pincode']}|{rr['city']}|{rr['state']}|{rr['country']}",
                                "score": 1.0,
                                "source": src,
                            }
                        )
                if all_possible_pincodes_db:
                    best = None
                    best_s = -1.0
                    for p in all_possible_pincodes_db:
                        for src, rr in pin_index.get(p, []):
                            sc = 0.6 * (
                                sim(rr["state"], in_state) if in_state else 0.0
                            ) + 0.4 * (
                                sim(rr["country"], in_country)
                                if in_country
                                else 0.0
                            )
                            if sc > best_s:
                                best = (
                                    p,
                                    rr["city"],
                                    rr["state"],
                                    rr["country"],
                                )
                                best_s = sc
                    if best:
                        out_pin, out_city, out_state, out_country = best
                        source_used = "city→pin"

            # 3) world cities fallback for city / country
            if (not out_city) and city_candidates:
                wc = world[world["city"].isin(city_candidates)].copy()
                if not wc.empty:
                    wc["kscore"] = wc["country"].apply(
                        lambda k: sim(k, in_country) if in_country else 0.0
                    )
                    idx = wc["kscore"].astype(float).idxmax()
                    out_city = wc.loc[idx, "city"]
                    out_country = wc.loc[idx, "country"]
                    if not out_state:
                        out_state = in_state
                    source_used = "world_cities"

            # ------- SCORE CALCULATIONS --------
            # Rule A: value-based (exact/fuzzy)
            city_score_val = 0.0
            state_score_val = 0.0
            country_score_val = 0.0

            if out_city:
                city_score_val = max(
                    sim(out_city, in_city) if in_city else 0.0,
                    max(
                        (sim(out_city, g) for g in grams_city_state), default=0.0
                    ),
                )

            if out_state:
                state_score_val = max(
                    sim(out_state, in_state_raw) if in_state_raw else 0.0,
                    sim(out_state, in_state) if in_state else 0.0,
                )

            if out_country:
                country_score_val = max(
                    sim(out_country, in_country) if in_country else 0.0,
                    max(
                        (sim(out_country, g) for g in grams_country),
                        default=0.0,
                    ),
                )

            # Rule B: combination vs pincode
            city_score_pin = 0.0
            state_score_pin = 0.0
            country_score_pin = 0.0

            if out_pin and out_pin in pin_index:
                # see if any record for that pincode lines up with outputs
                best_city = 0.0
                best_state = 0.0
                best_country = 0.0
                for src, rr in pin_index[out_pin]:
                    best_city = max(
                        best_city,
                        sim(out_city, rr["city"]) if out_city else 0.0,
                    )
                    best_state = max(
                        best_state,
                        sim(out_state, rr["state"]) if out_state else 0.0,
                    )
                    best_country = max(
                        best_country,
                        sim(out_country, rr["country"])
                        if out_country
                        else 0.0,
                    )
                city_score_pin = best_city
                state_score_pin = best_state
                country_score_pin = best_country

            # Rule C: ambiguity
            #  - 1 candidate -> 1.0
            #  - 2-3 -> 0.8
            #  - >3 -> 0.5
            def ambiguity_score(n_candidates: int) -> float:
                if n_candidates <= 1:
                    return 1.0
                if n_candidates <= 3:
                    return 0.8
                return 0.5

            city_score_amb = ambiguity_score(len(city_candidates))
            # for state and country: we don't keep full candidate sets,
            # so just check presence vs absence
            state_score_amb = 1.0 if out_state else 0.0
            country_score_amb = 1.0 if out_country else 0.0

            # convert scores to 0–100 for output
            city_score_a = round(city_score_val * 100, 2)
            city_score_b = round(city_score_pin * 100, 2)
            city_score_c = round(city_score_amb * 100, 2)

            state_score_a = round(state_score_val * 100, 2)
            state_score_b = round(state_score_pin * 100, 2)
            state_score_c = round(state_score_amb * 100, 2)

            country_score_a = round(country_score_val * 100, 2)
            country_score_b = round(country_score_pin * 100, 2)
            country_score_c = round(country_score_amb * 100, 2)

            all_scores = [
                city_score_a,
                city_score_b,
                city_score_c,
                state_score_a,
                state_score_b,
                state_score_c,
                country_score_a,
                country_score_b,
                country_score_c,
            ]
            overall_score = round(
                sum(all_scores) / (len(all_scores) or 1), 2
            )

            # ------- FLAGS & POSSIBLE LISTS -------
            t30_city_possible = 1 if (
                (out_city and Title(clean_output_city(out_city)) in t30_set)
                or any(
                    Title(clean_output_city(c)) in t30_set
                    for c in city_candidates
                )
            ) else 0

            foreign_country_possible = 1 if any(
                Title(k) != "India"
                for k in [out_country, in_country]
                if k
            ) else 0

            # possible lists – filter out blanks & NaN
            all_cities_list = sorted(
                {Title(c) for c in city_candidates if c}
            )
            all_states_raw = []
            if out_state:
                all_states_raw.append(Title(out_state))
            elif in_state:
                all_states_raw.append(Title(in_state))

            def not_nan(x: str) -> bool:
                if not x:
                    return False
                return str(x).strip().lower() not in {"nan", "none", "null"}

            all_states_list = sorted({s for s in all_states_raw if not_nan(s)})

            all_countries_list = sorted(
                {
                    Title(k)
                    for k in [out_country, in_country]
                    if k and not_nan(k)
                }
            )

            all_pincodes_text_list = sorted(pins_text)
            all_pincodes_union = sorted(
                set(all_pincodes_text_list) | set(all_possible_pincodes_db)
            )

            # Cleaned output city
            out_city_clean = clean_output_city(out_city)

            # Local address remainder – remove discovered entities & pincode
            remove_list = []
            if out_pin:
                remove_list.append(out_pin)
            if out_country:
                remove_list.append(out_country)
            if out_state:
                remove_list.append(out_state)
            for c in [out_city_clean] + list(city_candidates):
                if c:
                    remove_list.append(c)
            local_address = remove_terms(concat, remove_list)

            # -------- REASON COLUMN ----------
            reasons = []
            # thresholds – can tune
            LOW = 80.0

            if not out_pin:
                reasons.append("pincode_not_found")
            if city_score_a < LOW:
                reasons.append("low_city_value_match")
            if state_score_a < LOW:
                reasons.append("low_state_value_match")
            if country_score_a < LOW:
                reasons.append("low_country_value_match")
            if len(city_candidates) > 1:
                reasons.append("ambiguous_city_candidates")
            if foreign_country_possible:
                reasons.append("foreign_country_detected")
            reason_text = "; ".join(sorted(set(reasons))) if reasons else ""

            results.append(
                {
                    # Inputs
                    "input_id": input_id,
                    "address1": r.get("address1"),
                    "address2": r.get("address2"),
                    "address3": r.get("address3"),
                    "input_city": in_city,
                    "input_state_raw": in_state_raw,
                    "input_state": in_state,
                    "input_country": in_country,
                    "input_pincode": in_pin,
                    "concatenated_address": concat,
                    # Outputs
                    "output_pincode": out_pin,
                    "output_city": out_city_clean,
                    "output_state": out_state,
                    "output_country": out_country,
                    # NEW scores (0–100)
                    "city_score_rule_a": city_score_a,
                    "city_score_rule_b": city_score_b,
                    "city_score_rule_c": city_score_c,
                    "state_score_rule_a": state_score_a,
                    "state_score_rule_b": state_score_b,
                    "state_score_rule_c": state_score_c,
                    "country_score_rule_a": country_score_a,
                    "country_score_rule_b": country_score_b,
                    "country_score_rule_c": country_score_c,
                    "overall_score": overall_score,
                    "reason": reason_text,
                    # Flags
                    "t30_city_possible": t30_city_possible,
                    "foreign_country_possible": foreign_country_possible,
                    "pincode_found": 1 if out_pin else 0,
                    "source_used": source_used,
                    # all possible lists (NO all_possible_pincode_db)
                    "all_possible_countries": json.dumps(
                        all_countries_list, ensure_ascii=False
                    ),
                    "all_possible_states": json.dumps(
                        all_states_list, ensure_ascii=False
                    ),
                    "all_possible_cities": json.dumps(
                        all_cities_list, ensure_ascii=False
                    ),
                    "all_possible_pincodes_text": json.dumps(
                        all_pincodes_text_list, ensure_ascii=False
                    ),
                    "all_possible_pincodes": json.dumps(
                        all_pincodes_union, ensure_ascii=False
                    ),
                    # remainder
                    "local_address": local_address,
                }
            )

    result_df = pd.DataFrame(results)
    audit_df = (
        pd.DataFrame(audits)
        if audits
        else pd.DataFrame(
            columns=["input_id", "match_type", "candidate", "score", "source"]
        )
    )

    # Excel output
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    xls_path = excel or os.path.join(out_dir, "validation_results.xlsx")
    with pd.ExcelWriter(xls_path) as xl:
        result_df.to_excel(xl, index=False, sheet_name="results")
        audit_df.to_excel(xl, index=False, sheet_name="audit")

    # Persist to DB
    eng = get_engine()
    with eng.begin() as con:
        minimal = result_df.rename(
            columns={
                "output_pincode": "chosen_pincode",
                "output_city": "chosen_city",
                "output_state": "chosen_state",
                "output_country": "chosen_country",
            }
        )[
            [
                "input_id",
                "chosen_pincode",
                "chosen_city",
                "chosen_state",
                "chosen_country",
                "overall_score",
                "t30_city_possible",
                "foreign_country_possible",
                "pincode_found",
                "source_used",
            ]
        ]
        minimal.to_sql(
            "validation_result",
            con,
            schema="output",
            if_exists="append",
            index=False,
            method="multi",
        )
        result_df.to_sql(
            "validation_result_full",
            con,
            schema="output",
            if_exists="append",
            index=False,
            method="multi",
        )
        audit_df.to_sql(
            "audit_matches",
            con,
            schema="output",
            if_exists="append",
            index=False,
            method="multi",
        )

    print(f"✅ Validation done: {len(result_df)} rows. Excel → {xls_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel, batch_size=args.batch_size)
