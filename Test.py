import os
import re
import json
from typing import List, Dict, Tuple, Set

import pandas as pd
from sqlalchemy import text
from db_config import get_engine

# ---------------------------------------------------------
#  Similarity helpers
# ---------------------------------------------------------
try:
    from rapidfuzz import fuzz

    def sim(a, b) -> float:
        """0–100 similarity (token_set)."""
        return float(fuzz.token_set_ratio(str(a or ""), str(b or "")))
except Exception:
    import difflib

    def sim(a, b) -> float:
        """0–100 similarity using difflib if rapidfuzz unavailable."""
        return float(difflib.SequenceMatcher(None, str(a or "").lower(),
                                             str(b or "").lower()).ratio() * 100.0)


THRESH = 80.0   # fuzzy threshold for accepting candidate
PIN_RE = re.compile(r'(?<!\d)(\d{6})(?!\d)')

CITY_DIRECTIONS = {
    "north", "south", "east", "west", "n", "s", "e", "w",
    "northwest", "northeast", "southwest", "southeast", "nw", "ne", "sw", "se",
    "city", "moffusil", "division", "district", "zone", "sector", "block", "phase"
}


def Title(x: str) -> str:
    return str(x or "").strip().title()


def Upper(x: str) -> str:
    return str(x or "").strip().upper()


def norm_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokens_alpha(s: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z]+", str(s or "")) if t]


def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extract_pincodes(*fields) -> List[str]:
    pins: Set[str] = set()
    for f in fields:
        if not f:
            continue
        for m in PIN_RE.findall(str(f)):
            pins.add(m)
    return sorted(pins)


def clean_output_city(city: str) -> str:
    if not city:
        return city
    parts = [p for p in tokens_alpha(city) if p.lower() not in CITY_DIRECTIONS]
    return Title(" ".join(parts)) if parts else Title(city)


def remove_terms(text: str, terms: List[str]) -> str:
    if not text:
        return text
    s = " " + text + " "
    for t in terms:
        t = str(t or "").strip()
        if not t:
            continue
        s = re.sub(rf"(?i)\b{re.escape(t)}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------
#  State Abbreviation expansion (DB + static)
# ---------------------------------------------------------
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


def expand_state_abbrev(raw_state: str, states_df: pd.DataFrame) -> str:
    """Return canonical state from abbreviation / variants."""
    if not raw_state:
        return raw_state
    s_raw = str(raw_state).strip()
    s_norm = re.sub(r"[’'`“”]", "", s_raw)
    s_title = Title(s_norm)
    tok = _norm_token(s_norm)

    if states_df is None or states_df.empty:
        # Only static mapping
        for canon, aliases in STATE_ALIAS_STATIC.items():
            all_toks = {_norm_token(a) for a in aliases} | {_initials_of(canon), _norm_token(canon)}
            if tok in all_toks:
                return canon
        return s_title

    df = states_df.copy()
    df["state"] = df["state"].astype(str).map(Title)
    if "abbreviation" not in df.columns:
        df["abbreviation"] = ""
    df["abbreviation"] = df["abbreviation"].astype(str).fillna("")

    alias: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        canon = Title(row["state"])
        alias.setdefault(canon, set())
        for t in re.split(r"[^A-Za-z]+", str(row["abbreviation"])):
            if t:
                alias[canon].add(_norm_token(t))
        alias[canon].add(_initials_of(canon))
        alias[canon].add(_norm_token(canon))
        if canon in STATE_ALIAS_STATIC:
            alias[canon].update({_norm_token(x) for x in STATE_ALIAS_STATIC[canon]})

    for canon, toks in alias.items():
        if tok in toks:
            return canon

    if s_title in alias:
        return s_title

    # fuzzy to canonical name as last resort
    best_c, best_s = None, -1.0
    for canon in alias.keys():
        score = sim(canon, s_title)
        if score > best_s:
            best_c, best_s = canon, score
    return best_c if best_s >= THRESH else s_title


# ---------------------------------------------------------
#  Candidate discovery helpers (city/state/country)
# ---------------------------------------------------------
def discover_candidates(
    grams: Set[str],
    master_list: List[str],
    extra: List[str] = None,
    thr: float = THRESH,
) -> Dict[str, float]:
    """
    Return {canonical_value: best_similarity} from grams vs master_list.
    Capped by threshold.
    """
    cand_scores: Dict[str, float] = {}
    master_set = [Title(x) for x in master_list]
    grams_all: Set[str] = set(Title(g) for g in grams if g) | {Title(x) for x in (extra or []) if x}
    for g in grams_all:
        g_norm = g.strip()
        if not g_norm:
            continue
        for m in master_set:
            if not m:
                continue
            # quick first-letter filter to avoid crazy loops
            if m[0].lower() != g_norm[0].lower():
                continue
            score = sim(g_norm, m)
            if score >= thr:
                if score > cand_scores.get(m, 0.0):
                    cand_scores[m] = score
    return cand_scores


def build_grams_for_dimension(concat_addr: str, mode: str) -> Set[str]:
    """
    mode in {"city", "state", "country"}.
    - city/state: 1-gram + 2-gram
    - country: 1-gram + 2-gram + 3-gram + 4-gram
    """
    tks = tokens_alpha(concat_addr)
    grams: Set[str] = set()
    # 1-gram
    grams.update(tks)
    # 2-gram
    grams.update(ngrams(tks, 2))
    if mode == "country":
        grams.update(ngrams(tks, 3))
        grams.update(ngrams(tks, 4))
    return set(Title(g) for g in grams if g)


# ---------------------------------------------------------
#  Scoring
# ---------------------------------------------------------
def score_field(
    input_value: str,
    chosen_value: str,
    cand_scores: Dict[str, float],
    pin_record: Dict[str, str],
) -> Tuple[float, List[str]]:
    """
    Return (score, reasons_for_penalty) for one field (city/state/country).
    Rules:
      a) similarity of input vs chosen (if input present)
      b) consistency with pincode record (if present)
      c) ambiguity penalty if many candidates with similar scores
    """
    reasons = []
    # Rule (a): direct fuzzy
    rule_a = 0.0
    if chosen_value:
        if input_value:
            rule_a = sim(input_value, chosen_value)
        else:
            # no input, but we discovered from address
            rule_a = 70.0

    # Rule (b): pin consistency
    rule_b = 0.0
    if chosen_value and pin_record:
        key = "city" if "city" in pin_record and chosen_value == pin_record["city"] else None
        # we don't know which dimension we are scoring here, so check generically
        for dim in ("city", "state", "country"):
            if dim in pin_record and pin_record[dim] and chosen_value:
                if sim(chosen_value, pin_record[dim]) >= 95.0:
                    rule_b = 100.0
                    break
        if rule_b == 0.0:
            # partial match
            for dim in ("city", "state", "country"):
                if dim in pin_record and pin_record[dim] and chosen_value:
                    if sim(chosen_value, pin_record[dim]) >= THRESH:
                        rule_b = 70.0
                        break

    # Rule (c): ambiguity
    rule_c = 100.0
    if len(cand_scores) == 0:
        rule_c = 0.0
        reasons.append("No candidate derived for field")
    elif len(cand_scores) == 1:
        rule_c = 100.0
    else:
        # multiple candidates – if second-best close to best, penalize
        sorted_scores = sorted(cand_scores.values(), reverse=True)
        best = sorted_scores[0]
        if len(sorted_scores) > 1:
            second = sorted_scores[1]
            if best - second <= 5.0:
                rule_c = 40.0
                reasons.append("Multiple candidates with similar scores")
            else:
                rule_c = 80.0

    # Final weighted score for field
    score = 0.5 * rule_a + 0.3 * rule_b + 0.2 * rule_c
    return score, reasons


# ---------------------------------------------------------
#  Main validator
# ---------------------------------------------------------
def main(limit: int = 1000, excel: str = None, batch_size: int = 200):
    eng = get_engine()
    with eng.begin() as con:
        postal = pd.read_sql("SELECT city,state,pincode,country FROM ref.postal_pincode", con)
        rta = pd.read_sql("SELECT city,state,pincode,country FROM ref.rta_pincode", con)
        try:
            states_df = pd.read_sql("SELECT state,abbreviation FROM ref.indian_state_abbrev", con)
        except Exception:
            states_df = pd.DataFrame({"state": [], "abbreviation": []})
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

    # Normalise refs
    for df, cols in [(postal, ["city", "state", "country"]),
                     (rta, ["city", "state", "country"]),
                     (world, ["city", "country"])]:
        for c in cols:
            df[c] = df[c].astype(str).map(Title)

    # Master sets
    city_master = sorted(set(postal["city"]) | set(rta["city"]) | set(world["city"]))
    state_master = sorted(set(postal["state"]) | set(rta["state"]))
    country_master = sorted(set(postal["country"]) | set(rta["country"]) | set(world["country"]))

    # Pincode index: exact only, no fuzzy here
    pin_index: Dict[str, List[Dict[str, str]]] = {}
    for df in (postal, rta):
        for _, rr in df.iterrows():
            pin = str(rr["pincode"])
            pin_index.setdefault(pin, []).append(
                {
                    "pincode": pin,
                    "city": rr["city"],
                    "state": rr["state"],
                    "country": rr["country"],
                }
            )

    results = []
    audits = []

    t30_set = {Title(x) for x in t30}

    # --------- Process in batches ----------
    for start in range(0, len(inputs), batch_size):
        chunk = inputs.iloc[start:start + batch_size].copy()

        for _, row in chunk.iterrows():
            input_id = int(row["id"])
            in_a1 = row.get("address1")
            in_a2 = row.get("address2")
            in_a3 = row.get("address3")
            in_city_raw = row.get("city")
            in_state_raw = row.get("state")
            in_country_raw = row.get("country")
            in_pin_raw = row.get("pincode")

            in_city = Title(in_city_raw)
            in_state = expand_state_abbrev(Title(in_state_raw), states_df)
            in_country = Title(in_country_raw)
            in_pin = str(in_pin_raw or "").strip()

            concat_addr = norm_text(" ".join([
                str(in_a1 or ""), str(in_a2 or ""), str(in_a3 or ""),
                in_city, in_state, in_country, in_pin
            ]))

            # ----- PINCODES from text (exact only) -----
            pins_from_text = set(extract_pincodes(concat_addr, in_pin))
            # input pin is highest priority — keep it first if present
            if in_pin and re.fullmatch(r"\d{6}", in_pin):
                pins_from_text.add(in_pin)

            # ----- Candidate grams for each dimension -----
            grams_city = build_grams_for_dimension(concat_addr, "city")
            grams_state = build_grams_for_dimension(concat_addr, "state")
            grams_country = build_grams_for_dimension(concat_addr, "country")

            city_candidates = discover_candidates(
                grams_city, city_master, extra=[in_city], thr=THRESH
            )
            state_candidates = discover_candidates(
                grams_state, state_master, extra=[in_state], thr=THRESH
            )
            country_candidates = discover_candidates(
                grams_country, country_master, extra=[in_country], thr=THRESH
            )

            # Choose best city/state/country
            def pick_best(cand_scores: Dict[str, float]) -> str:
                if not cand_scores:
                    return None
                return max(cand_scores.items(), key=lambda kv: kv[1])[0]

            out_city = pick_best(city_candidates) or (in_city if in_city else None)
            out_state = pick_best(state_candidates) or (in_state if in_state else None)
            out_country = pick_best(country_candidates) or (in_country if in_country else None)

            out_city_clean = clean_output_city(out_city)

            # ----- PINCODES from DB based on discovered city -----
            pins_from_db: Set[str] = set()
            pin_record_for_scoring: Dict[str, str] = {}

            # First, if we have a pincode in text and it exists in DB,
            # use the first such as primary.
            out_pin = None
            for pin in sorted(pins_from_text):
                if pin in pin_index:
                    out_pin = pin
                    # pick first record for scoring (there can be many)
                    pin_record_for_scoring = pin_index[pin][0].copy()
                    break

            # If we still don't have any valid pin, try to derive from city/state/country
            if out_pin is None and out_city_clean:
                # exact city + (optional) state/country filters
                candidate_rows = []
                for df in (postal, rta):
                    sub = df[df["city"] == out_city]
                    if out_state:
                        sub = sub[sub["state"] == out_state]
                    if out_country:
                        sub = sub[sub["country"] == out_country]
                    candidate_rows.extend([
                        {
                            "pincode": str(r["pincode"]),
                            "city": r["city"],
                            "state": r["state"],
                            "country": r["country"],
                        } for _, r in sub.iterrows()
                    ])

                # fallback: relax filters stepwise if nothing found
                if not candidate_rows:
                    for df in (postal, rta):
                        sub = df[df["city"] == out_city]
                        candidate_rows.extend([
                            {
                                "pincode": str(r["pincode"]),
                                "city": r["city"],
                                "state": r["state"],
                                "country": r["country"],
                            } for _, r in sub.iterrows()
                        ])

                for rec in candidate_rows:
                    pins_from_db.add(rec["pincode"])

                if candidate_rows:
                    # choose pin whose state/country matches our chosen ones best
                    best_rec = None
                    best_score = -1.0
                    for rec in candidate_rows:
                        score = 0.0
                        if out_state:
                            score += sim(out_state, rec["state"])
                        if out_country:
                            score += sim(out_country, rec["country"])
                        if score > best_score:
                            best_score = score
                            best_rec = rec
                    if best_rec:
                        out_pin = best_rec["pincode"]
                        pin_record_for_scoring = best_rec.copy()

            # If still nothing: no pin_record_for_scoring
            # Possible pincodes = from text + from db
            all_possible_pins_text = sorted(pins_from_text)
            all_possible_pins_db = sorted(pins_from_db)
            all_possible_pins_union = sorted(set(all_possible_pins_text) | set(all_possible_pins_db))

            # record some audit info
            audits.append({
                "input_id": input_id,
                "match_type": "summary",
                "candidate": f"pins_text={list(all_possible_pins_text)}, pins_db={list(all_possible_pins_db)}",
                "score": 0,
                "source": "validator"
            })

            # ----- Scores for city/state/country -----
            city_score, city_reasons = score_field(
                in_city, out_city_clean, city_candidates, pin_record_for_scoring
            )
            state_score, state_reasons = score_field(
                in_state, out_state, state_candidates, pin_record_for_scoring
            )
            country_score, country_reasons = score_field(
                in_country, out_country, country_candidates, pin_record_for_scoring
            )

            # ----- Overall score = average of three -----
            overall_score = (city_score + state_score + country_score) / 3.0

            # ----- Reasons & ambiguity flag -----
            reasons: List[str] = []
            reasons.extend(city_reasons)
            reasons.extend(state_reasons)
            reasons.extend(country_reasons)

            if out_pin is None and all_possible_pins_union:
                reasons.append("Pincode not uniquely identified from candidates")
            if not all_possible_pins_union:
                reasons.append("No pincode discovered")

            ambiguous_flag = 1 if overall_score < 60.0 or "Multiple candidates" in " ".join(reasons) else 0

            if overall_score < 60.0 and not reasons:
                reasons.append("Overall confidence below threshold")

            reason_str = "; ".join(sorted(set(r for r in reasons if r)))

            # ----- Flags -----
            t30_city_possible = 1 if (
                out_city_clean and Title(out_city_clean) in t30_set
            ) or any(Title(clean_output_city(c)) in t30_set for c in city_candidates.keys()) else 0

            foreign_country_possible = 1 if any(
                Title(c) != "India" for c in [out_country] + list(country_candidates.keys())
                if c
            ) else 0

            pincode_found_flag = 1 if out_pin else 0

            # ----- All possible lists (with NaNs removed) -----
            all_countries_list = [Title(c) for c in country_candidates.keys() if c]
            if out_country and out_country not in all_countries_list:
                all_countries_list.append(Title(out_country))

            all_states_list = [Title(s) for s in state_candidates.keys() if s]
            if out_state and out_state not in all_states_list:
                all_states_list.append(Title(out_state))

            all_cities_list = [Title(c) for c in city_candidates.keys() if c]
            if out_city_clean and out_city_clean not in all_cities_list:
                all_cities_list.append(Title(out_city_clean))
            if in_city and in_city not in all_cities_list:
                all_cities_list.append(Title(in_city))

            all_countries_list = sorted(set([c for c in all_countries_list if c]))
            all_states_list = sorted(set([s for s in all_states_list if s]))
            all_cities_list = sorted(set([c for c in all_cities_list if c]))

            # ----- Local address (remove all detected entities & all pins) -----
            remove_list = []
            remove_list.extend(all_cities_list)
            remove_list.extend(all_states_list)
            remove_list.extend(all_countries_list)
            remove_list.extend(all_possible_pins_union)
            local_address = remove_terms(concat_addr, remove_list)

            # ----- Build result row (keep old columns; update scores/flags) -----
            results.append({
                "input_id": input_id,
                "address1": in_a1,
                "address2": in_a2,
                "address3": in_a3,
                "input_city": in_city,
                "input_state_raw": Title(in_state_raw),
                "input_state": in_state,
                "input_country": in_country,
                "input_pincode": in_pin,
                "concatenated_address": concat_addr,

                "output_pincode": out_pin,
                "output_city": out_city_clean,
                "output_state": out_state,
                "output_country": out_country,

                # NEW numeric scores (0–100)
                "city_score": round(city_score, 2),
                "state_score": round(state_score, 2),
                "country_score": round(country_score, 2),
                "overall_score": round(overall_score, 2),

                # Flags
                "t30_city_possible": t30_city_possible,
                "foreign_country_possible": foreign_country_possible,
                "pincode_found": pincode_found_flag,
                "ambiguous_address_flag": ambiguous_flag,

                # Reasons text
                "reason": reason_str,

                # All possible entities (only those actually discovered)
                "all_possible_countries": json.dumps(all_countries_list, ensure_ascii=False),
                "all_possible_states": json.dumps(all_states_list, ensure_ascii=False),
                "all_possible_cities": json.dumps(all_cities_list, ensure_ascii=False),
                "all_possible_pincodes_text": json.dumps(all_possible_pins_text, ensure_ascii=False),
                "all_possible_pincodes_db": json.dumps(all_possible_pins_db, ensure_ascii=False),
                "all_possible_pincodes": json.dumps(all_possible_pins_union, ensure_ascii=False),

                # Remaining local address
                "local_address": local_address,
            })

    result_df = pd.DataFrame(results)
    audit_df = pd.DataFrame(audits) if audits else pd.DataFrame(
        columns=["input_id", "match_type", "candidate", "score", "source"]
    )

    # -------- Excel output --------
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    xls_path = excel or os.path.join(out_dir, "validation_results.xlsx")
    with pd.ExcelWriter(xls_path) as xl:
        result_df.to_excel(xl, index=False, sheet_name="results")
        audit_df.to_excel(xl, index=False, sheet_name="audit")

    # -------- DB output (minimal + full + audit) --------
    eng = get_engine()
    with eng.begin() as con:
        minimal = result_df.rename(columns={
            "output_pincode": "chosen_pincode",
            "output_city": "chosen_city",
            "output_state": "chosen_state",
            "output_country": "chosen_country",
        })[[
            "input_id", "chosen_pincode", "chosen_city", "chosen_state",
            "chosen_country", "t30_city_possible", "foreign_country_possible",
            "pincode_found", "ambiguous_address_flag", "overall_score"
        ]]
        minimal.to_sql("validation_result", con, schema="output",
                       if_exists="append", index=False, method="multi")

        result_df.to_sql("validation_result_full", con, schema="output",
                         if_exists="append", index=False, method="multi")

        audit_df.to_sql("audit_matches", con, schema="output",
                        if_exists="append", index=False, method="multi")

    print(f"✅ Validation done: {len(result_df)} rows. Excel → {xls_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel, batch_size=args.batch_size)
