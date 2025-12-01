import os
import re
import json
import pandas as pd
import datetime
from .db_config import get_db_connection

"""
Address validator / normalizer.

This module exposes two call paths:

1) CLI batch mode (used for offline validation)
   - main(limit, excel, batch_size)

2) In-process API mode (used by FastAPI service)
   - validate_single_address_df(df)

Both paths share the same core logic implemented in _run_validation().
"""

# ---------------- Similarity helpers ----------------

try:
    from rapidfuzz import fuzz

    def sim(a, b):
        """0..1 similarity using RapidFuzz."""
        return fuzz.token_set_ratio(str(a or ""), str(b or "")) / 100.0

except ImportError:
    from difflib import SequenceMatcher

    def sim(a, b):
        """0..1 similarity fallback."""
        return SequenceMatcher(None, str(a or ""), str(b or "")).ratio()


def is_null_token(x):
    """Detect NaN / None / common null strings."""
    if x is None:
        return True
    s = str(x).strip().lower()
    if not s:
        return True
    return s in {"na", "n/a", "null", "none", "-"}


def Title(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return s
    return " ".join(w.capitalize() for w in s.split())


def norm_text(s):
    """Normalize free text: lowercase, collapse spaces."""
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def tokens_alpha(s):
    """Split text into alphabetic tokens only."""
    return re.findall(r"[A-Za-z]+", str(s or ""))


def tokens_alnum(s):
    """Split text into alphanumeric tokens."""
    return re.findall(r"[A-Za-z0-9]+", str(s or ""))


# ---------------- City cleanup helpers ----------------


def clean_output_city(city):
    """Heuristic city normalization (remove trailing 'Dist', etc.)."""
    if not city:
        return city
    w = city.strip()
    w = re.sub(r"\b(dist(rict)?|district)\b\.?", "", w, flags=re.IGNORECASE)
    w = re.sub(r"\s+", " ", w).strip()
    return Title(w)


# ---------------- Static country / state config ----------------

INDIA_SYNONYMS = {
    "IN",
    "INDIA",
    "BHARAT",
    "HINDUSTAN",
}

COUNTRY_ALIAS_STATIC = {
    "India": {"INDIA", "BHARAT", "HINDUSTAN", "IN"},
}

STATE_ALIAS_STATIC = {
    # Add more common abbreviations or aliases here as needed
    "Tamil Nadu": {"TN", "Tam Nadu"},
    "Uttar Pradesh": {"UP"},
    "Madhya Pradesh": {"MP"},
}


def _norm_token(s):
    return re.sub(r"[^A-Za-z]+", "", str(s or "").strip()).lower()


# ---------------- N-gram / candidate finders ----------------


def generate_ngrams(words, max_n=3):
    ngrams = []
    L = len(words)
    for n in range(1, max_n + 1):
        for i in range(L - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
    return ngrams


def find_candidates_from_ngrams(words, master, max_n=3, thresh=80, extra_seed=None):
    """
    From tokens 'words' and 'master' (set of strings), build candidate set
    using n-grams plus an extra seed (the raw input string).
    """
    ngrams = generate_ngrams(words, max_n=max_n)
    candidates = set()
    master = {Title(x) for x in master if not is_null_token(x)}

    for ng in ngrams:
        for m in master:
            if sim(ng, m) * 100 >= thresh:
                candidates.add(m)

    if extra_seed and not is_null_token(extra_seed):
        es = Title(extra_seed)
        for m in master:
            if sim(es, m) * 100 >= thresh:
                candidates.add(m)

    return candidates


# ---------------- State alias expansion ----------------


def build_state_alias(df_states):
    """
    Build alias dict for states from DB + static alias table.
    df_states is a DataFrame with columns: 'state', 'abbreviation' (optional).
    """
    alias = {}
    for _, row in df_states.iterrows():
        canon = Title(row["state"])
        if is_null_token(canon):
            continue
        alias.setdefault(canon, set())

        abbr = str(row.get("abbreviation", "") or "")
        for t in re.split(r"[^A-Za-z]+", abbr):
            if not t:
                continue
            alias[canon].add(_norm_token(t))

        alias[canon].add(_norm_token(canon))

    # Merge static aliases
    for canon, toks in STATE_ALIAS_STATIC.items():
        alias.setdefault(canon, set())
        for t in toks:
            alias[canon].add(_norm_token(t))

    return alias


def expand_state_abbrev(raw, alias_dict):
    """
    Expand a raw state token (e.g. 'TN') into canonical form ('Tamil Nadu')
    using alias_dict.
    """
    if not raw or is_null_token(raw):
        return None
    norm = _norm_token(raw)
    best = None
    best_score = -1.0
    for canon, toks in alias_dict.items():
        if norm in toks:
            # Exact token match
            return canon
        for t in toks:
            s = sim(norm, t)
            if s > best_score:
                best, best_score = canon, s
    return best or Title(raw)


# ---------------- Scoring helpers ----------------


def score_value_match(result, raw_input, best_ngram):
    """
    Score 0–100 for how well the chosen canonical result
    matches the input and the best n-gram from the address.
    """
    if not result and not raw_input and not best_ngram:
        return 0.0

    scores = []
    if result and raw_input:
        scores.append(sim(result, raw_input) * 100)
    if result and best_ngram:
        scores.append(sim(result, best_ngram) * 100)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


def score_consistency_with_pin(result, from_pin):
    if not result or not from_pin:
        return 100.0  # no evidence of inconsistency
    return round(sim(result, from_pin) * 100, 2)


def score_ambiguity(candidates):
    """
    Simple ambiguity penalty: more distinct candidates → lower score.
    1 candidate => 100, 2 => 80, 3 => 60, 4+ => 40.
    """
    n = len(candidates or [])
    if n <= 1:
        return 100.0
    if n == 2:
        return 80.0
    if n == 3:
        return 60.0
    return 40.0


# Thresholds
CITY_STATE_THRESH = 0.75
COUNTRY_THRESH = 0.75
FOREIGN_COUNTRY_STRICT_THRESH = 0.85


def detect_foreign_country_from_ngrams_strict(country_words, country_master_full):
    """
    Stricter version: looks for high-confidence match to foreign countries.
    """
    all_country_ngrams = generate_ngrams(country_words, max_n=4)
    foreign_master = {c for c in country_master_full if Title(c) != "India"}
    for ng in all_country_ngrams:
        best_ngram_sim = max(
            (sim(m, ng) for m in foreign_master), default=0.0
        )
        if best_ngram_sim >= FOREIGN_COUNTRY_STRICT_THRESH:
            return 1  # Found a high-confidence foreign match
    return 0


# ---------------- Core validation runner ----------------


def _run_validation(
    inputs: pd.DataFrame,
    postal: pd.DataFrame,
    rta: pd.DataFrame,
    world: pd.DataFrame,
    states_ref: pd.DataFrame,
    t30_list,
    batch_size: int = 200,
):
    """
    Core validator. Takes an input DataFrame (with columns address1,address2,address3,
    city,state,pincode,country, id) plus reference tables and returns:

      result_df, audit_df
    """
    # Normalise reference text
    for df, cols in [
        (postal, ["city", "state", "country"]),
        (rta, ["city", "state", "country"]),
        (world, ["city", "country"]),
    ]:
        for c in cols:
            df[c] = df[c].astype(str).map(Title)

    # Create master lists for both India-only and world-inclusive searches
    country_master_full = (
        set(postal["country"]).union(set(world["country"])).union({"India"})
    )

    city_master_india = set(postal["city"]).union(set(rta["city"]))
    city_master_world = city_master_india.union(set(world["city"]))

    # Build pin index for quick look-up (postal + rta)
    pin_index = {}
    for src, df in [("postal", postal), ("rta", rta)]:
        for _, row in df.iterrows():
            pin_index.setdefault(str(row["pincode"]), []).append((src, row))

    # Build state alias (DB + static)
    state_alias = build_state_alias(states_ref)

    # Build map city/state -> countries from reference for enrichment
    state_to_countries = {}
    for df in (postal, rta, world):
        if "state" not in df.columns:
            continue
        for _, row in df[["state", "country"]].drop_duplicates().iterrows():
            st = Title(row["state"])
            co = Title(row["country"])
            if is_null_token(st) or is_null_token(co):
                continue
            state_to_countries.setdefault(st, set()).add(co)

    city_to_countries = {}
    for df in (postal, rta, world):
        for _, row in df[["city", "country"]].drop_duplicates().iterrows():
            ci = Title(row["city"])
            co = Title(row["country"])
            if is_null_token(ci) or is_null_token(co):
                continue
            city_to_countries.setdefault(ci, set()).add(co)

    results = []
    audits = []
    t30_set = {Title(x) for x in t30_list}

    # Ensure id column exists
    if "id" not in inputs.columns:
        inputs = inputs.copy()
        inputs["id"] = range(1, len(inputs) + 1)

    for start in range(0, len(inputs), batch_size):
        chunk = inputs.iloc[start : start + batch_size].copy()

        for _, rec in chunk.iterrows():
            rid = int(rec["id"])

            a1 = rec.get("address1")
            a2 = rec.get("address2")
            a3 = rec.get("address3")

            in_city_raw = rec.get("city")
            in_city = Title(in_city_raw)
            in_state_raw = rec.get("state")
            in_state_title = Title(in_state_raw)
            in_state_expanded = (
                expand_state_abbrev(in_state_title, state_alias)
                if in_state_raw and not is_null_token(in_state_raw)
                else None
            )
            in_country_raw = rec.get("country")
            in_country = Title(in_country_raw)
            in_pin_raw = rec.get("pincode")
            in_pin = str(in_pin_raw or "").strip()
            invalid_pincode_format = bool(in_pin) and not re.fullmatch(r"\d{6}", in_pin)

            # 1) Whole address (concatenated)
            whole = norm_text(
                " ".join(
                    [
                        str(x or "")
                        for x in [
                            a1,
                            a2,
                            a3,
                            in_city,
                            in_state_expanded,
                            in_country,
                            in_pin,
                        ]
                    ]
                )
            )
            words = tokens_alpha(whole)

            # 2) N-grams: city/state 1–2; country 1–4
            city_words = [w for w in words if len(w) > 1]
            state_words = city_words[:]  # same tokens used for state
            country_words = city_words[:]

            # Detect possible foreign country strictly from country words
            foreign_country_possible = detect_foreign_country_from_ngrams_strict(
                country_words, country_master_full
            )

            # --- INDIA-FIRST LOGIC ---
            country_cands_full = find_candidates_from_ngrams(
                country_words,
                country_master_full,
                max_n=4,
                thresh=COUNTRY_THRESH,
                extra_seed=in_country,
            )

            has_foreign_country = any(c != "India" for c in country_cands_full)
            input_country_is_india = (
                in_country
                and not is_null_token(in_country)
                and Title(in_country) == "India"
            )

            # City master selection:
            if has_foreign_country and not input_country_is_india:
                city_master = city_master_world
            else:
                city_master = city_master_india

            country_master = country_master_full

            city_cands = find_candidates_from_ngrams(
                city_words,
                city_master,
                max_n=2,
                thresh=CITY_STATE_THRESH,
                extra_seed=in_city,
            )

            state_cands_raw = find_candidates_from_ngrams(
                state_words,
                set(states_ref["state"]),
                max_n=2,
                thresh=CITY_STATE_THRESH,
                extra_seed=in_state_expanded,
            )
            state_cands = set()
            for s in state_cands_raw:
                expanded = expand_state_abbrev(s, state_alias)
                if expanded and not is_null_token(expanded):
                    state_cands.add(expanded)

            country_cands = find_candidates_from_ngrams(
                country_words,
                country_master,
                max_n=4,
                thresh=COUNTRY_THRESH,
                extra_seed=in_country,
            )

            # 3) Pincode logic
            input_pin_not_in_master = False
            chosen_pin = None
            chosen_pin_row = None
            all_possible_pincodes_set = set()

            # Extract pincodes directly from free text
            pins_text = set(re.findall(r"\b\d{6}\b", whole))

            def _pins_from_city_state():
                """
                Use city/state candidates to retrieve all related pincodes from DB.
                """
                pins = set()
                for df_src, dfp in [("postal", postal), ("rta", rta)]:
                    sub = dfp[
                        dfp["city"].isin(city_cands)
                        | dfp["state"].isin(state_cands)
                        | dfp["country"].isin(country_cands)
                    ]
                    for _, row_pin in sub.iterrows():
                        pins.add(str(row_pin["pincode"]))
                        audits.append(
                            {
                                "input_id": rid,
                                "type": "city→pin",
                                "pincode": row_pin["pincode"],
                                "city": row_pin["city"],
                                "state": row_pin["state"],
                                "country": row_pin["country"],
                                "src": df_src,
                            }
                        )
                return pins

            input_pin_clean = (
                in_pin if (in_pin and re.fullmatch(r"\d{6}", in_pin)) else None
            )

            if input_pin_clean:
                rows = pin_index.get(input_pin_clean, [])
                if rows:
                    chosen_pin = input_pin_clean
                    postal_rows = [r for (src, r) in rows if src == "postal"]
                    if postal_rows:
                        chosen_pin_row = postal_rows[0]
                    else:
                        chosen_pin_row = rows[0][1]
                    all_possible_pincodes_set = set(pins_text) or {input_pin_clean}
                else:
                    input_pin_not_in_master = True
                    pins_from_city = _pins_from_city_state()
                    all_possible_pincodes_set = pins_from_city

            else:
                if pins_text:
                    if len(pins_text) == 1:
                        only_pin = next(iter(pins_text))
                        chosen_pin = only_pin
                        rows = pin_index.get(only_pin, [])
                        if rows:
                            postal_rows = [r for (src, r) in rows if src == "postal"]
                            if postal_rows:
                                chosen_pin_row = postal_rows[0]
                            else:
                                chosen_pin_row = rows[0][1]
                    else:
                        best = None
                        best_score = -1.0
                        for p in pins_text:
                            rows = pin_index.get(p, [])
                            if not rows:
                                if best is None:
                                    best = (p, None, 0.0)
                                continue
                            for src, row_pin in rows:
                                s_city = max(
                                    (sim(row_pin["city"], c) for c in city_cands),
                                    default=0.0,
                                )
                                s_state = max(
                                    (sim(row_pin["state"], s) for s in state_cands),
                                    default=0.0,
                                )
                                s_ctry = max(
                                    (sim(row_pin["country"], k) for k in country_cands),
                                    default=0.0,
                                )
                                score = 0.5 * s_city + 0.3 * s_state + 0.2 * s_ctry
                                if score > best_score:
                                    best = (p, row_pin, score)
                                    best_score = score
                        if best:
                            chosen_pin, chosen_pin_row, _ = best

                    all_possible_pincodes_set = set(pins_text)

                else:
                    pins_from_city = _pins_from_city_state()

                    if invalid_pincode_format:
                        # User provided a pincode, but its format is invalid (not 6 digits).
                        # Do NOT auto-select a single pincode – only suggest candidates.
                        all_possible_pincodes_set = set(pins_from_city)
                    else:
                        if pins_from_city:
                            best = None
                            best_score = -1.0
                            for p in pins_from_city:
                                for src, row_pin in pin_index.get(p, []):
                                    s_state = max(
                                        (
                                            sim(row_pin["state"], s)
                                            for s in state_cands
                                        ),
                                        default=0.0,
                                    )
                                    s_ctry = max(
                                        (
                                            sim(row_pin["country"], k)
                                            for k in country_cands
                                        ),
                                        default=0.0,
                                    )
                                    score = 0.6 * s_state + 0.4 * s_ctry
                                    if score > best_score:
                                        best = (p, row_pin, score)
                                        best_score = score
                            if best:
                                chosen_pin, chosen_pin_row, _ = best

                        all_possible_pincodes_set = set(pins_from_city)
                        if chosen_pin:
                            all_possible_pincodes_set.add(chosen_pin)

            # 4) Choose city/state/country
            def choose_best_entity(candidates, input_value, pin_value):
                candidates = {Title(x) for x in candidates if not is_null_token(x)}

                if pin_value and not is_null_token(pin_value):
                    return Title(pin_value)

                if not candidates and input_value and not is_null_token(input_value):
                    return Title(input_value)

                if candidates:
                    best_c = None
                    best_s = -1.0
                    inp = (
                        Title(input_value)
                        if (input_value and not is_null_token(input_value))
                        else None
                    )
                    for c in candidates:
                        s = sim(c, inp) if inp else 1.0
                        if s > best_s:
                            best_c, best_s = c, s
                    return best_c

                return (
                    Title(input_value)
                    if (input_value and not is_null_token(input_value))
                    else None
                )

            city_from_pin = chosen_pin_row["city"] if chosen_pin_row is not None else None
            state_from_pin = (
                chosen_pin_row["state"] if chosen_pin_row is not None else None
            )
            country_from_pin = (
                chosen_pin_row["country"] if chosen_pin_row is not None else None
            )

            chosen_city = choose_best_entity(city_cands, in_city, city_from_pin)
            chosen_state = choose_best_entity(
                state_cands, in_state_expanded, state_from_pin
            )
            chosen_country = choose_best_entity(
                country_cands, in_country, country_from_pin
            )

            chosen_city_clean = clean_output_city(chosen_city)

            # 5) Build all_possible_* (no NaN, and enrich from PIN + state/city→country)
            all_possible_cities = sorted(
                {
                    Title(clean_output_city(c))
                    for c in city_cands
                    if not is_null_token(c)
                }
            )
            if chosen_city_clean and not is_null_token(chosen_city_clean):
                if chosen_city_clean not in all_possible_cities:
                    all_possible_cities.append(chosen_city_clean)

            # enrich from chosen pincode (postal + rta)
            if chosen_pin:
                try:
                    pin_int = int(chosen_pin)
                except ValueError:
                    pin_int = None

                if pin_int is not None:
                    for df_src in (postal, rta):
                        sub = df_src[df_src["pincode"] == pin_int]
                        for _, row_pin in sub.iterrows():
                            c_title = Title(row_pin["city"])
                            if (
                                not is_null_token(c_title)
                                and c_title not in all_possible_cities
                            ):
                                all_possible_cities.append(c_title)

            all_possible_states = sorted(
                {Title(s) for s in state_cands if not is_null_token(s)}
            )
            if chosen_state and not is_null_token(chosen_state):
                cs = Title(chosen_state)
                if cs not in all_possible_states:
                    all_possible_states.append(cs)

            # Include raw input state pieces (e.g., "Rajasthan tamilnadu")
            if in_state_title and not is_null_token(in_state_title):
                for part in re.split(r"[^A-Za-z]+", in_state_title):
                    if part:
                        ptitle = Title(part)
                        if ptitle not in all_possible_states:
                            all_possible_states.append(ptitle)

            all_possible_countries = sorted(
                {Title(k) for k in country_cands if not is_null_token(k)}
            )
            if chosen_country and not is_null_token(chosen_country):
                cc = Title(chosen_country)
                if cc not in all_possible_countries:
                    all_possible_countries.append(cc)

            # Enrich countries using state/city mapping
            for st in list(all_possible_states):
                for co in state_to_countries.get(st, []):
                    if co not in all_possible_countries:
                        all_possible_countries.append(co)
            for ci in list(all_possible_cities):
                for co in city_to_countries.get(ci, []):
                    if co not in all_possible_countries:
                        all_possible_countries.append(co)

            all_possible_pincodes = sorted(
                {p for p in all_possible_pincodes_set if not is_null_token(p)}
            )

            # 6) local_address = whole – entities we already found
            remove_list = set()
            for p in all_possible_pincodes:
                remove_list.add(p)

            if chosen_country and not is_null_token(chosen_country):
                remove_list.add(chosen_country)
            if chosen_state and not is_null_token(chosen_state):
                remove_list.add(chosen_state)
            if chosen_city_clean and not is_null_token(chosen_city_clean):
                remove_list.add(chosen_city_clean)

            local_address_tokens = tokens_alnum(whole)
            filtered_tokens = []
            for t in local_address_tokens:
                skip = False
                for r in remove_list:
                    if r and sim(t, r) > 0.8:
                        skip = True
                        break
                if not skip:
                    filtered_tokens.append(t)

            local_address = " ".join(filtered_tokens).strip()
            if not local_address:
                local_address = None

            # 7) Scores
            all_city_ngrams = generate_ngrams(city_words, max_n=2)
            all_state_ngrams = generate_ngrams(state_words, max_n=2)
            all_country_ngrams = generate_ngrams(country_words, max_n=4)

            best_city_ng = max(
                all_city_ngrams, key=lambda ng: sim(chosen_city_clean, ng), default=None
            )
            best_state_ng = max(
                all_state_ngrams, key=lambda ng: sim(chosen_state, ng), default=None
            )
            best_country_ng = max(
                all_country_ngrams, key=lambda ng: sim(chosen_country, ng), default=None
            )

            city_value = score_value_match(chosen_city_clean, in_city, best_city_ng)
            state_value = score_value_match(
                chosen_state, in_state_expanded, best_state_ng
            )
            country_value = score_value_match(
                chosen_country, in_country, best_country_ng
            )

            city_cons = score_consistency_with_pin(chosen_city_clean, city_from_pin)
            state_cons = score_consistency_with_pin(chosen_state, state_from_pin)
            country_cons = score_consistency_with_pin(chosen_country, country_from_pin)

            city_amb = score_ambiguity(all_possible_cities)
            state_amb = score_ambiguity(all_possible_states)
            country_amb = score_ambiguity(all_possible_countries)

            def bundle(v, c, a):
                return 0.5 * v + 0.4 * c + 0.1 * a

            overall = round(
                (
                    bundle(city_value, city_cons, city_amb)
                    + bundle(state_value, state_cons, state_amb)
                    + bundle(country_value, country_cons, country_amb)
                )
                / 3.0,
                2,
            )

            # Reasons
            reasons = []

            invalid_pincode = False
            if in_pin and not re.fullmatch(r"\d{6}", in_pin):
                invalid_pincode = True
                reasons.append("Invalid Pincode")
            elif input_pin_not_in_master:
                invalid_pincode = True
                reasons.append("Invalid Pincode")

            if input_pin_not_in_master or (chosen_pin and chosen_pin_row is None):
                reasons.append("pincode_not_found_in_master")

            if chosen_pin_row is not None:
                if sim(chosen_city_clean, city_from_pin) < CITY_STATE_THRESH:
                    reasons.append("mismatch_city_vs_pincode")
                if sim(chosen_state, state_from_pin) < CITY_STATE_THRESH:
                    reasons.append("mismatch_state_vs_pincode")
                if sim(chosen_country, country_from_pin) < COUNTRY_THRESH:
                    reasons.append("mismatch_country_vs_pincode")

            if city_amb < 80:
                reasons.append("ambiguous_city_candidates")
            if state_amb < 80:
                reasons.append("ambiguous_state_candidates")
            if country_amb < 80:
                reasons.append("ambiguous_country_candidates")

            if city_value < 80:
                reasons.append("low_city_value_match")
            if state_value < 80:
                reasons.append("low_state_value_match")
            if country_value < 80:
                reasons.append("low_country_value_match")

            ambiguous_flag = 1 if overall < 85 else 0

            # formatted_address only if no issues
            has_issue = bool(reasons) or ambiguous_flag == 1
            if has_issue:
                formatted_address = ""
            else:
                parts = []
                if local_address and not is_null_token(local_address):
                    parts.append(str(local_address))
                if chosen_city_clean and not is_null_token(chosen_city_clean):
                    parts.append(str(chosen_city_clean))
                if chosen_state and not is_null_token(chosen_state):
                    parts.append(str(chosen_state))
                if chosen_country and not is_null_token(chosen_country):
                    parts.append(str(chosen_country))
                if chosen_pin and not is_null_token(chosen_pin):
                    parts.append(str(chosen_pin))
                formatted_address = ",".join(parts)

            results.append(
                {
                    "input_id": rid,
                    "address1": a1,
                    "address2": a2,
                    "address3": a3,
                    "input_city": in_city,
                    "input_state_raw": in_state_title,
                    "input_state": in_state_expanded,
                    "input_country": in_country,
                    "input_pincode": in_pin,
                    "concatenated_address": whole,
                    "output_pincode": chosen_pin,
                    "output_city": chosen_city_clean,
                    "output_state": chosen_state,
                    "output_country": chosen_country,
                    "t30_city_possible": 1
                    if (chosen_city_clean and Title(chosen_city_clean) in t30_set)
                    else 0,
                    "foreign_country_possible": foreign_country_possible,
                    # IMPORTANT: pincode_found is 1 if we have ANY candidate pincode
                    "pincode_found": 1 if all_possible_pincodes else 0,
                    "ambiguous_address_flag": ambiguous_flag,
                    "all_possible_countries": json.dumps(
                        all_possible_countries, ensure_ascii=False
                    ),
                    "all_possible_states": json.dumps(
                        all_possible_states, ensure_ascii=False
                    ),
                    "all_possible_cities": json.dumps(
                        all_possible_cities, ensure_ascii=False
                    ),
                    "all_possible_pincodes": json.dumps(
                        all_possible_pincodes, ensure_ascii=False
                    ),
                    "city_value_match": city_value,
                    "city_consistency_with_pincode": city_cons,
                    "city_ambiguity_penalty": city_amb,
                    "state_value_match": state_value,
                    "state_consistency_with_pincode": state_cons,
                    "state_ambiguity_penalty": state_amb,
                    "country_value_match": country_value,
                    "country_consistency_with_pincode": country_cons,
                    "country_ambiguity_penalty": country_amb,
                    "overall_score": overall,
                    "reason": "; ".join(reasons) if reasons else "",
                    "local_address": local_address,
                    "formatted_address": formatted_address,
                }
            )

    result_df = pd.DataFrame(results)
    audit_df = (
        pd.DataFrame(audits)
        if audits
        else pd.DataFrame(
            columns=["input_id", "type", "pincode", "city", "state", "country", "src"]
        )
    )
    return result_df, audit_df


# ---------------- Public APIs ----------------


def validate_single_address_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    API entry point used by address_validator.py.

    Expects a DataFrame with columns:
        address1,address2,address3,city,state,pincode,country
    Optionally an 'id' column; if missing it will be auto-assigned.

    Returns a result DataFrame in the same shape as produced by main().
    """
    eng = get_db_connection()
    with eng.begin() as con:
        postal = pd.read_sql(
            "SELECT city,state,pincode,country FROM ref.postal_pincode", con
        )
        rta = pd.read_sql(
            "SELECT city,state,pincode,country FROM ref.rta_pincode", con
        )
        world = pd.read_sql(
            "SELECT city,country FROM ref.world_cities", con
        )
        states_ref = pd.read_sql(
            "SELECT state,abbreviation FROM ref.states_alias", con
        )
        t30 = pd.read_sql(
            "SELECT city FROM ref.top_30_cities", con
        )["city"].tolist()

    result_df, _ = _run_validation(
        inputs=df,
        postal=postal,
        rta=rta,
        world=world,
        states_ref=states_ref,
        t30_list=t30,
        batch_size=max(len(df), 1),
    )
    return result_df


def main(limit=1000, excel=None, batch_size=200):
    eng = get_db_connection()

    with eng.begin() as con:
        postal = pd.read_sql(
            "SELECT city,state,pincode,country FROM ref.postal_pincode", con
        )
        rta = pd.read_sql(
            "SELECT city,state,pincode,country FROM ref.rta_pincode", con
        )
        world = pd.read_sql(
            "SELECT city,country FROM ref.world_cities", con
        )
        states_ref = pd.read_sql(
            "SELECT state,abbreviation FROM ref.states_alias", con
        )
        t30 = pd.read_sql(
            "SELECT city FROM ref.top_30_cities", con
        )["city"].tolist()
        inputs = pd.read_sql(
            "SELECT * FROM input.addresses ORDER BY id LIMIT {}".format(int(limit)), con
        )

    result_df, audit_df = _run_validation(
        inputs=inputs,
        postal=postal,
        rta=rta,
        world=world,
        states_ref=states_ref,
        t30_list=t30,
        batch_size=batch_size,
    )

    # Excel output
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    if excel:
        xls_path = excel
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        xls_path = os.path.join(out_dir, f"validation_results_{timestamp}.xlsx")

    try:
        with pd.ExcelWriter(xls_path) as xl:
            result_df.to_excel(xl, index=False, sheet_name="results")
            audit_df.to_excel(xl, index=False, sheet_name="audit")
    except PermissionError:
        print(
            f"ERROR: Could not write to Excel file. Is '{xls_path}' open in another program?"
        )

    # Persist full result to DB
    with eng.begin() as con:
        result_df.to_sql(
            "validation_result_full",
            con,
            schema="output",
            if_exists="replace",
            index=False,
        )
        audit_df.to_sql(
            "validation_audit",
            con,
            schema="output",
            if_exists="replace",
            index=False,
        )

    return result_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel, batch_size=args.batch_size)
