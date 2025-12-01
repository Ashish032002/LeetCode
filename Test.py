import os
import re
import json
import pandas as pd
import datetime
from .db_config import get_db_connection

# ---------------- Similarity helpers ----------------

try:
    from rapidfuzz import fuzz

    def sim(a, b):
        """0..1 similarity using RapidFuzz."""
        return fuzz.token_set_ratio(str(a or ""), str(b or "")) / 100.0

except Exception:  # fallback
    import difflib

    def sim(a, b):
        """0..1 similarity using difflib."""
        return difflib.SequenceMatcher(
            None, str(a or "").lower(), str(b or "").lower()
        ).ratio()


# Global thresholds
CITY_THRESH = 0.7
STATE_THRESH = 0.7
COUNTRY_THRESH = 0.7
CITY_STATE_THRESH = 0.7


PIN_RE = re.compile(r"\b(\d{6})\b")


NULL_TOKENS = {
    None,
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    "not applicable",
    "nil",
    "0",
    0,
}


def is_null_token(x):
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip().lower()
        return s in NULL_TOKENS
    return False


def Title(s: str | None) -> str | None:
    if s is None:
        return None
    if is_null_token(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    return s.title()


def load_master_dfs(conn=None):
    """
    Load postal and RTA pincode master from DB.
    """
    if conn is None:
        conn = get_db_connection()

    postal = pd.read_sql(
        """
        SELECT
            TRIM(LOWER(COALESCE(city, ''))) AS city,
            TRIM(LOWER(COALESCE(state, ''))) AS state,
            TRIM(LOWER(COALESCE(country, ''))) AS country,
            CAST(pincode AS TEXT) AS pincode
        FROM pincode_master_postal
        """,
        conn,
    )

    rta = pd.read_sql(
        """
        SELECT
            TRIM(LOWER(COALESCE(city, ''))) AS city,
            TRIM(LOWER(COALESCE(state, ''))) AS state,
            TRIM(LOWER(COALESCE(country, ''))) AS country,
            CAST(pincode AS TEXT) AS pincode
        FROM pincode_master_rta
        """,
        conn,
    )

    # Static top 30 (or any business list) cities for t30 flag
    t30 = pd.read_sql(
        """
        SELECT DISTINCT TRIM(LOWER(city)) AS city
        FROM t30_cities
        """,
        conn,
    )

    # state alias mapping
    state_alias = pd.read_sql(
        """
        SELECT
            TRIM(LOWER(canonical_state)) AS canonical_state,
            TRIM(LOWER(alias_state))      AS alias_state
        FROM state_alias_master
        """,
        conn,
    )

    country_alias = pd.read_sql(
        """
        SELECT
            TRIM(LOWER(canonical_country)) AS canonical_country,
            TRIM(LOWER(alias_country))     AS alias_country
        FROM country_alias_master
        """,
        conn,
    )

    city_alias = pd.read_sql(
        """
        SELECT
            TRIM(LOWER(canonical_city)) AS canonical_city,
            TRIM(LOWER(alias_city))     AS alias_city
        FROM city_alias_master
        """,
        conn,
    )

    return postal, rta, t30, state_alias, country_alias, city_alias


def build_indexes(postal: pd.DataFrame, rta: pd.DataFrame):
    """
    Convenience indexes and lookups on DB masters.
    """
    pin_index: dict[str, list[tuple[str, dict]]] = {}

    def add_rows(df, src):
        for _, row in df.iterrows():
            p = str(row["pincode"])
            ent = {
                "pincode": p,
                "city": str(row["city"]),
                "state": str(row["state"]),
                "country": str(row["country"]),
                "src": src,
            }
            pin_index.setdefault(p, []).append((src, ent))

    add_rows(postal, "postal")
    add_rows(rta, "rta")

    # City/State → Country map
    city_to_countries: dict[str, list[str]] = {}
    state_to_countries: dict[str, list[str]] = {}

    for _, row in postal.iterrows():
        c = str(row["city"])
        s = str(row["state"])
        co = str(row["country"])
        city_to_countries.setdefault(c, [])
        if co not in city_to_countries[c]:
            city_to_countries[c].append(co)
        state_to_countries.setdefault(s, [])
        if co not in state_to_countries[s]:
            state_to_countries[s].append(co)

    for _, row in rta.iterrows():
        c = str(row["city"])
        s = str(row["state"])
        co = str(row["country"])
        city_to_countries.setdefault(c, [])
        if co not in city_to_countries[c]:
            city_to_countries[c].append(co)
        state_to_countries.setdefault(s, [])
        if co not in state_to_countries[s]:
            state_to_countries[s].append(co)

    return pin_index, city_to_countries, state_to_countries


def build_alias_maps(state_alias, country_alias, city_alias):
    """
    Build alias → canonical maps for state, country, city.
    """
    STATE_ALIAS_STATIC: dict[str, list[str]] = {}
    for _, row in state_alias.iterrows():
        canon = row["canonical_state"]
        alias = row["alias_state"]
        STATE_ALIAS_STATIC.setdefault(canon, [])
        if alias not in STATE_ALIAS_STATIC[canon]:
            STATE_ALIAS_STATIC[canon].append(alias)

    COUNTRY_ALIAS_STATIC: dict[str, list[str]] = {}
    for _, row in country_alias.iterrows():
        canon = row["canonical_country"]
        alias = row["alias_country"]
        COUNTRY_ALIAS_STATIC.setdefault(canon, [])
        if alias not in COUNTRY_ALIAS_STATIC[canon]:
            COUNTRY_ALIAS_STATIC[canon].append(alias)

    CITY_ALIAS_STATIC: dict[str, list[str]] = {}
    for _, row in city_alias.iterrows():
        canon = row["canonical_city"]
        alias = row["alias_city"]
        CITY_ALIAS_STATIC.setdefault(canon, [])
        if alias not in CITY_ALIAS_STATIC[canon]:
            CITY_ALIAS_STATIC[canon].append(alias)

    # Build reverse alias maps
    state_alias_map: dict[str, str] = {}
    for canon, toks in STATE_ALIAS_STATIC.items():
        alias_list = [canon] + toks
        for a in alias_list:
            state_alias_map[a] = canon

    country_alias_map: dict[str, str] = {}
    for canon, toks in COUNTRY_ALIAS_STATIC.items():
        alias_list = [canon] + toks
        for a in alias_list:
            country_alias_map[a] = canon

    city_alias_map: dict[str, str] = {}
    for canon, toks in CITY_ALIAS_STATIC.items():
        alias_list = [canon] + toks
        for a in alias_list:
            city_alias_map[a] = canon

    return state_alias_map, country_alias_map, city_alias_map


def extract_candidates_from_text(
    whole: str,
    postal: pd.DataFrame,
    rta: pd.DataFrame,
    state_alias_map: dict[str, str],
    country_alias_map: dict[str, str],
    city_alias_map: dict[str, str],
    country_seed: str | None = None,
):
    """
    Extract possible cities, states, countries from free text.
    """
    tokens = re.split(r"[^A-Za-z0-9]+", whole.lower())
    tokens = [t for t in tokens if t]

    # Unique city/state/country from master
    city_set = set(postal["city"]).union(set(rta["city"]))
    state_set = set(postal["state"]).union(set(rta["state"]))
    country_set = set(postal["country"]).union(set(rta["country"]))

    # + alias keys
    city_set = city_set.union(set(city_alias_map.keys()))
    state_set = state_set.union(set(state_alias_map.keys()))
    country_set = country_set.union(set(country_alias_map.keys()))

    tok_str = " ".join(tokens)

    def find_match(candidates, thresh, extra_seed=None):
        found = set()
        for cand in candidates:
            if not cand:
                continue
            if cand in tok_str:
                found.add(cand)
                continue
            s = sim(tok_str, cand)
            if s >= thresh:
                found.add(cand)
        if extra_seed:
            extra = extra_seed.strip().lower()
            if extra and extra in candidates:
                found.add(extra)
        return found

    cities_found = find_match(city_set, CITY_THRESH)
    states_found = find_match(state_set, STATE_THRESH)
    countries_found = find_match(country_set, COUNTRY_THRESH, extra_seed=country_seed)

    # Canonicalise
    def canon_city(x):
        return city_alias_map.get(x, x)

    def canon_state(x):
        return state_alias_map.get(x, x)

    def canon_country(x):
        return country_alias_map.get(x, x)

    city_cands = sorted({canon_city(x) for x in cities_found})
    state_cands = sorted({canon_state(x) for x in states_found})
    country_cands = sorted({canon_country(x) for x in countries_found})

    return city_cands, state_cands, country_cands


def find_foreign_country_flag(country_cands, in_country, country_words):
    """
    Heuristic to find if foreign country possible.
    """
    in_country_norm = (in_country or "").strip().lower()
    if in_country_norm and in_country_norm not in {"india", "bharat", "republic of india"}:
        return 1

    for c in country_cands:
        if c not in {"india"}:
            return 1

    for w in country_words:
        wn = w.strip().lower()
        if wn and wn not in {"india", "bharat", "republic of india"}:
            return 1
    return 0


def remove_terms(text: str, remove_list: set[str]) -> str:
    """
    Remove country/state/city/pincodes from the full concatenated address
    to derive local_address.
    """
    if not text:
        return ""
    out = " " + text + " "
    for token in remove_list:
        if not token:
            continue
        t = str(token).strip()
        if not t:
            continue
        pattern = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
        out = pattern.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip(" ,")


def validate_addresses(df: pd.DataFrame, conn=None) -> pd.DataFrame:
    """
    Core validator: takes an input DF with address columns, returns enriched DF.
    """
    postal, rta, t30, state_alias, country_alias, city_alias = load_master_dfs(conn)
    pin_index, city_to_countries, state_to_countries = build_indexes(postal, rta)
    state_alias_map, country_alias_map, city_alias_map = build_alias_maps(
        state_alias, country_alias, city_alias
    )

    t30_set = {Title(x) for x in t30["city"].tolist() if not is_null_token(x)}

    rows_out = []
    audits = []

    for rid, row in df.iterrows():
        a1 = (row.get("address1") or "").strip()
        a2 = (row.get("address2") or "").strip()
        a3 = (row.get("address3") or "").strip()
        in_city = (row.get("city") or "").strip()
        in_state_raw = (row.get("state") or "").strip()
        in_country_raw = (row.get("country") or "").strip()
        in_pin = (row.get("pincode") or "").strip()

        # Build full text
        whole_parts = [a1, a2, a3, in_city, in_state_raw, in_country_raw, in_pin]
        whole = ", ".join([p for p in whole_parts if p])

        in_city_title = Title(in_city)
        in_state_title = Title(in_state_raw)
        in_country_title = Title(in_country_raw)

        in_state_expanded = None
        if in_state_title:
            key = in_state_title.lower()
            in_state_expanded = state_alias_map.get(key, key)

        in_country_expanded = None
        if in_country_title:
            key = in_country_title.lower()
            in_country_expanded = country_alias_map.get(key, key)

        # 2) Extract candidates
        city_cands, state_cands, country_cands = extract_candidates_from_text(
            whole,
            postal,
            rta,
            state_alias_map,
            country_alias_map,
            city_alias_map,
            country_seed=in_country_expanded,
        )

        # 3) Pincodes from text – ABSOLUTE priority, no fuzzy
        pins_text = set(PIN_RE.findall(whole))

        chosen_pin = None
        chosen_pin_row = None
        all_possible_pincodes_set = set()
        input_pin_not_in_master = False

        # Helper: derive pincodes from discovered cities via DB
        def _pins_from_city_state():
            pins = set()
            if city_cands:
                for src, df_src in [("postal", postal), ("rta", rta)]:
                    sub = df_src[df_src["city"].isin(city_cands)]
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
                                "src": src,
                            }
                        )
            return pins

        # --------- SMART PINCODE HANDLING (UPDATED) ---------
        # Normalise and classify the raw input pincode
        raw_pin = in_pin if in_pin is not None else ""
        raw_pin = raw_pin.strip()
        input_pin_clean = (
            raw_pin if (raw_pin and re.fullmatch(r"\d{6}", raw_pin)) else None
        )
        input_pin_syntax_invalid = bool(raw_pin) and input_pin_clean is None

        if input_pin_clean:
            # User has provided a 6-digit pincode in the input column
            rows_pin = pin_index.get(input_pin_clean, [])
            if rows_pin:
                # Valid pincode present in master -> choose it
                chosen_pin = input_pin_clean
                postal_rows = [r for (src, r) in rows_pin if src == "postal"]
                if postal_rows:
                    chosen_pin_row = postal_rows[0]
                else:
                    chosen_pin_row = rows_pin[0][1]

                # For a valid input pin, keep all pins seen in the address text if any,
                # otherwise at least the input pin itself.
                all_possible_pincodes_set = set(pins_text) or {input_pin_clean}
            else:
                # Input pincode is syntactically valid but NOT present in master.
                # Do NOT predict a single pincode – only return possible pins from DB.
                input_pin_not_in_master = True
                pins_from_city = _pins_from_city_state()
                all_possible_pincodes_set = pins_from_city

        elif input_pin_syntax_invalid:
            # Input pincode is present but has an invalid format (e.g. not 6 digits).
            # Treat as invalid: surface possible pincodes, but DO NOT force-pick one.
            input_pin_not_in_master = True
            pins_from_city = _pins_from_city_state()
            all_possible_pincodes_set = pins_from_city

        else:
            if pins_text:
                # No clean input pin but pincodes are present in free text.
                # Choose best matching pin using DB and keep all pins seen in text.
                if len(pins_text) == 1:
                    only_pin = next(iter(pins_text))
                    chosen_pin = only_pin
                    rows_pin = pin_index.get(only_pin, [])
                    if rows_pin:
                        postal_rows = [r for (src, r) in rows_pin if src == "postal"]
                        if postal_rows:
                            chosen_pin_row = postal_rows[0]
                        else:
                            chosen_pin_row = rows_pin[0][1]
                else:
                    best = None
                    best_score = -1.0
                    for p in pins_text:
                        rows_pin = pin_index.get(p, [])
                        if not rows_pin:
                            if best is None:
                                best = (p, None, 0.0)
                            continue
                        for src, row_pin in rows_pin:
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
                # No pincode in text at all – derive purely from city/state via DB.
                pins_from_city = _pins_from_city_state()

                if pins_from_city:
                    # Choose best pincode by state/country similarity
                    best = None
                    best_score = -1.0
                    for p in pins_from_city:
                        for src, row_pin in pin_index.get(p, []):
                            s_state = max(
                                (sim(row_pin["state"], s) for s in state_cands),
                                default=0.0,
                            )
                            s_ctry = max(
                                (sim(row_pin["country"], k) for k in country_cands),
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

        # 4) Choose city/state/country – PINCODE ROW HAS HIGHEST PRIORITY
        def choose_best_entity(candidates, input_value, pin_value):
            candidates = {Title(x) for x in candidates if not is_null_token(x)}

            # If we have a value coming from pincode master, ALWAYS use it.
            if pin_value and not is_null_token(pin_value):
                return Title(pin_value)

            # Else: pick best candidate vs input
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
                    sscore = sim(c, inp) if inp else 1.0  # any candidate if no input
                    if sscore > best_s:
                        best_c, best_s = c, sscore
                return best_c

            return Title(input_value) if (input_value and not is_null_token(input_value)) else None

        chosen_city_from_pin = None
        chosen_state_from_pin = None
        chosen_country_from_pin = None

        if chosen_pin and chosen_pin_row:
            chosen_city_from_pin = Title(chosen_pin_row["city"])
            chosen_state_from_pin = Title(chosen_pin_row["state"])
            chosen_country_from_pin = Title(chosen_pin_row["country"])

        chosen_city_clean = choose_best_entity(
            city_cands, in_city_title, chosen_city_from_pin
        )
        chosen_state = choose_best_entity(
            state_cands, in_state_title, chosen_state_from_pin
        )
        chosen_country = choose_best_entity(
            country_cands, in_country_title, chosen_country_from_pin
        )

        # 5) Build "all possibles" lists
        all_possible_countries = sorted(
            {c for c in country_cands if not is_null_token(c)}
        )
        all_possible_states = sorted(
            {s for s in state_cands if not is_null_token(s)}
        )
        all_possible_cities = sorted(
            {c for c in city_cands if not is_null_token(c)}
        )

        # Additional enrichment: if only city/state known, add matching countries
        foreign_country_possible = 0
        country_words = [w for w in re.split(r"[^A-Za-z]+", whole) if w]

        if not all_possible_countries and (all_possible_states or all_possible_cities):
            foreign_country_possible = find_foreign_country_flag(
                country_cands, in_country_title, country_words
            )
            is_confirmed_india = False

            # Enrich only for non-India
            for st in list(all_possible_states):
                for co in state_to_countries.get(st, []):
                    if co not in all_possible_countries:
                        all_possible_countries.append(co)
            for ci in list(all_possible_cities):
                for co in city_to_countries.get(ci, []):
                    if co not in all_possible_countries:
                        all_possible_countries.append(co)
        else:
            foreign_country_possible = find_foreign_country_flag(
                country_cands, in_country_title, country_words
            )
            is_confirmed_india = False

            # Enrich only for non-India
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
        # Add all found pincodes
        for p in all_possible_pincodes:
            remove_list.add(p)

        # Add chosen entities
        if chosen_country and not is_null_token(chosen_country):
            remove_list.add(chosen_country)
        if chosen_state and not is_null_token(chosen_state):
            remove_list.add(chosen_state)
        if chosen_city_clean and not is_null_token(chosen_city_clean):
            remove_list.add(chosen_city_clean)

        # Add original input entities to be sure they are removed
        if in_pin and not is_null_token(in_pin):
            remove_list.add(in_pin)
        if in_city and not is_null_token(in_city):
            remove_list.add(in_city)
        if in_state_title and not is_null_token(in_state_title):
            remove_list.add(in_state_title)
        if in_state_expanded and not is_null_token(in_state_expanded):
            remove_list.add(in_state_expanded)
        if in_country and not is_null_token(in_country):
            remove_list.add(in_country)

        local_address = remove_terms(whole, remove_list)

        # 7) Scores
        def score_entity(input_val, chosen_val, all_vals, row_from_pin):
            input_norm = Title(input_val)
            chosen_norm = Title(chosen_val)
            all_norm = {Title(x) for x in all_vals if not is_null_token(x)}

            if not chosen_norm and not all_norm:
                return 0.0, 0.0, 0.0

            # value_match
            value_match = 0.0
            if input_norm and chosen_norm:
                value_match = sim(input_norm, chosen_norm) * 100.0

            # consistency_with_pincode
            consistency = 0.0
            if row_from_pin and chosen_norm:
                # we already used row_from_pin to choose, this is high
                consistency = 100.0

            # ambiguity penalty
            amb_penalty = 0.0
            if len(all_norm) > 1:
                amb_penalty = 80.0
            elif len(all_norm) == 1:
                amb_penalty = 100.0
            else:
                amb_penalty = 0.0

            return value_match, consistency, amb_penalty

        city_from_pin = chosen_city_from_pin
        state_from_pin = chosen_state_from_pin
        country_from_pin = chosen_country_from_pin

        city_value, city_cons, city_amb = score_entity(
            in_city_title, chosen_city_clean, all_possible_cities, city_from_pin
        )
        state_value, state_cons, state_amb = score_entity(
            in_state_title, chosen_state, all_possible_states, state_from_pin
        )
        country_value, country_cons, country_amb = score_entity(
            in_country_title, chosen_country, all_possible_countries, country_from_pin
        )

        overall = round(
            (
                city_value
                + city_cons
                + city_amb
                + state_value
                + state_cons
                + state_amb
                + country_value
                + country_cons
                + country_amb
            )
            / 9.0,
            2,
        )

        # Reasons – concrete + understandable
        reasons = []
        if input_pin_not_in_master or (chosen_pin and chosen_pin_row is None):
            reasons.append("pincode_not_found_in_master")

        if chosen_pin_row is not None:
            if sim(chosen_city_clean, city_from_pin) < CITY_STATE_THRESH:
                reasons.append("mismatch_city_vs_pincode")
            if sim(chosen_state, state_from_pin) < CITY_STATE_THRESH:
                reasons.append("mismatch_state_vs_pincode")
            if sim(chosen_country, country_from_pin) < CITY_STATE_THRESH:
                reasons.append("mismatch_country_vs_pincode")

        # ambiguous flag
        ambiguous_flag = 0
        if (
            len(all_possible_cities) > 1
            or len(all_possible_states) > 1
            or len(all_possible_countries) > 1
        ):
            ambiguous_flag = 1

        rows_out.append(
            {
                # Inputs
                "input_id": rid,
                "address1": a1,
                "address2": a2,
                "address3": a3,
                "input_city": in_city,
                "input_state_raw": in_state_title,
                "input_state": in_state_expanded,
                "input_country": in_country_title,
                "input_pincode": in_pin,
                "concatenated_address": whole,
                # Outputs
                "output_pincode": chosen_pin,
                "output_city": chosen_city_clean,
                "output_state": chosen_state,
                "output_country": chosen_country,
                # Flags
                "t30_city_possible": 1
                if (chosen_city_clean and Title(chosen_city_clean) in t30_set)
                else 0,
                "foreign_country_possible": foreign_country_possible,
                # IMPORTANT: found if we have any possible pincodes at all
                "pincode_found": 1 if all_possible_pincodes else 0,
                "ambiguous_address_flag": ambiguous_flag,
                # All possibles
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
                # Scores (points, not 0/1) – 3 rules per entity
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
                # Local address
                "local_address": local_address,
            }
        )

    result_df = pd.DataFrame(rows_out)

    # Optionally you can also return audits if needed
    audit_df = (
        pd.DataFrame(audits)
        if audits
        else pd.DataFrame(
            columns=["input_id", "type", "pincode", "city", "state", "country", "src"]
        )
    )

    return result_df


def main(limit=1000, excel=None, batch_size=200):
    conn = get_db_connection()
    # Example: load from a staging table
    df_in = pd.read_sql(
        f"""
        SELECT id as input_id,
               address1,
               address2,
               address3,
               city,
               state,
               country,
               pincode
        FROM staging_addresses
        LIMIT {limit}
        """,
        conn,
    )
    df_in = df_in.set_index("input_id")

    result_df = validate_addresses(df_in, conn=conn)

    if excel:
        result_df.to_excel(excel, index=False)
    else:
        # fallback: print JSON lines
        print(result_df.to_json(orient="records", force_ascii=False))


if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel, batch_size=args.batch_size)
