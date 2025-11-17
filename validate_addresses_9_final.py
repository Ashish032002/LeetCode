import os
import re
import json
import pandas as pd
import datetime
from db_config import get_db_connection

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
CITY_STATE_THRESH = 0.88   # stricter to avoid wrong cities/states
COUNTRY_THRESH    = 0.85
FOREIGN_COUNTRY_STRICT_THRESH = 0.80 # Very high confidence for a foreign country
PIN_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


CITY_DIRECTIONS = {
    "north", "south", "east", "west", "n", "s", "e", "w",
    "northwest", "northeast", "southwest", "southeast", "nw", "ne", "sw", "se",
    "city", "moffusil", "division", "district", "zone", "sector", "block",
    "phase", "central", "suburban"
}


def Title(s):
    return str(s or "").strip().title()


def Upper(s):
    return str(s or "").strip().upper()


def is_null_token(x) -> bool:
    """Treat empty / NaN / None as null; avoid 'Nan' noise."""
    if x is None:
        return True
    s = str(x).strip().lower()
    return s in ("", "nan", "none", "null")


def norm_text(s):
    s = str(s or "")
    s = re.sub(r"[^0-9A-Za-z ,./-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokens_alpha(s):
    return [t for t in re.split(r"[^A-Za-z]+", str(s or "")) if t]


def ngrams(words, n):
    out = set()
    if n <= 0:
        return out
    L = len(words)
    for i in range(L - n + 1):
        out.add(Title(" ".join(words[i:i + n])))
    return out


def clean_city_tokens(words):
    return [w for w in words if w.lower() not in CITY_DIRECTIONS]


def clean_output_city(city):
    if not city:
        return city
    w = clean_city_tokens(tokens_alpha(city))
    return Title(" ".join(w)) if w else Title(city)


# ---------------- State abbreviation handling ----------------

def _norm_token(x):
    x = str(x or "").strip()
    x = re.sub(r"[^A-Za-z]", "", x)
    return x.upper()


STATE_ALIAS_STATIC = {
    "Andhra Pradesh": {"AP", "AD"},
    "Arunachal Pradesh": {"AR"},
    "Assam": {"AS"},
    "Bihar": {"BH", "BR"},
    "Chhattisgarh": {"CG", "CT"},
    "Goa": {"GO", "GA"},
    "Gujarat": {"GU", "GJ"},
    "Haryana": {"HA", "HR"},
    "Himachal Pradesh": {"HP"},
    "Jammu And Kashmir": {"JK", "JNK", "J&K"},
    "Jharkhand": {"JH", "JD"},
    "Karnataka": {"KA", "KAR"},
    "Kerala": {"KE", "KL"},
    "Madhya Pradesh": {"MP", "MD"},
    "Maharashtra": {"MA", "MH"},
    "Manipur": {"MN"},
    "Meghalaya": {"ML", "ME"},
    "Mizoram": {"MZ"},
    "Nagaland": {"NL"},
    "Odisha": {"OD", "OR", "ORISSA"},
    "Punjab": {"PU", "PB"},
    "Rajasthan": {"RA", "RJ"},
    "Sikkim": {"SK"},
    "Tamil Nadu": {"TN", "TM"},
    "Telangana": {"TG", "TS", "TE"},
    "Tripura": {"TR"},
    "Uttar Pradesh": {"UP"},
    "Uttarakhand": {"UK", "UA", "UC"},
    "West Bengal": {"WB", "W.B"},
    "Delhi": {"DL", "ND", "NCT"},
    "Chandigarh": {"CH"},
    "Puducherry": {"PO", "PY"},
    "Ladakh": {"LA"},
    "Lakshadweep": {"LD"},
}


def build_state_alias(states_df):
    """
    Build mapping canonical_state -> set of normalised tokens
    from both DB abbreviations and static aliases.
    """
    alias = {}

    for _, row in states_df.iterrows():
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
        alias.setdefault(canon, set()).update({_norm_token(x) for x in toks})

    return alias


def expand_state_abbrev(state_in, alias):
    """
    Expand AP → Andhra Pradesh etc.
    If nothing obvious, return title-cased input.
    """
    if not state_in or is_null_token(state_in):
        return state_in

    tok = _norm_token(state_in)
    for canon, toks in alias.items():
        if tok in toks:
            return canon

    st = Title(state_in)
    if st in alias:
        return st

    # small fuzzy fallback
    best_c, best_s = None, -1.0
    for canon in alias.keys():
        s = sim(canon, st)
        if s > best_s:
            best_c, best_s = canon, s
    return best_c if best_s >= CITY_STATE_THRESH else st


# ---------------- Rule scoring helpers ----------------

def score_value_match(chosen, input_value, best_ngram_match):
    """
    Rule (a): direct exact/fuzzy match vs input + best n-gram.
    Returns 0..100.
    """
    if not chosen:
        return 0.0
    s1 = sim(chosen, input_value) if (input_value and not is_null_token(input_value)) else 0.0
    s2 = sim(chosen, best_ngram_match) if best_ngram_match else 0.0
    return round(100.0 * max(s1, s2), 2)


def score_consistency_with_pin(chosen, field_from_pin):
    """
    Rule (b): consistency against chosen pincode row.
    Returns 0..100.
    """
    if not (chosen and field_from_pin):
        return 0.0
    return round(100.0 * sim(chosen, field_from_pin), 2)


def score_ambiguity(candidates):
    """
    Rule (c): ambiguity penalty.
    1 unique candidate -> 100.
    More candidates -> gradually lower but minimum 40.
    """
    uniq = {Title(x) for x in candidates if not is_null_token(x)}
    k = len(uniq)
    if k <= 1:
        return 100.0
    return float(max(40, 100 - 20 * (k - 1)))


def remove_terms(text, terms):
    """
    Remove already-identified pincode / city / state / country tokens
    from concatenated address to produce local_address.
    """
    s = " " + (text or "") + " "
    for t in terms:
        if not t or is_null_token(t):
            continue
        t = re.escape(str(t).strip())
        s = re.sub(r"(?i)\b" + t + r"\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------- Candidate finders ----------------

def find_candidates_from_ngrams(ngram_words, master_set, max_n, thresh, extra_seed=None):
    """
    Generic helper:
    - Build 1-gram, 2-gram (or up to 4 for country) from ngram_words.
    - For each n:
        * take exact matches.
        * plus fuzzy matches above thresh.
    - If we already found some at n=1, we do NOT go to higher n
      (your rule: no need to move forward if lower-order fuzzy already found).
    """
    master_list = [m for m in master_set if not is_null_token(m)]
    found = set()

    seed = Title(extra_seed) if (extra_seed and not is_null_token(extra_seed)) else None

    for n in range(1, max_n + 1):
        grams_n = ngrams(ngram_words, n)
        if seed:
            grams_n.add(seed)

        exact = {g for g in grams_n if g in master_set}
        fuzzy = set()

        for g in grams_n:
            g_clean = g.strip()
            if not g_clean or len(g_clean) < 3:
                continue

            # Only search near strings (same first letter) to avoid garbage.
            first = g_clean[0]
            best_c = None
            best_s = 0.0
            for cm in master_list:
                if not cm:
                    continue
                if cm[0] != first:
                    continue
                s = sim(g_clean, cm)
                if s > best_s:
                    best_c, best_s = cm, s

            if best_c and best_s >= thresh:
                fuzzy.add(best_c)

        level_cands = exact.union(fuzzy)

        if level_cands:
            # as soon as we have candidates at this n, we stop (no higher n)
            found.update(level_cands)
            break

    return {Title(x) for x in found if not is_null_token(x)}


def find_foreign_country_flag(country_cands, in_country, country_words):
    """
    Checks for high-confidence foreign countries. Returns 1 if a non-India
    country is found with a similarity score > FOREIGN_COUNTRY_STRICT_THRESH.
    """
    foreign_cands = {c for c in country_cands if c != "India"}
    if not foreign_cands:
        return 0

    # Check against input country first
    if in_country and not is_null_token(in_country) and in_country != "India":
        if sim(in_country, max(foreign_cands, key=lambda x: sim(x, in_country))) >= FOREIGN_COUNTRY_STRICT_THRESH:
            return 1

    # Check n-grams from address text
    all_country_ngrams = set()
    for n in (1, 2, 3, 4):
        all_country_ngrams |= ngrams(country_words, n)

    for cand in foreign_cands:
        best_ngram_sim = max((sim(cand, ng) for ng in all_country_ngrams), default=0.0)
        if best_ngram_sim >= FOREIGN_COUNTRY_STRICT_THRESH:
            return 1 # Found a high-confidence foreign match
    return 0

# ---------------- Main validator ----------------

def main(limit=1000, excel=None, batch_size=200):
    eng = get_db_connection()

    # Load reference tables
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
        t30 = (
            pd.read_sql("SELECT city FROM ref.t30_cities", con)["city"]
            .astype(str)
            .str.title()
            .unique()
            .tolist()
        )
        try:
            abbrev = pd.read_sql(
                "SELECT state, abbreviation FROM ref.indian_state_abbrev", con
            )
        except Exception:
            abbrev = pd.DataFrame({"state": [], "abbreviation": []})

        inputs = pd.read_sql(
            "SELECT * FROM input.addresses ORDER BY id LIMIT {}".format(int(limit)), con
        )

    # Normalise reference text
    for df, cols in [
        (postal, ["city", "state", "country"]),
        (rta, ["city", "state", "country"]),
        (world, ["city", "country"]),
    ]:
        for c in cols:
            df[c] = df[c].astype(str).map(Title)

    country_master = (
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
    alias_df = pd.DataFrame(
        {"state": list(STATE_ALIAS_STATIC.keys()), "abbreviation": [""] * len(STATE_ALIAS_STATIC)}
    )
    if not abbrev.empty:
        alias_df = pd.concat(
            [alias_df, abbrev[["state", "abbreviation"]]], ignore_index=True
        )
    state_alias = build_state_alias(alias_df)

    # Build state -> countries & city -> countries maps (postal + rta + world)
    state_to_countries = {}
    for df in (postal, rta):
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
    t30_set = {Title(x) for x in t30}

    for start in range(0, len(inputs), batch_size):
        chunk = inputs.iloc[start:start + batch_size].copy()

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

            # 1) Whole address (concatenated)
            whole = norm_text(
                " ".join(
                    [
                        str(x or "")
                        for x in [a1, a2, a3, in_city, in_state_expanded, in_country, in_pin]
                    ]
                )
            )
            words = tokens_alpha(whole)

            # 2) N-grams: city/state 1–2; country 1–4
            city_words = clean_city_tokens(words)
            state_words = words
            country_words = words

            # Determine if the address is likely Indian to decide which city master to use.
            # This is a pre-check before the more robust country selection later.
            temp_country_cands = find_candidates_from_ngrams(
                country_words,
                country_master,
                max_n=4,
                thresh=COUNTRY_THRESH,
                extra_seed=in_country,
            )
            probably_india = (not is_null_token(in_country) and Title(in_country) == "India") or ("India" in temp_country_cands)

            # Use a world-inclusive city master only if the country is not likely India.
            city_master = city_master_india if probably_india else city_master_world

            city_cands = find_candidates_from_ngrams(
                city_words,
                city_master,
                max_n=2,
                thresh=CITY_STATE_THRESH,
                extra_seed=in_city,
            )

            state_cands_raw = find_candidates_from_ngrams(
                state_words,
                set(state_alias.keys()),
                max_n=2,
                thresh=CITY_STATE_THRESH,
                extra_seed=in_state_expanded,
            )
            state_cands = set()
            for s in state_cands_raw:
                expanded = expand_state_abbrev(s, state_alias)
                if expanded and not is_null_token(expanded):
                    state_cands.add(expanded)

            country_cands = temp_country_cands

            # 3) Pincodes from text – ABSOLUTE priority, no fuzzy
            pins_text = set(PIN_RE.findall(whole))

            chosen_pin = None
            chosen_pin_row = None

            if pins_text:
                # Highest priority: if input_pincode is a valid 6-digit, keep it.
                if re.fullmatch(r"\d{6}", in_pin):
                    chosen_pin = in_pin
                    rows = pin_index.get(in_pin, [])
                    if rows:
                        # prefer postal if available, else first
                        postal_rows = [r for (src, r) in rows if src == "postal"]
                        if postal_rows:
                            chosen_pin_row = postal_rows[0]
                        else:
                            chosen_pin_row = rows[0][1]
                else:
                    # If multiple pins in text, choose one that best matches city/state/country via DB.
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

                # all_possible_pincodes: ONLY pins we literally saw in text
                all_possible_pincodes_set = set(pins_text)

            else:
                # No pin in text – derive from discovered cities via DB.
                pins_from_city = set()
                if city_cands:
                    for src, df_src in [("postal", postal), ("rta", rta)]:
                        sub = df_src[df_src["city"].isin(city_cands)]
                        for _, row_pin in sub.iterrows():
                            pins_from_city.add(str(row_pin["pincode"]))
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

                if pins_from_city:
                    # Choose best by state/country similarity
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
                    inp = Title(input_value) if (input_value and not is_null_token(input_value)) else None
                    for c in candidates:
                        s = sim(c, inp) if inp else 1.0  # any candidate if no input
                        if s > best_s:
                            best_c, best_s = c, s
                    return best_c

                return Title(input_value) if (input_value and not is_null_token(input_value)) else None

            city_from_pin = chosen_pin_row["city"] if chosen_pin_row is not None else None
            state_from_pin = chosen_pin_row["state"] if chosen_pin_row is not None else None
            country_from_pin = chosen_pin_row["country"] if chosen_pin_row is not None else None

            chosen_city = choose_best_entity(city_cands, in_city, city_from_pin)
            chosen_state = choose_best_entity(state_cands, in_state_expanded, state_from_pin)
            chosen_country = choose_best_entity(country_cands, in_country, country_from_pin)

            chosen_city_clean = clean_output_city(chosen_city)

            # 5) Build all_possible_* (no NaN, and enrich from PIN + state/city→country)
            all_possible_cities = sorted(
                {Title(clean_output_city(c)) for c in city_cands if not is_null_token(c)}
            )

            # Add chosen city
            if chosen_city_clean and not is_null_token(chosen_city_clean):
                if chosen_city_clean not in all_possible_cities:
                    all_possible_cities.append(chosen_city_clean)

            # Always also include the raw input city (locality like HSR Layout)
            if in_city and not is_null_token(in_city):
                in_city_clean = clean_output_city(in_city)
                if in_city_clean and in_city_clean not in all_possible_cities:
                    all_possible_cities.append(in_city_clean)

            # --- enrich all_possible_cities using chosen pincode (postal + rta) ---
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
                            if not is_null_token(c_title) and c_title not in all_possible_cities:
                                all_possible_cities.append(c_title)

            all_possible_states = sorted(
                {Title(s) for s in state_cands if not is_null_token(s)}
            )
            if chosen_state and not is_null_token(chosen_state):
                cs = Title(chosen_state)
                if cs not in all_possible_states:
                    all_possible_states.append(cs)

            all_possible_countries = sorted(
                {Title(k) for k in country_cands if not is_null_token(k)}
            )
            if chosen_country and not is_null_token(chosen_country):
                ck = Title(chosen_country)
                if ck not in all_possible_countries:
                    all_possible_countries.append(ck)

            # --- Smart Flag for Foreign Country & India Lock-in ---
            foreign_country_possible = find_foreign_country_flag(country_cands, in_country, country_words)
            is_confirmed_india = (chosen_country == "India" and not foreign_country_possible)

            # Only enrich countries from cities/states if the address is NOT confirmed as Indian.
            if not is_confirmed_india:
                for st in list(all_possible_states):
                    for co in state_to_countries.get(st, []):
                        if co not in all_possible_countries:
                            all_possible_countries.append(co)
                for ci in list(all_possible_cities):
                    for co in city_to_countries.get(ci, []):
                        if co not in all_possible_countries:
                            all_possible_countries.append(co)
            # FINAL OVERRIDE: If it's an Indian address, ensure only India is in the list.
            if is_confirmed_india:
                all_possible_countries = ["India"]

            all_possible_pincodes = sorted(
                {p for p in all_possible_pincodes_set if not is_null_token(p)}
            )

            # 6) local_address = whole – entities we already found
            remove_list = []
            remove_list.extend(all_possible_pincodes)
            if chosen_country and not is_null_token(chosen_country):
                remove_list.append(chosen_country)
            if chosen_state and not is_null_token(chosen_state):
                remove_list.append(chosen_state)
            if chosen_city_clean and not is_null_token(chosen_city_clean):
                remove_list.append(chosen_city_clean)
            local_address = remove_terms(whole, remove_list)

            # 7) Scores: 3 rules x 3 entities + overall
            # best n-gram per entity for rule (a)
            best_city_ng = ""
            if chosen_city_clean and not is_null_token(chosen_city_clean):
                all_city_ngrams = ngrams(city_words, 1).union(ngrams(city_words, 2))
                best_city_ng = max(
                    all_city_ngrams or {""},
                    key=lambda x: sim(x, chosen_city_clean),
                )

            best_state_ng = ""
            if chosen_state and not is_null_token(chosen_state):
                all_state_ngrams = ngrams(state_words, 1).union(ngrams(state_words, 2))
                best_state_ng = max(
                    all_state_ngrams or {""}, key=lambda x: sim(x, chosen_state)
                )

            best_country_ng = ""
            if chosen_country and not is_null_token(chosen_country):
                all_country_ngrams = set()
                for n in (1, 2, 3, 4):
                    all_country_ngrams |= ngrams(country_words, n)
                best_country_ng = max(
                    all_country_ngrams or {""},
                    key=lambda x: sim(x, chosen_country),
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
            country_cons = score_consistency_with_pin(
                chosen_country, country_from_pin
            )

            city_amb = score_ambiguity(all_possible_cities)
            state_amb = score_ambiguity(all_possible_states)
            country_amb = score_ambiguity(all_possible_countries)

            def bundle(v, c, a):
                # simple weighted mix for each entity
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

            # Reasons – concrete + understandable
            reasons = []
            if chosen_pin and chosen_pin_row is None:
                reasons.append("pincode_not_found_in_master")

            if chosen_pin_row is not None:
                if sim(chosen_city_clean, city_from_pin) < CITY_STATE_THRESH:
                    reasons.append("mismatch_city_vs_pincode")
                if sim(chosen_state, state_from_pin) < CITY_STATE_THRESH:
                    reasons.append("mismatch_state_vs_pincode")
                if sim(chosen_country, country_from_pin) < COUNTRY_THRESH:
                    reasons.append("mismatch_country_vs_pincode")

            if city_amb < 100:
                reasons.append("ambiguous_city_candidates")
            if state_amb < 100:
                reasons.append("ambiguous_state_candidates")
            if country_amb < 100:
                reasons.append("ambiguous_country_candidates")

            if city_value < 80:
                reasons.append("low_city_value_match")
            if state_value < 80:
                reasons.append("low_state_value_match")
            if country_value < 80:
                reasons.append("low_country_value_match")

            ambiguous_flag = 1 if overall < 85 else 0

            results.append(
                {
                    # Inputs
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
                    "pincode_found": 1 if chosen_pin else 0,
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

    result_df = pd.DataFrame(results)
    audit_df = (
        pd.DataFrame(audits)
        if audits
        else pd.DataFrame(
            columns=["input_id", "type", "pincode", "city", "state", "country", "src"]
        )
    )

    # Excel output
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    xls_path = excel or os.path.join(out_dir, "validation_results.xlsx")
    with pd.ExcelWriter(xls_path) as xl:
        result_df.to_excel(xl, index=False, sheet_name="results")
        audit_df.to_excel(xl, index=False, sheet_name="audit")

    # Persist full result to DB
    with eng.begin() as con:
        result_df.to_sql(
            "validation_result_full",
            con,
            schema="output",
            if_exists="append",
            index=False,
            method="multi",
        )

    print("Validation done: {} rows. Excel → {}".format(len(result_df), xls_path))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel, batch_size=args.batch_size)