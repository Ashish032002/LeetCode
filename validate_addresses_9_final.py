import os
import re
import json
import pandas as pd
import datetime
from .db_config import get_db_connection
from functools import lru_cache

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
FOREIGN_COUNTRY_STRICT_THRESH = 0.95 # Very high confidence for a foreign country
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
    - Builds n-grams (up to max_n) from ngram_words and extra_seed.
    - Collects all exact and fuzzy matches (above thresh) from the master_set.
    - Does not stop after the first level, but collects all possible candidates.
    """
    master_list = [m for m in master_set if not is_null_token(m)]
    found = set()

    # Process extra_seed first, as it often contains direct user input that might be multi-part
    if extra_seed and not is_null_token(extra_seed):
        seed_words = tokens_alpha(extra_seed)
        for n_seed in range(1, max_n + 1):
            seed_grams = ngrams(seed_words, n_seed)
            for sg in seed_grams:
                if sg in master_set:
                    found.add(sg)
                else:
                    # Fuzzy match for seed grams
                    first = sg[0] if sg else ''
                    best_c = None
                    best_s = 0.0
                    for cm in master_list:
                        if not cm or cm[0] != first:
                            continue
                        s = sim(sg, cm)
                        if s > best_s:
                            best_c, best_s = cm, s
                    if best_c and best_s >= thresh:
                        found.add(best_c)

    # Now process the main ngram_words from the concatenated address
    for n in range(1, max_n + 1):
        grams_n = ngrams(ngram_words, n)
        
        for g in grams_n: # g is already Title cased from ngrams function
            if g in master_set:
                found.add(g)
            else:
                # Fuzzy match for grams from ngram_words
                if not g or len(g) < 3: # g is already stripped and Title cased
                    continue
                first = g[0]
                best_c = None
                best_s = 0.0
                for cm in master_list:
                    if not cm:
                        continue
                    if cm[0] != first:
                        continue
                    s = sim(g, cm)
                    if s > best_s:
                        best_c, best_s = cm, s

                if best_c and best_s >= thresh:
                    found.add(best_c)
    
    return {x for x in found if not is_null_token(x)} # Items in 'found' are already Title cased


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
@lru_cache(maxsize=None)
def _load_and_prepare_reference_data():
    """
    Loads all reference data from the database and prepares it for validation.
    This function is cached to avoid repeated DB queries on every API call.
    """
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
    
    t30_set = {Title(x) for x in t30}

    return {
        "postal": postal,
        "rta": rta,
        "world": world,
        "t30_set": t30_set,
        "country_master_full": country_master_full,
        "city_master_india": city_master_india,
        "city_master_world": city_master_world,
        "pin_index": pin_index,
        "state_alias": state_alias,
        "state_to_countries": state_to_countries,
        "city_to_countries": city_to_countries,
    }

# ---------------- Main validator ----------------

def main(limit=1000, excel=None, batch_size=200):
    eng = get_db_connection()

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

            # --- INDIA-FIRST LOGIC ---
            # Detect country candidates from full master (always include world countries)
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
            # Only allow world_cities when there is a clear foreign country
            # and the input country is not explicitly India.
            if has_foreign_country and not input_country_is_india:
                city_master = city_master_world
            else:
                city_master = city_master_india

            # For countries, always use the full master when finding candidates
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

            # Now find country candidates using the correctly scoped master list
            country_cands = find_candidates_from_ngrams(
                country_words,
                country_master, # Uses India-only or full list based on 'probably_india'
                max_n=4,
                thresh=COUNTRY_THRESH,
                extra_seed=in_country,
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

            input_pin_clean = in_pin if (in_pin and re.fullmatch(r"\d{6}", in_pin)) else None

            if input_pin_clean:
                # User has provided a 6-digit pincode in the input column
                rows = pin_index.get(input_pin_clean, [])
                if rows:
                    # Valid pincode present in master -> choose it
                    chosen_pin = input_pin_clean
                    postal_rows = [r for (src, r) in rows if src == "postal"]
                    if postal_rows:
                        chosen_pin_row = postal_rows[0]
                    else:
                        chosen_pin_row = rows[0][1]

                    # For a valid input pin, keep all pins seen in the address text if any,
                    # otherwise at least the input pin itself.
                    all_possible_pincodes_set = set(pins_text) or {input_pin_clean}
                else:
                    # Input pincode is syntactically valid but NOT present in master.
                    # Do NOT predict a single pincode – only return possible pins from DB.
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
# 4) Choose city/state/country – PINCODE ROW HAS HIGHEST PRIORIT
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

            # Include raw input state pieces (e.g., "Rajasthan tamilnadu")
            if in_state_title and not is_null_token(in_state_title):
                for part in re.split(r"[^A-Za-z]+", in_state_title):
                    part = part.strip()
                    if not part:
                        continue
                    s_clean = Title(part)
                    if s_clean and s_clean not in all_possible_states:
                        all_possible_states.append(s_clean)

            all_possible_countries = sorted(
                {Title(k) for k in country_cands if not is_null_token(k)}
            )
            if chosen_country and not is_null_token(chosen_country):
                ck = Title(chosen_country)
                if ck not in all_possible_countries:
                    all_possible_countries.append(ck)

            # Include raw input country pieces (e.g., "India USA")
            if in_country and not is_null_token(in_country):
                for part in re.split(r"[^A-Za-z]+", in_country):
                    part = part.strip()
                    if not part:
                        continue
                    c_clean = Title(part)
                    if c_clean and c_clean not in all_possible_countries:
                        all_possible_countries.append(c_clean)

            # -------- India lock based on STATE / chosen country --------
            INDIAN_STATES = set(state_alias.keys())

            is_confirmed_india = False

            if chosen_state in INDIAN_STATES:
                # Force main country to India, but keep other possible countries in the list
                chosen_country = "India"
                if "India" not in all_possible_countries:
                    all_possible_countries.insert(0, "India")

                # Restrict cities to Indian cities only
                indian_city_set = set(city_master_india)
                all_possible_cities = [
                    c for c in all_possible_cities
                    if c in indian_city_set
                ]
                is_confirmed_india = True

            # Enrich countries from cities/states only when the final chosen country is not India
            if chosen_country == "India":
                foreign_country_possible = 0
                is_confirmed_india = True

                # Ensure cities are restricted to Indian list
                indian_city_set = set(city_master_india)
                all_possible_cities = [
                    c for c in all_possible_cities
                    if c in indian_city_set
                ]
            else:
                # Normal logic for non-India addresses
                foreign_country_possible = find_foreign_country_flag(
                    country_cands, in_country, country_words
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

            # Add all possible entities found during validation to the removal list.
            # This is more robust than just removing the final "chosen" one.
            for item_list in [
                all_possible_pincodes,
                all_possible_cities,
                all_possible_states,
                all_possible_countries,
            ]:
                for item in item_list:
                    remove_list.add(item)

            # Also add the raw, un-normalized inputs to ensure they are stripped out.
            for raw_input in [in_pin, in_city_raw, in_state_raw, in_country_raw]:
                if raw_input and not is_null_token(raw_input):
                    remove_list.add(raw_input)

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
        print(f"ERROR: Could not write to Excel file. Is '{xls_path}' open in another program?")
        # Exit gracefully without writing the file, but after DB operations

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




def validate_inputs_df(inputs: pd.DataFrame, batch_size: int = 200) -> pd.DataFrame:
    """
    Validates a DataFrame of addresses using cached reference data.
    """
    # Load reference data from cache (or from DB on first call)
    ref_data = _load_and_prepare_reference_data()
    postal = ref_data["postal"]
    rta = ref_data["rta"]
    world = ref_data["world"]
    t30_set = ref_data["t30_set"]
    country_master_full = ref_data["country_master_full"]
    city_master_india = ref_data["city_master_india"]
    city_master_world = ref_data["city_master_world"]
    pin_index = ref_data["pin_index"]
    state_alias = ref_data["state_alias"]
    state_to_countries = ref_data["state_to_countries"]
    city_to_countries = ref_data["city_to_countries"]
    
    results = []
    audits = []
    
    for start in range(0, len(inputs), batch_size):
        chunk = inputs.iloc[start:start + batch_size].copy()
    
        for _, rec in chunk.iterrows():
            rid = int(rec["id"])
            reasons = [] # Start with an empty list of reasons for each record
    
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
    
            # --- INDIA-FIRST LOGIC ---
            # Detect country candidates from full master (always include world countries)
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
            # Only allow world_cities when there is a clear foreign country
            # and the input country is not explicitly India.
            if has_foreign_country and not input_country_is_india:
                city_master = city_master_world
            else:
                city_master = city_master_india

            # For countries, always use the full master when finding candidates
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
    
            # Now find country candidates using the correctly scoped master list
            country_cands = find_candidates_from_ngrams(
                country_words,
                country_master, # Uses India-only or full list based on 'probably_india'
                max_n=4,
                thresh=COUNTRY_THRESH,
                extra_seed=in_country,
            )
    
            # 3) Pincodes from text – ABSOLUTE priority, no fuzzy
            pins_text = set(PIN_RE.findall(whole))

            # --- Stricter Pincode Logic ---
            # Only accept the input pincode if it's a valid 6-digit number and exists in the master data.
            chosen_pin = None
            chosen_pin_row = None
            all_possible_pincodes_set = set()
            
            input_pin_clean = in_pin if (in_pin and re.fullmatch(r"\d{6}", in_pin)) else None
            
            if input_pin_clean:
                # A 6-digit pincode was provided. Check if it's in the master data.
                rows = pin_index.get(input_pin_clean, [])
                if rows:
                    # Valid 6-digit pincode found in master data. This is the only case where we accept a pincode.
                    chosen_pin = input_pin_clean
                    postal_rows = [r for (src, r) in rows if src == "postal"]
                    chosen_pin_row = postal_rows[0] if postal_rows else rows[0][1]
                    all_possible_pincodes_set.add(chosen_pin)
                else:
                    # Pincode is 6 digits but not in master data.
                    reasons.append("pincode_not_found_in_master")
            elif in_pin and not is_null_token(in_pin):
                # Pincode was provided but not in the correct 6-digit format.
                reasons.append("invalid_pincode_format")
# 4) Choose city/state/country – PINCODE ROW HAS HIGHEST PRIORIT
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

            # Include raw input state pieces (e.g., "Rajasthan tamilnadu")
            if in_state_title and not is_null_token(in_state_title):
                for part in re.split(r"[^A-Za-z]+", in_state_title):
                    part = part.strip()
                    if not part:
                        continue
                    s_clean = Title(part)
                    if s_clean and s_clean not in all_possible_states:
                        all_possible_states.append(s_clean)

            all_possible_countries = sorted(
                {Title(k) for k in country_cands if not is_null_token(k)}
            )
            if chosen_country and not is_null_token(chosen_country):
                ck = Title(chosen_country)
                if ck not in all_possible_countries:
                    all_possible_countries.append(ck)

            # Include raw input country pieces (e.g., "India USA")
            if in_country and not is_null_token(in_country):
                for part in re.split(r"[^A-Za-z]+", in_country):
                    part = part.strip()
                    if not part:
                        continue
                    c_clean = Title(part)
                    if c_clean and c_clean not in all_possible_countries:
                        all_possible_countries.append(c_clean)

            # -------- India lock based on STATE / chosen country --------
            INDIAN_STATES = set(state_alias.keys())

            is_confirmed_india = False

            if chosen_state in INDIAN_STATES:
                # Force main country to India, but keep other possible countries in the list
                chosen_country = "India"
                if "India" not in all_possible_countries:
                    all_possible_countries.insert(0, "India")

                # Restrict cities to Indian cities only
                indian_city_set = set(city_master_india)
                all_possible_cities = [
                    c for c in all_possible_cities
                    if c in indian_city_set
                ]
                is_confirmed_india = True

            # Enrich countries from cities/states only when the final chosen country is not India
            if chosen_country == "India":
                foreign_country_possible = 0
                is_confirmed_india = True

                # Ensure cities are restricted to Indian list
                indian_city_set = set(city_master_india)
                all_possible_cities = [
                    c for c in all_possible_cities
                    if c in indian_city_set
                ]
            else:
                # Normal logic for non-India addresses
                foreign_country_possible = find_foreign_country_flag(
                    country_cands, in_country, country_words
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
            if chosen_country and not is_null_token(chosen_country): remove_list.add(chosen_country)
            if chosen_state and not is_null_token(chosen_state): remove_list.add(chosen_state)
            if chosen_city_clean and not is_null_token(chosen_city_clean): remove_list.add(chosen_city_clean)
            # Add all possible entities found during validation to the removal list.
            # This is more robust than just removing the final "chosen" one.
            for item_list in [
                all_possible_pincodes,
                all_possible_cities,
                all_possible_states,
                all_possible_countries,
            ]:
                for item in item_list:
                    remove_list.add(item)

            # Add original input entities to be sure they are removed
            if in_pin and not is_null_token(in_pin): remove_list.add(in_pin)
            if in_city and not is_null_token(in_city): remove_list.add(in_city)
            if in_state_title and not is_null_token(in_state_title): remove_list.add(in_state_title)
            if in_state_expanded and not is_null_token(in_state_expanded): remove_list.add(in_state_expanded)
            if in_country and not is_null_token(in_country): remove_list.add(in_country)
            # Also add the raw, un-normalized inputs to ensure they are stripped out.
            for raw_input in [in_pin, in_city_raw, in_state_raw, in_country_raw]:
                if raw_input and not is_null_token(raw_input):
                    remove_list.add(raw_input)

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
    
            # Add more reasons based on consistency and ambiguity checks
            if chosen_pin_row is not None:
                if sim(chosen_city_clean, city_from_pin) < CITY_STATE_THRESH:
                    reasons.append("city_inconsistent_with_pincode")
                if sim(chosen_state, state_from_pin) < CITY_STATE_THRESH:
                    reasons.append("state_inconsistent_with_pincode")
                if sim(chosen_country, country_from_pin) < COUNTRY_THRESH:
                    reasons.append("country_inconsistent_with_pincode")
    
            if city_amb < 80:
                reasons.append("ambiguous_city_candidates")
            if state_amb < 80:
                reasons.append("ambiguous_state_candidates")
    
            # Add a reason if the chosen country is not India but no high-confidence foreign country was found
            if chosen_country != "India" and not foreign_country_possible:
                 reasons.append("ambiguous_country_candidates")
    
            ambiguous_flag = 1 if overall < 85 else 0
            if ambiguous_flag and not reasons:
                reasons.append("low_overall_confidence")
    
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
    
    return result_df

if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()
    main(limit=args.limit, excel=args.excel, batch_size=args.batch_size)