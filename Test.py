# --- all_possible_cities / states (same as before) ---
all_possible_cities = sorted({Title(clean_output_city(c)) for c in city_cands if c})
if chosen_city_clean and chosen_city_clean not in all_possible_cities:
    all_possible_cities.append(chosen_city_clean)

all_possible_states = sorted({Title(s) for s in state_cands if s})
if chosen_state and Title(chosen_state) not in all_possible_states:
    all_possible_states.append(Title(chosen_state))

# --- all_possible_countries (original + NEW fallback from state/city) ---
all_possible_countries = sorted({Title(k) for k in country_cands if k})
if chosen_country and Title(chosen_country) not in all_possible_countries:
    all_possible_countries.append(Title(chosen_country))

if not all_possible_countries:
    inferred_countries = set()

    # 1) Infer countries from states via postal & rta
    states_for_lookup = {Title(s) for s in state_cands if s}
    if chosen_state:
        states_for_lookup.add(Title(chosen_state))

    for st in states_for_lookup:
        # Indian states shortcut
        if st in STATE_ALIAS_STATIC:   # all of these are India
            inferred_countries.add("India")

        # Also look up from master tables in case of foreign states
        for df_src in (postal, rta):
            sub = df_src[df_src["state"] == st]
            if not sub.empty:
                inferred_countries.update(sub["country"].astype(str).map(Title))

    # 2) If still empty, infer from cities
    if not inferred_countries:
        cities_for_lookup = {Title(c) for c in city_cands if c}
        if chosen_city:
            cities_for_lookup.add(Title(chosen_city))

        for ct in cities_for_lookup:
            # postal + rta + world for country inference
            for df_src in (postal, rta, world):
                if "country" not in df_src.columns:
                    continue
                sub = df_src[df_src["city"] == ct]
                if not sub.empty:
                    inferred_countries.update(sub["country"].astype(str).map(Title))

    all_possible_countries = sorted(inferred_countries)






# --- NEW: enrich all_possible_cities using chosen pincode (postal + rta) ---
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
                if c_title not in all_possible_cities:
                    all_possible_cities.append(c_title)





if chosen_pin_row is not None:
    # City mismatch – compare after cleaning and only if really different
    if chosen_city_clean and city_from_pin:
        city_db_clean = clean_output_city(city_from_pin)
        city_sim = sim(chosen_city_clean, city_db_clean)
        if city_sim < CITY_STATE_THRESH and Title(chosen_city_clean) != Title(city_db_clean):
            reasons.append("mismatch_city_vs_pincode")

    # State mismatch
    if chosen_state and state_from_pin:
        state_sim = sim(chosen_state, state_from_pin)
        if state_sim < CITY_STATE_THRESH and Title(chosen_state) != Title(state_from_pin):
            reasons.append("mismatch_state_vs_pincode")

    # Country mismatch
    if chosen_country and country_from_pin:
        country_sim = sim(chosen_country, country_from_pin)
        if country_sim < COUNTRY_THRESH and Title(chosen_country) != Title(country_from_pin):
            reasons.append("mismatch_country_vs_pincode")
