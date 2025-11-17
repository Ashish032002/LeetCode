    # Separate masters for India vs world
    indian_city_master = set(postal["city"]).union(rta["city"])
    world_city_master  = set(world["city"])

    indian_country_master = {"India"}
    world_country_master  = set(world["country"]) - {"India"}


        # Decide whether this address is clearly India – then we should NOT look into world cities
        lower_whole = whole.lower()
        is_india_context = (
            "india" in lower_whole
            or (in_country and in_country.lower() == "india")
        )

        if is_india_context:
            # Use only Indian masters
            city_master_local    = indian_city_master
            country_master_local = indian_country_master
        else:
            # Use Indian + world only when it's foreign / no India found
            city_master_local    = indian_city_master.union(world_city_master)
            country_master_local = indian_country_master.union(world_country_master)


        city_cands = find_candidates_from_ngrams(
            city_words, city_master_local, max_n=2, thresh=CITY_STATE_THRESH, extra_seed=in_city
        )

        state_cands_raw = find_candidates_from_ngrams(
            state_words, set(state_alias.keys()), max_n=2, thresh=CITY_STATE_THRESH, extra_seed=in_state_expanded
        )
        state_cands = set()
        for s in state_cands_raw:
            expanded = expand_state_abbrev(s, state_alias)
            state_cands.add(expanded or Title(s))

        country_cands = find_candidates_from_ngrams(
            country_words, country_master_local, max_n=4, thresh=COUNTRY_THRESH, extra_seed=in_country
        )
