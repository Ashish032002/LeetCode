# --- NEW: enrich all_possible_cities using chosen pincode (postal + rta) ---
if chosen_pin:
    pin_str = str(chosen_pin).strip()
    if pin_str:
        for df_src in (postal, rta):
            # compare as string, so it works whether column is int or object
            sub = df_src[df_src["pincode"].astype(str) == pin_str]
            for _, row_pin in sub.iterrows():
                c_title = Title(row_pin["city"])
                if c_title and c_title not in all_possible_cities:
                    all_possible_cities.append(c_title)
