import pandas as pd
import re
from rapidfuzz import fuzz, process
from db_config import get_db_connection

# Connect to database
engine = get_db_connection()
print("Connected to database.")

# Load master_data
master_df = pd.read_sql("SELECT division_name, pincode, state_name FROM master_data;", engine)
master_df['source'] = 'master_data'

# Load pincode_master
pincode_df = pd.read_sql("SELECT city, pincode, state FROM pincode_master;", engine)
pincode_df.rename(columns={'city': 'division_name', 'state': 'state_name'}, inplace=True)
pincode_df['source'] = 'pincode_master'

# Combine both masters
combined_master = pd.concat([master_df, pincode_df], ignore_index=True).drop_duplicates()
combined_master.dropna(subset=['division_name', 'state_name', 'pincode'], inplace=True)
combined_master['country'] = 'India'
print(f"Loaded {len(combined_master)} combined master rows.")

# Load abbreviation list for input normalization
abbrev_df = pd.read_sql("SELECT state_abbreviation, state FROM abbreviation_list;", engine)
abbrev_map = dict(zip(abbrev_df['state_abbreviation'].str.upper(), abbrev_df['state']))

# Load input addresses
input_df = pd.read_sql("SELECT * FROM input_addresses;", engine)
print(f"Loaded {len(input_df)} input addresses.")

# Helper functions
def fuzzy_match(value, choices):
    if pd.isna(value) or str(value).strip() == "":
        return None, 0
    match = process.extractOne(str(value), choices, scorer=fuzz.token_sort_ratio)
    return (match[0], match[1] / 100) if match else (None, 0)

def compute_overall_confidence(pin_score, city_score, state_score):
    score = (0.4 * pin_score + 0.3 * city_score + 0.3 * state_score)
    if score >= 0.85: return "High"
    elif score >= 0.6: return "Medium"
    elif score >= 0.4: return "Low"
    else: return "Rejected"

def extract_components_from_address(address, known_cities, known_states):
    city = state = pincode = None
    pin_match = re.search(r'\b\d{6}\b', address)
    if pin_match:
        pincode = pin_match.group()
    for s in known_states:
        if s.lower() in address.lower():
            state = s
            break
    for c in known_cities:
        if c.lower() in address.lower():
            city = c
            break
    return city, state, pincode

def clean_address(address, components):
    for part in components:
        if part and part in address:
            address = address.replace(part, '')
    return address.strip()

validated = []
rejected = []
known_cities = combined_master['division_name'].dropna().unique()
known_states = combined_master['state_name'].dropna().unique()

for _, row in input_df.iterrows():
    addr_parts = [str(row.get("address1", "")), str(row.get("address2", "")), str(row.get("address3", ""))]
    combined_address = " ".join([p for p in addr_parts if p.strip()])

    city_in = str(row.get("city", "")).strip()
    state_in = str(row.get("state", "")).strip()
    pincode_in = str(row.get("pincode", "")).strip()
    country_in = str(row.get("country", "India")).strip()

    # Normalize state abbreviation
    state_in_full = abbrev_map.get(state_in.upper(), state_in)

    # Extract missing components from address fields
    city_auto, state_auto, pin_auto = extract_components_from_address(combined_address, known_cities, known_states)
    city_final = city_in if city_in else city_auto
    state_final = state_in_full if state_in_full else state_auto
    pincode_final = pincode_in if pincode_in else pin_auto

    # Clean address1
    address1_clean = clean_address(combined_address, [city_final, state_final, pincode_final, country_in])

    result = {
        "Address1": address1_clean,
        "City": city_final,
        "State": state_final,
        "Pincode": pincode_final,
        "Country": country_in,
        "City_Confidence": 0,
        "State_Confidence": 0,
        "Country_Confidence": 0,
        "Overall_Confidence": "Rejected",
        "Flag": None,
        "Source": None
    }

    pin_score = city_score = state_score = country_score = 0
    source = set()

    # Step 1: Pincode validation
    if pincode_final and pincode_final.isdigit():
        match = combined_master[combined_master["pincode"] == int(pincode_final)]
        if not match.empty:
            if len(match) > 1:
                result["Flag"] = "MULTIPLE_PIN_MATCHES"
            division = match.iloc[0]["division_name"]
            state_db = match.iloc[0]["state_name"]
            country_db = match.iloc[0]["country"]
            source.update(match['source'].unique())

            city_score = fuzz.token_sort_ratio(city_final.lower(), division.lower()) / 100 if city_final else 0
            state_score = fuzz.token_sort_ratio(state_final.lower(), state_db.lower()) / 100 if state_final else 0
            country_score = 1 if country_in.lower() == country_db.lower() else 0
            pin_score = 1

            result.update({
                "City_Confidence": city_score,
                "State_Confidence": state_score,
                "Country_Confidence": country_score
            })

            if city_score < 0.6 or state_score < 0.6:
                result["Flag"] = "PIN_MISMATCH"
        else:
            result["Flag"] = "INVALID_PIN"
    else:
        # Step 2: City validation
        best_div, city_score = fuzzy_match(city_final, combined_master["division_name"].unique())
        if best_div:
            possible_states = combined_master[combined_master["division_name"] == best_div]["state_name"].unique()
            source.update(combined_master[combined_master["division_name"] == best_div]['source'].unique())
            if state_final.lower() in [s.lower() for s in possible_states]:
                state_score = 1
                country_score = 1
                result.update({
                    "City_Confidence": city_score,
                    "State_Confidence": state_score,
                    "Country_Confidence": country_score
                })
            else:
                result["Flag"] = "AMBIGUOUS_CITY"
        else:
            result["Flag"] = "CITY_NOT_FOUND"

    result["Overall_Confidence"] = compute_overall_confidence(pin_score, city_score, state_score)
    result["Source"] = "Both" if len(source) > 1 else (list(source)[0] if source else None)

    validated.append(result)
    if result["Overall_Confidence"] == "Rejected":
        rejected.append(result)

# Save outputs
validated_df = pd.DataFrame(validated)
validated_df.to_excel("validated_output.xlsx", index=False)

if rejected:
    rejected_df = pd.DataFrame(rejected)
    rejected_df.to_excel("rejected_addresses.xlsx", index=False)

# Print summary
summary = validated_df["Overall_Confidence"].value_counts().to_dict()
print("Validation Summary:")
for level in ["High", "Medium", "Low", "Rejected"]:
    print(f"{level}: {summary.get(level, 0)}")

if rejected:
    print("Top 5 Rejected Addresses:")
    print(rejected_df[["Address1", "City", "State", "Pincode", "Flag"]].head(5))