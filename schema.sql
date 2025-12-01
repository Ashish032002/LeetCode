
-- ==============================
-- Create Schemas
-- ==============================
CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS input;
CREATE SCHEMA IF NOT EXISTS output;

-- ==============================
-- Reference Tables
-- ==============================

CREATE TABLE IF NOT EXISTS ref.countries (
    id BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS ref.world_cities (
    id BIGSERIAL PRIMARY KEY,
    city TEXT,
    country TEXT,
    iso2 TEXT,
    iso3 TEXT
);

CREATE TABLE IF NOT EXISTS ref.t30_cities (
    id BIGSERIAL PRIMARY KEY,
    city TEXT,
    country TEXT DEFAULT 'India'
);

CREATE TABLE IF NOT EXISTS ref.indian_state_abbrev (
    id BIGSERIAL PRIMARY KEY,
    state TEXT,
    abbreviation TEXT
);


CREATE TABLE IF NOT EXISTS ref.postal_pincode (
    id BIGSERIAL PRIMARY KEY,
    city TEXT,
    state TEXT,
    pincode TEXT,
    country TEXT DEFAULT 'India'
);

CREATE TABLE IF NOT EXISTS ref.rta_pincode (
    id BIGSERIAL PRIMARY KEY,
    city TEXT,
    state TEXT,
    pincode TEXT,
    country TEXT DEFAULT 'India'
);

-- ==============================
-- Input Tables
-- ==============================

CREATE TABLE IF NOT EXISTS input.addresses (
    id BIGSERIAL PRIMARY KEY,
    address1 TEXT,
    address2 TEXT,
    address3 TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    pincode TEXT,
    raw_text TEXT
);

-- ==============================
-- Output Tables
-- ==============================

CREATE TABLE IF NOT EXISTS output.validation_result (
    id BIGSERIAL PRIMARY KEY,
    input_id INTEGER,
    output_pincode TEXT,
    output_city TEXT,
    output_state TEXT,
    output_country TEXT,
    t30_city_possible INTEGER,
    foreign_country_possible INTEGER,
    pincode_found INTEGER,
    ambiguous_address_flag INTEGER,
    all_possible_countries TEXT,
    all_possible_states TEXT,
    all_possible_cities TEXT,
    all_possible_pincodes TEXT,
    city_value_match REAL,
    city_consistency_with_pincode REAL,
    city_ambiguity_penalty REAL,
    state_value_match REAL,
    state_consistency_with_pincode REAL,
    state_ambiguity_penalty REAL,
    country_value_match REAL,
    country_consistency_with_pincode REAL,
    country_ambiguity_penalty REAL,
    overall_score REAL,
    reason TEXT,
    local_address TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);




-- ===========================================================
-- End of Schema Creation
-- ===========================================================

