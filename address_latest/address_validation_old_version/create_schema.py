from sqlalchemy import text
from db_config import get_db_connection

def main():
    eng=get_db_connection()
    with eng.begin() as con:
        con.execute(text('CREATE SCHEMA IF NOT EXISTS av'))
        con.execute(text('CREATE TABLE IF NOT EXISTS av.master_ref(city TEXT,state TEXT,pincode TEXT)'))
        con.execute(text('CREATE TABLE IF NOT EXISTS av.countries(name TEXT PRIMARY KEY)'))
        con.execute(text('CREATE TABLE IF NOT EXISTS av.pincode_ref(city TEXT,state TEXT,pincode TEXT)'))
        con.execute(text('CREATE TABLE IF NOT EXISTS av.input_addresses(id BIGSERIAL PRIMARY KEY,address1 TEXT,address2 TEXT,address3 TEXT,city TEXT,state TEXT,pincode TEXT)'))
        con.execute(text('DROP TABLE IF EXISTS av.validation_result_final CASCADE'))
        con.execute(text('''CREATE TABLE av.validation_result_final(
            out_id BIGSERIAL PRIMARY KEY,
            input_id BIGINT,
            address1 TEXT, city TEXT, state TEXT, pincode TEXT, country TEXT, country_confidence NUMERIC,
            city_confidence NUMERIC, state_confidence NUMERIC, overall_confidence NUMERIC, confidence_level TEXT, 
            flag TEXT, reason TEXT, ambiguity_type TEXT, source_used TEXT,
            master_city TEXT, master_state TEXT, pincode_city TEXT, pincode_state TEXT,
            locality TEXT, possible_addresses JSONB,
            in_address1 TEXT, in_address2 TEXT, in_address3 TEXT, in_city TEXT, in_state TEXT, in_pincode TEXT
        )'''))
    print('Schema ready.')

if __name__=='__main__': main()
