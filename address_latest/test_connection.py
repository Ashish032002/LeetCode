from db_config import get_db_connection

engine = get_db_connection()
try:
    with engine.connect() as conn:
        print("Connected successfully to local SQLite database.")
except Exception as e:
    print("Connection failed:", e)
    