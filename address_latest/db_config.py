from sqlalchemy import create_engine
import os
def get_db_connection():
    # Create DB file inside project directory
    db_path = os.path.join(os.getcwd(), "address_validation.db")
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return engine