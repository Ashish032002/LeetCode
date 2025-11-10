from db_config import get_db_connection
import pandas as pd

engine = get_db_connection()
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", engine)
print(tables)