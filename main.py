import time
import pandas as pd
from dotenv import load_dotenv
from db_config import get_postgres_connection, get_mysql_connection
from user_profiles import create_user_profiles
from product_recommendation import recommend_products

# Load environment variables from .env file
load_dotenv()

# Connect to the databases
conn_pg, cur_pg = get_postgres_connection()
conn_mysql, cur_mysql = get_mysql_connection()

# Read the CSV data
attestations_csv = pd.read_csv('receipts-attestations.csv')

# Create user profiles
create_user_profiles_start_time = time.time()
users = create_user_profiles(cur_pg, attestations_csv)
create_user_profiles_end_time = time.time()
print(f"Total time to run create_user_profiles: {create_user_profiles_end_time - create_user_profiles_start_time:.2f} seconds")

# Get products and recommend
recommend_products_start_time = time.time()
product_data = recommend_products(cur_mysql, conn_mysql, users)
recommend_products_end_time = time.time()
print(f"Total time to run recommend_products: {recommend_products_end_time - recommend_products_start_time:.2f} seconds")

# Clean up database connections
cur_pg.close()
conn_pg.close()
cur_mysql.close()
conn_mysql.close()
