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
attestations_csv = pd.read_csv('attestations.csv')

# Create user profiles
users = create_user_profiles(cur_pg, attestations_csv)

# Get products and recommend
product_data = recommend_products(cur_mysql, conn_mysql, users)

# Clean up database connections
cur_pg.close()
conn_pg.close()
cur_mysql.close()
conn_mysql.close()
