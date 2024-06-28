import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import psycopg2
import pymysql
import subprocess
import sys
import os
from dotenv import load_dotenv
from country_codes import country_code_to_name  # Import the dictionary

# Function to check if running in a Jupyter notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (likely not a notebook)
    except NameError:
        return False  # Definitely not a notebook

# Install packages if running in a notebook
if is_notebook():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymysql"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])

# Load environment variables from .env file
load_dotenv()

# Fetching PostgreSQL connection details from environment variables
db_params = {
    'dbname': os.getenv('PG_DBNAME'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST'),
    'port': os.getenv('PG_PORT')
}

# Fetching MySQL connection details from environment variables
mysql_params = {
    'host': os.getenv('MYSQL_HOST'),
    'port': os.getenv('MYSQL_PORT'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE')
}

# Connect to the PostgreSQL database
conn_pg = psycopg2.connect(**db_params)
cur_pg = conn_pg.cursor()

# Connect to the MySQL database
conn_mysql = pymysql.connect(**mysql_params)
cur_mysql = conn_mysql.cursor()

# Read the CSV data
attestations_csv = pd.read_csv('attestations.csv')
coinbaseone_csv = pd.read_csv('coinbaseone.csv')

# Function to get all wallet attestations from both PostgreSQL and CSV data
def get_all_wallet_attestations():
    query = """
    SELECT "recipient", "schemaId", COUNT(*) as count, MAX("decodedDataJson") as decodedDataJson
    FROM "Attestation"
    WHERE "schemaId" IN (
        '0x1801901fabd0e6189356b4fb52bb0ab855276d84f7ec140839fbd1f6801ca065',
        '0x0f5b217904f3c65ad40b7af3db62716daddf53bb5db04b1a3ddb730fda0a474b',
        '0xf8b05c79f090979bf4a80270aba232dff11a10d9ca55c4f88de95317970f0de9',
        '0x254bd1b63e0591fefa66818ca054c78627306f253f86be6023725a67ee6bf9f4'
    )
    GROUP BY "recipient", "schemaId";
    """
    cur_pg.execute(query)
    pg_attestations = cur_pg.fetchall()
    
    csv_attestations = attestations_csv[['recipient', 'schema.id', 'decodedDataJson']].values.tolist()
    coinbaseone_attestations = coinbaseone_csv[['recipient', 'schema.id', 'decodedDataJson']].values.tolist()
    
    return pg_attestations, csv_attestations, coinbaseone_attestations

def extract_country_from_json(decoded_data_json):
    try:
        data = json.loads(decoded_data_json)
        for item in data:
            if item['name'] == 'verifiedCountry':
                country_code = item['value']['value'].upper()
                country_name = country_code_to_name.get(country_code, country_code)
                return country_code, country_name
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return "", ""

# Function to create user profiles based on attestations from both sources
def create_user_profiles():
    pg_attestations, csv_attestations, coinbaseone_attestations = get_all_wallet_attestations()
    profiles = {}

    # Process PostgreSQL attestations
    for recipient, schema_id, count, decoded_data_json in pg_attestations:
        if recipient not in profiles:
            profiles[recipient] = {
                "wallet": recipient,
                "country_code": "",
                "country": "",
                "activities": {
                    "running": 0
                },
                "attended_events": [],
                "coinbase": False,
                "coinbase_one": False
            }
        
        profile = profiles[recipient]
        if schema_id == '0x1801901fabd0e6189356b4fb52bb0ab855276d84f7ec140839fbd1f6801ca065':
            country_code, country_name = extract_country_from_json(decoded_data_json)
            profile["country_code"] = country_code
            profile["country"] = country_name
        elif schema_id == '0x0f5b217904f3c65ad40b7af3db62716daddf53bb5db04b1a3ddb730fda0a474b':
            profile["activities"]["running"] += count
        elif schema_id == '0xf8b05c79f090979bf4a80270aba232dff11a10d9ca55c4f88de95317970f0de9':
            profile["coinbase"] = True
        elif schema_id == '0x254bd1b63e0591fefa66818ca054c78627306f253f86be6023725a67ee6bf9f4':
            profile["coinbase_one"] = True

    # Process CSV attestations
    for recipient, schema_id, decoded_data_json in csv_attestations:
        if recipient not in profiles:
            profiles[recipient] = {
                "wallet": recipient,
                "country_code": "",
                "country": "",
                "activities": {
                    "running": 0
                },
                "attended_events": [],
                "coinbase": False,
                "coinbase_one": False
            }
        
        profile = profiles[recipient]
        if schema_id == '0x1801901fabd0e6189356b4fb52bb0ab855276d84f7ec140839fbd1f6801ca065':
            country_code, country_name = extract_country_from_json(decoded_data_json)
            profile["country_code"] = country_code
            profile["country"] = country_name
        elif schema_id == '0x0f5b217904f3c65ad40b7af3db62716daddf53bb5db04b1a3ddb730fda0a474b':
            profile["activities"]["running"] += 1  # Increment by 1 as count is not present in CSV
        elif schema_id == '0xf8b05c79f090979bf4a80270aba232dff11a10d9ca55c4f88de95317970f0de9':
            profile["coinbase"] = True
        elif schema_id == '0x254bd1b63e0591fefa66818ca054c78627306f253f86be6023725a67ee6bf9f4':
            profile["coinbase_one"] = True

    # Process Coinbase One CSV attestations
    for recipient, schema_id, decoded_data_json in coinbaseone_attestations:
        if recipient not in profiles:
            profiles[recipient] = {
                "wallet": recipient,
                "country_code": "",
                "country": "",
                "activities": {
                    "running": 0
                },
                "attended_events": [],
                "coinbase": False,
                "coinbase_one": False
            }
        
        profile = profiles[recipient]
        if schema_id == '0x254bd1b63e0591fefa66818ca054c78627306f253f86be6023725a67ee6bf9f4':
            profile["coinbase_one"] = True
    
    return list(profiles.values())

# Create user profiles
users = create_user_profiles()

print("User Profiles:")
for user in users:
    print(user)

# Clean up PostgreSQL connection
cur_pg.close()
conn_pg.close()

# Function to get all products from the MySQL database
def get_all_products():
    query = """
    SELECT id, title, description, shop, handle, variantId, variantFormattedPrice, alt, image, createdAt
    FROM Product;
    """
    cur_mysql.execute(query)
    return cur_mysql.fetchall()

# Fetch products from MySQL database
products = get_all_products()

# Convert product data to a DataFrame
product_df = pd.DataFrame(products, columns=['id', 'title', 'description', 'shop', 'handle', 'variantId', 'variantFormattedPrice', 'alt', 'image', 'createdAt'])

# Step 2: Prepare data for analysis

# Select relevant columns for product information
product_data = product_df[['id', 'handle', 'title', 'description', 'alt']].copy()

# Display the first few rows to understand the structure
print(product_data.head())

# Function to clean text data
def clean_text(text):
    if isinstance(text, float):
        text = str(text)
    # Replace "Coinbase One" with "coinbaseone" before further cleaning
    text = text.replace("Coinbase One", "coinbaseone")
    # Remove HTML tags using regex
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

# Clean 'description' column
product_data.loc[:, 'description'] = product_data['description'].astype(str)
product_data.loc[:, 'cleaned_description'] = product_data['description'].apply(clean_text)

# Combine relevant text columns for TF-IDF vectorization
product_data.loc[:, 'combined_text'] = product_data['title'] + ' ' + product_data['cleaned_description'] + ' ' + product_data['alt'].fillna('')

# Convert combined_text to string type
product_data.loc[:, 'combined_text'] = product_data['combined_text'].astype(str)

# Define keywords and phrases for activity categories
activity_keywords = {
    "running_100": ["for frequent runners", "for daily running", "ideal for daily runners", "run every day"],
    "running_50": ["run multiple times a week", "run several times a week"],
    "running_10": ["run weekly", "for weekly running"],
    "running_5": ["occasional running", "run a few times a month"],
    "running_1": ["run occasionally"],
    "running_0": ["sedentary"]
}

# Function to infer activity category from text
def infer_activity_category(text, keywords):
    for category, phrases in keywords.items():
        for phrase in phrases:
            if phrase in text.lower():
                return category
    return "running_0" 

# Infer activity categories for products
product_data.loc[:, 'activity_category'] = product_data['combined_text'].apply(lambda x: infer_activity_category(x, activity_keywords))

# Update combined text with inferred activity category
product_data.loc[:, 'combined_text'] = product_data['combined_text'] + ' ' + product_data['activity_category']

# Step 3: Vectorize text data (TF-IDF)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
product_tfidf = vectorizer.fit_transform(product_data['combined_text'].values.astype('U'))

# Check the shape of the product TF-IDF matrix
print("Shape of product TF-IDF matrix:", product_tfidf.shape)

# Print the words for each product
feature_names = vectorizer.get_feature_names_out()
for i, product_text in enumerate(product_data['combined_text']):
    print(f"Product {i+1} words:")
    tfidf_scores = zip(feature_names, product_tfidf[i].toarray()[0])
    sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    for word, score in sorted_tfidf_scores:
        if score > 0:
            print(f"{word}: {score:.4f}")
    print()

# Function to categorize running sessions
def categorize_sessions(activity, sessions):
    if activity == "running":
        if sessions > 100:
            return "running_100"
        elif sessions > 50:
            return "running_50"
        elif sessions > 10:
            return "running_10"
        elif sessions > 5:
            return "running_5"
        elif sessions > 0:
            return "running_1"
        elif sessions == 0:
            return "running_0"
    return activity  # Keep original activity for non-running activities

# Preprocess user profiles for TF-IDF vectorization
def preprocess_user_profile(user):
    profile_text = user['country_code'] + ' ' + user['country'] + ' '
    profile_text += ' '.join([categorize_sessions(activity, sessions) for activity, sessions in user['activities'].items() if sessions > 0]) + ' '
    profile_text += ' '.join(user['attended_events']) + ' '
    profile_text += 'coinbaseone' if user['coinbase_one'] else 'nocoinbaseone'
    profile_text += ' coinbase' if user['coinbase'] else ' nocoinbase'
    return profile_text

user_profiles_with_wallets = [(user['wallet'], preprocess_user_profile(user)) for user in users]

# Verify user profiles
print("User Profiles:")
for wallet, profile in user_profiles_with_wallets:
    print(f"Wallet: {wallet}, Profile: {profile}")

# Extract only the profile texts for TF-IDF vectorization
user_profiles = [profile for _, profile in user_profiles_with_wallets]

# TF-IDF transformation for user profiles
user_tfidf = vectorizer.transform(user_profiles)

# Check the shape of the user TF-IDF matrix
print("Shape of user TF-IDF matrix:", user_tfidf.shape)

# Verify presence of "coinbaseone" in TF-IDF vocabulary
print("TF-IDF Vocabulary:")
print(vectorizer.vocabulary_)

# Step 5: Calculate similarities

# Compute cosine similarity between user profiles and product descriptions
similarities = cosine_similarity(user_tfidf, product_tfidf)

# Debug: Check similarity scores
print("Similarity Scores:")
print(similarities)

# Step 6: Recommend products

# Set a threshold for similarity
similarity_threshold = 0.25  # Adjust this value as needed

# Function to store recommendations in MySQL database
def store_recommendations(user_id, recommendations, frame_id):
    if any(recommendations):  # Only store if there's at least one valid recommendation
        values = [(user_id, frame_id, recommendations[0], recommendations[1], recommendations[2])]
        insert_query = """
        INSERT INTO UserProduct (walletAddress, frameId, productId1, productId2, productId3)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            productId1 = VALUES(productId1),
            productId2 = VALUES(productId2),
            productId3 = VALUES(productId3)
        """
        cur_mysql.executemany(insert_query, values)
        conn_mysql.commit()

# Recommend products and store in MySQL database
for i, (wallet, _) in enumerate(user_profiles_with_wallets):
    user_similarities = similarities[i]
    # Sort product indices by similarity score (descending order)
    sorted_indices = np.argsort(user_similarities)[::-1]
    recommendations = []
    for idx in sorted_indices[:3]:  # Get top 3 recommendations
        similarity_score = user_similarities[idx]
        if similarity_score >= similarity_threshold:  # Only recommend if above threshold
            product = product_data.iloc[idx]
            recommendations.append(product['id'])
    # Ensure recommendations list has exactly 3 items
    while len(recommendations) < 3:
        recommendations.append(None)
    if any(recommendations):  # Only store if there's at least one valid recommendation
        store_recommendations(wallet, recommendations, i + 1)
        
        print(f"Recommendations for User {i+1} (Wallet: {wallet}):")
        print(f"- {recommendations} (Top 3 recommendations)")

# Clean up MySQL connection
cur_mysql.close()
conn_mysql.close()
