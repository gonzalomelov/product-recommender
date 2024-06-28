import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import psycopg2
import subprocess
import sys
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

# Database connection details
db_params = {
    'dbname': 'eas-index',
    'user': 'postgres',
    'password': 'postgresPassword',
    'host': '127.0.0.1',
    'port': '5432'
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Function to get all wallet attestations from the database
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
    cur.execute(query)
    return cur.fetchall()

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

# Function to create user profiles based on attestations
def create_user_profiles():
    attestations = get_all_wallet_attestations()
    profiles = {}

    for recipient, schema_id, count, decoded_data_json in attestations:
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
    
    return list(profiles.values())

# Create user profiles
users = create_user_profiles()

print("User Profiles:")
for user in users:
    print(user)

# Clean up
cur.close()
conn.close()

# Download the product data file from Google Drive
import gdown
product_file_id = '1KLZGUwL0g86QvY5o9Zmyml_Qo1aqgtK9'  # Replace with your actual file ID
product_url = f'https://drive.google.com/uc?id={product_file_id}'
product_output = 'products_export_1.csv'
gdown.download(product_url, product_output, quiet=False)

# Load the data from CSV
product_df = pd.read_csv('products_export_1.csv')

# Step 2: Prepare data for analysis

# Select relevant columns for product information
product_data = product_df[['Handle', 'Title', 'Body (HTML)', 'Tags']]

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

# Clean 'Body (HTML)' column
product_data['Body (HTML)'] = product_data['Body (HTML)'].astype(str)
product_data['cleaned_body'] = product_data['Body (HTML)'].apply(clean_text)

# Combine relevant text columns for TF-IDF vectorization
product_data['combined_text'] = product_data['Title'] + ' ' + product_data['cleaned_body'] + ' ' + product_data['Tags'].fillna('')

# Convert combined_text to string type
product_data['combined_text'] = product_data['combined_text'].astype(str)

# Define keywords and phrases for activity categories
activity_keywords = {
    "running_100": ["for frequent runners", "for daily running", "ideal for daily runners", "run every day"],
    "running_50": ["run multiple times a week", "run several times a week"],
    "running_10": ["run weekly", "for weekly running"],
    "running_5": ["occasional running", "run a few times a month"],
    "running_1": ["run occasionally"]
}

# Function to infer activity category from text
def infer_activity_category(text, keywords):
    for category, phrases in keywords.items():
        for phrase in phrases:
            if phrase in text.lower():
                return category
    return "running_1"  # Default category if no match found

# Infer activity categories for products
product_data['activity_category'] = product_data['combined_text'].apply(lambda x: infer_activity_category(x, activity_keywords))

# Update combined text with inferred activity category
product_data['combined_text'] = product_data['combined_text'] + ' ' + product_data['activity_category']

# Step 3: Vectorize text data (TF-IDF)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
product_tfidf = vectorizer.fit_transform(product_data['combined_text'].values.astype('U'))

# Check the shape of the product TF-IDF matrix
print("Shape of product TF-IDF matrix:", product_tfidf.shape)

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
    return activity  # Keep original activity for non-running activities

# Preprocess user profiles for TF-IDF vectorization
def preprocess_user_profile(user):
    profile_text = user['country_code'] + ' ' + user['country'] + ' '
    profile_text += ' '.join([categorize_sessions(activity, sessions) for activity, sessions in user['activities'].items() if sessions > 0]) + ' '
    profile_text += ' '.join(user['attended_events']) + ' '
    profile_text += 'coinbaseone' if user['coinbase_one'] else 'nocoinbaseone'
    return profile_text

user_profiles = [preprocess_user_profile(user) for user in users]

# Verify user profiles
print("User Profiles:")
for profile in user_profiles:
    print(profile)

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

# Recommend products based on highest similarity
for i, user in enumerate(users):
    print(f"Recommendations for User {i+1}:")
    user_similarities = similarities[i]
    # Sort product indices by similarity score (descending order)
    sorted_indices = np.argsort(user_similarities)[::-1]
    for idx in sorted_indices:
        similarity_score = user_similarities[idx]
        if similarity_score >= similarity_threshold:  # Only recommend if above threshold
            product = product_data.iloc[idx]
            print(f"- {product['Title']} (Similarity Score: {similarity_score:.2f})")
    print()
