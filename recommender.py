# Jupyter Notebook: Product Recommendation System

# Importing necessary libraries

# Install gdown if not already installed
!pip install gdown

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Step 1: Load and inspect data

import gdown

# Download the product data file from Google Drive
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
    # Replace "Coinbase One" with "coinbaseone" before further cleaning
    text = text.replace("Coinbase One", "coinbaseone")
    # Remove HTML tags using regex
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

# Clean 'Body (HTML)' column
product_data['cleaned_body'] = product_data['Body (HTML)'].apply(clean_text)

# Combine relevant text columns for TF-IDF vectorization
product_data['combined_text'] = product_data['Title'] + ' ' + product_data['cleaned_body'] + ' ' + product_data['Tags'].fillna('')

# Step 3: Vectorize text data (TF-IDF)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
product_tfidf = vectorizer.fit_transform(product_data['combined_text'].values.astype('U'))

# Check the shape of the product TF-IDF matrix
print("Shape of product TF-IDF matrix:", product_tfidf.shape)

# Step 4: Define user profiles

# Example user profiles (replace with actual user data)
users = [
    {
        "country": "USA",
        "activities": ["running", "hiking"],
        "attended_events": ["Devcon"],
        "coinbase_one": True
    },
    {
        "country": "Canada",
        "activities": ["cycling", "yoga"],
        "attended_events": ["HealthNY", "Classic Cars 2024"],
        "coinbase_one": False
    },
]

# Preprocess user profiles
user_profiles = [
    f"{user['country']} {' '.join(user['activities'])} {' '.join(user['attended_events'])} {'coinbaseone' if user['coinbase_one'] else 'nocoinbaseone'}"
    for user in users
]

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

# Recommend products based on highest similarity
for i, user in enumerate(users):
    print(f"Recommendations for User {i+1}:")
    user_similarities = similarities[i]
    # Sort product indices by similarity score (descending order)
    sorted_indices = np.argsort(user_similarities)[::-1]
    for idx in sorted_indices[:5]:  # Adjust the number of recommendations as needed
        product = product_data.iloc[idx]
        similarity_score = user_similarities[idx]
        print(f"- {product['Title']} (Similarity Score: {similarity_score:.2f})")
    print()
