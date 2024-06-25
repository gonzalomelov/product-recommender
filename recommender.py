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

# Display the first few rows to understand the structure
product_df.head()

# Step 2: Prepare data for analysis

# Select relevant columns for product information
product_data = product_df[['Handle', 'Title', 'Body (HTML)', 'Tags']]

# Function to clean text data
def clean_text(text):
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
    f"{user['country']} {' '.join(user['activities'])} {' '.join(user['attended_events'])} {'coinbase_one' if user['coinbase_one'] else 'no_coinbase_one'}"
    for user in users
]

# TF-IDF transformation for user profiles
user_tfidf = vectorizer.transform(user_profiles)

# Step 5: Calculate similarities

# Compute cosine similarity between user profiles and product descriptions
similarities = cosine_similarity(user_tfidf, product_tfidf)

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
