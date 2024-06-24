# Install gdown if not already installed
!pip install gdown

import gdown

# Download the product data file from Google Drive
product_file_id = '1KLZGUwL0g86QvY5o9Zmyml_Qo1aqgtK9'  # Replace with your actual file ID
product_url = f'https://drive.google.com/uc?id={product_file_id}'
product_output = 'products_export_1.csv'
gdown.download(product_url, product_output, quiet=False)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the data from CSV
df = pd.read_csv('products_export_1.csv')

# Select relevant columns for product information
product_data = df[['Handle', 'Title', 'Body (HTML)', 'Tags']]

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

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
product_tfidf = vectorizer.fit_transform(product_data['combined_text'].values.astype('U'))

# Example user profiles (replace with actual user data)
users = [
    {"country": "USA", "activities": "running hiking", "attended_events": ["Devcon"]},
    {"country": "Canada", "activities": "cycling yoga", "attended_events": ["HealthNY", "Classic Cars 2024"]},
]

# Preprocess user profiles
user_profiles = [
    f"{user['country']} {user['activities']} {' '.join(user['attended_events'])}"
    for user in users
]

# TF-IDF transformation for user profiles
user_tfidf = vectorizer.transform(user_profiles)

# Compute cosine similarity between user profile and each product
similarities = cosine_similarity(user_tfidf, product_tfidf)

# Recommend products based on highest similarity
for i, user in enumerate(users):
    print(f"Recommendations for User {i+1}:")
    user_similarities = similarities[i]
    # Sort product indices by similarity score (descending order)
    sorted_indices = np.argsort(user_similarities)[::-1]
    for idx in sorted_indices:
        product = product_data.iloc[idx]
        similarity_score = user_similarities[idx]
        print(f"- {product['Title']} (Similarity Score: {similarity_score:.2f})")
    print()
