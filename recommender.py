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

# Step 4: Define user profiles

# Example user profiles (replace with actual user data)
users = [
    {
        "country": "USA",
        "activities": {
            "running": 120,  # Number of running sessions
            "hiking": 15
        },
        "attended_events": ["Devcon"],
        "coinbase_one": True
    },
    {
        "country": "Canada",
        "activities": {
            "cycling": 30,
            "yoga": 20
        },
        "attended_events": ["HealthNY", "Classic Cars 2024"],
        "coinbase_one": False
    },
]

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

# Preprocess user profiles
user_profiles = [
    f"{user['country']} " +
    ' '.join([categorize_sessions(activity, sessions) for activity, sessions in user['activities'].items()]) + ' ' +
    ' '.join(user['attended_events']) + ' ' +
    ('coinbaseone' if user['coinbase_one'] else 'nocoinbaseone')
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
