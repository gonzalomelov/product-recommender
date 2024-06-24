import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Updated sample data (replace with your actual data)
products = [
    {"title": "Running Shoes", "description": "High-quality running shoes for athletes", "tags": "running sports shoes"},
    {"title": "Yoga Mat", "description": "Eco-friendly yoga mat with non-slip surface", "tags": "yoga fitness mat"},
    {"title": "Cycling Helmet", "description": "Safety helmet for cycling enthusiasts", "tags": "cycling safety gear"},
    {"title": "Hiking Backpack", "description": "Durable backpack for hiking adventures", "tags": "hiking outdoors gear"},
    {"title": "Car Helmet", "description": "Durable helmet for your racing car", "tags": "racing car"},
]

users = [
    {"country": "USA", "activities": "running hiking", "attended_events": ["Devcon"]},
    {"country": "Canada", "activities": "cycling yoga", "attended_events": ["HealthNY", "Classic Cars 2024"]},
]

# Extract features from products and users, including attended events
product_descriptions = [prod['description'] for prod in products]
user_profiles = [
    f"{user['country']} {user['activities']} {' '.join(user['attended_events'])}"
    for user in users
]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
product_tfidf = vectorizer.fit_transform(product_descriptions)
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
        product = products[idx]
        similarity_score = user_similarities[idx]
        print(f"- {product['title']} (Similarity Score: {similarity_score:.2f})")
    print()
