import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data (replace with your actual data)
products = [
    {"title": "Product A", "description": "Description of Product A", "tags": "tag1 tag2"},
    {"title": "Product B", "description": "Description of Product B", "tags": "tag2 tag3"},
    {"title": "Product C", "description": "Description of Product C", "tags": "tag1 tag3"},
]

users = [
    {"country": "USA", "activities": "running hiking", "attended_event": "yes"},
    {"country": "Canada", "activities": "cycling yoga", "attended_event": "no"},
]

# Extract features from products and users
product_descriptions = [prod['description'] for prod in products]
user_profiles = [
    f"{user['country']} {user['activities']} {user['attended_event']}"
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
