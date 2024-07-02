import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import clean_text, infer_activity_category, categorize_sessions

def get_all_products(cur_mysql):
    query = """
    SELECT id, title, description, shop, handle, variantId, variantFormattedPrice, alt, image, createdAt
    FROM Product;
    """
    cur_mysql.execute(query)
    return cur_mysql.fetchall()

def store_recommendations(cur_mysql, conn_mysql, user_id, recommendations, frame_id):
    if any(recommendations):
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

def recommend_products(cur_mysql, conn_mysql, users):
    # Fetch products from MySQL database
    products = get_all_products(cur_mysql)

    # Convert product data to a DataFrame
    product_df = pd.DataFrame(products, columns=['id', 'title', 'description', 'shop', 'handle', 'variantId', 'variantFormattedPrice', 'alt', 'image', 'createdAt'])

    # Select relevant columns for product information
    product_data = product_df[['id', 'handle', 'title', 'description', 'alt']].copy()

    # Display the first few rows to understand the structure
    print(product_data.head())

    # Clean 'description' column
    product_data['description'] = product_data['description'].astype(str)
    product_data['cleaned_description'] = product_data['description'].apply(clean_text)

    # Combine relevant text columns for TF-IDF vectorization
    product_data['combined_text'] = product_data['title'] + ' ' + product_data['cleaned_description'] + ' ' + product_data['alt'].fillna('')

    # Define keywords and phrases for activity categories
    activity_keywords = {
        "running_100": ["for frequent runners", "for daily running", "ideal for daily runners", "run every day"],
        "running_50": ["run multiple times a week", "run several times a week"],
        "running_10": ["run weekly", "for weekly running"],
        "running_5": ["occasional running", "run a few times a month"],
        "running_1": ["run occasionally"],
        "running_0": ["sedentary"]
    }

    # Infer activity categories for products
    product_data['activity_category'] = product_data['combined_text'].apply(lambda x: infer_activity_category(x, activity_keywords))

    # Update combined text with inferred activity category
    product_data['combined_text'] = product_data['combined_text'] + ' ' + product_data['activity_category']

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

    # Compute cosine similarity between user profiles and product descriptions
    similarities = cosine_similarity(user_tfidf, product_tfidf)

    # Debug: Check similarity scores
    print("Similarity Scores:")
    print(similarities)

    # Set a threshold for similarity
    similarity_threshold = 0.25  # Adjust this value as needed

    # Recommend products and store in MySQL database
    for i, (wallet, _) in enumerate(user_profiles_with_wallets):
        user_similarities = similarities[i]
        sorted_indices = np.argsort(user_similarities)[::-1]
        recommendations = []
        for idx in sorted_indices[:3]:
            similarity_score = user_similarities[idx]
            if similarity_score >= similarity_threshold:
                product = product_data.iloc[idx]
                recommendations.append(product['id'])
        while len(recommendations) < 3:
            recommendations.append(None)
        if any(recommendations):
            store_recommendations(cur_mysql, conn_mysql, wallet, recommendations, i + 1)
            
            print(f"Recommendations for User {i+1} (Wallet: {wallet}):")
            print(f"- {recommendations} (Top 3 recommendations)")

    return product_data
