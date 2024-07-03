import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import clean_text, infer_activity_category, categorize_sessions

def get_all_frames(cur_mysql):
    query = """
    SELECT id, title, shop, matchingCriteria
    FROM Frame
    WHERE matchingCriteria = 'ALL';
    """
    cur_mysql.execute(query)
    return cur_mysql.fetchall()

def get_all_products(cur_mysql):
    query = """
    SELECT id, title, description, shop, handle, variantId, variantFormattedPrice, alt, image, createdAt
    FROM Product;
    """
    cur_mysql.execute(query)
    return cur_mysql.fetchall()

def store_recommendations_batch(cur_mysql, conn_mysql, recommendations_batch):
    if recommendations_batch:
        upsert_query = """
        INSERT INTO UserProduct (walletAddress, frameId, productId1, productId2, productId3)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            productId1 = VALUES(productId1),
            productId2 = VALUES(productId2),
            productId3 = VALUES(productId3)
        """
        cur_mysql.executemany(upsert_query, recommendations_batch)
        conn_mysql.commit()

def recommend_products(cur_mysql, conn_mysql, users):
    # Fetch frames from MySQL database
    frames = get_all_frames(cur_mysql)
    frames_df = pd.DataFrame(frames, columns=['id', 'title', 'shop', 'matchingCriteria'])
    print("Frames fetched from database:", frames_df)  # Debugging line

    # Fetch products from MySQL database
    products = get_all_products(cur_mysql)
    products_df = pd.DataFrame(products, columns=['id', 'title', 'description', 'shop', 'handle', 'variantId', 'variantFormattedPrice', 'alt', 'image', 'createdAt'])
    print("Products fetched from database:", products_df)  # Debugging line

    # Define keywords and phrases for activity categories
    activity_keywords = {
        "running_100": ["for frequent runners", "for daily running", "ideal for daily runners", "run every day"],
        "running_50": ["run multiple times a week", "run several times a week"],
        "running_10": ["run weekly", "for weekly running"],
        "running_5": ["occasional running", "run a few times a month"],
        "running_1": ["run occasionally"],
        "running_0": ["sedentary"]
    }

    for _, frame in frames_df.iterrows():
        frame_start_time = time.time()
        frame_shop = frame['shop']
        print(f"Processing frame ID: {frame['id']} with shop: {frame_shop}")

        # Filter products by the frame's shop
        frame_products = products_df[products_df['shop'] == frame_shop].copy()

        # Select relevant columns for product information
        product_data = frame_products[['id', 'handle', 'title', 'description', 'alt']].copy()

        # Clean 'description' column
        product_data['description'] = product_data['description'].astype(str)
        product_data['cleaned_description'] = product_data['description'].apply(clean_text)

        # Combine relevant text columns for TF-IDF vectorization
        product_data['combined_text'] = product_data['title'] + ' ' + product_data['cleaned_description'] + ' ' + product_data['alt'].fillna('')

        # Infer activity categories for products
        product_data['activity_category'] = product_data['combined_text'].apply(lambda x: infer_activity_category(x, activity_keywords))
        
        # Update combined text with inferred activity category
        product_data['combined_text'] = product_data['combined_text'] + ' ' + product_data['activity_category']

        # TF-IDF vectorization
        tfidf_vectorization_start = time.time()
        vectorizer = TfidfVectorizer(stop_words='english')
        product_tfidf = vectorizer.fit_transform(product_data['combined_text'].values.astype('U'))
        print(f"Time for TF-IDF vectorization for frame ID {frame['id']}: {time.time() - tfidf_vectorization_start:.2f} seconds")

        # Check the shape of the product TF-IDF matrix
        print(f"Shape of product TF-IDF matrix for frame ID {frame['id']}:", product_tfidf.shape)

        # Print the words for each product
        feature_names = vectorizer.get_feature_names_out()
        for i, product_text in enumerate(product_data['combined_text']):
            print(f"Product {i+1} words in frame ID {frame['id']}:")
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
        user_tfidf_start = time.time()
        user_profiles = [profile for _, profile in user_profiles_with_wallets]
        # TF-IDF transformation for user profiles
        user_tfidf = vectorizer.transform(user_profiles)
        print(f"Time for TF-IDF transformation of user profiles: {time.time() - user_tfidf_start:.2f} seconds")

        # Check the shape of the user TF-IDF matrix
        print(f"Shape of user TF-IDF matrix for frame ID {frame['id']}:", user_tfidf.shape)

        # Verify presence of "coinbaseone" in TF-IDF vocabulary
        print("TF-IDF Vocabulary:")
        print(vectorizer.vocabulary_)

        # Compute cosine similarity between user profiles and product descriptions
        similarities = cosine_similarity(user_tfidf, product_tfidf)

        # Debug: Check similarity scores
        print(f"Similarity Scores for frame ID {frame['id']}:")
        print(similarities)
        
        # Set a threshold for similarity and recommend products
        similarity_threshold = 0.25  # Adjust this value as needed

        recommendations_start_time = time.time()
        recommendations_batch = []
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
                recommendations_batch.append((wallet, frame['id'], recommendations[0], recommendations[1], recommendations[2]))

                print(f"Recommendations for User {i+1} (Wallet: {wallet}) in Frame {frame['id']}:")
                print(f"- {recommendations} (Top 3 recommendations)")

        if recommendations_batch:
            store_recommendations_batch(cur_mysql, conn_mysql, recommendations_batch)
            print(f"Recommendations stored for frame ID {frame['id']}")

        print(f"Time to process recommendations for frame ID {frame['id']}: {time.time() - recommendations_start_time:.2f} seconds")
        print(f"Total time to process frame ID {frame['id']}: {time.time() - frame_start_time:.2f} seconds")

    return products_df
