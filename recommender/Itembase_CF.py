# ------------------------------------------------------------
# SmartFlix - Item-Based Collaborative Filtering 
# ------------------------------------------------------------
# Core Model: Handles training, normalization, similarity, prediction, and recommendation
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from .TMDb_connect import get_movie_info


# ------------------------------------------------------------
# 1. Load MovieLens Data
# ------------------------------------------------------------
num_users = 943
num_items = 1682

# Training set
df_train = pd.read_csv("D:/Final_project/Smartflix/data/u1.base", sep="\t",
                       names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
values = df_train.values
values[:, 0:2] -= 1
train_data = sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])),
                               shape=(num_users, num_items), dtype=np.float64)

# Test set
df_test = pd.read_csv("D:/Final_project/Smartflix/data/u1.test", sep="\t",
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
values = df_test.values
values[:, 0:2] -= 1
test_data = sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])),
                              shape=(num_users, num_items), dtype=np.float64)

# Movie titles
movies = pd.read_csv("D:/Final_project/Smartflix/data/u.item", sep="|",
                     names=range(24), encoding="latin-1", usecols=[0, 1])
movies.columns = ['MovieID', 'Title']
movie_dict = dict(zip(movies['MovieID'] - 1, movies['Title']))

# ------------------------------------------------------------
# 2. Normalize by User Means
# ------------------------------------------------------------
train_copy = train_data.copy().tolil()
user_means = np.zeros(num_users)

for u in range(num_users):
    user_ratings = train_copy.getrow(u).data
    if len(user_ratings) > 0:
        mean_rating = np.mean(user_ratings)
        user_means[u] = mean_rating
        cols = train_copy.getrow(u).nonzero()[1]
        for c in cols:
            train_copy[u, c] -= mean_rating

normalized_matrix = train_copy.toarray()

# ------------------------------------------------------------
# 3. Compute Item-Item Similarity
# ------------------------------------------------------------
item_similarity = np.zeros((num_items, num_items))

for i in range(num_items):
    for j in range(i, num_items):
        item_i = normalized_matrix[:, i]
        item_j = normalized_matrix[:, j]
        common_users = (item_i != 0) & (item_j != 0)
        if np.sum(common_users) > 0:
            ratings_i = item_i[common_users]
            ratings_j = item_j[common_users]
            norm_i = np.linalg.norm(ratings_i)
            norm_j = np.linalg.norm(ratings_j)
            if norm_i > 0 and norm_j > 0:
                sim = np.dot(ratings_i, ratings_j) / (norm_i * norm_j)
                item_similarity[i, j] = sim
                item_similarity[j, i] = sim

print("✅ Item-item similarity computed.")


# ------------------------------------------------------------
# 4. Rating Prediction
# ------------------------------------------------------------
def predict(user, item, k=10):
    user_ratings = train_data.getrow(user)
    rated_items = user_ratings.nonzero()[1]
    if len(rated_items) == 0:
        return user_means[user] if user_means[user] > 0 else 3.0

    sims = item_similarity[item, rated_items]
    ratings = user_ratings.data
    if len(sims) > 0:
        sorted_idx = np.argsort(sims)[::-1]
        k = min(k, len(sorted_idx))
        top_k_idx = sorted_idx[:k]
        top_k_sims = sims[top_k_idx]
        top_k_ratings = ratings[top_k_idx]
        pos_mask = top_k_sims > 0
        if np.sum(pos_mask) > 0:
            pred = np.dot(top_k_sims[pos_mask], top_k_ratings[pos_mask]) / np.sum(np.abs(top_k_sims[pos_mask]))
            return pred
    return user_means[user] if user_means[user] > 0 else 3.0


# ------------------------------------------------------------
# 5. MAE Evaluation
# ------------------------------------------------------------
def compute_mae(k=10):
    rows, cols = test_data.nonzero()
    errors = []
    for idx in range(len(rows)):
        actual = test_data[rows[idx], cols[idx]]
        pred = predict(rows[idx], cols[idx], k)
        errors.append(abs(actual - pred))
    return np.mean(errors)


def test_eval():
    k_values = [1, 5, 10, 50, 100]
    maes = [compute_mae(k) for k in k_values]

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, maes, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Item-Based CF - MAE vs k')
    plt.grid(True, alpha=0.3)
    plt.show()

    for k, mae in zip(k_values, maes):
        print(f"k={k}, MAE={mae:.4f}")


# ------------------------------------------------------------
# 6. Recommendation Function
# ------------------------------------------------------------
def show_recommendations_for_user(user_id, k=50, top_n=5):
    print(f"\n🎬 Recommendations for User {user_id}:\n")

    user_ratings = train_data.getrow(user_id)
    unrated_items = np.where(user_ratings.toarray()[0] == 0)[0]

    predictions = [(item, predict(user_id, item, k)) for item in unrated_items]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:top_n]

    for movie_id, rating in top_movies:
        title = movies[movies["MovieID"] == movie_id + 1]["Title"].values[0]
        info = get_movie_info(title)
        print(f"🎞️ Title: {info['title']}")
        print(f"⭐ Predicted Rating: {rating:.2f}")
        print(f"🗓️ Release Date: {info['release_date']}")
        print(f"📝 Overview: {info['overview']}")
        print(f"🖼️ Poster URL: {info['poster']}\n")


def recommend_based_on_movie(target_movie_id, top_n=5):
    """Return top-N similar movies based on item-item similarity."""
    sims = item_similarity[target_movie_id]
    similar_movies = np.argsort(sims)[::-1]  # descending order
    similar_movies = similar_movies[1: top_n + 1]  # skip itself
    return similar_movies, sims[similar_movies]
