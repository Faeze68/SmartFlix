# ------------------------------------------------------------
# SmartFlix - Streamlit Movie Recommendation App
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
from recommender.Itembase_CF import predict, train_data, item_similarity
from recommender.TMDb_connect import get_movie_info
# ------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="SmartFlix", layout="wide")

st.title("🎬 SmartFlix: AI-Powered Movie Recommendation System")
st.markdown("""
Welcome to **SmartFlix**!  
This app provides two ways to get movie recommendations:  
1️⃣ **By User ID** — Get top movies recommended for a specific user.  
2️⃣ **By Movie Title** — Find movies similar to your favorite title.  
All powered by **Item-Based Collaborative Filtering** and enriched with **TMDb movie data**.
""")

# ------------------------------------------------------------
# Load Movie Data
# ------------------------------------------------------------
movies = pd.read_csv(
    DATA_DIR / "u.item",
    sep="|",
    header=None,
    encoding="latin-1",
    usecols=[0, 1]
)
movies.columns = ["MovieID", "Title"]

# ------------------------------------------------------------
# Sidebar Layout
# ------------------------------------------------------------
st.sidebar.header(" SmartFlix Controls")

# --- User ID Recommendation Section ---
st.sidebar.subheader("👤 Recommend by User ID")
user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    max_value=943,
    value=1,
    step=1
)
top_n_user = st.sidebar.slider("Number of Recommendations (User):", 1, 10, 5)
k_user = st.sidebar.slider("Neighbors (k) for User:", 10, 100, 50, step=10)
user_button = st.sidebar.button("🎥 Recommend for User")

st.sidebar.markdown("---")

# --- Movie Title Recommendation Section ---
st.sidebar.subheader("🎬 Recommend by Movie Title")
search_title = st.sidebar.text_input("Enter a movie title (e.g. The Matrix)")
top_n_title = st.sidebar.slider("Number of Similar Movies (Title):", 1, 10, 5, key="title_slider")
title_button = st.sidebar.button("🔍 Recommend by Title")

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def get_recommendations_by_user(user_id, k=50, top_n=5):
    """Return top-N movie recommendations for a given user."""
    user_ratings = train_data.getrow(user_id)
    unrated_items = np.where(user_ratings.toarray()[0] == 0)[0]

    predictions = []
    for item in unrated_items:
        pred_rating = predict(user_id, item, k)
        predictions.append((item, pred_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


def get_recommendations_by_title(movie_title, top_n=5):
    """Find top-N similar movies using cosine similarity."""
    movie_match = movies[movies["Title"].str.contains(movie_title, case=False, na=False)]
    if movie_match.empty:
        return None, []

    movie_id = movie_match.iloc[0]["MovieID"] - 1
    similarities = list(enumerate(item_similarity[movie_id]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for idx, score in similarities:
        title = movies[movies["MovieID"] == idx + 1]["Title"].values[0]
        results.append((title, score))
    return movie_match.iloc[0]["Title"], results


# ------------------------------------------------------------
# Main Display Area (Center)
# ------------------------------------------------------------
if user_button:
    st.subheader(f"🎞️ Top {top_n_user} Recommendations for User {user_id}")
    top_movies = get_recommendations_by_user(user_id - 1, k_user, top_n_user)

    for movie_id, rating in top_movies:
        title = movies[movies["MovieID"] == movie_id + 1]["Title"].values[0]
        info = get_movie_info(title)
        st.markdown(f"### 🎬 {info['title']}")
        if info["poster"]:
            st.image(info["poster"], width=200)
        
        st.write(f"🗓️ **Release Date:** {info['release_date']}")
        st.write(f"📝 {info['overview']}")
        st.divider()

elif title_button:
    if not search_title.strip():
        st.warning("⚠️ Please enter a movie title before searching.")
    else:
        selected_title, similar_movies = get_recommendations_by_title(search_title, top_n_title)
        if similar_movies:
            st.subheader(f"🎬 Movies similar to {selected_title}:")
            for title, score in similar_movies:
                info = get_movie_info(title)
                st.markdown(f"### 🎞️ {info['title']}")
                if info["poster"]:
                    st.image(info["poster"], width=200)
                
                st.write(f"🗓️ **Release Date:** {info['release_date']}")
                st.write(f"📝 {info['overview']}")
                st.divider()
        else:
            st.warning("No similar movies found. Try another title!")
