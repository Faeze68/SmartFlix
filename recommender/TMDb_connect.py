import requests
import streamlit as st

API_KEY = "5d13e375e7a6142f1c5a4e870217bd22"
BASE_URL = "https://api.themoviedb.org/3"

def clean_title(title):
    """Clean up movie title for better TMDb search results."""
    return title.split("(")[0].strip()

@st.cache_data(show_spinner=False)
def get_movie_info(movie_title):
    """Fetch movie details (title, overview, poster, release date) from TMDb."""
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": clean_title(movie_title)}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("results"):
        movie = data["results"][0]
        return {
            "title": movie["title"],
            "overview": movie.get("overview", "No overview available."),
            "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
            "release_date": movie.get("release_date", "Unknown"),
        }
    else:
        return {"title": movie_title, "overview": "Not found.", "poster": None, "release_date": "Unknown"}

# Fetch similar movies
@st.cache_data(show_spinner=False)
def get_similar_movies(movie_title, top_n=5):
    """Fetch similar movies from TMDb based on a movie title."""
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": clean_title(movie_title)}
    response = requests.get(url, params=params)
    data = response.json()

    if not data.get("results"):
        return []

    movie_id = data["results"][0]["id"]
    similar_url = f"{BASE_URL}/movie/{movie_id}/similar"
    response = requests.get(similar_url, params={"api_key": API_KEY})
    similar_data = response.json()

    results = []
    for movie in similar_data.get("results", [])[:top_n]:
        results.append({
            "title": movie.get("title"),
            "overview": movie.get("overview", "No overview available."),
            "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
            "release_date": movie.get("release_date", "Unknown"),
        })

    return results
