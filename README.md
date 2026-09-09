# SmartFlix: AI-Powered Movie Recommendation System

SmartFlix is a movie recommendation system developed using Python and Item-Based Collaborative Filtering. The application provides personalized movie recommendations based on user ratings and also allows users to discover movies similar to a selected title.

## Features

- Personalized movie recommendations by User ID
- Movie similarity recommendations by title
- Item-Based Collaborative Filtering
- Cosine similarity for measuring movie similarity
- TMDb API integration for movie information, posters, release dates, and descriptions
- Interactive Streamlit web interface

## Technologies

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Streamlit
- TMDb API
- python-dotenv

## Dataset

The project uses the MovieLens 100K dataset for user ratings and movie information.

## Project Structure

```text
SmartFlix/
├── data/
├── notebooks/
├── recommender/
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```
## How to Run

Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Run the application:
```Bash
streamlit run app.py
```
## Author
Roqia Alirezaei
