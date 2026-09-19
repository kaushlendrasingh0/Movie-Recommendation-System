import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load movie datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Combine both datasets
movies = movies.merge(credits, left_on="id", right_on="movie_id")

# Keep only useful columns
movies = movies[["movie_id", "title_x", "overview", "genres", "keywords", "cast", "crew"]]
movies = movies.rename(columns={"title_x": "title"})

# Convert JSON-like columns into simple text

def convert_to_list(text):
    items = ast.literal_eval(text)
    return [item["name"] for item in items]


movies["genres"] = movies["genres"].apply(convert_to_list)
movies["keywords"] = movies["keywords"].apply(convert_to_list)

# Extract top 3 cast members
def get_cast(text):
    items = ast.literal_eval(text)
    return [item["name"] for item in items[:3]]


# Extract director name
def get_director(text):
    items = ast.literal_eval(text)
    for item in items:
        if item["job"] == "Director":
            return [item["name"]]
    return []


movies["cast"] = movies["cast"].apply(get_cast)
movies["crew"] = movies["crew"].apply(get_director)

# Replace missing values with empty lists
movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].apply(lambda x: x if isinstance(x, list) else [])
movies["keywords"] = movies["keywords"].apply(lambda x: x if isinstance(x, list) else [])
movies["cast"] = movies["cast"].apply(lambda x: x if isinstance(x, list) else [])
movies["crew"] = movies["crew"].apply(lambda x: x if isinstance(x, list) else [])

# Create a single tags column
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"].apply(lambda x: " ".join(x))
)

# Convert movie tags into numerical vectors using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["tags"])

# Calculate similarity between all movies
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(movie_title):
    movie_title = movie_title.lower()

    matches = movies[movies["title"].str.lower() == movie_title]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    recommendations = []

    for i, score in movie_list:
        recommendations.append(
            (movies.iloc[i]["title"], round(score * 100, 2))
        )

    return recommendations


# Streamlit application
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite movie using Machine Learning.")

movie_titles = sorted(movies["title"].dropna().unique())

selected_movie = st.selectbox(
    "🎥 Select a movie:",
    movie_titles
)

# Recommend similar movies
if st.button("🎬 Recommend Movies"):
    recommendations = recommend(selected_movie)

    if recommendations:
        st.subheader("🍿 Recommended Movies")

        for movie, score in recommendations:
            st.write(f"**{movie}** — Similarity: {score}%")
    else:
        st.warning("Movie not found.")