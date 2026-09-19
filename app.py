import ast
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def parse_names(value, limit=None):
    """Extract names from TMDB JSON-like columns."""
    if pd.isna(value):
        return ""

    try:
        data = ast.literal_eval(value)
        if isinstance(data, list):
            names = []

            for item in data:
                if isinstance(item, dict) and "name" in item:
                    names.append(str(item["name"]))

            if limit:
                names = names[:limit]

            return " ".join(names)

    except (ValueError, SyntaxError, TypeError):
        return ""

    return ""


def get_director(value):
    """Extract director name from crew column."""
    if pd.isna(value):
        return ""

    try:
        data = ast.literal_eval(value)

        if isinstance(data, list):
            for item in data:
                if (
                    isinstance(item, dict)
                    and item.get("job") == "Director"
                    and item.get("name")
                ):
                    return str(item["name"])

    except (ValueError, SyntaxError, TypeError):
        return ""

    return ""


def clean_text(text):
    """Clean text for TF-IDF processing."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    # Keep only required columns
    movies = movies[
        ["id", "title", "overview", "genres", "keywords"]
    ].copy()

    credits = credits[
        ["movie_id", "cast", "crew"]
    ].copy()

    # Merge datasets
    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="left"
    )

    # Remove duplicate movie IDs if any
    movies = movies.drop_duplicates(subset="id").reset_index(drop=True)

    # Fill missing values
    for column in ["title", "overview", "genres", "keywords", "cast", "crew"]:
        movies[column] = movies[column].fillna("")

    # Convert JSON-like columns into useful text
    movies["genres_text"] = movies["genres"].apply(parse_names)
    movies["keywords_text"] = movies["keywords"].apply(parse_names)
    movies["cast_text"] = movies["cast"].apply(
        lambda x: parse_names(x, limit=5)
    )
    movies["director_text"] = movies["crew"].apply(get_director)

    # Create combined recommendation features
    movies["combined_features"] = (
        movies["overview"]
        + " "
        + movies["genres_text"]
        + " "
        + movies["keywords_text"]
        + " "
        + movies["cast_text"]
        + " "
        + movies["director_text"]
    )

    movies["combined_features"] = movies["combined_features"].apply(
        clean_text
    )

    return movies


# ---------------------------------------------------------
# Build recommendation model
# ---------------------------------------------------------
@st.cache_resource
def build_model(movies):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    feature_matrix = vectorizer.fit_transform(
        movies["combined_features"]
    )

    similarity_matrix = cosine_similarity(feature_matrix)

    return similarity_matrix


# ---------------------------------------------------------
# Recommendation function
# ---------------------------------------------------------
def recommend_movies(movie_title, movies, similarity_matrix, number=8):
    matches = movies.index[
        movies["title"].str.lower() == movie_title.lower()
    ].tolist()

    if not matches:
        return []

    movie_index = matches[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:number + 1]:
        recommendations.append(
            {
                "title": movies.iloc[index]["title"],
                "score": round(float(score) * 100, 2)
            }
        )

    return recommendations


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
movies = load_data()
similarity_matrix = build_model(movies)


# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------
st.title("Movie Recommendation System")

st.markdown(
    "Find movies similar to your favorite movie using "
    "Machine Learning."
)

st.divider()

st.subheader("Select a Movie")

movie_titles = sorted(
    movies["title"].dropna().unique().tolist()
)

selected_movie = st.selectbox(
    "Choose a movie:",
    movie_titles
)


# ---------------------------------------------------------
# Recommendation button
# ---------------------------------------------------------
if st.button("Recommend Movies", type="primary"):

    recommendations = recommend_movies(
        selected_movie,
        movies,
        similarity_matrix,
        number=8
    )

    if recommendations:

        st.divider()
        st.subheader("Recommended Movies")

        st.success(
            f"Movies similar to: {selected_movie}"
        )

        for number, movie in enumerate(recommendations, start=1):

            st.markdown(
                f"### {number}. {movie['title']}"
            )

            st.progress(
                min(movie["score"] / 100, 1.0)
            )

            st.caption(
                f"Similarity Score: {movie['score']:.2f}%"
            )

            st.divider()

    else:
        st.warning(
            "Sorry, the selected movie was not found."
        )


# ---------------------------------------------------------
# Project information
# ---------------------------------------------------------
with st.expander("About this Project"):

    st.write(
        """
        This project uses a Content-Based Recommendation System.

        Machine Learning techniques used:

        - TF-IDF Vectorization
        - Cosine Similarity
        - Natural Language Processing
        - Content-Based Filtering

        The system analyzes movie information such as
        genres, keywords, overview, cast and director to
        recommend similar movies.
        """
    )

st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)