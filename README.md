\# 🎬 Movie Recommendation System



A Machine Learning based Movie Recommendation System that recommends movies similar to the movie selected by the user.



\## 📌 Project Overview



This project uses a Content-Based Recommendation approach.



The system analyzes movie information such as:

\- Movie overview

\- Genres

\- Keywords

\- Cast

\- Director



It then uses \*\*TF-IDF Vectorization\*\* and \*\*Cosine Similarity\*\* to find movies that are most similar to the selected movie.



\## 🚀 Features



\- Select a movie from the list

\- Get similar movie recommendations

\- Display similarity percentage

\- Simple and user-friendly Streamlit interface

\- Uses real movie metadata



\## 🛠️ Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Streamlit

\- TF-IDF

\- Cosine Similarity



\## 🧠 Machine Learning Method



\### TF-IDF



TF-IDF converts movie text information into numerical vectors.



\### Cosine Similarity



Cosine Similarity compares the numerical vectors and calculates how similar two movies are.



\## 📂 Project Structure



```text

Movie-Recommendation-System/

│

├── app.py

├── requirements.txt

├── README.md

│

├── data/

│   ├── tmdb\_5000\_movies.csv

│   └── tmdb\_5000\_credits.csv

│

└── venv/

