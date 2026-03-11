import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
# In app.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the parent directory path
parent_dir = Path(__file__).parent.parent

# Load .env file from parent directory
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    st.error(".env file not found in parent directory!")
    st.stop()

API_KEY = os.getenv("OMDB_KEY")

# Fix for MacOS
nltk.download('stopwords', download_dir='/tmp/nltk_data')
nltk.data.path.append('/tmp/nltk_data')

# Set page config
st.set_page_config(page_title="Movie Recommendation System", layout="centered")

# Add title and description
st.title("🎬 Movie Recommendation System")
st.write("Type a movie name to get similar recommendations")

@st.cache_data
def load_data():
    df = pd.read_csv('data/movies.csv')
    data = df[['title','genres','keywords','overview','cast','director']]
    data = data.dropna()
    data['combined'] = data['genres'] + ' ' + data['keywords'] + ' ' + data['overview'] + ' ' + data['cast'] + ' ' + data['director']
    return data

@st.cache_data
def preprocess_data(data):
    stop_words = set(stopwords.words('english'))
    
    def pre_process(text):
        text = re.sub(r"[^a-zA-Z0-9\s]","",text)
        text = text.lower()
        words = text.split(" ")
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    
    data['cleaned'] = data['combined'].apply(pre_process)
    return data

@st.cache_resource
def create_similarity_matrix(data):
    Tfidf = TfidfVectorizer(max_features=5000)
    Tfidf_mat = Tfidf.fit_transform(data['cleaned'])
    return cosine_similarity(Tfidf_mat, Tfidf_mat)

def recommended_movies(movie_name, data, similarity, n=5):
    ind = data[data['title'].str.lower() == movie_name.lower()].index
    if len(ind) == 0:
        return None
    else:
        ind = ind[0]
        scores = list(enumerate(similarity[ind]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True) 
        scores = scores[1:n+1]
        movind = [score[0] for score in scores]
        return data['title'].iloc[movind]
#c5bfc45b
def get_movie_details(title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        res = response.json()
        if res.get("Response") == "True":
            return res.get("Plot", "N/A"), res.get("Poster", "N/A")
    except requests.exceptions.RequestException:
        pass
    return "Plot not available", None

# Load and process data
with st.spinner('Loading movie database...'):
    data = load_data()
    data = preprocess_data(data)
    cosine_sim = create_similarity_matrix(data)

# User input
movie_name = st.text_input("Enter a movie name:", "The Dark Knight Rises")
num_recommendations = st.slider("Number of recommendations", 1, 10, 5)

# Get recommendations
if st.button("Get Recommendations"):
    with st.spinner('Finding similar movies...'):
        recommendations = recommended_movies(movie_name, data, cosine_sim, num_recommendations)
        
        if recommendations is None:
            st.error("Movie not found in database. Please check the spelling.")
        else:
            st.success(f"🎥 Movies similar to '{movie_name}':")
            
            # Display original movie details
            plot, poster = get_movie_details(movie_name)
            if poster != "N/A":
                st.subheader(f"About '{movie_name}':")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(poster, width=200)
                with col2:
                    st.write(plot)
                st.markdown("---")
            
            # Display recommendations
            st.subheader("Recommended Movies:")
            for i, movie in enumerate(recommendations, 1):
                plot, poster = get_movie_details(movie)
                cols = st.columns([1, 3])
                with cols[0]:
                    if poster and poster != "N/A":
                        st.image(poster, width=150)
                    else:
                        st.write("No poster available")
                with cols[1]:
                    st.markdown(f"**{i}. {movie}**")
                    st.caption(plot)
                st.markdown("---")

# Add some styling
st.markdown("""
<style>
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 12px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMarkdown h3 {
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)