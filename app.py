import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests

# ===================================================================
# BAGIAN 1: KONFIGURASI DAN FUNGSI PEMUATAN ASET
# ===================================================================

# GANTI DENGAN API KEY ANDA YANG SEBENARNYA DARI TMDB
TMDB_API_KEY = "4359622a966ba5b04ead2088a11c9e4b"

@st.cache_data
def load_assets():
    """Memuat model, dan memproses semua DataFrame yang dibutuhkan oleh aplikasi."""
    try:
        model_path = 'recommender_model.pkl'
        features_path = "features.parquet"
        movies_path = "movies.csv"

        model_bundle = joblib.load(model_path)
        movies_df_features = pd.read_parquet(features_path)
        movies_df_original = pd.read_csv(movies_path)
        
        movies_2020 = movies_df_original[movies_df_original['movieId'].isin(movies_df_features['movieId'])].copy()
        movies_2020['genres_list'] = movies_2020['genres'].str.split('|')
        
        return model_bundle, movies_df_original, movies_2020, movies_df_features
    except FileNotFoundError as e:
        st.error(f"Error: File tidak ditemukan. Pastikan file '{e.filename}' ada di repository GitHub.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Terjadi error saat memuat aset: {e}")
        return None, None, None, None

# ===================================================================
# BAGIAN 2: FUNGSI BARU UNTUK MENGAMBIL DETAIL FILM DARI TMDB
# ===================================================================
@st.cache_data
def get_movie_details(movie_title):
    """Mengambil sinopsis dan URL poster dari API TMDb."""
    try:
        title_only = movie_title.rsplit('(', 1)[0].strip()
        year = movie_title.rsplit('(', 1)[1].replace(')', '')
    except:
        title_only = movie_title
        year = None
    
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title_only}&year={year}"
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            movie_data = data['results'][0]
            overview = movie_data.get('overview', 'Sinopsis tidak tersedia.')
            poster_path = movie_data.get('poster_path', '')
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else None
            return overview, poster_url
    except requests.exceptions.RequestException:
        pass # Abaikan error API agar aplikasi tidak crash
    return "Sinopsis tidak tersedia.", None

# ===================================================================
# BAGIAN 3: FUNGSI REKOMENDASI (VERSI LENGKAP DAN BENAR)
# ===================================================================
def get_recommendations(model_bundle, movies_2020, movies_2020_features, input_title_clean):
    """Memberikan rekomendasi film menggunakan logika K-Means dan post-filtering."""
    try:
        # Cari film yang diinput
        movie_row = movies_2020_features[movies_2020_features['title_clean'] == input_title_clean]
        if movie_row.empty:
            return pd.DataFrame()

        # Ambil cluster dari film tersebut
        movie_cluster = movie_row['cluster'].values[0]
        cluster_movies = movies_2020_features[movies_2020_features['cluster'] == movie_cluster].copy()
        
        # Ambil kandidat awal
        initial_candidates = cluster_movies.sort_values(by='mean_rating', ascending=False).head(20)

        # Lakukan post-filtering
        input_title_original = movie_row['title'].values[0]
        input_genres = set(movies_2020[movies_2020['title'] == input_title_original]['genres_list'].iloc[0])

        relevance_scores = []
        for index, row in initial_candidates.iterrows():
            candidate_genres = set(movies_2020[movies_2020['title'] == row['title']]['genres_list'].iloc[0])
            intersection = len(input_genres.intersection(candidate_genres))
            union = len(input_genres.union(candidate_genres))
            score = intersection / union if union != 0 else 0
            relevance_scores.append(score)

        initial_candidates['jaccard_score'] = relevance_scores
        filtered_recommendations = initial_candidates[initial_candidates['jaccard_score'] >= 0.25]
        filtered_recommendations = filtered_recommendations[filtered_recommendations['title_clean'] != input_title_clean]
        final_recommendations = filtered_recommendations.head(5)

        # Gabungkan dengan info genre untuk ditampilkan
        final_recommendations_with_genres = pd.merge(
            final_recommendations,
            movies_2020[['title', 'genres']],
            on='title',
            how='left'
        )
        return final_recommendations_with_genres[['title', 'genres', 'mean_rating']]
    except Exception as e:
        st.error(f"Terjadi error pada logika rekomendasi: {e}")
        return pd.DataFrame()


# ===================================================================
# BAGIAN 4: TAMPILAN UTAMA APLIKASI STREAMLIT
# ===================================================================
st.set_page_config(page_title="Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")
st.write("Masukkan judul film favorit Anda untuk menemukan film serupa, atau cari judul dari daftar di samping.")

model_bundle, movies_raw, movies_2020, movies_2020_features = load_assets()

if model_bundle:
    st.sidebar.title("Daftar Film Tersedia (2020+)")
    search_term = st.sidebar.text_input("Cari judul film di sini:")
    if search_term:
        available_movies = movies_2020[movies_2020['title'].str.contains(search_term, case=False)]
    else:
        available_movies = movies_2020
    st.sidebar.dataframe(available_movies[['title', 'genres']], height=400)

    movie_title_input = st.text_input("Ketik judul film (contoh: waves):", "waves")

    if st.button("Cari Rekomendasi", type="primary"):
        if movie_title_input:
            with st.spinner(f"Mencari film yang mirip dengan '{movie_title_input}'..."):
                recommendations = get_recommendations(model_bundle, movies_2020, movies_2020_features, movie_title_input.lower())
                st.subheader("Berikut adalah hasil rekomendasinya:")
                if not recommendations.empty:
                    for index, row in recommendations.iterrows():
                        st.write("---")
                        col1, col2 = st.columns([1, 4])
                        synopsis, poster_url = get_movie_details(row['title'])
                        with col1:
                            if poster_url:
                                st.image(poster_url)
                            else:
                                st.image("https://via.placeholder.com/200x300.png?text=No+Image")
                        with col2:
                            st.subheader(row['title'])
                            st.caption(f"Genre: {row['genres']} | Rating: {row['mean_rating']:.2f} ⭐")
                            st.write(synopsis)
                else:
                    st.warning("Maaf, tidak ada rekomendasi yang ditemukan untuk judul tersebut.")
        else:
            st.error("Harap masukkan judul film terlebih dahulu.")