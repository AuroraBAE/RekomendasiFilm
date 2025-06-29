import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests

# ===================================================================
# BAGIAN 1: KONFIGURASI DAN FUNGSI PEMUATAN ASET
# ===================================================================

TMDB_API_KEY = "4359622a966ba5b04ead2088a11c9e4b"

@st.cache_data
def load_assets():
    """Memuat semua aset: model bundle dan dataframes."""
    try:
        model_bundle = joblib.load('model_bundle.pkl')
        movies_df_features = pd.read_parquet("features.parquet")
        movies_df_original = pd.read_csv("movies.csv")
        
        # Ekstrak model individual dari bundle
        scaler = model_bundle['scaler']
        pca = model_bundle['pca']
        kmeans = model_bundle['kmeans']
        genre_weight = model_bundle.get('genre_weight', 3.0) 
        
        # Siapkan dataframe movies_2020 untuk genres_list
        movies_2020 = movies_df_original[movies_df_original['movieId'].isin(movies_df_features['movieId'])].copy()
        movies_2020['genres_list'] = movies_2020['genres'].str.split('|')
        
        # Ambil daftar kolom genre dari dataframe fitur
        non_feature_cols = ['movieId', 'title', 'year', 'title_clean', 'genre_encoded', 
                            'mean_rating', 'num_ratings', 'decoded_genre', 'cluster']
        genre_cols = [col for col in movies_df_features.columns if col not in non_feature_cols]

        return kmeans, scaler, pca, movies_2020, movies_df_features, genre_cols, genre_weight
    
    except FileNotFoundError as e:
        st.error(f"Error: File aset tidak ditemukan. Pastikan file '{e.filename}' ada di direktori aplikasi Anda.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Terjadi error saat memuat aset: {e}")
        return None, None, None, None, None, None

# ===================================================================
# BAGIAN 2: FUNGSI UNTUK MENGAMBIL DETAIL FILM DARI TMDB 
# ===================================================================
@st.cache_data
def get_movie_details(movie_title):
    """Mengambil sinopsis dan URL poster dari API TMDb."""
    try:
        title_only = movie_title.rsplit('(', 1)[0].strip()
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title_only}"
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            movie_data = data['results'][0]
            overview = movie_data.get('overview', 'Sinopsis tidak tersedia.')
            poster_path = movie_data.get('poster_path', '')
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else "https://via.placeholder.com/200x300.png?text=No+Image"
            return overview, poster_url
    except Exception:
        pass
    return "Sinopsis tidak tersedia.", "https://via.placeholder.com/200x300.png?text=No+Image"

# ===================================================================
# BAGIAN 3: FUNGSI REKOMENDASI
# ===================================================================
def get_recommendations_final(input_title_clean, df_labeled, kmeans_model, scaler, pca, genre_cols,
                             original_movies_df, n_initial_candidates=30, n_recommendations=5, 
                             similarity_threshold=0.5):
    """
    Fungsi rekomendasi FINAL dengan logika centroid dan fallback.
    """
    movie_row = df_labeled[df_labeled['title_clean'] == input_title_clean.lower()]
    if movie_row.empty:
        return pd.DataFrame()

    target_cluster = movie_row['cluster'].iloc[0]
    input_movie_id = movie_row['movieId'].iloc[0]
    
    # 1. Ambil kandidat berdasarkan jarak ke centroid
    cluster_df = df_labeled[df_labeled['cluster'] == target_cluster].copy()
    features = cluster_df[genre_cols + ['mean_rating', 'num_ratings']].fillna(0)
    
    # Terapkan bobot yang sama seperti saat training
    features[genre_cols] *= genre_weight
    
    scaled_features = scaler.transform(features)
    reduced_features = pca.transform(scaled_features)
    
    distances = kmeans_model.transform(reduced_features)[:, target_cluster]
    cluster_df['distance'] = distances
    initial_candidates = cluster_df.sort_values(by='distance').head(n_initial_candidates)

    # 2. Post-Filtering dengan Jaccard Similarity
    input_genres = set(original_movies_df[original_movies_df['movieId'] == input_movie_id]['genres_list'].iloc[0])
    relevance_scores = []
    for index, row in initial_candidates.iterrows():
        try:
            candidate_genres = set(original_movies_df[original_movies_df['movieId'] == row['movieId']]['genres_list'].iloc[0])
            intersection = len(input_genres.intersection(candidate_genres))
            union = len(input_genres.union(candidate_genres))
            score = intersection / union if union != 0 else 0
            relevance_scores.append(score)
        except IndexError:
            relevance_scores.append(0)
            
    initial_candidates['jaccard_score'] = relevance_scores
    
    filtered = initial_candidates[initial_candidates['jaccard_score'] >= similarity_threshold]
    filtered = filtered[filtered['movieId'] != input_movie_id]
    
    # 3. Logika Fallback
    if not filtered.empty:
        final_recommendations = filtered.head(n_recommendations)
    else:
        fallback_recommendations = initial_candidates[initial_candidates['movieId'] != input_movie_id]
        final_recommendations = fallback_recommendations.head(n_recommendations)

    return final_recommendations

# ===================================================================
# BAGIAN 4: TAMPILAN UTAMA APLIKASI STREAMLIT
# ===================================================================
st.set_page_config(page_title="Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")
st.write("Temukan film serupa berdasarkan kemiripan genre dan popularitas menggunakan Klasterisasi K-Means.")

# Memuat semua aset saat aplikasi dimulai
assets = load_assets()
if all(asset is not None for asset in assets):
    kmeans_model, scaler_model, pca_model, movies_2020, movies_2020_features, genre_cols, genre_weight = assets
    
    # --- Sidebar untuk mencari film ---
    st.sidebar.title("Daftar Film Tersedia (2020+)")
    search_term = st.sidebar.text_input("Cari judul film di sini:")
    if search_term:
        available_movies = movies_2020[movies_2020['title'].str.contains(search_term, case=False)]
    else:
        # Tampilkan 1000 film pertama jika tidak ada pencarian
        available_movies = movies_2020
    st.sidebar.dataframe(available_movies[['title', 'genres']].head(1000), height=400)

    # --- Input utama dari pengguna ---
    movie_title_input = st.selectbox(
    "Pilih atau ketik judul film favorit Anda:",
    options=movies_2020_features['title_clean'].unique(),
    index=list(movies_2020_features['title_clean'].unique()).index('waves') 
    )

    if st.button("Cari Rekomendasi", type="primary", use_container_width=True):
        if movie_title_input:
            with st.spinner(f"Mencari film yang mirip dengan '{movie_title_input}'..."):
                
                # --- Panggilan fungsi rekomendasi yang sudah diperbaiki ---
                recommendations = get_recommendations_final(
                    input_title_clean=movie_title_input,
                    df_labeled=movies_2020_features,
                    kmeans_model=kmeans_model,
                    scaler=scaler_model,
                    pca=pca_model,
                    genre_cols=genre_cols,
                    genre_weight=genre_weight,
                    original_movies_df=movies_2020
                )
                
                st.subheader("Berikut adalah hasil rekomendasinya:")
                if not recommendations.empty:
                    for index, row in recommendations.iterrows():
                        st.write("---")
                        col1, col2 = st.columns([1, 4])
                        synopsis, poster_url = get_movie_details(row['title'])
                        with col1:
                            st.image(poster_url)
                        with col2:
                            st.subheader(row['title'])
                            genres_display = movies_2020[movies_2020['movieId'] == row['movieId']]['genres'].iloc[0]
                            st.caption(f"Genre: {genres_display} | Rating: {row['mean_rating']:.2f} ⭐ | Jarak Klaster: {row.get('distance', 'N/A'):.4f}")
                            st.write(synopsis)
                else:
                    st.warning("Maaf, tidak ada rekomendasi yang ditemukan untuk judul tersebut.")
        else:
            st.error("Harap masukkan judul film terlebih dahulu.")