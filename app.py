import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests

# ===================================================================
# BAGIAN 1: FUNGSI UNTUK MEMUAT SEMUA ASET (MODEL & DATA)
# ===================================================================

TMDB_API_KEY = "4359622a966ba5b04ead2088a11c9e4b"

@st.cache_data
def load_assets():
    """Memuat model dan data yang sudah diproses."""
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
    except Exception as e:
        st.error(f"Error saat memuat aset: {e}")
        return None, None, None, None
    

@st.cache_data
def get_movie_details(movie_title):
    """Mengambil sinopsis dan URL poster dari API TMDb."""
    # Ekstrak judul bersih dan tahun
    try:
        title_only = movie_title.rsplit('(', 1)[0].strip()
        year = movie_title.rsplit('(', 1)[1].replace(')', '')
    except:
        title_only = movie_title
        year = None
    
    # URL untuk mencari film di TMDb
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title_only}&year={year}"
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        
        if data['results']:
            # Ambil film pertama dari hasil pencarian
            movie_data = data['results'][0]
            overview = movie_data.get('overview', 'Sinopsis tidak tersedia.')
            poster_path = movie_data.get('poster_path', '')
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else None
            return overview, poster_url
    except requests.exceptions.RequestException as e:
        # Jangan tampilkan error API ke pengguna, cukup kembalikan nilai default
        print(f"API Error: {e}")

    return "Sinopsis tidak tersedia.", None


# ===================================================================
# BAGIAN 2: FUNGSI UNTUK MENDAPATKAN REKOMENDASI
# ===================================================================
def get_recommendations(model_bundle, movies_2020, movies_2020_features, input_title_clean):
    """
    Memberikan rekomendasi film menggunakan model dan data yang sudah dimuat.
    Fungsi ini tidak lagi melakukan feature engineering ulang.
    """
    # Ekstrak model dari bundle
    kmeans = model_bundle['kmeans']
    scaler = model_bundle['scaler']
    pca = model_bundle['pca']

    # --- PERBAIKAN: HAPUS SEMUA PROSES PEMBUATAN FITUR ULANG ---
    # Kita asumsikan 'movies_2020_features' yang dimuat dari .parquet sudah siap.
    # Kolom 'cluster' juga sudah ada dari proses training.
    
    # Cari baris film yang diinput oleh pengguna
    movie_row = movies_2020_features[movies_2020_features['title_clean'] == input_title_clean]
    if movie_row.empty:
        return pd.DataFrame() # Kembalikan DataFrame kosong jika film tidak ditemukan

    # Langsung ambil cluster dari film yang dipilih
    # Pastikan kolom 'cluster' ada di file features.parquet Anda
    try:
        movie_cluster = movie_row['cluster'].values[0]
    except KeyError:
        st.error("Error: Kolom 'cluster' tidak ada di file 'features.parquet'. Pastikan Anda menyimpan dataframe setelah training.")
        return pd.DataFrame()

    # Ambil film lain dalam cluster yang sama
    cluster_movies = movies_2020_features[movies_2020_features['cluster'] == movie_cluster].copy()
    
    # --- LOGIKA POST-FILTERING TETAP SAMA ---
    initial_candidates = cluster_movies.sort_values(by='mean_rating', ascending=False).head(20)

    input_title_original = movie_row['title'].values[0]
    input_genres = set(movies_2020[movies_2020['title'] == input_title_original]['genres_list'].iloc[0])

    relevance_scores = []
    for index, row in initial_candidates.iterrows():
        try:
            candidate_genres = set(movies_2020[movies_2020['title'] == row['title']]['genres_list'].iloc[0])
            intersection = len(input_genres.intersection(candidate_genres))
            union = len(input_genres.union(candidate_genres))
            score = intersection / union if union != 0 else 0
            relevance_scores.append(score)
        except (IndexError, TypeError):
            relevance_scores.append(0)

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

    # Kembalikan hasil akhir
    return final_recommendations_with_genres[['title', 'genres', 'mean_rating', 'jaccard_score']]

# ===================================================================
# BAGIAN 3: TAMPILAN UTAMA APLIKASI STREAMLIT
# ===================================================================
st.set_page_config(page_title="Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")
st.write("Masukkan judul film favorit Anda untuk menemukan film serupa.")

# Muat semua aset
model_bundle, movies_raw, movies_2020, movies_2020_features = load_assets()

if model_bundle:
    # --- SIDEBAR ---
    st.sidebar.title("Daftar Film Tersedia (2020+)")
    search_term = st.sidebar.text_input("Cari judul film di sini:")
    if search_term:
        available_movies = movies_2020[movies_2020['title'].str.contains(search_term, case=False)]
    else:
        available_movies = movies_2020
    st.sidebar.dataframe(available_movies[['title', 'genres']], height=400)

    # --- KONTEN UTAMA ---
    movie_title_input = st.text_input("Ketik judul film (contoh: waves):", "waves")

    if st.button("Cari Rekomendasi", type="primary"):
        if movie_title_input:
            with st.spinner(f"Mencari film yang mirip dengan '{movie_title_input}'..."):
                recommendations = get_recommendations(model_bundle, movies_2020, movies_2020_features, movie_title_input.lower())

                st.subheader("Berikut adalah hasil rekomendasinya:")
                if not recommendations.empty:
                    # --- TAMPILAN BARU MENGGUNAKAN KOLOM ---
                    for index, row in recommendations.iterrows():
                        st.write("---")
                        # Buat 2 kolom: satu untuk poster, satu untuk info
                        col1, col2 = st.columns([1, 4])
                        
                        # Ambil sinopsis dan poster dari API
                        synopsis, poster_url = get_movie_details(row['title'])
                        
                        with col1:
                            if poster_url:
                                st.image(poster_url)
                        
                        with col2:
                            st.subheader(row['title'])
                            st.caption(f"Genre: {row['genres']} | Rating: {row['mean_rating']:.2f} ⭐")
                            st.write(synopsis)
                else:
                    st.warning("Maaf, tidak ada rekomendasi yang ditemukan untuk judul tersebut.")
        else:
            st.error("Harap masukkan judul film terlebih dahulu.")