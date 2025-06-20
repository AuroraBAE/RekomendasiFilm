import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ===================================================================
# BAGIAN 1: FUNGSI UNTUK MEMUAT SEMUA ASET (MODEL & DATA)
# ===================================================================

@st.cache_data
def load_assets():
    """
    Memuat model dan data yang sudah diproses sebelumnya.
    """
    try:
        model_path = 'recommender_model.pkl'
        movies_path = "movies.csv"
        features_path = "features.parquet" # <-- Path ke file fitur baru

        model_bundle = joblib.load(model_path)
        movies_raw = pd.read_csv(movies_path)
        movies_2020_features = pd.read_parquet(features_path) # <-- Muat file parquet
        
        # Buat movies_2020 untuk lookup genre (jika masih diperlukan)
        movies_2020 = movies_raw[movies_raw['movieId'].isin(movies_2020_features['movieId'])].copy()
        movies_2020['genres_list'] = movies_2020['genres'].str.split('|')
        
        return model_bundle, movies_raw, movies_2020, movies_2020_features

    except FileNotFoundError as e:
        st.error(f"Error: File tidak ditemukan. Pastikan file '{e.filename}' ada di repository GitHub.")
        return None, None, None, None


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
st.title("🎬 Sistem Rekomendasi Film Berdasarkan Genre")
st.write("Masukkan judul film favorit Anda untuk menemukan film serupa, atau cari judul dari daftar di samping.")

# Muat semua aset yang dibutuhkan
model_bundle, movies_raw, movies_2020, movies_2020_features = load_assets()

if model_bundle:
    st.sidebar.title("Daftar Film Tersedia (2020+)") 
    search_term = st.sidebar.text_input("Cari judul film di sini:")

    # Gunakan movies_2020 sebagai sumber data untuk sidebar
    if search_term:
        available_movies = movies_2020[movies_2020['title'].str.contains(search_term, case=False)]
    else:
        available_movies = movies_2020

    st.sidebar.dataframe(available_movies[['title', 'genres']], height=400)

    # --- KONTEN UTAMA ---
    movie_title = st.text_input("Ketik judul film (lowercase) ")

    if st.button("Cari Rekomendasi", type="primary"):
        if movie_title:
            with st.spinner(f"Mencari film yang mirip dengan '{movie_title}'..."):
                recommendations = get_recommendations(model_bundle, movies_2020, movies_2020_features, movie_title.lower())
                st.subheader("Berikut adalah hasil rekomendasinya:")
                if not recommendations.empty:
                    st.dataframe(recommendations)
                else:
                    st.warning("Maaf, tidak ada rekomendasi yang ditemukan untuk judul tersebut.")
        else:
            st.error("Harap masukkan judul film terlebih dahulu.")