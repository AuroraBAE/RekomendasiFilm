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
    Memuat model, dan memproses semua DataFrame yang dibutuhkan oleh aplikasi.
    """
    try:
        # Gunakan Path Relatif
        model_path = 'recommender_model.pkl'
        movies_path = "movies.csv"
        ratings_path = "ratings.csv"

        # Muat model bundle
        model_bundle = joblib.load(model_path)
        
        # Muat data mentah
        movies_raw = pd.read_csv(movies_path)
        ratings_raw = pd.read_csv(ratings_path)
        
        # --- PERBAIKAN DIMULAI DI SINI ---
        
        # 1. Buat kolom 'year' TERLEBIH DAHULU dari 'title'
        movies_raw['year'] = movies_raw['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # 2. BARU saring data menggunakan kolom 'year' yang sudah ada
        movies_2020 = movies_raw[
            (movies_raw['year'] >= 2020) & (movies_raw['genres'] != '(no genres listed)')
        ].copy()
        
        # --- PERBAIKAN SELESAI ---

        # Sisa kode untuk membuat movies_2020_features sama seperti sebelumnya
        movies_2020['title_clean'] = movies_2020['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip().str.lower()
        movies_2020['genres_list'] = movies_2020['genres'].str.split('|')
        
        genre_dummies = movies_2020['genres_list'].explode().str.get_dummies().groupby(level=0).sum()
        movies_2020_features = pd.concat(
            [movies_2020[['movieId', 'title', 'title_clean', 'year']], genre_dummies], axis=1
        )
        mean_rating = ratings_raw.groupby('movieId')['rating'].mean().reset_index(name='mean_rating')
        count_rating = ratings_raw.groupby('movieId')['rating'].count().reset_index(name='num_ratings')
        rating_stats = pd.merge(mean_rating, count_rating, on='movieId')
        movies_2020_features = pd.merge(movies_2020_features, rating_stats, on='movieId', how='left')
        movies_2020_features.dropna(subset=['mean_rating', 'num_ratings'], inplace=True)
        movies_2020_features.reset_index(drop=True, inplace=True)
        
        # Kembalikan semua yang dibutuhkan aplikasi
        return model_bundle, movies_raw, movies_2020, movies_2020_features

    except FileNotFoundError:
        st.error("Error: Salah satu file (recommender_model.pkl, movies.csv, atau ratings.csv) tidak ditemukan.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Terjadi error saat memuat aset: {e}")
        return None, None, None, None


# ===================================================================
# BAGIAN 2: FUNGSI UNTUK MENDAPATKAN REKOMENDASI
# ===================================================================
# GANTI FUNGSI LAMA ANDA DENGAN VERSI LENGKAP INI
def get_recommendations(model_bundle, movies_2020, movies_2020_features, input_title_clean):
    """
    Memberikan rekomendasi film menggunakan model dan data yang sudah dimuat.
    """
    # Ekstrak model dari bundle
    kmeans = model_bundle['kmeans']
    scaler = model_bundle['scaler']
    pca = model_bundle['pca']

    # Siapkan fitur untuk prediksi dari dataframe yang sudah diproses
    genre_cols = [col for col in movies_2020_features.columns if col not in ['movieId', 'title', 'title_clean', 'year', 'mean_rating', 'num_ratings']]
    feature_data = movies_2020_features[genre_cols + ['mean_rating', 'num_ratings']].copy()

    # Terapkan pembobotan, scaling, dan PCA yang sama persis seperti saat training
    weight = 2.0
    feature_data[genre_cols] = feature_data[genre_cols] * weight
    scaled_features = scaler.transform(feature_data)
    reduced_features = pca.transform(scaled_features)

    # Prediksi cluster untuk SEMUA film
    cluster_labels = kmeans.predict(reduced_features)
    movies_2020_features['cluster'] = cluster_labels

    # Jalankan logika post-filtering
    movie_row = movies_2020_features[movies_2020_features['title_clean'] == input_title_clean]
    if movie_row.empty:
        return pd.DataFrame() # Kembalikan DataFrame kosong jika film tidak ditemukan

    movie_cluster = movie_row['cluster'].values[0]
    cluster_movies = movies_2020_features[movies_2020_features['cluster'] == movie_cluster].copy()
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