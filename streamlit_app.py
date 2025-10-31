# ============================================================
# 🎧 Spotify User Segmentation Dashboard (EDA + K-Prototypes)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1️⃣ SETUP DASHBOARD
# ------------------------------------------------------------
st.set_page_config(page_title="Spotify User Clustering", layout="wide")
st.title("🎧 Spotify User Segmentation Dashboard")
st.markdown("""
Dashboard ini menampilkan:
- EDA (distribusi, korelasi, perilaku pengguna)
- Pembersihan data dan transformasi log
- K-Prototypes Clustering
- Analisis churn per cluster
""")

# ------------------------------------------------------------
# 2️⃣ LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("spotify_churn_dataset.csv")

df = load_data()
st.sidebar.success("✅ Data berhasil dimuat!")
st.sidebar.write(f"Jumlah baris: {df.shape[0]} | Jumlah kolom: {df.shape[1]}")

# ------------------------------------------------------------
# 3️⃣ CEK DUPLIKAT DAN MISSING VALUE
# ------------------------------------------------------------
st.subheader("📋 Cek Data Awal")
col1, col2 = st.columns(2)
with col1:
    st.write("**Cek duplikat:**", round(len(df.drop_duplicates()) / len(df), 2))
with col2:
    st.write("**Missing values per kolom:**")
    st.write(df.isnull().sum())

# ------------------------------------------------------------
# 4️⃣ PEMBENTUKAN KOLOM AGE GROUP
# ------------------------------------------------------------
if "age" in df.columns:
    bins = [15, 25, 35, 45, 60]
    labels = ['16-25', '26-35', '36-45', '46-59']
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)
    st.success("✅ Kolom `age_group` berhasil dibuat.")
else:
    st.warning("⚠️ Kolom `age` tidak ditemukan di dataset.")

# Tabs utama
tab1, tab2, tab3 = st.tabs(["📊 EDA", "⚙️ Data Cleaning + Transformasi", "🧩 K-Prototypes Clustering"])

# ============================================================
# 📊 TAB 1: EDA
# ============================================================
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")

    # Distribusi churn
    if "is_churned" in df.columns:
        churn_counts = df["is_churned"].value_counts(normalize=True)
        labels = ['Churned', 'Not Churned']
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=['salmon', 'skyblue'], startangle=90, explode=(0.05, 0))
        ax.set_title('Distribusi Pelanggan Churn vs Tidak Churn')
        st.pyplot(fig)

    # Pola perilaku churn vs non-churn
    st.subheader("🎵 Perbandingan Rata-rata Perilaku antara Churned vs Non-Churned")
    behavior_cols = ['listening_time', 'songs_played_per_day', 'skip_rate', 'ads_listened_per_week', 'offline_listening']
    if all(col in df.columns for col in behavior_cols):
        mean_behavior = df.groupby("is_churned")[behavior_cols].mean().T
        mean_behavior.plot(kind='bar', figsize=(8,5), colormap='viridis')
        plt.title("Perbandingan Rata-rata Perilaku antara Churned vs Non-Churned")
        st.pyplot(plt)

    # Perilaku berdasarkan tipe langganan
    st.subheader("💎 Perilaku Pengguna Berdasarkan Tipe Langganan")
    if "subscription_type" in df.columns:
        mean_subs = df.groupby("subscription_type")[behavior_cols].mean()
        mean_subs.plot(kind='bar', figsize=(8,5), colormap='plasma')
        plt.title("Perilaku Pengguna Berdasarkan Tipe Langganan")
        st.pyplot(plt)

    # Usia vs rata-rata lagu
    st.subheader("🧑‍🎤 Rata-rata Lagu yang Diputar per Hari berdasarkan Kelompok Usia")
    if "age_group" in df.columns:
        age_activity = df.groupby('age_group')['songs_played_per_day'].mean().reset_index()
        age_activity = age_activity.sort_values(by='songs_played_per_day', ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.barplot(x='age_group', y='songs_played_per_day', data=age_activity, palette='viridis')
        ax2.set_title('Rata-rata Lagu yang Diputar per Hari berdasarkan Kelompok Usia (Diurutkan)')
        st.pyplot(fig2)

    # Heatmap korelasi
    st.subheader("🔥 Heatmap Korelasi Variabel Numerik")
    corr = df.corr(numeric_only=True)
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

# ============================================================
# ⚙️ TAB 2: CLEANING & TRANSFORMASI
# ============================================================
with tab2:
    st.header("🧹 Data Cleaning dan Transformasi Log")

    df_clean = df.copy()

    # Missing values
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    st.success("✅ Missing value telah ditangani.")

    # Log transform
    st.subheader("📈 Log Transform")
    log_cols = ['ads_listened_per_week']
    for col in log_cols:
        if col in df_clean.columns:
            df_clean[f'{col}_log'] = np.log1p(df_clean[col])
    st.write("Kolom hasil log transform:", log_cols)

# ============================================================
# 🧩 TAB 3: CLUSTERING
# ============================================================
with tab3:
    st.header("🧩 K-Prototypes Clustering")

    # Gabung data numerik + kategorikal
    categorical_cols = ['gender','country','subscription_type','device_type']
    numerical_cols = ['age','listening_time','songs_played_per_day','skip_rate','ads_listened_per_week','offline_listening']

     # Scaling
    st.subheader("⚖️ StandardScaler untuk fitur numerik")
    num_cols = ['age','listening_time','songs_played_per_day','skip_rate','ads_listened_per_week','offline_listening']
    scaler = StandardScaler()
    df_scaled_num = pd.DataFrame(scaler.fit_transform(df_clean[num_cols]), columns=num_cols)
    df_scaled = pd.concat([df_clean, df_scaled_num.add_suffix('_scaled')], axis=1)
    st.dataframe(df_scaled.head())

    st.info("✅ Data siap untuk clustering.")
    df_cluster = pd.concat([df[categorical_cols], df_scaled_num], axis=1)
    cat_idx = [df_cluster.columns.get_loc(col) for col in categorical_cols]

    # Pengaturan parameter
    st.markdown("### ⚙️ Parameter Clustering")
    n_clusters = st.slider("Jumlah Cluster (K)", 2, 10, 7)
    gamma = st.number_input("Nilai Gamma", 0.0, 2.0, 0.1, 0.1)

    if st.button("🚀 Jalankan K-Prototypes"):
        try:
            kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Cao', random_state=42)
            clusters = kproto.fit_predict(df_cluster, categorical=cat_idx)
            df_cluster["Cluster"] = clusters
            st.success("✅ Clustering selesai!")
             # ------------------------------------------------------------
            # Nama cluster (berdasarkan insight kamu)
            # ------------------------------------------------------------
            cluster_map = {
                0: "🧠 Risky Premiums (High-Skip Listeners)",
                1: "💼 Steady Premium Users",
                2: "🎵 Loyal Premium Listeners",
                3: "🌍 Engaged Free Explorers",
                4: "⏸️ Passive Free Listeners",
                5: "🎧 Moderate Free Users",
                6: "💎 Premium Loyalists"
            }
          
            # Profil tiap cluster
            st.subheader("📊 Profil Tiap Cluster (Numerik)")
            num_summary = df_cluster.groupby("Cluster")[numerical_cols].mean().round(2)
            st.dataframe(num_summary)

            st.subheader("💬 Profil Kategorikal")
            cat_summary = df_cluster.groupby("Cluster")[categorical_cols].agg(lambda x: x.mode().iloc[0])
            st.dataframe(cat_summary)

            # Distribusi cluster
            fig4, ax4 = plt.subplots(figsize=(6,4))
            sns.countplot(x="Cluster", data=df_cluster, palette="Set2", ax=ax4)
            ax4.set_title("Distribusi Spotify Users berdasarkan Cluster")
            st.pyplot(fig4)

            # Analisis churn
            if "is_churned" in df.columns:
                churn_merge = df_cluster.join(df["is_churned"])
                churn_rate = churn_merge.groupby("Cluster")["is_churned"].mean() * 100
                st.subheader("📉 Tingkat Churn per Cluster (%)")
                st.bar_chart(churn_rate)

            # Download hasil
            csv = df_cluster.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Unduh Hasil Cluster (CSV)", csv, "hasil_cluster_spotify.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
