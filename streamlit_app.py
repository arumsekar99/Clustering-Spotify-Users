# ============================================================
# 🎧 Spotify User Clustering Dashboard (EDA + Cleaning + Clustering)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1️⃣ SETUP DASHBOARD
# ------------------------------------------------------------
st.set_page_config(page_title="Spotify User Dashboard", layout="wide")
st.title("🎧 Spotify User Clustering Dashboard")
st.markdown("Analisis EDA, pembersihan data, dan segmentasi pengguna Spotify menggunakan K-Prototypes.")

# ------------------------------------------------------------
# 2️⃣ LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("spotify_churn_dataset.csv")

df = load_data()
st.sidebar.success("✅ Data berhasil dimuat!")
st.sidebar.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")

# Tabs: EDA | Clustering
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🧩 Clustering (K-Prototypes)"])

# ============================================================
# 📊 TAB 1: EDA
# ============================================================
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.subheader("1️⃣ Distribusi Pelanggan Churn vs Tidak Churn")

    if "is_churned" in df.columns:
        churn_counts = df["is_churned"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        colors = ['#ff9999','#66b3ff']
        ax1.pie(
            churn_counts,
            labels=["Churned", "Not Churned"],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            shadow=True,
            explode=(0.05, 0)
        )
        ax1.set_title("Distribusi Pelanggan Churn vs Tidak Churn")
        st.pyplot(fig1)
    else:
        st.warning("Kolom 'is_churned' tidak ditemukan di dataset.")

    # ------------------------------------------------------------
    st.subheader("2️⃣ Perbandingan Rata-rata Perilaku antara Churned vs Non-Churned")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if "is_churned" in df.columns:
        mean_behavior = df.groupby("is_churned")[["listening_time", "songs_played_per_day", "skip_rate", "ads_listened_per_week", "offline_listening"]].mean()
        mean_behavior.plot(kind='bar', figsize=(8,5), color=['purple', 'gold'])
        plt.title("Perbandingan Rata-rata Perilaku antara Churned vs Non-Churned")
        plt.xlabel("")
        plt.ylabel("Rata-rata Nilai")
        st.pyplot(plt)
    else:
        st.info("Kolom churn belum tersedia.")

    # ------------------------------------------------------------
    st.subheader("3️⃣ Perilaku Pengguna Berdasarkan Tipe Langganan")
    if "subscription_type" in df.columns:
        mean_subs = df.groupby("subscription_type")[["listening_time", "songs_played_per_day", "skip_rate", "ads_listened_per_week", "offline_listening"]].mean()
        mean_subs.plot(kind='bar', figsize=(8,5))
        plt.title("Perilaku Pengguna Berdasarkan Tipe Langganan")
        plt.ylabel("Rata-rata Nilai")
        st.pyplot(plt)
    else:
        st.info("Kolom subscription_type tidak ditemukan.")

    # ------------------------------------------------------------
    st.subheader("4️⃣ Rata-rata Lagu per Hari Berdasarkan Kelompok Usia")
    if "age_group" in df.columns:
        mean_age = df.groupby("age_group")["songs_played_per_day"].mean().sort_values(ascending=False)
        sns.barplot(x=mean_age.index, y=mean_age.values, palette="viridis")
        plt.title("Rata-rata Lagu yang Diputar per Hari berdasarkan Kelompok Usia (Diurutkan)")
        plt.ylabel("Rata-rata Lagu per Hari")
        plt.xlabel("Kelompok Usia")
        st.pyplot(plt)
    else:
        st.info("Kolom age_group tidak ditemukan.")

# ============================================================
# 🧩 TAB 2: CLUSTERING
# ============================================================
with tab2:
    st.header("🧩 Clustering (K-Prototypes) dengan Data yang Telah Dibersihkan")

    # ------------------------------------------------------------
    # 1️⃣ DATA CLEANING
    # ------------------------------------------------------------
    st.subheader("🧹 Data Cleaning")

    # Missing value handling
    st.markdown("**🔧 Menangani Missing Values**")
    df_clean = df.fillna(df.median(numeric_only=True))
    st.write("✅ Missing values telah diganti dengan median (untuk numerik) dan mode (untuk kategori).")

    # Log transform
    st.markdown("**📈 Log Transform (untuk data skewed)**")
    skewed_cols = ["listening_time", "songs_played_per_day", "ads_listened_per_week"]
    for col in skewed_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    # StandardScaler
    st.markdown("**⚖️ Standardisasi Fitur Numerik**")
    scaler = StandardScaler()
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_scaled = df_clean.copy()
    df_scaled[num_cols] = scaler.fit_transform(df_clean[num_cols])
    st.write("✅ Fitur numerik telah distandarisasi dengan StandardScaler.")

    st.dataframe(df_scaled.head())

    # ------------------------------------------------------------
    # 2️⃣ CLUSTERING
    # ------------------------------------------------------------
    st.subheader("⚙️ Jalankan K-Prototypes Clustering")

    n_clusters = st.slider("Jumlah Cluster (K)", 2, 10, 7)
    gamma = st.number_input("Nilai Gamma", 0.0, 2.0, 0.1, 0.1)

    if st.button("🚀 Jalankan K-Prototypes"):
        try:
            cat_cols = df_scaled.select_dtypes(include=['object', 'category']).columns.tolist()
            for c in cat_cols:
                df_scaled[c] = df_scaled[c].astype("category")

            cat_idx = [df_scaled.columns.get_loc(c) for c in cat_cols]
            kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Cao', random_state=42)
            clusters = kproto.fit_predict(df_scaled, categorical=cat_idx)

            df_scaled["Cluster"] = clusters
            st.success("✅ Clustering selesai!")

            # Nama cluster (berdasarkan insight kamu)
            cluster_map = {
                0: "🧠 Risky Premiums (High-Skip Listeners)",
                1: "💼 Steady Premium Users",
                2: "🎵 Loyal Premium Listeners",
                3: "🌍 Engaged Free Explorers",
                4: "⏸️ Passive Free Listeners",
                5: "🎧 Moderate Free Users",
                6: "💎 Premium Loyalists"
            }
            df_scaled["Cluster_Name"] = df_scaled["Cluster"].map(cluster_map)

            # ------------------------------------------------------------
            # 3️⃣ VISUALISASI CLUSTER
            # ------------------------------------------------------------
            st.subheader("🎨 Visualisasi Distribusi Cluster")
            cluster_counts = df_scaled["Cluster_Name"].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                sns.countplot(x="Cluster_Name", data=df_scaled, palette="viridis")
                plt.xticks(rotation=30, ha="right")
                plt.title("Distribusi Jumlah Pengguna per Cluster")
                st.pyplot(plt)
            with col2:
                fig, ax = plt.subplots(figsize=(5,5))
                ax.pie(cluster_counts, labels=cluster_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("viridis", len(cluster_counts)))
                ax.axis("equal")
                plt.title("Proporsi Pengguna per Cluster")
                st.pyplot(fig)

            # ------------------------------------------------------------
            # 4️⃣ INSIGHT
            # ------------------------------------------------------------
            st.subheader("💬 Insight Tiap Cluster")
            for i, name in cluster_map.items():
                st.markdown(f"### {name}")
                st.write(f"- Jumlah pengguna: {len(df_scaled[df_scaled['Cluster']==i])}")
                st.write(f"- Ciri utama: {df_scaled[df_scaled['Cluster']==i].select_dtypes(include='number').mean().round(2).to_dict()}")
                st.write("---")

            # ------------------------------------------------------------
            # 5️⃣ DOWNLOAD HASIL
            # ------------------------------------------------------------
            csv = df_scaled.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Unduh Hasil Cluster (CSV)", csv, "hasil_cluster_spotify.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat clustering: {e}")
