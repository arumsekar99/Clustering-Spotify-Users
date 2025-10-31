# ============================================================
# 🎧 Spotify User Segmentation Dashboard (EDA + K-Prototypes)
# ============================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

# ------------------------------------------------------------
# 1️⃣ SETUP DASHBOARD
# ------------------------------------------------------------
st.set_page_config(page_title="Spotify User Segmentation Dashboard", layout="wide")
st.title("🎧 Spotify User Segmentation Dashboard (EDA + K-Prototypes)")
st.markdown("Analisis segmentasi pengguna Spotify berdasarkan perilaku mendengarkan dan karakteristik demografis.")

# ------------------------------------------------------------
# 2️⃣ LOAD DATA
# ------------------------------------------------------------
try:
    df = pd.read_csv("spotify_churn_dataset.csv")
    st.subheader("📂 Data Awal")
    st.dataframe(df.head())
    st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

# ------------------------------------------------------------
# 3️⃣ EDA (Exploratory Data Analysis)
# ------------------------------------------------------------
st.header("🔍 Exploratory Data Analysis (EDA)")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🔢 Kolom Numerik:** " + ", ".join(num_cols))
with col2:
    st.markdown("**🔠 Kolom Kategorikal:** " + ", ".join(cat_cols))

# Ringkasan Statistik
st.subheader("📊 Ringkasan Statistik Numerik")
st.dataframe(df[num_cols].describe().T)

# Distribusi Fitur Numerik
st.subheader("📈 Distribusi Fitur Numerik")
selected_num = st.selectbox("Pilih fitur numerik:", num_cols)
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df[selected_num], kde=True, color='teal', ax=ax)
ax.set_title(f"Distribusi {selected_num}")
st.pyplot(fig)

# Distribusi Kategorikal
st.subheader("📦 Distribusi Fitur Kategorikal")
selected_cat = st.selectbox("Pilih fitur kategorikal:", cat_cols)
fig, ax = plt.subplots(figsize=(6, 4))
df[selected_cat].value_counts().plot(kind='bar', color='orange', ax=ax)
ax.set_title(f"Distribusi {selected_cat}")
st.pyplot(fig)

# ------------------------------------------------------------
# 4️⃣ CLUSTERING (K-PROTOTYPES)
# ------------------------------------------------------------
st.header("🧩 K-Prototypes Clustering")

n_clusters = st.slider("Jumlah Cluster (K)", 2, 10, 7)
gamma = st.number_input("Nilai Gamma (pengaruh kategori terhadap cluster)", min_value=0.0, max_value=2.0, value=0.1, step=0.1)

if st.button("🚀 Jalankan Clustering"):
    try:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
        for col in cat_cols:
            df_scaled[col] = df_scaled[col].astype('category')
        cat_idx = [df_scaled.columns.get_loc(c) for c in cat_cols]

        kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Cao', random_state=42)
        clusters = kproto.fit_predict(df_scaled, categorical=cat_idx)
        df['Cluster'] = clusters

        st.success("✅ Clustering selesai!")

        # ------------------------------------------------------------
        # 5️⃣ NAMA CLUSTER (dari hasil insight kamu)
        # ------------------------------------------------------------
        cluster_name_map = {
            0: "🧠 Risky Premiums (High-Skip Listeners)",
            1: "💼 Steady Premium Users",
            2: "🎵 Loyal Premium Listeners",
            3: "🌍 Engaged Free Explorers",
            4: "⏸️ Passive Free Listeners",
            5: "🎧 Moderate Free Users",
            6: "💎 Premium Loyalists"
        }
        df["Cluster_Name"] = df["Cluster"].map(cluster_name_map)

        # ------------------------------------------------------------
        # 6️⃣ RINGKASAN CLUSTER
        # ------------------------------------------------------------
        st.subheader("📋 Ringkasan Tiap Cluster")
        summary_num = df.groupby("Cluster_Name")[num_cols].mean().round(2)
        summary_cat = df.groupby("Cluster_Name")[cat_cols].agg(lambda x: x.mode().iloc[0])
        st.dataframe(summary_num)
        st.dataframe(summary_cat)

        # ------------------------------------------------------------
        # 7️⃣ VISUALISASI CLUSTER
        # ------------------------------------------------------------
        st.subheader("🎨 Visualisasi Distribusi Cluster")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.countplot(x='Cluster_Name', data=df, palette='viridis', ax=ax1)
            plt.xticks(rotation=30, ha='right')
            ax1.set_title("Distribusi Jumlah Pengguna per Cluster")
            st.pyplot(fig1)
        with col2:
            cluster_counts = df["Cluster_Name"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(5,5))
            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(cluster_counts)))
            ax2.axis('equal')
            st.pyplot(fig2)

        # ------------------------------------------------------------
        # 8️⃣ HEATMAP
        # ------------------------------------------------------------
        st.subheader("🔥 Heatmap Fitur Numerik per Cluster")
        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.heatmap(summary_num, annot=True, cmap='YlGnBu', ax=ax3)
        ax3.set_title("Rata-rata Fitur Numerik per Cluster")
        st.pyplot(fig3)

        # ------------------------------------------------------------
        # 9️⃣ INSIGHT OTOMATIS
        # ------------------------------------------------------------
        st.subheader("💬 Insight Singkat Tiap Cluster")
        cluster_insights = {
            0: "Sering skip dan jarang mendengar lama → indikasi ketidakpuasan terhadap playlist.",
            1: "Pengguna stabil dengan engagement sedang.",
            2: "Loyal Premium, aktif mendengarkan musik.",
            3: "Free user dengan engagement tinggi → churn rendah.",
            4: "Aktivitas rendah, potensi churn sedang.",
            5: "Engagement sedang, churn menengah.",
            6: "Aktif dan stabil, loyal terhadap layanan Premium."
        }

        for idx, name in cluster_name_map.items():
            st.markdown(f"### {name}")
            st.write(cluster_insights[idx])
            st.write("---")

        # ------------------------------------------------------------
        # 🔟 FILTER INTERAKTIF
        # ------------------------------------------------------------
        st.subheader("🔍 Eksplorasi Interaktif per Cluster")
        selected_cluster = st.selectbox("Pilih cluster:", list(cluster_name_map.values()))
        filtered = df[df["Cluster_Name"] == selected_cluster]
        st.dataframe(filtered)

        num_choice = st.selectbox("Pilih fitur numerik untuk lihat distribusi:", num_cols)
        fig4, ax4 = plt.subplots(figsize=(6,4))
        sns.histplot(filtered[num_choice], kde=True, color='teal', ax=ax4)
        ax4.set_title(f"Distribusi {num_choice} - {selected_cluster}")
        st.pyplot(fig4)

        # ------------------------------------------------------------
        # 11️⃣ COMPARE CLUSTERS
        # ------------------------------------------------------------
        st.subheader("⚖️ Bandingkan Dua Cluster")
        col_a, col_b = st.columns(2)
        with col_a:
            c1 = st.selectbox("Cluster 1:", list(cluster_name_map.values()), key="c1")
        with col_b:
            c2 = st.selectbox("Cluster 2:", list(cluster_name_map.values()), key="c2")

        compare_num = pd.concat([
            summary_num.loc[[c1]].T.rename(columns={c1: "Cluster 1"}),
            summary_num.loc[[c2]].T.rename(columns={c2: "Cluster 2"})
        ], axis=1)
        st.dataframe(compare_num)

        # ------------------------------------------------------------
        # 12️⃣ DOWNLOAD HASIL
        # ------------------------------------------------------------
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Unduh Hasil Cluster (CSV)", csv, "hasil_cluster_spotify.csv", "text/csv")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
