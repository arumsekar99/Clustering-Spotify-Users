# ============================================================
# 🎧 Spotify User Clustering Dashboard (Full Interactive + Insights)
# ============================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1️⃣ SETUP DASHBOARD
# ------------------------------------------------------------
st.set_page_config(page_title="Spotify User Clustering Dashboard", layout="wide")
st.title("🎧 Spotify User Clustering Dashboard (K-Prototypes)")
st.markdown("Segmentasi pengguna Spotify berdasarkan perilaku mendengarkan dan karakteristik demografis.")

# ------------------------------------------------------------
# 2️⃣ AUTO LOAD DATASET
# ------------------------------------------------------------
try:
    # 🔹 Pilih salah satu:
    df = pd.read_csv("spotify_churn_dataset.csv")  # lokal

    # atau load dari GitHub:
    # df = pd.read_csv("https://raw.githubusercontent.com/username/repo-name/main/spotify_churn_dataset.csv")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
    st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

# ------------------------------------------------------------
# 3️⃣ IDENTIFIKASI KOLOM
# ------------------------------------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

st.markdown(f"**🔢 Kolom Numerik:** {', '.join(num_cols)}")
st.markdown(f"**🔠 Kolom Kategorikal:** {', '.join(cat_cols)}")

# ------------------------------------------------------------
# 4️⃣ PENGATURAN CLUSTERING
# ------------------------------------------------------------
st.subheader("⚙️ Pengaturan Clustering")
n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=4)
gamma = st.number_input("Nilai Gamma (pengaruh kategori terhadap cluster)", min_value=0.0, max_value=2.0, value=0.1, step=0.1)

if st.button("🚀 Jalankan K-Prototypes"):
    try:
        # ------------------------------------------------------------
        # 5️⃣ STANDARISASI & MODEL
        # ------------------------------------------------------------
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

        for col in cat_cols:
            df_scaled[col] = df_scaled[col].astype('category')

        cat_idx = [df_scaled.columns.get_loc(c) for c in cat_cols]

        kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Cao', random_state=42)
        clusters = kproto.fit_predict(df_scaled, categorical=cat_idx)

        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        st.success("✅ Clustering selesai!")

        # ------------------------------------------------------------
        # 6️⃣ RINGKASAN CLUSTER
        # ------------------------------------------------------------
        summary_num = df_clustered.groupby('Cluster').mean(numeric_only=True).round(2)
        summary_cat = (
            df_clustered.groupby('Cluster')[cat_cols]
            .agg(lambda x: x.mode().iloc[0])
        )

        st.subheader("📊 Ringkasan Tiap Cluster")
        st.dataframe(summary_num)
        st.dataframe(summary_cat)

        # ------------------------------------------------------------
        # 7️⃣ PENAMAAN & INSIGHT OTOMATIS
        # ------------------------------------------------------------
        st.subheader("💬 Penamaan & Insight Otomatis")

        cluster_names = {}
        cluster_insights = {}

        for c in sorted(df_clustered['Cluster'].unique()):
            avg_vals = summary_num.loc[c]
            cat_vals = summary_cat.loc[c]
            name_parts, insight_parts = [], []

            # Penamaan otomatis
            if 'subscription_type' in cat_vals:
                if cat_vals['subscription_type'].lower() == 'premium':
                    name_parts.append("Premium")
                    insight_parts.append("mayoritas pengguna premium")
                else:
                    name_parts.append("Free")
                    insight_parts.append("mayoritas pengguna akun gratis")

            if 'skip_rate' in avg_vals:
                if avg_vals['skip_rate'] > df['skip_rate'].mean():
                    name_parts.append("Skipers Berat")
                    insight_parts.append("cenderung sering melewati lagu")
                else:
                    name_parts.append("Pendengar Sabar")
                    insight_parts.append("jarang melakukan skip lagu")

            if 'listening_time' in avg_vals:
                if avg_vals['listening_time'] > df['listening_time'].mean():
                    name_parts.append("Pendengar Aktif")
                    insight_parts.append("memiliki durasi mendengarkan yang tinggi")
                else:
                    name_parts.append("Pendengar Santai")
                    insight_parts.append("memiliki durasi mendengarkan yang rendah")

            cluster_name = " ".join(name_parts) if name_parts else f"Cluster {c}"
            cluster_names[c] = cluster_name

            sentence = (
                f"Cluster {c} disebut **{cluster_name}**, karena {', '.join(insight_parts)}. "
                f"Rata-rata waktu dengar: {avg_vals.get('listening_time', 'N/A')}, "
                f"rata-rata skip rate: {avg_vals.get('skip_rate', 'N/A')}."
            )
            cluster_insights[c] = sentence

        df_clustered['Cluster_Name'] = df_clustered['Cluster'].map(cluster_names)

        # ------------------------------------------------------------
        # 8️⃣ VISUALISASI CLUSTER + PIE CHART
        # ------------------------------------------------------------
        st.subheader("🎨 Visualisasi Distribusi Cluster")

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.countplot(x='Cluster_Name', data=df_clustered, palette='viridis', ax=ax1)
            plt.xticks(rotation=30)
            ax1.set_title('Distribusi Jumlah Anggota per Cluster')
            st.pyplot(fig1)

        with col2:
            cluster_counts = df_clustered['Cluster_Name'].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(5,5))
            ax_pie.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(cluster_counts)))
            ax_pie.axis('equal')
            plt.title("Proporsi Pengguna per Cluster")
            st.pyplot(fig_pie)

        # ------------------------------------------------------------
        # 9️⃣ HEATMAP NUMERIK
        # ------------------------------------------------------------
        st.subheader("🔥 Heatmap Rata-rata Fitur Numerik")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.heatmap(summary_num, annot=True, cmap='YlGnBu', ax=ax2)
        ax2.set_title('Rata-rata Fitur Numerik per Cluster')
        st.pyplot(fig2)

        # ------------------------------------------------------------
        # 🔟 INSIGHT NATURAL-LANGUAGE
        # ------------------------------------------------------------
        st.subheader("🧠 Insight Natural Tiap Cluster")
        for c in sorted(df_clustered['Cluster'].unique()):
            st.markdown(f"### {cluster_names[c]} (Cluster {c})")
            st.write(cluster_insights[c])
            st.write("- **Kategori dominan:**", summary_cat.loc[c].to_dict())
            st.write("- **Rata-rata numerik:**", summary_num.loc[c].to_dict())
            st.write("---")

        # ------------------------------------------------------------
        # 11️⃣ FILTER INTERAKTIF PER CLUSTER
        # ------------------------------------------------------------
        st.subheader("🔍 Eksplorasi Interaktif per Cluster")
        selected_cluster = st.selectbox("Pilih Cluster untuk dianalisis:", sorted(df_clustered['Cluster'].unique()))
        filtered_data = df_clustered[df_clustered['Cluster'] == selected_cluster]

        st.markdown(f"**Menampilkan {len(filtered_data)} pengguna dari {cluster_names[selected_cluster]} (Cluster {selected_cluster})**")
        st.dataframe(filtered_data)

        # Plot distribusi numerik di cluster terpilih
        st.markdown("**Distribusi Fitur Numerik di Cluster ini:**")
        num_col_choice = st.selectbox("Pilih fitur numerik untuk dilihat distribusinya:", num_cols)
        fig3, ax3 = plt.subplots(figsize=(6,4))
        sns.histplot(filtered_data[num_col_choice], bins=20, kde=True, color='teal', ax=ax3)
        ax3.set_title(f"Distribusi {num_col_choice} - {cluster_names[selected_cluster]}")
        st.pyplot(fig3)

        # ------------------------------------------------------------
        # 12️⃣ DOWNLOAD HASIL
        # ------------------------------------------------------------
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Unduh Hasil Cluster (CSV)",
            data=csv,
            file_name="hasil_cluster_spotify.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan clustering: {e}")
