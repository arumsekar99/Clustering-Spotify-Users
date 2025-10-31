# ============================================================
# 🎧 Spotify User Clustering Dashboard (K-Prototypes + Naming)
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
# 2️⃣ UPLOAD DATA
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload file CSV kamu di sini", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")

    # Pisahkan kolom numerik dan kategorikal
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown(f"**🔢 Kolom Numerik:** {', '.join(num_cols)}")
    st.markdown(f"**🔠 Kolom Kategorikal:** {', '.join(cat_cols)}")

    # ------------------------------------------------------------
    # 3️⃣ PARAMETER CLUSTERING
    # ------------------------------------------------------------
    st.subheader("⚙️ Pengaturan Clustering")
    n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=4)
    gamma = st.number_input("Nilai Gamma (pengaruh kategori terhadap cluster)", min_value=0.0, max_value=2.0, value=0.1, step=0.1)

    if st.button("🚀 Jalankan K-Prototypes"):
        try:
            # Standarisasi kolom numerik
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

            # Konversi kategorikal ke tipe 'category'
            for col in cat_cols:
                df_scaled[col] = df_scaled[col].astype('category')

            cat_idx = [df_scaled.columns.get_loc(c) for c in cat_cols]

            # ------------------------------------------------------------
            # 4️⃣ JALANKAN K-PROTOTYPES
            # ------------------------------------------------------------
            kproto = KPrototypes(n_clusters=n_clusters, gamma=gamma, init='Cao', random_state=42)
            clusters = kproto.fit_predict(df_scaled, categorical=cat_idx)
            df_clustered = df.copy()
            df_clustered['Cluster'] = clusters

            st.success("✅ Clustering selesai!")

            # ------------------------------------------------------------
            # 5️⃣ RINGKASAN CLUSTER
            # ------------------------------------------------------------
            st.subheader("📊 Ringkasan Tiap Cluster")

            summary_num = df_clustered.groupby('Cluster').mean(numeric_only=True).round(2)
            summary_cat = (
                df_clustered.groupby('Cluster')[cat_cols]
                .agg(lambda x: x.mode().iloc[0])
            )

            st.markdown("**📈 Rata-rata Fitur Numerik per Cluster**")
            st.dataframe(summary_num)

            st.markdown("**🔠 Fitur Kategorikal Dominan per Cluster**")
            st.dataframe(summary_cat)

            # ------------------------------------------------------------
            # 6️⃣ PENAMAAN OTOMATIS CLUSTER
            # ------------------------------------------------------------
            st.subheader("🏷️ Penamaan Otomatis Cluster")

            cluster_names = {}
            for c in sorted(df_clustered['Cluster'].unique()):
                avg_vals = summary_num.loc[c]
                cat_vals = summary_cat.loc[c]

                # Heuristic sederhana buat nama cluster
                name_parts = []
                if 'subscription_type' in cat_vals:
                    name_parts.append("Premium" if cat_vals['subscription_type'] == 'premium' else "Free")
                if 'skip_rate' in avg_vals and avg_vals['skip_rate'] > df['skip_rate'].mean():
                    name_parts.append("Skipers Berat")
                elif 'skip_rate' in avg_vals:
                    name_parts.append("Pendengar Sabar")
                if 'listening_time' in avg_vals and avg_vals['listening_time'] > df['listening_time'].mean():
                    name_parts.append("Pendengar Aktif")
                else:
                    name_parts.append("Pendengar Santai")

                cluster_name = " ".join(name_parts) if name_parts else f"Cluster {c}"
                cluster_names[c] = cluster_name

            df_clustered['Cluster_Name'] = df_clustered['Cluster'].map(cluster_names)

            st.dataframe(df_clustered[['Cluster', 'Cluster_Name']].drop_duplicates().reset_index(drop=True))

            # ------------------------------------------------------------
            # 7️⃣ VISUALISASI
            # ------------------------------------------------------------
            st.subheader("🎨 Visualisasi Distribusi Cluster")

            # Distribusi jumlah anggota per cluster
            fig1, ax1 = plt.subplots(figsize=(7,4))
            sns.countplot(x='Cluster_Name', data=df_clustered, palette='viridis', ax=ax1)
            plt.xticks(rotation=30)
            ax1.set_title('Distribusi Jumlah Anggota per Cluster')
            st.pyplot(fig1)

            # Heatmap rata-rata numerik
            fig2, ax2 = plt.subplots(figsize=(10,6))
            sns.heatmap(summary_num, annot=True, cmap='YlGnBu', ax=ax2)
            ax2.set_title('Rata-rata Fitur Numerik per Cluster')
            st.pyplot(fig2)

            # ------------------------------------------------------------
            # 8️⃣ INSIGHT OTOMATIS
            # ------------------------------------------------------------
            st.subheader("💬 Insight Otomatis Tiap Cluster")

            for c in sorted(df_clustered['Cluster'].unique()):
                st.markdown(f"**{cluster_names[c]} (Cluster {c})**")
                cat_desc = summary_cat.loc[c].to_dict()
                num_desc = summary_num.loc[c].to_dict()
                st.write("- Kategori dominan:", cat_desc)
                st.write("- Rata-rata numerik:", num_desc)
                st.write("---")

            # ------------------------------------------------------------
            # 9️⃣ DOWNLOAD HASIL
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

else:
    st.info("👆 Upload file CSV untuk memulai analisis.")
