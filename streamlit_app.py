import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

# ======================
# 1. SETUP DASHBOARD
# ======================
st.set_page_config(page_title="Spotify User Clustering Dashboard", layout="wide")
st.title("🎧 Spotify User Clustering Dashboard (K-Prototypes)")
st.markdown("Analisis segmentasi pengguna Spotify berdasarkan perilaku mendengarkan dan karakteristik demografis.")

# ======================
# 2. UPLOAD DATA
# ======================
uploaded_file = st.file_uploader("spotify_churn_dataset.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
    
    st.write("**Info Data:**")
    st.write(f"- Jumlah baris: {df.shape[0]}")
    st.write(f"- Jumlah kolom: {df.shape[1]}")
    
    # Pisahkan kolom numerik dan kategorikal
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.markdown("**🔢 Kolom Numerik:** " + ", ".join(num_cols))
    st.markdown("**🔠 Kolom Kategorikal:** " + ", ".join(cat_cols))
    
    # ======================
    # 3. PARAMETER CLUSTERING
    # ======================
    st.subheader("⚙️ Pengaturan Clustering")
    n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=4)
    st.write("**Catatan:** Pastikan kolom kategorikal dan numerik sudah benar sebelum menjalankan clustering.")
    
    if st.button("🚀 Jalankan K-Prototypes"):
        try:
            # Standarisasi numerik
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

            # Konversi kategorikal ke tipe 'category'
            for col in cat_cols:
                df_scaled[col] = df_scaled[col].astype('category')
            
            # Jalankan K-Prototypes
            kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=0, random_state=42)
            clusters = kproto.fit_predict(df_scaled, categorical=[df_scaled.columns.get_loc(c) for c in cat_cols])
            df['Cluster'] = clusters
            
            st.success("✅ Clustering selesai!")
            
            # ======================
            # 4. VISUALISASI HASIL
            # ======================
            st.subheader("🧩 Hasil Klustering")
            st.dataframe(df.head())
            
            cluster_counts = df['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            st.markdown("**Distribusi Cluster:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Cluster', data=df, palette='viridis')
            ax.set_title("Distribusi Jumlah Anggota per Cluster")
            st.pyplot(fig)
            
            # Rata-rata tiap fitur numerik per cluster
            st.markdown("**📈 Rata-rata Fitur Numerik per Cluster**")
            st.dataframe(df.groupby('Cluster')[num_cols].mean().round(2))
            
            # ======================
            # 5. INSIGHT SINGKAT
            # ======================
            st.subheader("💬 Insight Tiap Cluster")
            for c in sorted(df['Cluster'].unique()):
                st.markdown(f"**Cluster {c}:**")
                st.write("- Karakteristik umum berdasarkan rata-rata fitur numerik dan dominasi kategori.")
                st.write(df[df['Cluster'] == c][cat_cols].mode().iloc[0].to_dict())
                st.write("---")
            
            # Unduh hasil cluster
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Unduh Hasil Cluster (CSV)",
                data=csv,
                file_name="hasil_cluster_spotify.csv",
                mime="text/csv",
            )
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👆 Upload file CSV untuk memulai analisis.")
