import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import segmentasi_jeruk, hitung_kematangan

# Konfigurasi Halaman
st.set_page_config(
    page_title="Klasifikasi Jeruk - PCD",
    page_icon="🍊",
    layout="centered"
)

# Kustomisasi CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🍊 Sistem Klasifikasi Kematangan Jeruk")
st.write("Proyek Pengantar Citra Digital - Deteksi Berdasarkan Ruang Warna HSV")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("📁 Menu Utama")
uploaded_file = st.sidebar.file_uploader("Unggah Foto Jeruk", type=["jpg", "jpeg", "png"])

st.sidebar.info("""
**Instruksi:**
1. Unggah foto jeruk dengan latar belakang polos.
2. Pastikan cahaya cukup (tidak terlalu gelap).
3. Aplikasi akan menghitung nilai Hue rata-rata pada objek.
""")

# --- MAIN CONTENT ---
if uploaded_file is not None:
    # 1. Konversi file upload ke format yang dikenali OpenCV
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Karena PIL menggunakan RGB dan OpenCV menggunakan BGR
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 2. Proses Citra (Memanggil fungsi dari utils.py)
    hsv, mask, hasil_seg = segmentasi_jeruk(img_bgr)
    avg_hue = hitung_kematangan(hsv, mask)

    # 3. Logika Klasifikasi Berdasarkan Nilai Hue (H)
    # Range Hue OpenCV (0-180): 
    # Hijau biasanya > 35, Oranye/Kuning biasanya 0-25
    if avg_hue == 0:
        status = "Objek Tidak Terdeteksi"
        warna_box = "error"
        saran = "Coba gunakan latar belakang yang lebih kontras atau pastikan objek adalah jeruk."
    elif avg_hue > 35:
        status = "MENTAH (HIJAU)"
        warna_box = "error" # Merah di Streamlit (st.error)
        saran = "Buah masih hijau pekat, belum siap panen."
    elif 22 <= avg_hue <= 35:
        status = "SETENGAH MATANG (ORANYE KEKUNINGAN)"
        warna_box = "warning" # Kuning di Streamlit (st.warning)
        saran = "Buah dalam masa transisi kematangan."
    else:
        status = "MATANG (ORANYE/KUNING)"
        warna_box = "success" # Hijau di Streamlit (st.success)
        saran = "Buah sudah matang sempurna dan siap dikonsumsi."

    # --- TAMPILAN UI ---
    
    # Baris 1: Hasil Klasifikasi
    st.subheader("📊 Hasil Analisis")
    if warna_box == "success":
        st.success(f"### **{status}**")
    elif warna_box == "warning":
        st.warning(f"### **{status}**")
    else:
        st.error(f"### **{status}**")
    
    st.write(f"**Rata-rata Nilai Hue:** `{avg_hue:.2f}`")
    st.caption(f"Keterangan: {saran}")
    
    st.divider()

    # Baris 2: Visualisasi Proses PCD
    st.subheader("🖼️ Visualisasi Pengolahan Citra")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="1. Citra Asli", use_container_width=True)
    
    with col2:
        st.image(mask, caption="2. Masking (Segmentasi)", use_container_width=True)
    
    with col3:
        # Konversi BGR ke RGB untuk tampilan Streamlit yang benar
        hasil_seg_rgb = cv2.cvtColor(hasil_seg, cv2.COLOR_BGR2RGB)
        st.image(hasil_seg_rgb, caption="3. Hasil Ekstraksi", use_container_width=True)

    # Baris 3: Detail Teknis untuk Laporan
    with st.expander("Lihat Detail Teknis (Data Ruang Warna)"):
        st.write("Analisis Ruang Warna HSV (Hue, Saturation, Value):")
        st.image(hsv, caption="Visualisasi Komponen HSV", use_container_width=True)
        st.info("""
        **Penjelasan:** 
        Proses ini memisahkan warna jeruk dari background menggunakan metode *Color Thresholding*. 
        Nilai Hue yang didapat adalah representasi sudut warna dalam lingkaran warna (0-180 di OpenCV).
        """)

else:
    # Tampilan awal saat belum ada file
    st.image("https://images.unsplash.com/photo-1582979512210-99b6a53386f9?auto=format&fit=crop&q=80&w=800", use_container_width=True)
    st.warning("👈 Silakan unggah foto jeruk pada sidebar untuk memulai.")

st.sidebar.divider()
st.sidebar.caption("Dibuat untuk Tugas Pengantar Citra Digital")