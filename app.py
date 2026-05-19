import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import deteksi_cerdas, isolasi_objek

# Konfigurasi Halaman Web
st.set_page_config(page_title="Multi-Orange Classifier Pro", layout="wide")

st.title("Sistem Klasifikasi Tingkat Kematangan Buah Jeruk")
st.write("Sistem Klasifikasi Jenis Jeruk & Tingkat Kematangan Berdasarkan Fitur Geometri (Circularity) dan Warna (HSV).")
st.divider()

# Sidebar untuk Unggah Gambar
uploaded_file = st.sidebar.file_uploader("Upload Gambar Buah", type=["jpg", "png", "jpeg"])

st.sidebar.info("""
**💡 Petunjuk Pengujian:**
1. Unggah foto jeruk dengan latar belakang kontras/polos.
2. Sistem akan memproses gambar secara bertahap (Pipeline PCD).
3. Hasil ekstraksi geometri dan warna akan disimpulkan di panel kanan.
""")

if uploaded_file:
    # Mengubah gambar yang diunggah ke format OpenCV (BGR)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 1. Jalankan Pemrosesan Citra Digital via utils.py
    mask, hasil_deteksi = deteksi_cerdas(img_bgr)
    img_isolasi = isolasi_objek(img_bgr, mask)
    
    # Membagi layout menjadi 2 kolom besar (Kiri: Proses Foto, Kanan: Hasil & Kesimpulan)
    col_proses, col_kesimpulan = st.columns([3, 2])

    with col_proses:
        st.subheader("🖼️ Tahapan Operasi Pengolahan Citra (Pipeline)")
        
        # Buat grid 2x2 untuk menampilkan langkah-langkah operasi citra
        grid1, grid2 = st.columns(2)
        with grid1:
            st.image(image, use_container_width=True, caption="Langkah 1: Citra Asli (RGB Input)")
            
            # Membuat visualisasi HSV untuk ditampilkan
            hsv_visual = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            st.image(hsv_visual, use_container_width=True, caption="Langkah 2: Konversi Ruang Warna HSV")
            
        with grid2:
            st.image(mask, use_container_width=True, caption="Langkah 3: Binarisasi & Masking (Segmentasi)")
            
            # Menggambar Bounding Box di atas gambar isolasi untuk hasil akhir
            img_display = img_isolasi.copy()
            for i, buah in enumerate(hasil_deteksi):
                x, y, w, h = buah['box']
                cv2.rectangle(img_display, (x, y), (x+w, y+h), buah['warna'], 5)
                cv2.putText(img_display, f"ID:{i+1}", (x, y-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, buah['warna'], 3)
            
            st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Langkah 4: Isolasi Objek & ROI Bounding Box")

    with col_kesimpulan:
        st.subheader("📊 Hasil Analisis & Kesimpulan Akhir")
        
        if hasil_deteksi:
            # 1. Tampilkan Ringkasan Inventaris
            stats = {}
            for b in hasil_deteksi:
                stats[b['jenis']] = stats.get(b['jenis'], 0) + 1
            
            for jenis, jumlah in stats.items():
                st.metric(label=f"Total Terdeteksi ({jenis})", value=f"{jumlah} Buah")
            
            st.divider()
            
            # 2. Tampilkan Box Kesimpulan Besar untuk Setiap Buah yang Terdeteksi
            for i, b in enumerate(hasil_deteksi):
                st.markdown(f"### **Buah ID: {i+1}**")
                
                # Mengubah warna tampilan box Streamlit sesuai status kematangan
                if b['status'] == "Matang":
                    st.success(f"**VARIETAS:** {b['jenis']} \n\n **KESIMPULAN:** KONDISI {b['status'].upper()}")
                elif b['status'] == "Setengah Matang":
                    st.warning(f"**VARIETAS:** {b['jenis']} \n\n **KESIMPULAN:** KONDISI {b['status'].upper()}")
                else:
                    st.error(f"**VARIETAS:** {b['jenis']} \n\n **KESIMPULAN:** KONDISI {b['status'].upper()}")
                
                # Tampilkan detail angka ekstraksi fiturnya di bawah box kesimpulan
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.caption(f"**Nilai Hue (Warna):** {b['hue']:.1f}")
                    st.caption(f"**Luas Area:** {b['area']:.0f} px")
                with col_meta2:
                    st.caption(f"**Rasio Aspek (W/H):** {b['aspect_ratio']:.2f}")
                st.divider()
                
            # 3. Tabel Data Komparasi untuk Laporan
            st.write("**Tabel Parameter Fitur PCD:**")
            table_data = []
            for i, b in enumerate(hasil_deteksi):
                table_data.append({
                    "ID": i+1,
                    "Varietas Jenis": b['jenis'],
                    "Status Kematangan": b['status'],
                    "Nilai Hue": f"{b['hue']:.1f}",
                    "Area Size": f"{b['area']:.0f}"
                })
            st.dataframe(table_data, use_container_width=True)
            
        else:
            st.error("Sistem gagal menyimpulkan. Objek buah tidak terisolasi dengan baik oleh Masking. Periksa kondisi latar belakang foto.")

    # Bagian Teori Pendukung Sidang/Tugas
    with st.expander("🔬 Penjelasan Alur Operasi Matematika PCD"):
        st.markdown("""
        Bagaimana sistem mengambil kesimpulan? Berikut urutan operasinya:
        1. **Ekstraksi Citra Input:** Gambar dibaca dalam matriks BGR oleh OpenCV.
        2. **Filter Gaussian Blur:** Meratakan intensitas piksel tetangga untuk mereduksi *noise* halus.
        3. **Operasi Ambang Batas (Thresholding HSV):** Memisahkan rentang nilai *Hue* jeruk dari latar belakang putih.
        4. **Operasi Morfologi (Open & Close):** Menambal area kosong di dalam buah (*Close*) dan memotong piksel liar di luar buah (*Open*).
        5. **Ekstraksi Geometri & Warna:** Menghitung luas kontur untuk mendeteksi varietas ukuran (seperti Jeruk Bali) serta mendeteksi bentuk lonjong melalui *circularity* (Lemon). Rata-rata *Hue* dihitung pada area *masking* murni untuk menyimpulkan tingkat kematangan (**Matang / Setengah Matang / Belum Matang**).
        """)
else:
    # Tampilan awal saat web baru dibuka dan belum ada gambar
    st.info("👋 Selamat Datang! Silakan unggah foto sampel buah jeruk pada menu di sebelah kiri untuk melihat proses operasi citra digital.")