import cv2
import numpy as np

def segmentasi_jeruk(img_bgr):
    # 1. Konversi ke HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. Tentukan rentang warna jeruk (kulit jeruk umumnya berada di sini)
    # Kita ambil rentang luas dari hijau muda ke oranye tua
    lower_orange = np.array([0, 50, 50])
    upper_orange = np.array([100, 255, 255])
    
    # 3. Buat Mask (Topeng)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # 4. Operasi Morfologi (untuk menutup lubang kecil pada deteksi)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. Terapkan Mask ke gambar asli
    hasil_segmentasi = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    
    return hsv, mask, hasil_segmentasi

def hitung_kematangan(hsv_img, mask):
    # Hanya ambil nilai Hue pada area yang ada mask-nya (bukan hitam)
    hue_channel = hsv_img[:, :, 0]
    pixels_jeruk = hue_channel[mask > 0]
    
    if len(pixels_jeruk) == 0:
        return 0
    
    avg_hue = np.mean(pixels_jeruk)
    return avg_hue