import cv2
import numpy as np

def deteksi_cerdas(img_bgr):
    # 1. Preprocessing (Smoothing & Blur)
    blurred = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 2. Adaptive Masking (Mengambil spektrum hijau hingga oranye pekat)
    lower = np.array([0, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # 3. Morfologi untuk membersihkan noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 4. Deteksi Kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    daftar_buah = []
    mask_isolasi = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0: continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        if area > 1500 and extent > 0.6:
            
            roi_hsv = hsv[y:y+h, x:x+w]
            roi_mask = mask[y:y+h, x:x+w]
            
            if cv2.countNonZero(roi_mask) == 0:
                continue
                
            avg_hue = np.sum(roi_hsv[:, :, 0] * (roi_mask / 255)) / np.sum(roi_mask / 255)
            avg_sat = np.sum(roi_hsv[:, :, 1] * (roi_mask / 255)) / np.sum(roi_mask / 255)
            
            if area > 50000 and circularity > 0.80 and avg_hue > 25:
                jenis = "Jeruk Bali"
            
            elif (circularity < 0.82 or aspect_ratio < 0.82 or aspect_ratio > 1.12) and avg_hue > 22:
                jenis = "Lemon"
            
            elif area < 8000 and circularity >= 0.82 and avg_hue > 35 and avg_sat > 100:
                jenis = "Jeruk Nipis / Limau"
            
            else:
                jenis = "Jeruk Umum / Sunkist"

            if avg_hue > 45:
                status = "Belum Matang"
                warna = (0, 0, 255)
            elif 25 <= avg_hue <= 45:
                status = "Setengah Matang"
                warna = (0, 255, 255)
            else:
                status = "Matang"
                warna = (0, 255, 0)

            daftar_buah.append({
                'box': (x, y, w, h),
                'jenis': jenis,
                'status': status,
                'warna': warna,
                'hue': avg_hue,
                'aspect_ratio': aspect_ratio,
                'area': area
            })
            
            cv2.drawContours(mask_isolasi, [cnt], -1, 255, -1)

    return mask_isolasi, daftar_buah

def isolasi_objek(img_bgr, mask):
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)