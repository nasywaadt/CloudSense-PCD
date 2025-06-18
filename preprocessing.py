import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def enhance_image(image):
    # Ubah ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE (untuk meningkatkan kontras lokal)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Bilateral filter (untuk meredam noise sambil menjaga edge)
    enhanced = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    return enhanced


def segment_image(enhanced_img):
    _, binary = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def extract_features(image):
    features = []

    # Ubah ke BGR
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize untuk standarisasi fitur
    gray = cv2.resize(gray, (128, 128))

    # Histogram HSV (warna)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    hist = np.concatenate((hist_h, hist_s, hist_v)).flatten()
    hist = hist / np.sum(hist)
    features.extend(hist)

    # LBP (tekstur)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-6)
    features.extend(hist_lbp)

    # Hu Moments (bentuk)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(np.log1p(np.abs(hu_moments)))

    # Kontur utama (fitur bentuk)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2 + 1e-6))
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        convexity = area / (hull_area + 1e-6)   

        x, y, w, h = cv2.boundingRect(c)
        extent = area / (w * h + 1e-6)  # Fitur baru: extent ratio
    else:
        circularity = convexity = extent = 0

    features.extend([circularity, convexity, extent])

    # Fitur baru: brightness
    mean_brightness = np.mean(gray)
    features.append(mean_brightness)

    return np.array(features)

