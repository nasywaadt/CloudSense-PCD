import cv2
import joblib
import numpy as np
from preprocessing import enhance_image, segment_image, extract_features

# Load model hanya sekali saat file di-import
model = joblib.load('model.pkl')

# Tambahkan info cuaca sesuai label awan
weather_map = {
    'Cirrus': 'Cerah/tidak hujan',
    'Cumulonimbus': 'Hujan deras atau badai petir',
    'Cumulus': 'Cerah sebagian atau berawan ringan',
    'Stratus': 'Berawan, kemungkinan gerimis'
}


def classify_cloud(image_cv):
    enhanced = enhance_image(image_cv)
    segmented = segment_image(enhanced)
    features = extract_features(segmented).reshape(1, -1)

    label = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features)) * 100
    weather = weather_map.get(label, "Informasi cuaca tidak tersedia")

    return label, confidence, weather
