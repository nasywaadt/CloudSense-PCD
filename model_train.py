import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from preprocessing import enhance_image, segment_image, extract_features

def load_dataset(folder_path):
    X, y = [], []
    print(f"📂 Memuat data dari folder: {folder_path}")
    for label in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, label)
        if not os.path.isdir(class_dir):
            continue
        print(f"  🔍 Memproses kelas: {label}")
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"    ⚠️ Gagal membaca gambar: {file}")
                continue
            # Preprocessing & Ekstraksi fitur
            enhanced = enhance_image(img)
            segmented = segment_image(enhanced)
            features = extract_features(segmented)
            X.append(features)
            y.append(label)
    print(f"✅ Selesai memuat {len(X)} data dari {folder_path}\n")
    return np.array(X), np.array(y)

# Load training data
print("🚀 Memulai pelatihan model...")
X_train, y_train = load_dataset('dataset/train')

# Buat pipeline model
print("🔧 Membuat pipeline model...")
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', class_weight='balanced', probability=True)
)

# Konfigurasi GridSearchCV
print("🔍 Melakukan pencarian hyperparameter (GridSearchCV)...")
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 0.01, 0.001],
    'svc__kernel': ['rbf']
}

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print("\n🏆 Best Parameters:", grid.best_params_)

# Simpan model
joblib.dump(grid.best_estimator_, 'model.pkl')
print("💾 Model terbaik disimpan sebagai model.pkl\n")

# Load testing data
print("🧪 Memuat data uji...")
X_test, y_test = load_dataset('dataset/test')

# Prediksi
print("📈 Melakukan prediksi..")
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Confusion Matrix
print("📊 Menampilkan Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()
plt.show()

# Evaluasi
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

print(f"🎯 Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
