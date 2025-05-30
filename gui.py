import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk  # ImageEnhance tidak lagi diperlukan untuk resize kualitas tinggi dengan Image.LANCZOS
import cv2
import os
import numpy as np
from PIL import ImageDraw, ImageFont  # Tambahkan impor ini untuk placeholder jika tidak ada file

# Pastikan classify_cloud.py berada di direktori yang sama atau di PATH
from classify_cloud import classify_cloud


class CloudClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("☁️ Aplikasi Klasifikasi Awan")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        self.root.configure(bg="#e0f2f7")

        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Gaya untuk Frame (utama dan content_frame)
        style.configure("TFrame", background="#e0f2f7")

        # --- PERBAIKAN UTAMA DI SINI ---
        # Gaya baru untuk result_container
        # Nama style harus unik, misal "Result.TFrame"
        style.configure("ResultContainer.TFrame",
                        background="#ffffff",  # Warna latar belakang yang diinginkan
                        padding=15,
                        relief="flat",
                        borderwidth=0)  # borderwidth 0 karena relief "flat"

        # Gaya untuk Tombol
        style.configure("TButton",
                        font=("Segoe UI", 12, "bold"),
                        background="#007bff",
                        foreground="white",
                        padding=10)
        style.map("TButton",
                  background=[('active', '#0056b3')],
                  foreground=[('active', 'white')])

        # Gaya untuk Label Judul
        style.configure("Title.TLabel",
                        font=("Segoe UI", 24, "bold"),
                        background="#e0f2f7",
                        foreground="#333333",
                        padding=(0, 10))

        # Gaya untuk Label Biasa (status_label)
        style.configure("TLabel",
                        font=("Segoe UI", 12),
                        background="#e0f2f7",
                        foreground="#333333")

        # Gaya untuk Label Hasil (lebih menonjol)
        style.configure("ResultValue.TLabel",  # Ubah nama style agar tidak bentrok dengan T.Label
                        # Atau buat style baru jika ingin padding dan border khusus
                        font=("Segoe UI", 14, "bold"),
                        background="#ffffff",  # Latar putih untuk hasil
                        foreground="#0056b3",  # Biru gelap
                        padding=15,
                        relief="solid",  # Border
                        borderwidth=1,
                        anchor="center")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="30 20 30 30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        title_label = ttk.Label(main_frame, text="☁️ APLIKASI KLASIFIKASI AWAN ☁️", style="Title.TLabel")
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.upload_button = ttk.Button(main_frame, text="📂 Pilih Gambar Awan", command=self.load_image)
        self.upload_button.grid(row=1, column=0, columnspan=2, pady=20, sticky="n")

        content_frame = ttk.Frame(main_frame, padding=15, relief="groove")
        content_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=10)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(content_frame, background="#cccccc", relief="solid", borderwidth=1)
        self.image_label.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        # --- PERBAIKAN KEDUA DI SINI ---
        # Terapkan style "ResultContainer.TFrame" yang baru dibuat
        result_container = ttk.Frame(content_frame, style="ResultContainer.TFrame")
        result_container.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        result_container.columnconfigure(0, weight=1)

        self.result_title_label = ttk.Label(result_container, text="HASIL KLASIFIKASI",
                                            font=("Segoe UI", 16, "bold"), foreground="#007bff",
                                            background="#ffffff")  # Label ini juga harus pakai background
        self.result_title_label.pack(pady=(0, 10))

        self.cloud_class_label = ttk.Label(result_container, text="☁️ Kelas Awan: -",
                                           style="ResultValue.TLabel", wraplength=250)  # Pakai ResultValue.TLabel
        self.cloud_class_label.pack(pady=5, fill=tk.X)

        self.confidence_label = ttk.Label(result_container, text="🔍 Kepercayaan: -",
                                          style="ResultValue.TLabel", wraplength=250)  # Pakai ResultValue.TLabel
        self.confidence_label.pack(pady=5, fill=tk.X)

        self.weather_label = ttk.Label(result_container, text="🗒️ Prediksi Cuaca: -",
                                       style="ResultValue.TLabel", wraplength=250)  # Pakai ResultValue.TLabel
        self.weather_label.pack(pady=5, fill=tk.X)

        self.status_label = ttk.Label(main_frame, text="Siap menerima gambar...", font=("Segoe UI", 10, "italic"),
                                      foreground="#666666", background="#e0f2f7")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)

        self.load_placeholder_image()

    def load_placeholder_image(self):
        placeholder_path = os.path.join(os.path.dirname(__file__), "assets", "placeholder.png")
        if not os.path.exists(placeholder_path):
            # Jika placeholder tidak ada, buat placeholder sederhana secara programmatis
            # Pastikan ImageDraw dan ImageFont diimpor dari PIL
            img = Image.new('RGB', (300, 300), color='#dddddd')
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()  # Fallback font
            d.text((50, 140), "Pilih Gambar Awan", fill=(0, 0, 0), font=font)
            self.photo = ImageTk.PhotoImage(img)
        else:
            img = Image.open(placeholder_path)
            img_resized = img.resize((300, 300), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_resized)

        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
        )
        if file_path:
            self.status_label.config(text="Memuat gambar dan memulai klasifikasi...")
            self.root.update_idletasks()

            img = Image.open(file_path)
            img_resized = img.resize((300, 300), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_resized)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

            self.cloud_class_label.config(text="☁️ Kelas Awan: Memproses...", foreground="#555555")
            self.confidence_label.config(text="🔍 Kepercayaan: Memproses...", foreground="#555555")
            self.weather_label.config(text="🗒️ Prediksi Cuaca: Memproses...", foreground="#555555")
            self.root.update_idletasks()

            image_cv = cv2.imread(file_path)
            if image_cv is None:
                img_data = np.fromfile(file_path, np.uint8)
                image_cv = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            if image_cv is None:
                self.status_label.config(text="Error: Gagal membaca gambar.", foreground="red")
                self.cloud_class_label.config(text="☁️ Kelas Awan: ERROR", foreground="red")
                self.confidence_label.config(text="🔍 Kepercayaan: N/A", foreground="red")  # Tambah ini
                self.weather_label.config(text="🗒️ Prediksi Cuaca: N/A", foreground="red")  # Tambah ini
                return

            import threading
            threading.Thread(target=self.run_classification, args=(image_cv,)).start()

    def run_classification(self, image_cv):
        try:
            label, confidence, weather = classify_cloud(image_cv)

            self.root.after(0, self.update_results, label, confidence, weather)
        except Exception as e:
            self.root.after(0, self.update_results_error, f"Klasifikasi gagal: {e}")

    def update_results(self, label, confidence, weather):
        self.cloud_class_label.config(text=f"☁️ Kelas Awan: {label}", foreground="#0056b3")
        self.confidence_label.config(text=f"🔍 Kepercayaan: {confidence:.2f}%", foreground="#0056b3")
        self.weather_label.config(text=f"🗒️ Prediksi Cuaca: {weather}", foreground="#0056b3")
        self.status_label.config(text="Klasifikasi selesai!", foreground="#008000")

    def update_results_error(self, message):
        self.cloud_class_label.config(text="☁️ Kelas Awan: ERROR", foreground="red")
        self.confidence_label.config(text="🔍 Kepercayaan: N/A", foreground="red")
        self.weather_label.config(text="🗒️ Prediksi Cuaca: N/A", foreground="red")
        self.status_label.config(text=f"Error: {message}", foreground="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = CloudClassifierApp(root)
    root.mainloop()