import pandas as pd
import re
import joblib
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Setup logging agar kita bisa melihat prosesnya di terminal (Robustness)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(fake_path, true_path):
    """
    Fungsi untuk membaca data, memberi label, dan menggabungkannya.
    Label 0 untuk Fake News, Label 1 untuk True News.
    """
    logging.info("Memulai proses Data Ingestion...")
    try:
        # Membaca dataset
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)

        # Memberikan label
        df_fake['label'] = 0
        df_true['label'] = 1

        # Menggabungkan dataset
        df = pd.concat([df_fake, df_true], ignore_index=True)

        # Mengacak data (shuffle) agar model belajar dengan seimbang
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logging.info(f"Data berhasil dimuat. Total baris: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Gagal memuat data: {e}")
        raise


def preprocess_text(text):
    """
    Fungsi untuk membersihkan teks.
    """
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus karakter non-alfabet (tanda baca, angka)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_pipeline():
    """
    Fungsi utama yang menjalankan seluruh MLOps Pipeline secara otomatis.
    """
    # 1. Tentukan path file
    fake_data_path = '../data/Fake.csv'
    true_data_path = '../data/True.csv'
    model_save_path = '../models/fake_news_model.pkl'

    # 2. Data Ingestion
    df = load_data(fake_data_path, true_data_path)

    # 3. Preprocessing (Membersihkan kolom teks)
    logging.info("Memulai Preprocessing teks...")
    # Kita gabungkan judul dan isi berita untuk konteks yang lebih kaya
    df['text'] = df['title'] + " " + df['text']
    # Terapkan fungsi pembersihan teks
    df['clean_text'] = df['text'].apply(preprocess_text)

    # 4. Memisahkan data untuk Training dan Validation
    logging.info("Memisahkan data Training dan Testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42
    )

    # 5. Membangun Automated Pipeline (Integrasi Vectorizer & Model)
    logging.info("Membangun dan melatih model Machine Learning...")
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),  # Mengubah teks menjadi angka
        ('clf', LogisticRegression(max_iter=1000))  # Model klasifikasi
    ])

    # Melatih model (Training)
    model_pipeline.fit(X_train, y_train)

    # 6. Validasi Model
    logging.info("Melakukan validasi model...")
    y_pred = model_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\n--- Classification Report ---\n")
    print(report)

    # 7. Menyimpan Model (Model Serialization)
    logging.info("Menyimpan model ke dalam folder models/...")
    # Pastikan folder models/ ada
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model_pipeline, model_save_path)
    logging.info(f"Model berhasil disimpan di: {model_save_path}")
    logging.info("ML Pipeline Selesai dengan sukses!")


if __name__ == "__main__":
    # Pindahkan working directory ke lokasi script ini berada agar path file selalu benar
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline()