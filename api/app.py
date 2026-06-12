from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import joblib
import os
import logging
import sys
import warnings

app = Flask(__name__)

# Izinkan CORS dari semua origin (Vercel, localhost, dll)
CORS(app, resources={r"/*": {"origins": "*"}})

# Inisialisasi Swagger untuk dokumentasi API interaktif
swagger = Swagger(app)

# Setup logging ke stdout agar Railway bisa membaca
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=== FakeNews BE API Starting ===")

# Coba semua kemungkinan path model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(BASE_DIR, '../models/fake_news_model.pkl'),
    os.path.join(BASE_DIR, 'models/fake_news_model.pkl'),
    os.path.join(BASE_DIR, 'fake_news_model.pkl'),
    os.path.join(os.getcwd(), 'models/fake_news_model.pkl'),
]

MODEL_PATH = None
for p in possible_paths:
    resolved = os.path.normpath(p)
    if os.path.exists(resolved):
        MODEL_PATH = resolved
        logger.info(f"Model ditemukan di: {MODEL_PATH}")
        break
    else:
        logger.info(f"Tidak ada di: {resolved}")

model_pipeline = None
if MODEL_PATH:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_pipeline = joblib.load(MODEL_PATH)
        # Sanity check
        _ = model_pipeline.predict(["test"])
        logger.info("Model berhasil dimuat dan diverifikasi.")
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        model_pipeline = None
else:
    logger.error("File model tidak ditemukan di semua path!")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint untuk mengecek status kesehatan API.
    ---
    responses:
      200:
        description: API dan Model berjalan dengan baik
      503:
        description: API berjalan tapi Model gagal dimuat
    """
    if model_pipeline:
        return jsonify({"status": "healthy", "message": "API and Model are ready!"}), 200
    else:
        return jsonify({"status": "unhealthy", "message": "Model not loaded!"}), 503


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk memprediksi berita Fake atau Real.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              example: "Washington - The president signed a new bill today to improve infrastructure..."
    responses:
      200:
        description: Prediksi berhasil dilakukan
      400:
        description: Bad Request (Input JSON tidak valid atau kosong)
      500:
        description: Internal Server Error (Kendala pada server)
    """
    # Error Handling 1: Jika model belum siap
    if model_pipeline is None:
        return jsonify({"error": "Model tidak tersedia di server."}), 500

    try:
        data = request.get_json()

        # Error Handling 2: Jika user tidak mengirim format JSON yang benar
        if not data or 'text' not in data:
            return jsonify({"error": "Format salah. Gunakan format JSON dengan key 'text'."}), 400

        text_input = data['text']

        # Error Handling 3: Jika text kosong atau bukan string
        if not isinstance(text_input, str) or len(text_input.strip()) == 0:
            return jsonify({"error": "Input 'text' tidak boleh kosong dan harus berupa teks."}), 400

        # Melakukan prediksi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model_pipeline.predict([text_input])[0]
            probabilities = model_pipeline.predict_proba([text_input])[0]

        # Menyusun response
        label = "Real News" if prediction == 1 else "Fake News"
        response = {
            "prediction": label,
            "confidence": {
                "fake_probability_percent": round(float(probabilities[0]) * 100, 2),
                "real_probability_percent": round(float(probabilities[1]) * 100, 2)
            }
        }
        logger.info(f"Prediksi: {label} | panjang teks: {len(text_input)}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error saat prediksi: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Baca PORT dari environment Railway secara dinamis
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)