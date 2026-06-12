# Menggunakan image Python versi slim agar ukurannya kecil dan efisien (Optimized)
FROM python:3.9-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt terlebih dahulu
# Ini adalah best practice Docker untuk memanfaatkan 'layer caching'
COPY requirements.txt .

# Menginstal library tanpa menyimpan cache untuk menghemat ruang
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh file proyek (data, models, src, api) ke dalam container
COPY . .

# Memberitahu Docker bahwa container ini menggunakan port 5001
EXPOSE 5001

# Perintah yang dijalankan saat container menyala
CMD ["python", "api/app.py"]