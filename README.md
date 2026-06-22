# Gradewise Backend

Backend FastAPI untuk memprediksi nilai akademik mahasiswa menggunakan model machine learning yang sudah dilatih. Model yang digunakan adalah `RandomForestRegressor` dengan 19 fitur akademik, personal, dan lingkungan belajar.

## Teknologi

- Python 3
- FastAPI dan Uvicorn
- Pydantic
- pandas
- scikit-learn 1.6.1
- joblib

## Struktur proyek

```text
gradewise-be/
|-- main.py
|-- requirements.txt
|-- .env.example
|-- feature_names.pkl
|-- label_encoders.pkl
|-- scaler.pkl
`-- model_student_performance.pkl
```

Fungsi setiap artefak model:

- `feature_names.pkl`: menyimpan nama dan urutan 19 fitur model.
- `label_encoders.pkl`: mengubah data kategorikal menjadi angka.
- `scaler.pkl`: melakukan standardisasi seluruh fitur.
- `model_student_performance.pkl`: model `RandomForestRegressor` yang menghasilkan prediksi nilai.

Semua file `.pkl` wajib tersedia di folder yang sama dengan `main.py`. Jangan memuat file pickle dari sumber yang tidak dipercaya.

## Menjalankan secara lokal

### 1. Clone dan masuk ke proyek

```bash
git clone https://github.com/MahendraSaputraa/gradewise-be.git
cd gradewise-be
```

Jika repository sudah tersedia, langsung masuk ke folder proyek.

### 2. Buat virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instal dependency

```bash
python -m pip install -r requirements.txt
```

Versi scikit-learn dikunci ke `1.6.1` karena artefak model dibuat menggunakan versi tersebut.

### 4. Konfigurasi environment

Salin `.env.example` menjadi `.env` jika perlu mengatur origin frontend:

```env
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://gradewise-seven.vercel.app
```

Pisahkan beberapa origin menggunakan koma dan tulis origin tanpa trailing slash.

### 5. Jalankan server

```bash
python -m uvicorn main:app --reload
```

Server akan tersedia di `http://localhost:8000`.

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

Hentikan server dengan `Ctrl+C`.

## Dokumentasi API

### Health check

```http
GET /health
```

Contoh respons:

```json
{
  "status": "ok",
  "message": "API and prediction model are ready",
  "model_loaded": true
}
```

### Prediksi nilai

```http
POST /predict
Content-Type: application/json
```

Contoh request:

```json
{
  "hours_studied": 20,
  "attendance": 85,
  "parental_involvement": "Medium",
  "access_to_resources": "High",
  "extracurricular_activities": "Yes",
  "sleep_hours": 7,
  "previous_scores": 75,
  "motivation_level": "Medium",
  "internet_access": "Yes",
  "tutoring_sessions": 2,
  "family_income": "Medium",
  "teacher_quality": "High",
  "school_type": "Public",
  "peer_influence": "Positive",
  "physical_activity": 3,
  "learning_disabilities": "No",
  "parental_education_level": "College",
  "distance_from_home": "Near",
  "gender": "Female"
}
```

Contoh respons:

```json
{
  "predicted_score": 68.9,
  "is_dummy": false,
  "message": "Prediksi berhasil dibuat menggunakan model student performance."
}
```

`predicted_score` adalah hasil regresi dari model, bukan probabilitas atau label kelulusan. Nilainya dibulatkan menjadi dua angka desimal.

## Referensi input

| Field | Tipe | Nilai yang diterima |
|---|---|---|
| `hours_studied` | number | 0–168 |
| `attendance` | number | 0–100 |
| `parental_involvement` | string | `Low`, `Medium`, `High` |
| `access_to_resources` | string | `Low`, `Medium`, `High` |
| `extracurricular_activities` | string | `No`, `Yes` |
| `sleep_hours` | number | 0–24 |
| `previous_scores` | number | 0–100 |
| `motivation_level` | string | `Low`, `Medium`, `High` |
| `internet_access` | string | `No`, `Yes` |
| `tutoring_sessions` | integer | Minimal 0 |
| `family_income` | string | `Low`, `Medium`, `High` |
| `teacher_quality` | string | `Low`, `Medium`, `High` |
| `school_type` | string | `Private`, `Public` |
| `peer_influence` | string | `Negative`, `Neutral`, `Positive` |
| `physical_activity` | number | 0–24 |
| `learning_disabilities` | string | `No`, `Yes` |
| `parental_education_level` | string | `High School`, `College`, `Postgraduate` |
| `distance_from_home` | string | `Near`, `Moderate`, `Far` |
| `gender` | string | `Female`, `Male` |

Nilai kategorikal bersifat case-sensitive. Sebagai contoh, kirim `Medium`, bukan `medium`.

## Alur prediksi

1. FastAPI memvalidasi request menggunakan Pydantic.
2. Nama field API dipetakan ke nama fitur yang digunakan saat training.
3. Fitur kategorikal ditransformasi menggunakan label encoder hasil training.
4. Semua fitur ditransformasi menggunakan scaler hasil training.
5. Data disusun berdasarkan urutan dalam `feature_names.pkl`.
6. Model menghasilkan prediksi nilai dan API mengembalikannya sebagai JSON.

Seluruh artefak dimuat satu kali ketika aplikasi mulai, bukan pada setiap request.

## Integrasi frontend

Simpan URL backend sebagai environment variable frontend, misalnya:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Frontend sebaiknya memakai input angka untuk fitur numerik dan dropdown untuk fitur kategorikal. Saat request berlangsung, tampilkan loading state dan tangani status `422` sebagai kesalahan validasi input.

Contoh JavaScript:

```javascript
const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

if (!response.ok) {
  throw new Error("Prediksi gagal");
}

const result = await response.json();
console.log(result.predicted_score);
```

## Validasi dan error

- `200 OK`: prediksi berhasil.
- `422 Unprocessable Entity`: field hilang, tipe data salah, nilai numerik di luar batas, atau pilihan kategori tidak valid.
- `500 Internal Server Error`: masalah internal, misalnya artefak model tidak tersedia atau tidak kompatibel.

Gunakan Swagger UI di `/docs` untuk melihat schema terbaru dan mencoba endpoint tanpa frontend atau Postman.

## Catatan pengembangan

- Jika model dilatih ulang, ganti keempat file `.pkl` secara bersamaan agar model, encoder, scaler, dan urutan fitur tetap konsisten.
- Jika fitur training berubah, schema `StudentInput` dan pemetaan fitur di `main.py` juga harus diperbarui.
- Jangan membuat encoder atau scaler baru ketika melakukan prediksi.
- Origin production baru harus ditambahkan ke `ALLOWED_ORIGINS`.
- Jangan mengaktifkan `--reload` pada server production.

Contoh start command untuk deployment:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```
