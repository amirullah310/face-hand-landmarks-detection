# Praktikum Pengolahan Citra Digital - Face & Hand Landmarks Detection

Repository ini berisi implementasi **deteksi landmark wajah dan kedua tangan secara real-time** dari webcam menggunakan **MediaPipe Holistic Landmarker** dan **OpenCV**. Program mendeteksi landmark wajah, tangan kiri, dan tangan kanan pada setiap frame, menggambarnya beserta garis penghubung, lalu menampilkan FPS.

Sumber referensi:
* [Face and Hand Landmarks Detection using Python - Mediapipe, OpenCV — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/)
* [Holistic landmarks detection task guide — Google AI Edge MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker)

---

## 📌 Tentang Program

Tutorial GeeksforGeeks memakai `mp.solutions.holistic`, tetapi modul `mp.solutions` **sudah tidak tersedia** pada MediaPipe versi terbaru (0.10.30 ke atas). Karena itu program ini memakai `HolisticLandmarker` dari **MediaPipe Tasks API**, dengan langkah dan nama variabel yang tetap mengikuti tutorial.

| Tutorial (`mp.solutions`) | Program ini (Tasks API) |
| :--- | :--- |
| `mp_holistic.Holistic(...)` | `vision.HolisticLandmarker.create_from_options(...)` |
| `static_image_mode=False` | `running_mode=vision.RunningMode.VIDEO` |
| `min_detection_confidence=0.5` | `min_face_detection_confidence`, `min_pose_detection_confidence` = `0.5` |
| `min_tracking_confidence=0.5` | `min_face_landmarks_confidence`, `min_pose_landmarks_confidence`, `min_hand_landmarks_confidence` = `0.5` |
| `holistic_model.process(image)` | `holistic_model.detect_for_video(mp_image, timestamp_ms)` |
| `mp.solutions.drawing_utils` | `vision.drawing_utils` |
| `mp_holistic.FACEMESH_CONTOURS` | `vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS` |
| `mp_holistic.HAND_CONNECTIONS` | `vision.HandLandmarksConnections.HAND_CONNECTIONS` |
| `results.left_hand_landmarks.landmark[0]` | `results.left_hand_landmarks[0]` |

### Langkah-langkah
1. **Import library** — `cv2`, `mediapipe`, serta `BaseOptions`, `vision`, dan `hand_landmarker` dari `mediapipe.tasks.python`.
2. **Inisialisasi model** — model `holistic_landmarker.task` diunduh otomatis jika belum ada, lalu `HolisticLandmarker` dibuat dengan running mode `VIDEO` dan ambang kepercayaan `0.5`.
3. **Akses indeks landmark** — mencetak nama dan indeks 21 landmark tangan dari enum `HandLandmark` (`WRIST = 0` sampai `PINKY_TIP = 20`).
4. **Baca frame** — `cv2.VideoCapture(0)` membuka webcam, setiap frame di-resize menjadi `800×600`.
5. **Deteksi** — frame dikonversi BGR → RGB, dibungkus menjadi `mp.Image`, lalu diproses dengan `detect_for_video()`. Hasilnya diakses melalui `results.face_landmarks`, `results.left_hand_landmarks`, dan `results.right_hand_landmarks`.
6. **Gambar landmark** — `draw_landmarks()` menggambar titik wajah berwarna magenta dengan kontur kuning, serta kerangka kedua tangan (titik merah, garis abu-abu terang).
7. **FPS** — dihitung dengan `1 / (currentTime - previousTime)` dan ditampilkan di pojok kiri atas.
8. **Simpan hasil** — jika wajah dan kedua tangan terdeteksi selama 15 frame berturut-turut, frame disimpan sebagai `output.png` dan program berhenti otomatis.

### Jumlah Landmark
| Bagian | Jumlah | Keterangan |
| :--- | :---: | :--- |
| Wajah | 478 | 468 titik *face mesh* + 10 titik iris (tutorial menyebut 468 karena versi lama tanpa iris) |
| Tangan kiri | 21 | Indeks 0–20 sesuai enum `HandLandmark` |
| Tangan kanan | 21 | Indeks 0–20 sesuai enum `HandLandmark` |
| Pose | 33 | Dihitung model, tetapi tidak digambar |

### Indeks Landmark Tangan
| Indeks | Landmark | Indeks | Landmark |
| :---: | :--- | :---: | :--- |
| 0 | `WRIST` | 11 | `MIDDLE_FINGER_DIP` |
| 1 | `THUMB_CMC` | 12 | `MIDDLE_FINGER_TIP` |
| 2 | `THUMB_MCP` | 13 | `RING_FINGER_MCP` |
| 3 | `THUMB_IP` | 14 | `RING_FINGER_PIP` |
| 4 | `THUMB_TIP` | 15 | `RING_FINGER_DIP` |
| 5 | `INDEX_FINGER_MCP` | 16 | `RING_FINGER_TIP` |
| 6 | `INDEX_FINGER_PIP` | 17 | `PINKY_MCP` |
| 7 | `INDEX_FINGER_DIP` | 18 | `PINKY_PIP` |
| 8 | `INDEX_FINGER_TIP` | 19 | `PINKY_DIP` |
| 9 | `MIDDLE_FINGER_MCP` | 20 | `PINKY_TIP` |
| 10 | `MIDDLE_FINGER_PIP` | | |

---

## 📂 Struktur Direktori

```text
├── face_hand_landmarks.py     # Script deteksi landmark wajah & tangan real-time
├── holistic_landmarker.task   # Model MediaPipe (diunduh otomatis, tidak di-upload)
├── output.png                 # Hasil tangkapan layar (dibuat otomatis, tidak di-upload)
├── requirements.txt           # Dependensi pustaka Python
├── .gitignore                 # Aturan ignore file git
└── README.md                  # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan

### 1. Prasyarat & Instalasi
Pastikan Python telah terpasang (diuji dengan Python 3.12) dan webcam tersedia. Install dependensi dengan perintah:
```bash
pip install -r requirements.txt
```

Model `holistic_landmarker.task` (±13,7 MB) akan diunduh otomatis saat program pertama kali dijalankan. Jika ingin mengunduh manual:
```bash
curl -L -o holistic_landmarker.task https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

### 2. Eksekusi Script
```bash
python face_hand_landmarks.py
```

Hadapkan wajah dan kedua tangan yang terbuka ke kamera. Setelah ketiganya terdeteksi stabil (±1,5 detik pada 10 fps), hasil tersimpan ke `output.png` dan jendela tertutup otomatis.

Kontrol keyboard (pada jendela video):
* `q` — keluar tanpa menyimpan

### 3. Catatan
* Jika muncul pesan `Kamera tidak bisa dibuka`, pastikan webcam aktif (cek tombol kamera pada laptop) dan tidak sedang dipakai aplikasi lain seperti Zoom atau Teams.
* Pada pengujian dengan CPU laptop, program berjalan sekitar **10 fps**.
* Pencahayaan yang merata dan tidak membelakangi jendela membuat deteksi lebih stabil.

---

## 👤 Penulis
* **Amirullah**
* Praktikum Pengolahan Citra Digital
* Politeknik Negeri Lhokseumawe (PNL)
