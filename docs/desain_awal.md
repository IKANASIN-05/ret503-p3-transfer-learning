# Dokumen Desain Awal Pipeline Persepsi

Kelompok: Willy

| Bagian | Isi |
|---|---|
| Misi projek | Robot edukasi yang direncanakan beroperasi di lingkungan perpustakaan. Kamera robot berfungsi mendeteksi objek di sekitarnya, dan botol serta box dipilih sebagai objek yang mewakili benda yang umum ditemui di sana. Persepsi bertugas mengenali kedua objek itu agar robot dapat mengetahui apa yang ada di sekitarnya. |
| Kelas objek | `botol` dan `box`, masing-masing 50 citra (`dataset_raw/`, daftar di `metadata.csv`). |
| Kamera & dudukan | Intel RealSense (tipe dan resolusi belum diisi), dipasang di bagian atas robot pada ketinggian sekitar 30 sampai 50 cm dari permukaan lantai. Sudut dan jarak kerja belum diisi. |
| Unit komputasi | NVIDIA Jetson (tipe, memori, dan mode daya belum diisi). |
| Target kinerja | Akurasi validasi >= 90%; inferensi <= 35 ms per frame (anggaran 15 FPS pada slide 16). Hasil uji laptop: MobileNetV3-Small 17,2 ms (58 FPS); pengukuran ulang di Jetson dilakukan setelah perangkat tersedia. |
| Kandidat model | MobileNetV3-Small (±2,5 juta parameter, ringan). Pembanding: ResNet-18 yang terukur 104,2 ms di laptop sehingga melebihi anggaran 35 ms. Karena perangkat target adalah Jetson, EfficientNet-B0 (kelas Jetson pada slide 14) menjadi kandidat lanjutan. |
| Strategi TL | Feature extraction (backbone beku, hanya classifier dilatih) karena data sedikit (80 citra latih). Pada uji: feature 100%, partial 100%, scratch 50%. |
| Rencana data | Sudah 50 citra/kelas. Berikutnya tambah variasi jarak, posisi, orientasi, cahaya, dan latar perpustakaan (rak, meja, lantai), serta kasus sulit (occlusion, motion blur). Ambil kedua kelas dengan kamera RealSense pada ketinggian pemasangan sebenarnya. |

## Risiko dan mitigasi

1. **Sumber foto kedua kelas berbeda** (botol 200x150 piksel, box 512x384 piksel); model bisa belajar karakter foto, bukan objek. Mitigasi: ambil ulang kedua kelas dengan kamera robot yang sama.
2. **Data leakage** dari frame beruntun. Mitigasi: bagi data train/val menurut urutan waktu atau sesi pengambilan.
3. **Data validasi hanya 20 citra**, akurasi kurang andal. Mitigasi: tambah data validasi dan uji di kondisi lapangan.
4. **Variasi cahaya dan latar perpustakaan** berbeda dari data saat ini. Mitigasi: ambil data pada berbagai kondisi cahaya dan gunakan ColorJitter.
5. **Kecepatan di Jetson belum terukur** (angka saat ini dari laptop). Mitigasi: jalankan `scripts/latency.py` di Jetson sebelum memutuskan model final.
