# Dokumen Desain Awal Pipeline Persepsi (maks. 2 halaman)

Kelompok: <!-- ISI nama kelompok & anggota -->

> Bagian bertanda ISI wajib diisi sesuai proyek kelompok. Bagian "draf" sudah terisi dari dataset/eksperimen dan perlu disesuaikan.

| Bagian | Isi |
|---|---|
| Misi projek | <!-- ISI: peran persepsi dalam 2-3 kalimat --> |
| Kelas objek | `botol`, `box` (50 citra/kelas, lihat `dataset_raw/`). <!-- ISI: tambahkan contoh foto --> |
| Kamera & dudukan | <!-- ISI: resolusi, tinggi, sudut, jarak kerja --> |
| Unit komputasi | <!-- ISI: perangkat, mode daya, memori --> |
| Target kinerja | <!-- ISI: akurasi/mAP, FPS onboard, latensi ROS2 --> Referensi anggaran: 15 FPS = 67 ms/frame, inferensi 35 ms. |
| Kandidat model | MobileNetV3-Small (±2,5 juta parameter, ±0,06 GFLOPs, ±67,7% top-1 ImageNet; cocok untuk Raspberry Pi tanpa GPU). Pembanding: ResNet-18. |
| Strategi TL | Mulai dari feature extraction, lanjut fine-tuning parsial bila akurasi belum cukup; pembanding pelatihan dari nol (hasil di README). |
| Rencana data | Minimal 50 citra/kelas, tambah variasi jarak, posisi, orientasi, cahaya, latar, dan kasus sulit (occlusion, motion blur). |
| Risiko (draf) | 1) Data tidak seragam: resolusi dan sumber citra kedua kelas berbeda, model bisa belajar "sumber foto", bukan objek. Mitigasi: ambil ulang semua kelas dengan kamera robot yang sama. 2) Data leakage dari frame beruntun. Mitigasi: split berdasarkan urutan waktu/sesi. 3) Variasi cahaya di lapangan. Mitigasi: ambil data di berbagai kondisi cahaya dan gunakan ColorJitter. |
