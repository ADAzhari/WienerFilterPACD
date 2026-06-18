# Wiener Filter Image Restoration

Aplikasi berbasis Streamlit untuk mendemonstrasikan restorasi citra menggunakan metode Wiener Filter. Dibuat untuk keperluan tugas/pembelajaran Pengolahan Citra Digital oleh Achmad Dzaki Azhari (NPM 140810230034).

## Instalasi

Pastikan Python 3.8 atau yang lebih baru sudah terinstal. Library yang dibutuhkan:

```bash
pip install streamlit numpy scikit-image scipy matplotlib
```

## Cara Menjalankan

1. Buka terminal di direktori proyek ini.
2. Jalankan perintah:

```bash
streamlit run 230034_WienerFilterGUI.py
```

3. Akses antarmuka web melalui browser di `http://localhost:8501`.

## Penggunaan

- **Upload Gambar:** Masukkan gambar yang ingin diproses (format PNG, JPG, JPEG, BMP, atau TIFF). Gambar akan otomatis diubah menjadi grayscale.
- **Parameter Degradasi:**
  - **Ukuran Blur:** Mengatur seberapa besar kernel blur yang diterapkan pada citra.
  - **Level Noise:** Menentukan intensitas Gaussian noise yang ditambahkan.
- **Restorasi:**
  - Aplikasi akan mencoba mencari nilai balance optimal secara otomatis (Unsupervised Wiener).
  - Anda juga dapat melakukan penyesuaian (tuning) manual pada nilai balance untuk melihat perbandingannya.
- **Analisis:** Aplikasi menampilkan metrik kualitas citra seperti PSNR dan SSIM. Tersedia juga visualisasi 3D spektrum frekuensi untuk masing-masing tahapan pemrosesan.

## Troubleshooting

- Jika muncul `ModuleNotFoundError`, pastikan semua package yang disebutkan di atas sudah terinstal (terutama `scikit-image` untuk memecahkan masalah modul `skimage`).
- Apabila proses restorasi atau visualisasi terasa sangat lambat, coba gunakan gambar dengan resolusi yang lebih kecil, karena perhitungan citra ukuran besar memakan cukup banyak waktu komputasi.
