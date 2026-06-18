# Image Restoration — Chapter 3

## Restoring an Image with the Wiener Filter

The **Wiener filter** adalah filter *Mean Squared Error* (MSE) yang menggabungkan fungsi degradasi dan karakteristik statistik noise. Asumsi dasarnya adalah noise dan gambar tidak berkorelasi. Filter ini mengoptimalkan sehingga MSE diminimalkan.

Dalam resep ini, kamu akan belajar cara mengimplementasikan Wiener filter menggunakan fungsi dari modul `scikit-image restoration` dan cara menerapkannya untuk merestorasi gambar yang terdegradasi, baik secara *supervised* maupun *unsupervised*.

---

## Getting Ready

Pada resep ini, kita akan menggunakan gambar kaktus sebagai input dan merusaknya dengan noise/blur. Import semua library yang diperlukan terlebih dahulu:

```python
from skimage.io import imread
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
```

---

## How to Do It

Ikuti langkah-langkah berikut untuk mengimplementasikan Wiener filter dan menerapkannya pada gambar yang rusak akibat blur/noise:

### 1. Baca gambar dan ubah ke grayscale

```python
im = color.rgb2gray(imread('images/cactus.png'))
```

### 2. Definisikan `convolve2d()` dan buat kernel blur

Definisikan fungsi `convolve2d()` untuk mengimplementasikan konvolusi di domain frekuensi (menggunakan *convolution theorem* — implementasi ini jauh lebih cepat daripada di domain spasial). Buat *point-spread function* (PSF) rata-rata 7×7 (box-blur) dan gunakan sebagai kernel konvolusi:

```python
def convolve2d(im, psf, k):
    M, N = im.shape
    freq = fp.fft2(im)
    # asumsi: min(M,N) > k > 0, k ganjil
    psf = np.pad(psf, (((M-k)//2, (M-k)//2+1),
                       ((N-k)//2, (N-k)//2+1)), mode='constant')
    freq_kernel = fp.fft2(fp.ifftshift(psf))
    return np.abs(fp.ifft2(freq * freq_kernel))

k = 5
psf = np.ones((k, k)) / k**2  # box blur
im1 = convolution2d(im, psf, k)
```

### 3. Tambahkan noise ke gambar yang diblur

```python
im1 += 0.2 * im.std() * np.random.standard_normal(im.shape)
```

### 4. Terapkan Unsupervised Wiener Filter

Terapkan *unsupervised Wiener filter* pada gambar yang terdegradasi beserta kernel sebagai input untuk merestorasi gambar:

```python
im2, _ = restoration.unsupervised_wiener(im1, psf)
```

### 5. Terapkan Wiener Filter dengan parameter `balance`

Terapkan Wiener filter (kali ini dengan parameter `balance`) pada gambar yang terdegradasi:

```python
im3 = restoration.wiener(im1, psf, balance=0.25)
```

> **Catatan:** Wiener restoration dengan parameter `balance` menghasilkan kualitas yang lebih baik dibandingkan unsupervised Wiener restoration. Kualitas gambar yang direstorasi diukur menggunakan metrik **Peak Signal-to-Noise Ratio (PSNR)**.

### 6. Plot spektrum frekuensi 3D

Plot spektrum frekuensi dari gambar input/output dan kernel dalam 3D:

```python
def plot_freq_spec_3d(freq):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    Y = np.arange(-freq.shape[0]//2, freq.shape[0] - freq.shape[0]//2)
    X = np.arange(-freq.shape[1]//2, freq.shape[1] - freq.shape[1]//2)
    X, Y = np.meshgrid(X, Y)
    Z = (20 * np.log10(0.01 + fp.fftshift(freq))).real
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()

plot_freq_spec_3d(fp.fft2(im))
plot_freq_spec_3d(fp.fft2(im1))
plot_freq_spec_3d(fp.fft2(im2))
plot_freq_spec_3d(fp.fft2(im3))
```

---

## How It Works

Wiener filter adalah filter yang memiliki fungsi objektif untuk **meminimalkan MSE** antara gambar yang direstorasi dan gambar aslinya.

Filter ini diformulasikan sebagai berikut:

```
Objektif: min E[(f - f̂)²]
         = min E[|F(u,v) - F̂(u,v)|²]   (by Parseval's Theorem)

s.t. F̂(u,v) = G(α,v) W(u,v)

Solusi:
         H*(u,v)               H*(u,v)
W(u,v) = ─────────────────── = ──────────────────
         |H(u,v)|² + Sn/Sf     |H(u,v)|² + K

         1     |H(u,v)|²
       = ─── × ─────────────────────
         H(u,v) |H(u,v)|² + Sn/Sf
         └─────┘ └───────────────┘
       Inverse filter       1/SNR
```

Beberapa poin penting:

- Wiener filter menjadi **inverse filter** ketika tidak ada noise (N=0).
- **K** adalah konstanta (tidak bergantung pada u, v) yang dipilih berdasarkan pengetahuan awal tentang noise.

### `unsupervised_wiener()`

Fungsi `unsupervised_wiener()` dari modul `skimage.restoration` digunakan untuk melakukan denoising pada gambar yang terdegradasi menggunakan dekonvolusi dengan pendekatan *Wiener-Hunt*, di mana hiperparameter diestimasi secara otomatis.

- Estimasi gambar didefinisikan sebagai *posterior mean* dari analisis Bayesian.
- Karena penjumlahan eksak tidak dapat dilakukan, digunakan metode **Markov Chain Monte Carlo (MCMC)** (Gibbs sampler).
- Gibbs sampler lebih sering menarik gambar yang sangat mungkin (karena kontribusinya lebih besar terhadap mean).
- Mean empiris dari sampel ini digunakan sebagai estimasi.
- **Laplacian** digunakan sebagai operator regularisasi secara default.

### `wiener()` dengan parameter `balance`

Fungsi `wiener()` dari modul `skimage.restoration` digunakan dengan parameter `balance` tambahan untuk melakukan denoising menggunakan Wiener filter dengan pendekatan *Wiener-Hunt* (diagonalisasi Fourier).

- **Parameter regularisasi** (`balance` atau λ) mengubah keseimbangan antara kecukupan data dan kecukupan prior.
  - Kecukupan data membantu meningkatkan restorasi frekuensi.
  - Kecukupan prior membantu mengurangi restorasi frekuensi untuk menghindari artefak noise.
- **Laplacian** digunakan sebagai operator regularisasi default.

---

## See Also

- [scikit-image — `unsupervised_wiener` API](https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.unsupervised_wiener)
- [Lecture Notes: Digital Image Processing — Wiener Filter](http://sce2.umkc.edu/csee/lizhu/teaching/2018.fall.digital-imageprocessing/notes/lec11.pdf)
