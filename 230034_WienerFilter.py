import numpy as np
import matplotlib.pylab as plt
from skimage import color, restoration
from skimage.io import imread
from scipy.signal import convolve2d

# 1. Persiapan: Fungsi untuk konvolusi di ranah frekuensi (agar lebih cepat)
def convolve(image, kernel):
    return np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel, s=image.shape)).real

# 2. Load gambar dan ubah ke grayscale
# Ganti 'images/cactus.png' dengan path gambar yang kamu miliki
try:
    img = color.rgb2gray(imread('images/cactus.png'))
except FileNotFoundError:
    # Jika file tidak ada, kita gunakan data sampel dari skimage untuk demo
    from skimage import data
    img = color.rgb2gray(data.astronaut())

# 3. Membuat Point Spread Function (PSF) - Box Blur 7x7
psf = np.ones((7, 7)) / 49

# 4. Proses Merusak Gambar (Simulasi Degradasi)
# Menambahkan Blur
img_blurred = convolve(img, psf)

# Menambahkan Noise (Gaussian Noise)
rng = np.random.default_rng()
img_noisy = img_blurred.copy()
img_noisy += 0.1 * img_noisy.std() * rng.standard_normal(img_noisy.shape)

# 5. Proses Restorasi dengan Wiener Filter
# Metode A: Unsupervised Wiener
# Algoritma ini mengestimasi parameter secara otomatis
deconvolved_unsupervised, _ = restoration.unsupervised_wiener(img_noisy, psf)

# Metode B: Wiener dengan Parameter Balance (Supervised)
# Kita tentukan nilai balance (lambda) secara manual, misalnya 1100
balance = 1100
deconvolved_supervised = restoration.wiener(img_noisy, psf, balance)

# 6. Visualisasi Hasil
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].set_title('Gambar Asli (Original)')

ax[0, 1].imshow(img_noisy, cmap='gray')
ax[0, 1].set_title('Gambar Rusak (Blur + Noise)')

ax[1, 0].imshow(deconvolved_unsupervised, cmap='gray')
ax[1, 0].set_title('Restorasi: Unsupervised Wiener')

ax[1, 1].imshow(deconvolved_supervised, cmap='gray')
ax[1, 1].set_title(f'Restorasi: Wiener (Balance={balance})')

for a in ax.ravel():
    a.axis('off')

plt.tight_layout()
plt.show()

# Opsional: Menampilkan Skor Kualitas (PSNR)
from skimage.metrics import peak_signal_noise_ratio as psnr
print(f"PSNR Gambar Rusak: {psnr(img, img_noisy):.3f}")
print(f"PSNR Unsupervised: {psnr(img, deconvolved_unsupervised):.3f}")
print(f"PSNR Supervised (Balance {balance}): {psnr(img, deconvolved_supervised):.3f}")