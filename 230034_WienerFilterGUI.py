import io
import streamlit as st
import numpy as np
from skimage import color, restoration, metrics
from skimage.io import imread
from scipy import fftpack as fp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

MAX_DIM = 512
# ============================================================
# PAGE CONFIG & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="Wiener Filter  — Image Restoration",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0820 0%, #12103a 40%, #1a1830 100%);
}
[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e0c28 0%, #151333 100%);
    border-right: 1px solid rgba(108,99,255,0.12);
}
[data-testid="stSidebar"] * {
    color: #d4d4f7 !important;
}

[data-testid="stSidebar"] h3 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    color: #e0e0ff !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] h4 {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px;
    color: #6c63ff !important;
}

.sidebar-group {
    background: rgba(108,99,255,0.06);
    border: 1px solid rgba(108,99,255,0.14);
    border-radius: 12px;
    padding: 14px 16px 10px;
    margin-bottom: 12px;
}
.sidebar-group-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6c63ff;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-group-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(108,99,255,0.2);
}

[data-testid="collapsedControl"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
[data-testid="stBaseButton-headerNoPadding"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

[data-testid="stShortcutLabel"] {
    display: none !important;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(108,99,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 4px;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.2);
    border-color: rgba(108,99,255,0.25);
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7070a0;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #48c6ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.82rem;
    color: #7a7aaa;
    margin-top: 6px;
}

.balance-display {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.18);
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.balance-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #7070a0;
}
.balance-value {
    font-family: 'Space Grotesk', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #48c6ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-header {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6c63ff;
    padding-bottom: 10px;
    margin-bottom: 16px;
    border-bottom: 1px solid rgba(108,99,255,0.2);
    display: flex;
    align-items: center;
    gap: 8px;
}

.img-container {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 12px;
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
    overflow: hidden;
}
.img-container:hover {
    box-shadow: 0 0 24px rgba(108,99,255,0.18);
    border-color: rgba(108,99,255,0.2);
}

.hero-wrap {
    padding: 10px 0 24px;
    border-bottom: 1px solid rgba(108,99,255,0.12);
    margin-bottom: 28px;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6c63ff;
    margin-bottom: 8px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #a8a0ff 60%, #48c6ef 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #6868a0;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-green { background: rgba(34,197,94,0.15); color: #22c55e; }
.badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; }
.badge-red   { background: rgba(239,68,68,0.15); color: #ef4444; }
.badge-blue  { background: rgba(59,130,246,0.15); color: #3b82f6; }

.tip-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(72,198,239,0.08));
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    color: #c8c8f0;
    font-size: 0.92rem;
    line-height: 1.6;
}
.tip-box strong {
    color: #a8a0ff;
}

.optimal-banner {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(16,185,129,0.08));
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 12px;
    padding: 14px 20px;
    color: #a7f3d0;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-bottom: 12px;
}
.optimal-banner strong {
    color: #34d399;
}

.welcome-card {
    background: rgba(108,99,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 20px;
    padding: 56px 48px;
    text-align: center;
    max-width: 580px;
    margin: 60px auto;
}
.welcome-card h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e0e0ff;
    margin-bottom: 12px;
    letter-spacing: -0.3px;
}
.welcome-card p {
    color: #7070a0;
    font-size: 0.93rem;
    line-height: 1.75;
}
.welcome-card p strong {
    color: #a8a0ff;
}
.welcome-icon {
    font-size: 3rem;
    margin-bottom: 20px;
    filter: drop-shadow(0 0 20px rgba(108,99,255,0.5));
}
.welcome-steps {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 24px;
    flex-wrap: wrap;
}
.welcome-step {
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #9090cc;
    display: flex;
    align-items: center;
    gap: 6px;
}

[data-testid="stFileUploader"] section {
    background: rgba(108,99,255,0.06) !important;
    border: 1.5px dashed rgba(108,99,255,0.35) !important;
    border-radius: 12px !important;
    padding: 20px 16px !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    background: rgba(108,99,255,0.1) !important;
    border-color: rgba(108,99,255,0.6) !important;
}

[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section p {
    color: #8888cc !important;
    font-size: 0.85rem !important;
}

[data-testid="stFileUploader"] section button,
[data-testid="stFileUploaderDropzoneInput"] + div button {
    background: rgba(108,99,255,0.2) !important;
    border: 1px solid rgba(108,99,255,0.45) !important;
    border-radius: 8px !important;
    color: #c8c8f0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stFileUploader"] section button:hover {
    background: rgba(108,99,255,0.38) !important;
    box-shadow: 0 0 14px rgba(108,99,255,0.3) !important;
    border-color: rgba(108,99,255,0.7) !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(108,99,255,0.08) !important;
    border: 1px solid rgba(108,99,255,0.18) !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button {
    color: #9090cc !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover {
    color: #ef4444 !important;
    background: rgba(239,68,68,0.1) !important;
    border-radius: 6px !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Image Processing · Wiener Filter</div>
    <p class="hero-title">Restorasi Gambar</p>
    <p class="hero-subtitle">Degradasi, restorasi, dan perbandingan secara real-time menggunakan Wiener Filter.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def convolve2d(im, psf, k_size):
    """Frequency-domain 2D convolution with properly centered PSF."""
    M, N = im.shape
    freq = fp.fft2(im)

    pad_top = (M - k_size) // 2
    pad_bottom = M - k_size - pad_top
    pad_left = (N - k_size) // 2
    pad_right = N - k_size - pad_left

    psf_pad = np.pad(psf, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    freq_kernel = fp.fft2(fp.ifftshift(psf_pad))
    return np.abs(fp.ifft2(freq * freq_kernel))

def load_as_gray(uploaded):
    """Load an uploaded image and safely convert to grayscale float64 [0-1]."""
    img = imread(uploaded)

    if img.ndim == 2:
        gray = img.astype(np.float64)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = color.rgb2gray(color.rgba2rgb(img))
    elif img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        raise ValueError(f"Format gambar tidak didukung: shape={img.shape}")

    if gray.max() > 1.0:
        gray = gray / 255.0

    return gray.astype(np.float64)

def psnr_badge(psnr):
    """Return an HTML badge colored by PSNR quality."""
    if psnr >= 30:
        return '<span class="badge badge-green">Excellent</span>'
    elif psnr >= 20:
        return '<span class="badge badge-amber">Fair</span>'
    else:
        return '<span class="badge badge-red">Poor</span>'

def find_optimal_balance(im_original, im_noisy, psf, search_min=-3.0, search_max=3.0):
    """
    Find the optimal Wiener balance via scipy.optimize.minimize_scalar.
    We minimise negative PSNR (= maximise PSNR) over log10(balance).
    Returns (optimal_log_balance, optimal_psnr).
    """
    def neg_psnr(log_b):
        balance = 10 ** log_b
        restored = restoration.wiener(im_noisy, psf, balance=balance)
        return -metrics.peak_signal_noise_ratio(im_original, restored)

    result = minimize_scalar(neg_psnr, bounds=(search_min, search_max), method="bounded",
                             options={"xatol": 0.05, "maxiter": 30})

    optimal_log = float(result.x)
    optimal_psnr = float(-result.fun)

    optimal_log = round(optimal_log * 10) / 10
    optimal_log = max(search_min, min(search_max, optimal_log))

    return optimal_log, optimal_psnr

def plot_freq_spec_3d(freq, title="", max_size=64):
    """
    Plot 3D frequency spectrum (log magnitude) of an image's FFT.
    Downsamples to max_size×max_size for rendering performance.
    """
    Z_full = (20 * np.log10(0.01 + np.abs(fp.fftshift(freq)))).real

    step_y = max(1, Z_full.shape[0] // max_size)
    step_x = max(1, Z_full.shape[1] // max_size)
    Z = Z_full[::step_y, ::step_x]

    rows, cols = Z.shape
    Y = np.arange(rows) - rows // 2
    X = np.arange(cols) - cols // 2
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.set_title(title, color="#e0e0ff", fontsize=11, fontweight=700, pad=12)
    fig.patch.set_facecolor("#1a1a40")
    ax.set_facecolor("#16163a")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color("#9d9dcc")
        axis.set_tick_params(labelcolor="#7a7aaa", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#2a2a5e")
    ax.yaxis.pane.set_edgecolor("#2a2a5e")
    ax.zaxis.pane.set_edgecolor("#2a2a5e")
    ax.set_xlabel("u", fontsize=9)
    ax.set_ylabel("v", fontsize=9)
    ax.set_zlabel("dB", fontsize=9)

    plt.tight_layout()
    return fig

# ============================================================
# SESSION STATE INITIALISATION
# ============================================================
if "log_val" not in st.session_state:
    st.session_state.log_val = -0.6
if "prev_file_id" not in st.session_state:
    st.session_state.prev_file_id = None
if "prev_kernel" not in st.session_state:
    st.session_state.prev_kernel = None
if "prev_noise" not in st.session_state:
    st.session_state.prev_noise = None
if "optimal_psnr" not in st.session_state:
    st.session_state.optimal_psnr = None
if "auto_optimized" not in st.session_state:
    st.session_state.auto_optimized = False
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### Konfigurasi")
    st.markdown('<div class="sidebar-group-label">Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Format yang didukung: PNG, JPG, JPEG, BMP, TIFF",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-group-label">Parameter Degradasi</div>', unsafe_allow_html=True)
    kernel_size = st.slider("Ukuran Blur (Kernel)", 3, 15, 5, step=2,
                            help="Semakin besar → gambar semakin buram")
    noise_level = st.slider("Level Noise (σ)", 0.0, 0.5, 0.05, step=0.01,
                            help="Intensitas Gaussian noise yang ditambahkan")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-group-label">Parameter Restorasi</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------
    # AUTO-OPTIMIZE: detect new upload OR changed degradation params
    # ----------------------------------------------------------
    needs_optimize = False

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if file_id != st.session_state.prev_file_id:
            st.session_state.file_bytes = uploaded_file.read()
            needs_optimize = True
            st.session_state.prev_file_id = file_id
            st.session_state.prev_kernel = kernel_size
            st.session_state.prev_noise = noise_level

    if needs_optimize:
        with st.spinner("Mencari balance optimal..."):
            _img = load_as_gray(io.BytesIO(st.session_state.file_bytes))
            _psf = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
            _blurred = convolve2d(_img, _psf, kernel_size)
            _rng = np.random.default_rng(42)
            _noisy = _blurred + noise_level * _img.std() * _rng.standard_normal(_img.shape)
            _noisy = np.clip(_noisy, 0, 1)

            opt_log, opt_psnr = find_optimal_balance(_img, _noisy, _psf)

            st.session_state.log_val = opt_log
            st.session_state.optimal_psnr = opt_psnr
            st.session_state.auto_optimized = True

    if st.session_state.get("pending_reoptimize"):
        st.session_state.log_val = st.session_state.pending_reoptimize["log"]
        st.session_state.optimal_psnr = st.session_state.pending_reoptimize["psnr"]
        st.session_state.auto_optimized = True
        del st.session_state["pending_reoptimize"]

    log_val = st.slider("Log₁₀ Balance", -3.0, 3.0, step=0.1, key="log_val",
                        help="Kontrol trade-off antara noise suppression dan detail preservation")
    balance_val = 10 ** log_val

    if st.session_state.auto_optimized and st.session_state.optimal_psnr is not None:
        st.markdown(f"""
        <div class="optimal-banner">
            <strong>Auto-optimized!</strong><br>
            Balance optimal ditemukan di <strong>10^({st.session_state.log_val:.1f}) = {10**st.session_state.log_val:.4f}</strong>
            dengan PSNR <strong>{st.session_state.optimal_psnr:.2f} dB</strong>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("Re-optimize Balance", use_container_width=True,
                     help="Cari ulang balance optimal dengan parameter degradasi saat ini"):
            with st.spinner("Mencari balance optimal..."):
                _img = load_as_gray(io.BytesIO(st.session_state.file_bytes))
                _psf = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
                _blurred = convolve2d(_img, _psf, kernel_size)
                _rng = np.random.default_rng(42)
                _noisy = _blurred + noise_level * _img.std() * _rng.standard_normal(_img.shape)
                _noisy = np.clip(_noisy, 0, 1)

                opt_log, opt_psnr = find_optimal_balance(_img, _noisy, _psf)

                st.session_state.pending_reoptimize = {"log": opt_log, "psnr": opt_psnr}
                st.rerun()

    st.markdown(
        f"""
        <div class="balance-display">
            <span class="balance-label">Balance</span>
            <span class="balance-value">{balance_val:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MAIN PROCESSING
# ============================================================
if uploaded_file is not None and st.session_state.file_bytes is not None:
    try:
      with st.spinner("Memproses gambar — harap tunggu..."):
        im_original = load_as_gray(io.BytesIO(st.session_state.file_bytes))
        h, w = im_original.shape
        was_resized = max(h, w) == MAX_DIM

        psf = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
        im_blurred = convolve2d(im_original, psf, kernel_size)

        rng = np.random.default_rng(42)
        std = im_original.std()
        noise_sigma = noise_level * std if std > 1e-8 else noise_level * 0.1
        im_noisy = im_blurred + noise_sigma * rng.standard_normal(im_original.shape)
        im_noisy = np.clip(im_noisy, 0, 1)

        try:
            im_unsup, _ = restoration.unsupervised_wiener(im_noisy, psf, clip=True)
            im_unsup = np.clip(im_unsup, 0, 1)
        except Exception:
            im_unsup = im_noisy.copy()

        im_restored = np.clip(
            restoration.wiener(im_noisy, psf, balance=balance_val), 0, 1
        )

        def safe_psnr(ref, img):
            try:
                val = metrics.peak_signal_noise_ratio(ref, img, data_range=1.0)
                return val if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        def safe_ssim(ref, img):
            try:
                val = metrics.structural_similarity(ref, img, data_range=1.0)
                return val if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        psnr_noisy    = safe_psnr(im_original, im_noisy)
        psnr_unsup    = safe_psnr(im_original, im_unsup)
        psnr_restored = safe_psnr(im_original, im_restored)

        ssim_noisy    = safe_ssim(im_original, im_noisy)
        ssim_unsup    = safe_ssim(im_original, im_unsup)
        ssim_restored = safe_ssim(im_original, im_restored)

    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
        st.stop()

    # ----------------------------------------------------------
    # RESIZE NOTICE
    # ----------------------------------------------------------
    if was_resized:
        h, w = im_original.shape
        st.markdown(
            f'''<div class="tip-box" style="margin-bottom:16px;">
                <strong>Gambar di-resize otomatis ke {w}x{h}px</strong> untuk menjaga performa.
                Wiener filter bekerja optimal di resolusi ini.
            </div>''',
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------
    # METRIC CARDS ROW
    # ----------------------------------------------------------
    st.markdown('<div class="section-header">Perbandingan Metrik</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Degraded</div>
            <div class="metric-value">{psnr_noisy:.2f} <span style="font-size:0.95rem">dB</span></div>
            <div class="metric-unit">PSNR &nbsp;|&nbsp; SSIM {ssim_noisy:.3f} &nbsp; {psnr_badge(psnr_noisy)}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unsupervised Wiener</div>
            <div class="metric-value">{psnr_unsup:.2f} <span style="font-size:0.95rem">dB</span></div>
            <div class="metric-unit">PSNR &nbsp;|&nbsp; SSIM {ssim_unsup:.3f} &nbsp; {psnr_badge(psnr_unsup)}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Wiener (Balance {balance_val:.4f})</div>
            <div class="metric-value">{psnr_restored:.2f} <span style="font-size:0.95rem">dB</span></div>
            <div class="metric-unit">PSNR &nbsp;|&nbsp; SSIM {ssim_restored:.3f} &nbsp; {psnr_badge(psnr_restored)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # ----------------------------------------------------------
    # IMAGE GRID
    # ----------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Kondisi Input</div>', unsafe_allow_html=True)

        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(im_original, caption="Original", use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(im_noisy, caption=f"Degraded — PSNR {psnr_noisy:.2f} dB | SSIM {ssim_noisy:.3f}",
                 use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Hasil Restorasi</div>', unsafe_allow_html=True)

        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(im_unsup, caption=f"Unsupervised Wiener — PSNR {psnr_unsup:.2f} dB | SSIM {ssim_unsup:.3f}",
                 use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(im_restored, caption=f"Wiener (Balance {balance_val:.4f}) — PSNR {psnr_restored:.2f} dB | SSIM {ssim_restored:.3f}",
                 use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------
    # 3D FREQUENCY SPECTRUM
    # ----------------------------------------------------------
    with st.expander("Spektrum Frekuensi 3D", expanded=False):
        st.markdown(
            '<p style="color:#6868a0; font-size:0.85rem; margin-bottom:16px;">'
            'Visualisasi log-magnitude spektrum frekuensi (dalam dB) dari setiap tahap pemrosesan.</p>',
            unsafe_allow_html=True,
        )

        fc1, fc2 = st.columns(2)
        with fc1:
            fig1 = plot_freq_spec_3d(fp.fft2(im_original), "Original")
            st.pyplot(fig1)
            plt.close(fig1)

            fig2 = plot_freq_spec_3d(fp.fft2(im_noisy), "Degraded (Blur + Noise)")
            st.pyplot(fig2)
            plt.close(fig2)

        with fc2:
            fig3 = plot_freq_spec_3d(fp.fft2(im_unsup), "Unsupervised Wiener")
            st.pyplot(fig3)
            plt.close(fig3)

            fig4 = plot_freq_spec_3d(fp.fft2(im_restored), f"Wiener (Balance {balance_val:.4f})")
            st.pyplot(fig4)
            plt.close(fig4)

    # ----------------------------------------------------------
    # TIP BOX
    # ----------------------------------------------------------
    best_psnr = max(psnr_unsup, psnr_restored)
    best_method = "Unsupervised Wiener" if psnr_unsup >= psnr_restored else f"Wiener (Balance {balance_val:.4f})"

    st.markdown(f"""
    <div class="tip-box">
        <strong>Tips:</strong> Geser slider <em>Log₁₀ Balance</em> di sidebar untuk menemukan PSNR tertinggi.
        Saat ini metode terbaik adalah <strong>{best_method}</strong> dengan PSNR <strong>{best_psnr:.2f} dB</strong>.
        <br>Klik tombol <strong>Re-optimize Balance</strong> di sidebar setelah mengubah parameter degradasi untuk mencari ulang balance optimal.
    </div>
    """, unsafe_allow_html=True)

else:
    # ----------------------------------------------------------
    # WELCOME STATE
    # ----------------------------------------------------------
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon"></div>
        <h2>Wiener Filter Dashboard</h2>
        <p>
            Upload gambar di panel kiri untuk memulai.
            Dashboard ini akan <strong>mendegradasi</strong> gambar dengan blur &amp; noise,
            lalu <strong>merestorasi</strong>-nya menggunakan Wiener Filter — semua secara real-time.
        </p>
        <div class="welcome-steps">
            <div class="welcome-step">Upload gambar</div>
            <div class="welcome-step">Atur degradasi</div>
            <div class="welcome-step">Tuning balance</div>
            <div class="welcome-step">Bandingkan PSNR</div>
        </div>
    </div>
    """, unsafe_allow_html=True)