"""
SAR Image Denoising — Interactive Prototype
Internship Project: Space Applications Centre (SAC), ISRO
Author: Jani Pruthak Maulik | L.D. College of Engineering | GTU
Supervisor: Dr. Bhaskar Dubey, Scientist/Engineer 'SF', SAC-ISRO
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import (
    uniform_filter, gaussian_filter, median_filter, convolve
)
from skimage.restoration import denoise_nl_means, denoise_bilateral, estimate_sigma
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from skimage.feature import canny
import io
import warnings
warnings.filterwarnings("ignore")
from streamlit_image_comparison import image_comparison

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SAR Denoising — Pruthak Jani",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem; font-weight: 800;
        color: #000000;
        margin-bottom: 0;
    }
    .subtitle {
        color: #555; font-size: 0.95rem; margin-top: 0.2rem;
    }
    .info-box {
        background: #e8f4fd; border-left: 4px solid #1a73e8;
        padding: 0.7rem 1rem; border-radius: 4px; margin: 0.5rem 0;
    }
    .eq-box {
        background: #f8f9fa; border: 1px solid #dee2e6;
        padding: 0.6rem 1rem; border-radius: 6px;
        font-family: 'Courier New', monospace; font-size: 0.85rem;
        margin: 0.3rem 0;
    }
    .metric-good { color: #28a745; font-weight: bold; }
    .metric-warn { color: #fd7e14; font-weight: bold; }
    .section-header {
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 4px; margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🛰️ SAR Image Denoising — Interactive Prototype</p>', unsafe_allow_html=True)
st.markdown("""
<p class="subtitle">
  <b>Jani Pruthak Maulik</b> | B.E. AI&ML, L.D. College of Engineering, GTU &nbsp;|&nbsp;
  Internship: <b>Space Applications Centre (SAC), ISRO, Ahmedabad</b> &nbsp;|&nbsp;
  Guide: <b>Dr. Bhaskar Dubey</b>, Scientist/Engineer 'SF'
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", [
    "📖 Project Overview",
    "📂 Load & Visualise Image",
    "🧪 Add Speckle Noise",
    "🧹 Denoise",
    "🎨 Texture Enhancement",
    "📊 Compare & Evaluate"
])

# ═══════════════════════════════════════════════════════════════════════════
# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_sar_bin(data_bytes, H, W):
    arr = np.frombuffer(data_bytes, dtype=np.float32)
    return arr.reshape(H, W)

def load_png(data_bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(data_bytes)).convert("L")
    return np.array(img, dtype=np.float32)

def display_img(img, pct_lo, pct_hi, log_scale):
    """Return uint8 for st.image display given display settings."""
    work = img.copy().astype(np.float32)
    if log_scale:
        mn = work.min()
        if mn <= 0:
            work = work - mn + 1e-5
        work = 10.0 * np.log10(work + 1e-8)
    lo = np.percentile(work, pct_lo)
    hi = np.percentile(work, pct_hi)
    if hi == lo:
        hi = lo + 1e-5
    clipped = np.clip(work, lo, hi)
    u8 = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    return u8

def sidebar_display_controls(key_prefix=""):
    """Render sidebar display controls and return (log, lo_pct, hi_pct)."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ Display Controls")
    log = st.sidebar.toggle("Log Scale (dB)", value=False, key=f"{key_prefix}_log")
    lo, hi = st.sidebar.slider(
        "Percentile Clipping (display only)",
        0.0, 100.0, (0.0, 99.0), 0.1,
        key=f"{key_prefix}_pct"
    )
    return log, lo, hi

def show_histogram(img_flat, lo_val, hi_val, key):
    samp = np.random.choice(img_flat, size=min(50000, len(img_flat)), replace=False)
    fig, ax = plt.subplots(figsize=(4, 1.8))
    ax.hist(samp, bins=60, color='steelblue', alpha=0.7)
    ax.axvline(lo_val, color='red', linestyle='--', linewidth=1.2, label='Lo cut')
    ax.axvline(hi_val, color='orange', linestyle='--', linewidth=1.2, label='Hi cut')
    ax.get_yaxis().set_visible(False)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    st.sidebar.pyplot(fig)
    plt.close(fig)

def compute_image_stats(img):
    img = img.astype(np.float32)
    mean = float(np.mean(img))
    var = float(np.var(img))
    std = float(np.std(img))
    return {
        "Shape": str(img.shape),
        "Min": float(np.min(img)),
        "Max": float(np.max(img)),
        "Mean": mean,
        "Std": std,
        "ENL": float((mean ** 2) / (var + 1e-10)),
        "CV": float(std / (mean + 1e-10)),
    }

def add_speckle_noise(img, looks=4.0):
    """Add multiplicative Gamma speckle noise with mean 1 and variance 1/L."""
    img = img.astype(np.float32)
    looks = max(float(looks), 1e-3)
    noise = np.random.gamma(shape=looks, scale=1.0 / looks, size=img.shape).astype(np.float32)
    noisy = img * noise
    return np.clip(noisy, 0, None).astype(np.float32), noise

# ───────────── DENOISING ALGORITHMS ─────────────

def gaussian_denoise(img, sigma=1.0):
    return gaussian_filter(img.astype(np.float32), sigma=sigma)

def median_denoise(img, size=3):
    return median_filter(img.astype(np.float32), size=size)

def wiener_denoise(img, size=5):
    """Simple local Wiener filter implementation."""
    img = img.astype(np.float32)
    local_mean = uniform_filter(img, size=size)
    local_sq_mean = uniform_filter(img**2, size=size)
    local_var = local_sq_mean - local_mean**2
    noise_var = np.mean(local_var)
    w = np.maximum(local_var - noise_var, 0) / (local_var + 1e-10)
    return local_mean + w * (img - local_mean)

def lee_filter(img, size=7, looks=1.0):
    """Classic Lee filter for SAR speckle reduction."""
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img**2, size=size)
    var = np.maximum(mean_sq - mean**2, 0.0)
    noise_var = (mean**2) / max(looks, 1e-5)
    w = var / (var + noise_var + 1e-10)
    return mean + w * (img - mean)

def enhanced_lee_filter(img, window_size=5, looks=3.7, cu=1.0):
    """Enhanced Lee filter: homogeneous → mean, heterogeneous → Lee, point → preserve."""
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=window_size, mode='reflect')
    mean_sq = uniform_filter(img**2, size=window_size, mode='reflect')
    var = np.maximum(mean_sq - mean**2, 0.0)
    cv = np.sqrt(var) / (mean + 1e-6)
    cv_img = np.sqrt(1.0 / max(looks, 1e-5))
    result = np.zeros_like(img)
    mask_h = cv <= cv_img
    result[mask_h] = mean[mask_h]
    mask_het = (cv > cv_img) & (cv < cu)
    if np.any(mask_het):
        denom = var[mask_het] + mean[mask_het]**2 / looks + 1e-6
        w = var[mask_het] / denom
        result[mask_het] = mean[mask_het] + w * (img[mask_het] - mean[mask_het])
    mask_point = cv >= cu
    result[mask_point] = img[mask_point]
    return result

def frost_filter(img, window_size=5, damping=2.0):
    """Frost adaptive exponential filter for SAR."""
    img = img.astype(np.float32)
    local_mean = uniform_filter(img, size=window_size)
    local_sq_mean = uniform_filter(img**2, size=window_size)
    local_var = local_sq_mean - local_mean**2
    C = local_var / (local_mean**2 + 1e-12)
    alpha = damping * C.mean()
    pad = window_size // 2
    x, y = np.meshgrid(np.arange(-pad, pad + 1), np.arange(-pad, pad + 1))
    dist = np.sqrt(x**2 + y**2)
    kernel = np.exp(-alpha * dist)
    kernel /= kernel.sum()
    return convolve(img, kernel, mode='reflect').astype(np.float32)

def blpf_filter(img, D0=30, order=2):
    """Butterworth Low-Pass Filter in frequency domain."""
    img = img.astype(np.float32)
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
    rows, cols = img.shape
    F = fftshift(fft2(img))
    u, v = np.meshgrid(
        np.arange(-cols // 2, cols // 2),
        np.arange(-rows // 2, rows // 2)
    )
    D = np.sqrt(u**2 + v**2)
    H = 1.0 / (1.0 + (D / (D0 + 1e-8))**(2 * order))
    filtered = np.real(ifft2(ifftshift(F * H)))
    return filtered.astype(np.float32)

def bilateral_denoise(img, sigma_s=3.0, sigma_r=0.15):
    """Log-domain bilateral filter for SAR."""
    img = img.astype(np.float32)
    log_img = np.log(img + 1e-6)
    filt = denoise_bilateral(log_img, sigma_color=sigma_r, sigma_spatial=sigma_s)
    return np.exp(filt).astype(np.float32)

def nlm_denoise(img, h=None, patch_size=3, patch_distance=5, use_h_cv=True, alpha=0.55):
    """Non-Local Means denoising.

    Parameters added:
    - use_h_cv: if True prefer h_cv estimate, else use sigma_est (scaled by alpha)
    - alpha: multiplier for sigma_est (so sigma_est = alpha * mean(estimate_sigma(...)))
    """
    img = img.astype(np.float32)
    # mn, mx = img.min(), img.max()
    # norm = (img - mn) / (mx - mn + 1e-8)
    norm = img

    sigma_est_base = float(np.mean(estimate_sigma(norm, channel_axis=None)))
    sigma_est = float(alpha) * sigma_est_base
    mean = uniform_filter(norm, 7)
    sq_mean = uniform_filter(norm ** 2, 7)
    var = sq_mean - mean ** 2
    cv = np.sqrt(np.maximum(var, 0.0)) / (mean + 1e-8)
    h_cv = float(np.mean(0.5 * cv * mean))

    if h is None:
        if use_h_cv:
            h = h_cv if np.isfinite(h_cv) and h_cv > 0 else sigma_est
        else:
            h = sigma_est
    else:
        h = float(h)

    out = denoise_nl_means(norm, h=h, patch_size=patch_size,
                        patch_distance=patch_distance, fast_mode=True,
                        channel_axis=None)
    # return (out * (mx - mn) + mn).astype(np.float32)
    return out.astype(np.float32)

def vpde_denoise(img, lam=2.0, rho=0.8, alpha=1.3, beta=0.7,
                delta=4.0, zeta=0.5, eta=0.2, max_iter=5, tol=0.05):
    """
    Variational PDE denoising (VariationalSARDespeckle).
    Parabolic PDE: ∂u/∂t = λ·∇·(ξ(|∇u|)∇u) + ρ·(u₀ - u)
    """
    img = img.astype(np.float32)
    u0 = np.log(np.clip(img, 1e-8, None))
    u = u0.copy()
    prev_grad_norm = None

    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1, 1:-1] = (arr[2:, 1:-1] + arr[:-2, 1:-1] +
                        arr[1:-1, 2:] + arr[1:-1, :-2] - 4*arr[1:-1, 1:-1])
        return lap

    def mixed_deriv(arr):
        uxy = np.zeros_like(arr)
        uxy[1:-1, 1:-1] = (arr[2:, 2:] - arr[2:, :-2] -
                           arr[:-2, 2:] + arr[:-2, :-2]) * 0.25
        return uxy

    def cross_grad(arr):
        cg = np.zeros_like(arr)
        cg[1:-1, 1:-1] = (arr[2:, 1:-1] - arr[:-2, 1:-1] +
                        arr[1:-1, 2:] - arr[1:-1, :-2])
        return cg

    for t in range(max_iter):
        ux = np.zeros_like(u); uy = np.zeros_like(u)
        ux[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
        uy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * 0.5
        grad_mag = np.sqrt(ux**2 + uy**2)
        gamma = alpha * grad_mag.mean() + eta * t
        s = grad_mag + gamma + 1e-8
        log_term = np.log(np.clip(s, 1e-8, None))
        denom = beta * (log_term**3) + delta
        xi_val = zeta * gamma / denom
        xi_prime = (-3*zeta*gamma*beta*(log_term**2)) / (denom**2 * s)
        lap = laplacian(u)
        uxy = mixed_deriv(u)
        cg = cross_grad(u)
        u_new = (u*(1-rho) + lam*xi_val*lap +
                lam*xi_prime*uxy*cg/8.0 + rho*u0)
        grad_norm = np.linalg.norm(grad_mag)
        if prev_grad_norm is not None:
            if abs(grad_norm - prev_grad_norm) / (prev_grad_norm + 1e-12) < tol:
                break
            if grad_norm > prev_grad_norm:
                break
        prev_grad_norm = grad_norm
        u = u_new

    return np.exp(u).astype(np.float32)

def hpde_denoise(img, lam=2.4, gamma_h=1.5, alpha=1.8,
                zeta=0.15, dt=0.2, N=3):
    """
    Hyperbolic PDE denoising (StableHyperbolicPDE).
    Second-order wave-like equation:
    u^{n+1} = (2 - γ²Δt)u^n - (1 - γ²Δt/2)u^{n-1} + Δt²(α·Δu - ζ(u-u₀))
    """
    img = img.astype(np.float32)
    u0_full = np.log(np.clip(img, 1e-8, None))
    # Pad for boundary stability
    U = np.zeros((img.shape[0]+10, img.shape[1]+10), dtype=np.float32)
    U[5:-5, 5:-5] = u0_full
    u0 = U.copy()
    u_prev = u0.copy()

    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1, 1:-1] = (arr[2:, 1:-1] + arr[:-2, 1:-1] +
                        arr[1:-1, 2:] + arr[1:-1, :-2] - 4*arr[1:-1, 1:-1])
        return lap

    lap0 = laplacian(u0)
    u = u0 + 0.5 * dt**2 * (alpha * lap0 - zeta * (u0 - u0))

    for n in range(1, N):
        lap = laplacian(u)
        u_next = ((2 - gamma_h**2 * dt) * u
                  - (1 - gamma_h**2 * dt / 2) * u_prev
                  + dt**2 * (alpha * lap - zeta * (u - u0)))
        if np.isnan(u_next).any() or np.isinf(u_next).any():
            break
        u_next = np.clip(u_next, -20, 20)
        u_prev = u
        u = u_next

    u = u[5:-5, 5:-5]
    return np.exp(u).astype(np.float32)

# ───────────── TEXTURE ENHANCEMENT ─────────────

def sobel_enhancement(orig, denoised, amount=0.75):
    """Sobel-based sharpening: inject high-frequency edges back."""
    gx = sobel(orig, axis=0, mode='reflect')
    gy = sobel(orig, axis=1, mode='reflect')
    edges = np.hypot(gx, gy)
    out = denoised + amount * edges
    return out.astype(np.float32)

def canny_enhancement(orig, denoised, sigma=1.0, amount=0.5):
    """Canny edge map injection for texture enhancement."""
    mn, mx = orig.min(), orig.max()
    norm = (orig - mn) / (mx - mn + 1e-8)
    edges = canny(norm, sigma=sigma).astype(np.float32)
    edges = edges * (mx - mn)
    out = denoised + amount * edges
    return out.astype(np.float32)

def structure_tensor_enhancement(orig, denoised, sigma=2.0, alpha_inj=0.3):
    """
    New Injection (Structure Tensor) — gradient-weighted high-freq injection.
    Extracts structure tensor components and injects texture proportional
    to local anisotropy.
    """
    orig = orig.astype(np.float32)
    denoised = denoised.astype(np.float32)
    EPS = 1e-8

    # Work in log domain
    log_orig = np.log(orig + EPS)
    low = gaussian_filter(log_orig, sigma=sigma)
    high_freq = log_orig - low

    # Structure tensor
    Ix = sobel(log_orig, axis=1, mode='reflect')
    Iy = sobel(log_orig, axis=0, mode='reflect')
    Jxx = gaussian_filter(Ix**2, sigma=sigma)
    Jxy = gaussian_filter(Ix*Iy, sigma=sigma)
    Jyy = gaussian_filter(Iy**2, sigma=sigma)

    # Coherence / anisotropy measure
    trace = Jxx + Jyy + EPS
    coherence = np.sqrt((Jxx - Jyy)**2 + 4*Jxy**2) / trace
    coherence = np.clip(coherence, 0, 1)

    log_den = np.log(denoised + EPS)
    enhanced_log = log_den + alpha_inj * high_freq * coherence
    return np.exp(enhanced_log).astype(np.float32)

def hf_injection(orig, denoised, sigma=10.0, alpha_inj=1.0, grad_power=0.15):
    """
    High-frequency component injection.
    Extracts HF = orig - smooth(orig) in log domain,
    weights by gradient magnitude, injects into denoised.
    """
    orig = orig.astype(np.float32)
    denoised = denoised.astype(np.float32)
    EPS = 1e-8

    log_orig = np.log(orig + EPS)
    low = gaussian_filter(log_orig, sigma=sigma)
    high_freq = log_orig - low

    gx = sobel(log_orig, axis=1)
    gy = sobel(log_orig, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag = grad_mag / (grad_mag.max() + EPS)
    grad_mag = grad_mag ** grad_power

    log_den = np.log(denoised + EPS)
    mo = np.mean(orig)
    enhanced_log = log_den + alpha_inj * high_freq * grad_mag
    enhanced = np.exp(enhanced_log)
    enhanced = enhanced - np.mean(enhanced) + mo
    return np.clip(enhanced, 0, None).astype(np.float32)

# ───────────── METRICS ─────────────

def compute_metrics(ref, img, nesz_patches=None):
    eps = 1e-10
    ref = ref.astype(np.float32)
    img = img.astype(np.float32)
    mn = img.min(); mx = img.max()
    mean = np.mean(img); var = np.var(img); std = np.std(img)
    mean_ref = np.mean(ref); std_ref = np.std(ref)

    ENL = (mean**2) / (var + eps)
    CV = std / (mean + eps)
    SNR = 10 * np.log10(1 + mean / (std + eps))
    PSNR = 10 * np.log10((mx**2) / (var + eps))
    SSIM_val = ssim(ref, img, data_range=max(img.max(), ref.max()))
    SSI = (std / (mean + eps)) / (std_ref / (mean_ref + eps))

    # EPI
    grad_orig = sobel(ref); grad_den = sobel(img)
    et = np.percentile(grad_orig, 90)
    mask = grad_orig > et
    if mask.sum() > 0:
        EPI = grad_den[mask].mean() / (grad_orig[mask].mean() + eps)
    else:
        EPI = 0.0

    # ESI
    eps2 = 1e-10
    orig_h = ref[:, :-1] / (ref[:, 1:] + eps2)
    denoised_h = img[:, :-1] / (img[:, 1:] + eps2)
    orig_v = ref[:-1, :] / (ref[1:, :] + eps2)
    denoised_v = img[:-1, :] / (img[1:, :] + eps2)
    num = np.sum(np.abs(orig_h - denoised_h)) + np.sum(np.abs(orig_v - denoised_v))
    denom = np.sum(orig_h) + np.sum(orig_v)
    ESI = float(np.clip(1 - num / (denom + eps2), 0, 1))

    # NESZ
    if nesz_patches:
        NESZ = 0
        for patch in nesz_patches:
            p = img[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]]
            p = p**2
            p_range = (img.max()**2 - img.min()**2)
            p = ((p - p.min()) / (p.max() - p.min() + eps)) * p_range + img.min()**2
            NESZ += 10 * np.log10(1 + np.var(p))
        NESZ /= len(nesz_patches)
    else:
        # Auto estimate from image corners
        ps = min(64, img.shape[0]//8, img.shape[1]//8)
        auto_patches = [
            [[0, ps], [0, ps]],
            [[0, ps], [img.shape[1]-ps, img.shape[1]]],
            [[img.shape[0]-ps, img.shape[0]], [0, ps]],
            [[img.shape[0]-ps, img.shape[0]], [img.shape[1]-ps, img.shape[1]]]
        ]
        NESZ = 0
        for patch in auto_patches:
            p = img[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]]
            p = p**2
            p_range = (img.max()**2 - img.min()**2)
            p = ((p - p.min()) / (p.max() - p.min() + eps)) * p_range + img.min()**2
            NESZ += 10 * np.log10(1 + np.var(p))
        NESZ /= 4

    return {
        "MIN": float(mn), "MAX": float(mx),
        "MEAN": float(mean), "SD": float(std),
        "ENL": float(ENL), "CV": float(CV),
        "SNR (dB)": float(SNR), "PSNR (dB)": float(PSNR),
        "SSIM": float(SSIM_val), "SSI": float(SSI),
        "EPI": float(EPI), "ESI": float(ESI),
        "NESZ (dB)": float(NESZ)
    }

# ───────────── SESSION STATE ─────────────
if "img_raw" not in st.session_state:
    st.session_state.img_raw = None
if "img_denoised" not in st.session_state:
    st.session_state.img_denoised = None
if "denoise_method" not in st.session_state:
    st.session_state.denoise_method = None
if "img_enhanced" not in st.session_state:
    st.session_state.img_enhanced = None
if "img_clean" not in st.session_state:
    st.session_state.img_clean = None
if "img_noisy" not in st.session_state:
    st.session_state.img_noisy = None
if "speckle_looks" not in st.session_state:
    st.session_state.speckle_looks = 4.0

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📖 Project Overview":

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 About & Internship",
        "📡 SAR & Speckle",
        "🔬 Denoising Algorithms",
        "🎨 Texture & Pipeline",
        "📏 Metrics"
    ])

    with tab1:
        st.markdown("## 🏢 About This Project")
        st.markdown("""
<div class="info-box">
<b>Title:</b> SAR Image Denoising using Classical, Statistical, and PDE-based Methods<br>
<b>Organization:</b> Space Applications Centre (SAC), ISRO, Ahmedabad<br>
<b>Division:</b> MQCD (Microwave Data Quality Evaluation & Calibration Division) — DQCG/SIPA<br>
<b>External Guide:</b> Dr. Bhaskar Dubey, Scientist/Engineer 'SF'<br>
<b>Internal Guide:</b> Prof. Hitesh D. Rajput, L.D. College of Engineering<br>
<b>Student:</b> Jani Pruthak Maulik | Enroll: 220280152023 | B.E. AI&ML, 8th Sem
</div>
""", unsafe_allow_html=True)

        st.markdown("### 🎯 Internship Objectives")
        st.markdown("""
- Study and implement classical SAR speckle reduction algorithms (Gaussian, Median, Lee, Frost, Bilateral, NLM)
- Investigate PDE-based models: Variational PDE (VPDE) and Hyperbolic PDE (HPDE)
- Develop texture preservation and enhancement pipeline (Sobel, Canny, Structure Tensor, HF Injection)
- Apply post-processing: Histogram Matching for radiometric consistency
- Quantitatively evaluate methods using ENL, CV, SSIM, PSNR, SNR, NESZ, EPI, ESI metrics
- Explore deep learning: Residual UNet with Vision Transformer (ViT) for learned denoising
- Process NISAR/SSAR RSLC binary SAR data in float32 format
""")

        st.markdown("### 🛠️ Tools & Technologies")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Data Processing**\n- NumPy, SciPy\n- scikit-image\n- Matplotlib")
        with cols[1]:
            st.markdown("**Deep Learning**\n- PyTorch, CUDA\n- Residual UNet + ViT\n- Attention Mechanisms")
        with cols[2]:
            st.markdown("**HPC Environment**\n- Linux workstation\n- GPU-accelerated compute\n- NISAR SSAR data")

        st.markdown("### 🗺️ Project Workflow")
        st.markdown("""
```
RAW SAR BINARY (.bin/.img float32)
        │
        ▼
┌──────────────────┐
│  Data Loading    │  np.fromfile(path, dtype='float32').reshape(H, W)
│  & Preprocessing │  Log transform, Percentile clipping, Impulse removal
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Denoising      │  Gaussian → Median → BLPF → Wiener → Lee → Enhanced Lee
│   Methods        │  Frost → Bilateral → NLM → VPDE → HPDE
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Texture         │  Sobel → Canny → Structure Tensor → HF Injection
│  Enhancement     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Post-processing │  Histogram Matching (radiometric correction)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Evaluation      │  ENL, CV, SSIM, PSNR, SNR, NESZ, EPI, ESI, SSI
└──────────────────┘
```
""")

    with tab2:
        st.markdown("## 📡 SAR Imaging & Speckle Noise")

        st.markdown("### What is SAR?")
        
# - **Complex baseband return (continuous model):** for a scene reflectivity \\(\\sigma(x,y)\\) the
#     received baseband signal (range τ, azimuth/time η) can be expressed as

#     $$
#     r(\\tau,\\eta)=\\iint \\sigma(x,y)\\,e^{-j\\frac{4\\pi}{\\lambda}R(x,y;\\eta)}\\,dx\\,dy
#     $$

# - **Radar equation (monostatic, power form):** received power decreases with range; a common form is

#     $$P_{r}\\propto \\dfrac{P_{t}G^{2}\\lambda^{2}\\sigma_{0}}{(4\\pi)^{3}R^{4}}$$
        st.markdown("""
**Synthetic Aperture Radar (SAR)** is an active, coherent microwave imaging system that transmits
pulsed microwave energy and records the complex-valued backscatter (amplitude + phase). Because
SAR measures coherent returns, it forms high-resolution images independent of daylight or cloud cover.

Key concepts and relations:

- **Range (slant-range) and timing:** the two-way travel time τ from transmit to target and back
    gives slant range:

    $$R(\\tau)=\\dfrac{c\\,\\tau}{2}$$

- **Range resolution:** determined by the transmitted pulse bandwidth B (or chirp bandwidth):

    $$\\Delta R=\\dfrac{c}{2B}$$

- **Azimuth (along-track) / Synthetic aperture:** by moving the sensor along-track and coherently
    combining many pulses, SAR synthesises a long aperture. After Doppler processing (azimuth
    compression) the azimuth resolution becomes effectively independent of range and depends on the
    synthetic aperture and wavelength — this is what gives SAR its fine along-track resolution.

Processing chain (typical SAR imaging pipeline):

1. **Range compression (matched filtering):** deconvolves the transmit pulse (chirp) to obtain
    high range resolution (implements $\\Delta R = c/(2B)$).
2. **Range cell migration correction (RCMC):** corrects range walk of targets across pulses.
3. **Azimuth compression (Doppler processing):** focusses energy in azimuth by exploiting Doppler
    shift of moving platform; produces the synthetic aperture effect.
4. **Multilooking:** incoherent averaging of independent looks to reduce speckle variance at the
    expense of radiometric resolution (L looks ⇒ speckle variance ≈ 1/L).
5. **Calibration & geocoding:** convert image coordinates to map coordinates and apply radiometric
    calibration factors.
    """)
        st.markdown("### ⚡ Speckle Noise — The Core Problem")
        st.markdown("""
Synthetic Aperture Radar (SAR) imagery fundamentally suffers from **speckle**, an intrinsic granular artifact arising from its coherent nature. When an active radar wave illuminates a target area, a single resolution cell contains numerous unresolvable sub-resolution microscopic scatterers. The backscattered waves undergo random phase shifts, leading to localized constructive and destructive interference. This produces a signal-dependent, "salt-and-pepper" visual distortion modeled mathematically as non-additive multiplicative noise. The commonly used intensity model is:

$$
I(x,y)=S(x,y)\\cdot N(x,y)
$$

where $S(x,y)$ is the underlying scene intensity (reflectivity) and $N(x,y)$ is speckle noise.
For fully developed speckle, the multiplicative noise follows a Gamma distribution for an L-look image:
$N\sim\mathrm{Gamma}(L,1/L)$ with mean 1 and variance $1/L$.

Common practices to handle speckle:

- Work in the log-domain: $\\ln I = \\ln S + \\ln N$ converts multiplicative speckle to additive noise.
- Use multilooking or adaptive filters (Lee, Frost, Enhanced Lee) that model speckle statistics.
- Use non-local or variational methods (NLM, VPDE/HPDE) to preserve texture and edges while
    suppressing speckle.
    """)

#         st.markdown("### ⚡ Speckle Noise — The Core Problem")
#         st.markdown("""
# SAR images suffer from **speckle**, a granular interference pattern caused by
# coherent summation of backscattered waves from multiple scatterers within a resolution cell.
# """)
# # #         st.markdown('<div class="eq-box">Multiplicative Model:&nbsp;&nbsp; I(x,y) = S(x,y) · N(x,y)</div>', unsafe_allow_html=True)
# #         st.markdown("""
# # $$
# # I(x,y) = S(x,y)\cdot N(x,y)
# # $$

# # Where:
# # - `I(x,y)` = Observed SAR intensity
# # - `S(x,y)` = True ground reflectivity
# # - `N(x,y)` = Multiplicative speckle noise

# # **Key Statistical Properties of Speckle:**
# # - Fully developed speckle follows a **Gamma distribution**
# # - Variance of speckle = `σ²_n = 1/L` where L = number of looks
# # - **ENL (Equivalent Number of Looks)** = `μ² / σ²` measures speckle amount

# # **Challenge:** Denoising must separate S from N without blurring edges or destroying texture.
# # The multiplicative nature means simple additive filters (Gaussian, Median) are suboptimal —
# # log transform converts it to additive: `ln(I) = ln(S) + ln(N)`.
# # """)

#         st.markdown("### SAR signal model & speckle statistics")
#         # st.markdown('<div class="eq-box">Multiplicative model: I(x,y) = S(x,y) \\times N(x,y)</div>', unsafe_allow_html=True)
#         st.markdown("""
#     Observed intensity is modelled as multiplicative speckle:

#     $$
#     I(x,y) = S(x,y)\cdot N(x,y)
#     $$

#     For fully developed speckle, $N$ follows a Gamma distribution for an $L$-look image:
#     $N\sim\mathrm{Gamma}(L,\,1/L)$ with mean 1 and variance $1/L$. Important derived measures:

#     - ENL (Equivalent Number of Looks): $\mathrm{ENL}=\mu^2/\sigma^2$ (higher ENL ⇒ less speckle)
#     - CV (coefficient of variation): $\mathrm{CV}=\sigma/\mu$

# **Challenge:** Denoising must separate S from N without blurring edges or destroying texture.
# The multiplicative nature means simple additive filters (Gaussian, Median) are suboptimal —
# log transform converts it to additive: ($\ln I = \ln S + \ln N$), enabling more effective denoising strategies.
#     """)

        st.markdown("### SAR modes, sensors & data products")
        st.markdown(r"""
        - **Modes:** Stripmap, Spotlight, ScanSAR, Sliding‑Spotlight, and Polarimetric SAR (PolSAR).
        - **InSAR:** interferometric SAR measures phase difference between acquisitions:
        $$\Delta\phi = \frac{4\pi}{\lambda}(R_2 - R_1)$$
        used for topography and deformation.
        - **Example platforms:** Sentinel‑1 (ESA, C‑band), TerraSAR‑X / TanDEM‑X (DLR, X‑band),
        RADARSAT (Canada, C‑band), COSMO‑SkyMed (ASI, X‑band), ALOS‑2 (JAXA, L‑band), and NISAR
        (NASA‑ISRO; L & S bands).
        - **Products at SAC/ISRO:** RSLC (range single-look complex), multilooked intensity, and
        geocoded imagery; binary float32 arrays are commonly used for algorithm development.
        """)


        st.markdown("### Applications of SAR & role of denoising")
        st.markdown("""
    SAR is used for topography (DEM), deformation monitoring (DInSAR), land cover, forestry,
    agriculture, maritime surveillance, and urban mapping. Denoising contributes to these tasks by:

    - Increasing radiometric stability (better classification & change detection).
    - Improving coherence and phase quality for InSAR (reducing phase noise and aiding unwrapping).
    - Enhancing feature detectability (roads, buildings, ships) by suppressing granular speckle.

    This app focuses on methods that preserve edges and texture (NLM, bilateral, VPDE/HPDE)
    so that downstream analytics (InSAR, classification) benefit from improved SNR without
    losing structural detail.
    """)
        st.markdown("### 🌐 SAR Data at SAC-ISRO")
        st.markdown("""
- **Sensor:** NISAR (NASA-ISRO SAR) / SSAR — L-band and S-band
- **Product:** RSLC (Range Single Look Complex), GCOV (Ground Covariance)
- **Polarization:** HH (Horizontal-Horizontal), HV, VH, VV
- **Data Format:** Binary float32, size typically 2048×2048 to 16384×16384
- **Reading:** `np.fromfile(path, dtype='float32').reshape(H, W)`
""")


    with tab3:
        st.markdown("## 🔬 Denoising Algorithms")

        algos = {
            "Gaussian Filter": {
                "desc": "Convolves image with a Gaussian kernel. Reduces speckle by weighted averaging.",
                "eq": "$$G(x,y)=\\frac{1}{2\\pi\\sigma^{2}}\\exp\\left(-\\frac{x^{2}+y^{2}}{2\\sigma^{2}}\\right)\\quad\\to\\quad\\hat{S}=I\\ast G$$",
                "pros": "Fast, simple", "cons": "Blurs edges, not SAR-specific"
            },
            "Median Filter": {
                "desc": "Replaces each pixel with the median of its neighbourhood. Non-linear, preserves edges better.",
                "eq": "$$\\hat{S}(x,y)=\\mathrm{median}\{I(x+i,y+j):(i,j)\\in W\\}$$",
                "pros": "Edge-preserving, removes impulse noise", "cons": "Can remove fine detail"
            },
            "BLPF (Butterworth LPF)": {
                "desc": "Frequency domain low-pass filter. Suppresses high-frequency noise components.",
                "eq": "$$H(u,v)=\\frac{1}{1+\\left(\\frac{D(u,v)}{D_{0}}\\right)^{2n}}\\quad\\to\\quad\\hat{S}=\\mathcal{F}^{-1}\{H\\cdot\\mathcal{F}(I)\\}$$",
                "pros": "Smooth roll-off, no ringing vs ideal LPF", "cons": "Blurs textures"
            },
            "Wiener Filter": {
                "desc": "Optimal linear filter minimising MSE. Adapts based on local noise-to-signal ratio.",
                "eq": "$$\\hat{S}=\\mu + \\frac{\\sigma^{2}_{\\text{local}}}{\\sigma^{2}_{\\text{local}}+\\sigma^{2}_{\\text{noise}}}(I-\\mu)$$",
                "pros": "Statistically optimal (MMSE)", "cons": "Assumes additive Gaussian noise"
            },
            "Lee Filter": {
                "desc": "Classic SAR adaptive filter. Preserves mean, adapts to local statistics.",
                "eq": "$$\\hat{S}=\\mu + W\\cdot (I-\\mu),\\quad W=\\frac{\\sigma^{2}_{\\text{local}}}{\\sigma^{2}_{\\text{local}}+\\sigma^{2}_{n}\\,\\mu^{2}/L}$$",
                "pros": "SAR-specific, edge-adaptive", "cons": "Square window causes blocky artifacts"
            },
            "Enhanced Lee Filter": {
                "desc": "Three-regime adaptive: homogeneous→mean, heterogeneous→Lee, point target→preserve.",
                "eq": "$$\\hat{S}=\\begin{cases}\\mu & CV\\le CV_{n}\\\\ \\mu+W(I-\\mu) & CV_{n}<CV<C_{u}\\\\ I & CV\\ge C_{u}\\end{cases}$$",
                "pros": "Point target preservation", "cons": "Threshold sensitive"
            },
            "Frost Filter": {
                "desc": "Exponential adaptive weighted average. Damping adapts to local coefficient of variation.",
                "eq": "$$\\hat{S}=\\frac{\\sum w(d)I(d)}{\\sum w(d)},\\quad w(d)=\\exp(-\\alpha\\,CV^{2}d),\\quad \\alpha=\\text{damping}\\cdot CV^{2}$$",
                "pros": "Exponential kernel, smooth adaptation", "cons": "Slower than Lee"
            },
            "Bilateral Filter": {
                "desc": "Edge-preserving smoothing: spatial + intensity Gaussian weights. Applied in log domain for SAR.",
                "eq": "$$\\hat{S}(p)=\\frac{\\sum_{q}G_{s}(\\|p-q\\|)G_{r}(|I(p)-I(q)|)I(q)}{\\sum_{q}G_{s}(\\|p-q\\|)G_{r}(|I(p)-I(q)|)}$$",
                "pros": "Strong edge preservation", "cons": "Computationally heavy"
            },
            "NLM (Non-Local Means)": {
                "desc": "Averages similar patches across entire image, not just local neighbourhood.",
                "eq": "$$\\hat{S}(x)=\\sum_{y} w(x,y) I(y),\\quad w(x,y)\\propto\\exp\\left(-\\frac{\\|P_{x}-P_{y}\\|^{2}}{h^{2}}\\right)$$",
                "pros": "Excellent texture preservation, state-of-art", "cons": "Very slow O(N^{2}P^{2})"
            },
            "VPDE (Variational PDE)": {
                "desc": "Parabolic PDE denoising with adaptive diffusion coefficient ξ(|∇u|). Fidelity term ρ(u₀-u) prevents over-smoothing.",
                "eq": "$$\\frac{\\partial u}{\\partial t}=\\lambda\\nabla\\cdot(\\xi(|\\nabla u|)\\nabla u)+\\rho(u_{0}-u),\\quad \\xi(s)=\\frac{\\zeta\\gamma}{\\beta\\ln^{3}(s+\\gamma)+\\delta}$$",
                "pros": "Anisotropic, edge-adaptive, theoretically sound", "cons": "Iterative, slow"
            },
            "HPDE (Hyperbolic PDE)": {
                "desc": "Second-order wave-type PDE. Introduces inertia/momentum preventing over-diffusion.",
                "eq": "$$u^{n+1}=(2-\\gamma^{2}\\Delta t)u^{n}-(1-\\tfrac{\\gamma^{2}\\Delta t}{2})u^{n-1}+\\Delta t^{2}(\\alpha\\Delta u-\\zeta(u-u_{0}))$$",
                "pros": "Better detail retention than VPDE", "cons": "Stability constraints on dt"
            }
        }

        for name, info in algos.items():
            with st.expander(f"**{name}**", expanded=False):
                st.markdown(f"**Description:** {info['desc']}")
                st.markdown(f'📐{info["eq"]}', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.success(f"✅ **Pros:** {info['pros']}")
                c2.error(f"⚠️ **Cons:** {info['cons']}")

    with tab4:
        st.markdown("## 🎨 Texture Enhancement & Pipeline")

        st.markdown("### Why Texture Enhancement?")
        st.markdown("""
Denoising algorithms suppress speckle but can also remove fine structural/textural detail.
Texture enhancement **re-injects** structural information from the original into the denoised output.
""")

        methods = {
            "Sobel Enhancement": {
                "desc": "Computes gradient magnitude using Sobel operator, injects scaled edges back.",
                "eq": r"$$\hat{S}_{\mathrm{enh}}=\hat{S}_{\mathrm{den}}+\alpha\,|\nabla I_{\mathrm{orig}}|,\quad \nabla I=\left(\frac{\partial I}{\partial x},\frac{\partial I}{\partial y}\right)$$"
            },
            "Canny Enhancement": {
                "desc": "Canny edge detector finds thin, precise edges. Injects binary edge map scaled by intensity range.",
                "eq": "$$E=\\mathrm{Canny}(I_{\\mathrm{orig}},\\sigma),\\quad \\\hat{S}_{\\mathrm{enh}}=\\hat{S}+\\alpha\\,E\\,(I_{\\max}-I_{\\min})$$"
            },
            "Structure Tensor": {
                "desc": "Computes structure tensor J = ∇u⊗∇u smoothed by Gaussian. Uses coherence as injection weight.",
                "eq": "$$J=\\nabla u\\otimes\\nabla u\\text{ (smoothed)},\\quad C=\\frac{\\sqrt{(J_{xx}-J_{yy})^{2}+4J_{xy}^{2}}}{J_{xx}+J_{yy}},\\quad \\\hat{S}=\\exp(\\log\\hat{S}+\\alpha\\,HF\\,C)$$"
            },
            "HF Component Injection": {
                "desc": "Extracts high-frequency layer (orig minus smooth) in log domain, injects weighted by gradient magnitude.",
                "eq": "$$HF=\\log I - G_{\\sigma}\\ast\\log I,\\quad w=\\frac{|\\nabla\\log I|^{p}}{\\max},\\quad \\\hat{S}=\\exp(\\log\\hat{S}+\\alpha\\,HF\\,w)$$"
            }
        }

        for name, info in methods.items():
            with st.expander(f"**{name}**"):
                st.markdown(info["desc"])
                st.markdown(f'📐{info["eq"]}', unsafe_allow_html=True)

        st.markdown("### 📊 Histogram Matching")
        st.markdown("""
After denoising, the pixel intensity distribution may shift.
**Histogram Matching** corrects radiometric consistency:
""")
        st.markdown('<div class="eq-box">For each pixel in denoised: find CDF match → map to reference CDF value</div>', unsafe_allow_html=True)
        st.markdown("""
This preserves the radiometric properties of the original SAR image
while benefiting from the denoised structure — critical for quantitative SAR analysis.
""")

    with tab5:
        st.markdown("## 📏 Evaluation Metrics")

        metrics = {
            "MIN / MAX / MEAN / SD": "Basic statistical descriptors of intensity distribution.",
            "ENL — Equivalent Number of Looks": r"$$\mathrm{ENL}=\frac{\mu^{2}}{\sigma^{2}}$$  Higher ENL = less speckle. Homogeneous region should have ENL \approx L (number of looks).",
            "CV — Coefficient of Variation": r"$$\mathrm{CV}=\frac{\sigma}{\mu}$$  Lower CV = smoother image. For speckle, CV \approx 1/\sqrt{L}",
            "SNR — Signal-to-Noise Ratio": r"$$\mathrm{SNR}=10\log_{10}\left(1+\frac{\mu}{\sigma}\right)$$  Higher is better.",
            "PSNR — Peak SNR": r"$$\mathrm{PSNR}=10\log_{10}\left(\frac{\mathrm{MAX}^{2}}{\sigma^{2}}\right)$$  Standard image quality measure. >30 dB typically good.",
            "SSIM — Structural Similarity": "SSIM ∈ [0,1]: compares luminance, contrast, structure between ref and denoised. Closer to 1 = better.",
            "SSI — Speckle Suppression Index": r"$$\mathrm{SSI}=\frac{\mathrm{CV}_{\mathrm{denoised}}}{\mathrm{CV}_{\mathrm{original}}}$$  < 1 indicates speckle suppression.",
            "NESZ — Noise Equivalent Sigma Zero": "NESZ: patch-based measure of radiometric noise; lower is cleaner homogeneous regions.",
            "EPI — Edge Preservation Index": r"$$\mathrm{EPI} = \frac{\sum G_{\mathrm{denoised}}}{\sum G_{\mathrm{original}}}$$ Evaluated on edge pixels; values closer to $1$ indicate optimal edge preservation.",
            "ESI — Edge Strength Index (EPD-ROA)": r"$$\mathrm{ESI} = 1 - \frac{\sum|\mathrm{ratio}_{\mathrm{orig}} - \mathrm{ratio}_{\mathrm{den}}|}{\sum \mathrm{ratio}_{\mathrm{orig}}}$$ Metric ranges within $[0,1]$; higher values signify superior edge profile performance.",
        }


        for name, desc in metrics.items():
            with st.expander(f"**{name}**"):
                st.markdown(f'📐 {desc}', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: LOAD & VISUALISE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📂 Load & Visualise Image":

    st.markdown("## 📂 Load SAR Image")
    st.markdown("""
Upload a SAR image in **`.bin` / `.img` (float32 binary)** or **`.png`** format.
- Binary files: raw float32 amplitude/intensity values, no header
- PNG: converted to float32 grayscale automatically
""")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload SAR Image", type=["bin", "img", "png"])
    with c2:
        H = st.number_input("Height (rows)", value=2048, min_value=1, step=1)
        W = st.number_input("Width (cols)", value=2048, min_value=1, step=1)

    if uploaded:
        fname = uploaded.name
        data = uploaded.read()
        try:
            with st.spinner("Loading image..."):
                if fname.endswith(".png"):
                    img = load_png(data)
                else:
                    img = load_sar_bin(data, int(H), int(W))
            st.session_state.img_clean = img
            st.session_state.img_raw = img
            st.session_state.img_denoised = None
            st.session_state.img_enhanced = None
            st.session_state.img_noisy = None
            st.success(f"✅ Loaded: `{fname}` | Shape: {img.shape} | dtype: {img.dtype}")
        except Exception as e:
            st.error(f"Error loading: {e}")

    if st.session_state.img_raw is not None:
        img = st.session_state.img_raw

        # Display controls
        log, lo_pct, hi_pct = sidebar_display_controls("load")
        st.sidebar.markdown("---")
        st.sidebar.metric("Raw Min", f"{img.min():.4f}")
        st.sidebar.metric("Raw Max", f"{img.max():.4f}")
        st.sidebar.metric("Raw Mean", f"{img.mean():.4f}")
        st.sidebar.metric("Raw Std", f"{img.std():.4f}")

        # Compute display
        work = img.copy()
        if log:
            mn = work.min()
            if mn <= 0:
                work = work - mn + 1e-5
            work = 10.0 * np.log10(work + 1e-8)
        lo_val = np.percentile(work, lo_pct)
        hi_val = np.percentile(work, hi_pct)
        show_histogram(work.flatten(), lo_val, hi_val, "hist_load")
        st.sidebar.metric("Display Lo", f"{lo_val:.4f}")
        st.sidebar.metric("Display Hi", f"{hi_val:.4f}")

        u8 = display_img(img, lo_pct, hi_pct, log)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(u8, caption=f"SAR Image — {'Log (dB)' if log else 'Linear'} scale, {lo_pct}–{hi_pct}% clipped",
                    use_container_width=True, clamp=True)
        with col2:
            st.subheader("Statistics")
            st.write(f"**Shape:** {img.shape}")
            st.write(f"**Min:** {img.min():.4f}")
            st.write(f"**Max:** {img.max():.4f}")
            st.write(f"**Mean:** {img.mean():.4f}")
            st.write(f"**Std:** {img.std():.4f}")
            ENL = (img.mean()**2) / (img.var() + 1e-10)
            CV = img.std() / (img.mean() + 1e-10)
            st.write(f"**ENL:** {ENL:.2f}")
            st.write(f"**CV:** {CV:.4f}")

        # Pixel value info
        st.markdown("""
<div class="info-box">
📌 <b>Note on display:</b> Percentile clipping is for <i>display only</i> — 
it does not modify the underlying data passed to denoising algorithms. 
Log scale converts multiplicative speckle to additive for better visual inspection.
</div>
""", unsafe_allow_html=True)

    else:
        st.info("⬆️ Please upload an image to begin.")
        st.markdown("""
**Supported formats:**
- `.bin` / `.img` — Raw binary float32 SAR data (NISAR/SSAR format). Provide correct H×W dimensions.
- `.png` — PNG image converted to float32 grayscale.
""")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: ADD SPECKLE NOISE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧪 Add Speckle Noise":

    st.markdown("## 🧪 Add Speckle Noise")
    st.markdown("""
Generate multiplicative SAR speckle noise on the loaded clean image.
Speckle is modeled as Gamma noise with number of looks $L$:

$$n \sim \Gamma(L, 1/L), \quad I_{noisy} = I_{clean} \cdot n$$

Lower $L$ means stronger speckle; higher $L$ means a cleaner image.
""")

    if st.session_state.img_raw is None:
        st.warning("⚠️ Please load a clean image first on the **Load & Visualise** page.")
        st.stop()

    clean_img = st.session_state.img_clean if st.session_state.img_clean is not None else st.session_state.img_raw

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌫️ Speckle Settings")

    level = st.sidebar.selectbox(
        "Noise level",
        ["Very noisy", "Moderate", "Mild", "Custom"],
        index=1
    )
    preset_looks = {
        "Very noisy": 1.0,
        "Moderate": 3.0,
        "Mild": 8.0,
    }
    if level == "Custom":
        looks = st.sidebar.slider("Looks (L)", 0.5, 20.0, float(st.session_state.speckle_looks), 0.5)
    else:
        looks = preset_looks[level]
        st.sidebar.metric("Looks (L)", f"{looks:.1f}")

    st.sidebar.caption("Lower looks = stronger Gamma speckle.")

    if st.sidebar.button("🎲 Generate Speckle Noise", use_container_width=True):
        noisy_img, noise_map = add_speckle_noise(clean_img, looks=looks)
        st.session_state.img_noisy = noisy_img
        st.session_state.speckle_looks = float(looks)
        st.session_state.denoise_method = None
        st.success(f"✅ Speckle noise generated with L={looks:.2f}")

    current_noisy = st.session_state.img_noisy
    if current_noisy is None or abs(float(st.session_state.speckle_looks) - float(looks)) > 1e-6:
        preview_noisy, _ = add_speckle_noise(clean_img, looks=looks)
    else:
        preview_noisy = current_noisy

    log, lo_pct, hi_pct = sidebar_display_controls("speckle")

    st.markdown("### Visual Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clean")
        st.image(display_img(clean_img, lo_pct, hi_pct, log), use_container_width=True, clamp=True)
    with col2:
        st.subheader(f"Noisy (L={looks:.2f})")
        st.image(display_img(preview_noisy, lo_pct, hi_pct, log), use_container_width=True, clamp=True)

    st.markdown("### Statistics")
    import pandas as pd
    stats_df = pd.DataFrame({
        "Clean": compute_image_stats(clean_img),
        f"Noisy (L={looks:.2f})": compute_image_stats(preview_noisy),
    })
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("### Download")
    clean_buf = io.BytesIO()
    clean_buf.write(clean_img.astype(np.float32).tobytes())
    clean_buf.seek(0)
    noisy_buf = io.BytesIO()
    noisy_buf.write(preview_noisy.astype(np.float32).tobytes())
    noisy_buf.seek(0)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Download Clean .img",
            data=clean_buf.getvalue(),
            file_name="clean_sar.img",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "⬇️ Download Noisy .img",
            data=noisy_buf.getvalue(),
            file_name=f"speckle_noisy_L{looks:.2f}.img",
            mime="application/octet-stream",
            use_container_width=True,
        )

    st.markdown("---")
    if st.button("Use Noisy Image for Denoising", use_container_width=True):
        st.session_state.img_raw = preview_noisy
        st.session_state.img_denoised = None
        st.session_state.img_enhanced = None
        st.session_state.denoise_method = None
        st.success("Noisy image is now the active image for denoising.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: DENOISE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧹 Denoise":

    st.markdown("## 🧹 Speckle Denoising")

    if st.session_state.img_raw is None:
        st.warning("⚠️ Please load an image first on the **Load & Visualise** page.")
        st.stop()

    img = st.session_state.img_raw
    ref_img = st.session_state.img_clean if st.session_state.get("img_clean") is not None else st.session_state.img_raw

    # Sidebar algorithm selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Algorithm")

    method = st.sidebar.selectbox("Select Denoising Method", [
        "Gaussian", "Median", "BLPF", "Wiener",
        "Lee", "Enhanced Lee", "Frost",
        "Bilateral", "NLM", "VPDE", "HPDE"
    ])

    # Per-algorithm parameters
    params = {}
    st.sidebar.markdown("**Parameters:**")

    if method == "Gaussian":
        params["sigma"] = st.sidebar.slider("Sigma (σ)", 0.3, 5.0, 1.0, 0.1)
        algo_info = "Gaussian Kernel: G(x,y) = (1/2πσ²)·exp(-(x²+y²)/(2σ²))"
    elif method == "Median":
        params["size"] = st.sidebar.slider("Window Size", 3, 15, 5, 2)
        algo_info = "Median: Ŝ = median{ I(x+i,y+j) } for (i,j) in window W"
    elif method == "BLPF":
        params["D0"] = st.sidebar.slider("Cutoff Freq D₀", 5, 200, 30, 5)
        params["order"] = st.sidebar.slider("Order n", 1, 5, 2, 1)
        algo_info = "Butterworth LPF: H(u,v) = 1/(1+(D/D₀)^2n)"
    elif method == "Wiener":
        params["size"] = st.sidebar.slider("Window Size", 3, 15, 5, 2)
        algo_info = "Wiener: Ŝ = μ + (σ²_l/(σ²_l+σ²_n))·(I-μ)"
    elif method == "Lee":
        params["size"] = st.sidebar.slider("Window Size", 3, 15, 7, 2)
        params["looks"] = st.sidebar.slider("Number of Looks (L)", 0.5, 10.0, 1.0, 0.5)
        algo_info = "Lee: W = σ²/(σ²+μ²/L),  Ŝ = μ + W·(I-μ)"
    elif method == "Enhanced Lee":
        params["window_size"] = st.sidebar.slider("Window Size", 3, 15, 5, 2)
        params["looks"] = st.sidebar.slider("Looks (L)", 0.5, 10.0, 3.7, 0.1)
        params["cu"] = st.sidebar.slider("Max CV threshold C_u", 0.5, 3.0, 1.0, 0.1)
        algo_info = "Enhanced Lee: 3-regime based on CV vs CV_n and C_u"
    elif method == "Frost":
        params["window_size"] = st.sidebar.slider("Window Size", 3, 15, 5, 2)
        params["damping"] = st.sidebar.slider("Damping Factor", 0.5, 10.0, 2.0, 0.5)
        algo_info = "Frost: w(d)=exp(-α·CV²·d),  α=damping·CV²"
    elif method == "Bilateral":
        params["sigma_s"] = st.sidebar.slider("Spatial σ_s", 0.5, 15.0, 3.0, 0.5)
        params["sigma_r"] = st.sidebar.slider("Range σ_r", 0.01, 1.0, 0.15, 0.01)
        algo_info = "Log-bilateral: applies bilateral in log(I) domain, back-transforms"
    elif method == "NLM":
        nlm_mode = st.sidebar.radio("h selection", ("h_cv", "sigma_est"), index=0)
        params["use_h_cv"] = nlm_mode == "h_cv"
        params["alpha"] = st.sidebar.slider("α × σ_est", 0.0, 2.0, 0.55, 0.05)
        params["patch_size"] = st.sidebar.slider("Patch Size", 3, 11, 5, 2)
        params["patch_distance"] = st.sidebar.slider("Search Distance", 3, 15, 6, 1)
        algo_info = "NLM: choose h_cv or sigma_est (scaled by α), then denoise with w∝exp(-||Px-Py||²/h²)"
    elif method == "VPDE":
        params["lam"] = st.sidebar.slider("λ (diffusion weight)", 0.5, 5.0, 2.0, 0.1)
        params["rho"] = st.sidebar.slider("ρ (fidelity)", 0.1, 1.5, 0.8, 0.05)
        params["alpha"] = st.sidebar.slider("α (gamma scale)", 0.5, 3.0, 1.3, 0.1)
        params["max_iter"] = st.sidebar.slider("Iterations", 1, 20, 5, 1)
        algo_info = "VPDE: ∂u/∂t=λ·∇·(ξ∇u)+ρ·(u₀-u), ξ=ζγ/(β·ln³(s+γ)+δ)"
    elif method == "HPDE":
        params["lam"] = st.sidebar.slider("λ", 0.5, 5.0, 2.4, 0.1)
        params["gamma_h"] = st.sidebar.slider("γ (wave speed)", 0.5, 3.0, 1.5, 0.1)
        params["alpha"] = st.sidebar.slider("α (diffusion)", 0.5, 3.0, 1.8, 0.1)
        params["N"] = st.sidebar.slider("Iterations", 1, 10, 3, 1)
        algo_info = "HPDE: u^{n+1}=(2-γ²Δt)u^n-(1-γ²Δt/2)u^{n-1}+Δt²(α·Δu-ζ(u-u₀))"

    # Display controls
    log, lo_pct, hi_pct = sidebar_display_controls("denoise")

    # Histogram matching option
    st.sidebar.markdown("---")
    apply_hist_match = st.sidebar.toggle("Apply Histogram Matching", value=False)

    # Info box
    st.markdown(f'<div class="info-box">📐 <b>{method}:</b> {algo_info}</div>', unsafe_allow_html=True)

    run_btn = st.button(f"🚀 Run {method} Denoising", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running {method}... (may take a moment for large images)"):
            try:
                fn_map = {
                    "Gaussian": lambda: gaussian_denoise(img, **params),
                    "Median": lambda: median_denoise(img, **params),
                    "BLPF": lambda: blpf_filter(img, **params),
                    "Wiener": lambda: wiener_denoise(img, **params),
                    "Lee": lambda: lee_filter(img, **params),
                    "Enhanced Lee": lambda: enhanced_lee_filter(img, **params),
                    "Frost": lambda: frost_filter(img, **params),
                    "Bilateral": lambda: bilateral_denoise(img, **params),
                    "NLM": lambda: nlm_denoise(img, **params),
                    "VPDE": lambda: vpde_denoise(img, **params),
                    "HPDE": lambda: hpde_denoise(img, **params),
                }
                denoised = fn_map[method]()
                if apply_hist_match:
                    denoised = match_histograms(denoised, img)
                st.session_state.img_denoised = denoised
                st.session_state.denoise_method = method
                st.session_state.img_enhanced = None
                st.success(f"✅ {method} complete!")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Display results
    if st.session_state.img_denoised is not None:
        denoised = st.session_state.img_denoised

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original (Reference)")
            u8_orig = display_img(ref_img, lo_pct, hi_pct, log)
            st.image(u8_orig, caption="Original SAR (clean)", use_container_width=True, clamp=True)
        with col2:
            st.subheader(f"Denoised ({st.session_state.denoise_method})")
            u8_den = display_img(denoised, lo_pct, hi_pct, log)
            st.image(u8_den, caption=f"After {st.session_state.denoise_method}", use_container_width=True, clamp=True)

        # Quick metrics
        st.markdown("### Quick Metrics")
        m = compute_metrics(ref_img, denoised)
        cols = st.columns(4)
        cols[0].metric("ENL", f"{m['ENL']:.2f}", delta=f"{m['ENL'] - (ref_img.mean()**2/(ref_img.var()+1e-10)):.2f}")
        cols[1].metric("CV", f"{m['CV']:.4f}")
        cols[2].metric("SSIM", f"{m['SSIM']:.4f}")
        cols[3].metric("PSNR", f"{m['PSNR (dB)']:.2f} dB")

        # Download
        st.markdown("### 💾 Download Denoised Image")
        buf = io.BytesIO()  # Ensure buf is a BytesIO object
        buf.write(denoised.astype(np.float32).tobytes())  # Write the bytes directly to the buffer
        buf.seek(0)  # Reset the buffer position to the beginning before reading its content
        st.download_button(
            "⬇️ Download .img (float32 binary)",
            data=buf.getvalue(),
            file_name=f"denoised_{st.session_state.denoise_method}.img",
            mime="application/octet-stream"
        )
        # Save as PNG for quick preview
        u8_save = display_img(denoised, 1.0, 99.0, False)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(u8_save, mode='L')
        png_buf = io.BytesIO()
        pil_img.save(png_buf, format="PNG")
        st.download_button(
            "⬇️ Download .png (display)",
            data=png_buf.getvalue(),
            file_name=f"denoised_{st.session_state.denoise_method}.png",
            mime="image/png"
        )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: TEXTURE ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎨 Texture Enhancement":

    st.markdown("## 🎨 Texture Enhancement")

    if st.session_state.img_raw is None:
        st.warning("⚠️ Please load an image first.")
        st.stop()

    img = st.session_state.img_raw
    denoised = st.session_state.img_denoised

    if denoised is None:
        st.info("ℹ️ No denoised image found. You can apply texture enhancement directly on the original, or go to Denoise first.")
        base = img
        base_label = "Original"
    else:
        use_den = st.radio("Apply enhancement on:", ["Denoised Image", "Original Image"], horizontal=True)
        base = denoised if use_den == "Denoised Image" else img
        base_label = use_den

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Enhancement Method")

    enh_method = st.sidebar.selectbox("Method", [
        "Sobel", "Canny",
        "Structure Tensor",
        "High-Frequency Injection"
    ])

    if enh_method == "Sobel":
        amount = st.sidebar.slider("Injection Strength α", 0.0, 2.0, 0.75, 0.05)
        enh_params = {"amount": amount}
        info = "Sobel: inject gradient magnitude from original back into smoothed image"
    elif enh_method == "Canny":
        sigma = st.sidebar.slider("Canny Sigma", 0.5, 5.0, 1.0, 0.1)
        amount = st.sidebar.slider("Injection Strength α", 0.0, 2.0, 0.5, 0.05)
        enh_params = {"sigma": sigma, "amount": amount}
        info = "Canny: precise thin edge map injected — preserves linear features"
    elif enh_method == "Structure Tensor":
        sigma = st.sidebar.slider("Tensor Smoothing σ", 0.5, 10.0, 2.0, 0.5)
        alpha_inj = st.sidebar.slider("Injection Strength α", 0.0, 1.0, 0.3, 0.05)
        enh_params = {"sigma": sigma, "alpha_inj": alpha_inj}
        info = "Structure Tensor: coherence-weighted HF injection — preserves texture orientation"
    elif enh_method == "High-Frequency Injection":
        sigma = st.sidebar.slider("Smoothing σ (HF scale)", 1.0, 30.0, 10.0, 0.5)
        alpha_inj = st.sidebar.slider("Injection Strength α", 0.0, 3.0, 1.0, 0.1)
        grad_power = st.sidebar.slider("Gradient Power p", 0.05, 1.0, 0.15, 0.05)
        enh_params = {"sigma": sigma, "alpha_inj": alpha_inj, "grad_power": grad_power}
        info = "HF Injection: log-domain high-freq layer, gradient-weighted, injected into denoised"

    st.sidebar.markdown("---")
    apply_hist_match = st.sidebar.toggle("Histogram Matching after enhancement", value=False)
    log, lo_pct, hi_pct = sidebar_display_controls("enh")

    st.markdown(f'<div class="info-box">📐 <b>{enh_method}:</b> {info}</div>', unsafe_allow_html=True)

    if st.button("🚀 Apply Texture Enhancement", type="primary", use_container_width=True):
        with st.spinner(f"Applying {enh_method}..."):
            try:
                fn_map = {
                    "Sobel": lambda: sobel_enhancement(img, base, **enh_params),
                    "Canny": lambda: canny_enhancement(img, base, **enh_params),
                    "Structure Tensor": lambda: structure_tensor_enhancement(img, base, **enh_params),
                    "High-Frequency Injection": lambda: hf_injection(img, base, **enh_params),
                }
                enhanced = fn_map[enh_method]()
                if apply_hist_match:
                    enhanced = match_histograms(enhanced, img)
                st.session_state.img_enhanced = enhanced
                st.success("✅ Enhancement applied!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.img_enhanced is not None:
        enhanced = st.session_state.img_enhanced

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original")
            st.image(display_img(img, lo_pct, hi_pct, log),
                     caption="Original", use_container_width=True, clamp=True)
        with col2:
            st.subheader(base_label)
            st.image(display_img(base, lo_pct, hi_pct, log),
                     caption=base_label, use_container_width=True, clamp=True)
        with col3:
            st.subheader(f"Enhanced ({enh_method})")
            st.image(display_img(enhanced, lo_pct, hi_pct, log),
                     caption=f"Enhanced", use_container_width=True, clamp=True)

        # Image Comparison Slider
        st.markdown("---")
        st.markdown("### 🖼️ Compare Original vs Enhanced (Final) Slider")
        try:
            original_display = display_img(img, lo_pct, hi_pct, log)
            enhanced_display = display_img(enhanced, lo_pct, hi_pct, log)
            image_comparison(
                img1=original_display,
                img2=enhanced_display,
                label1="Original",
                label2="Enhanced",
                width=900
            )
        except Exception as e:
            st.error(f"Error displaying comparison slider: {e}")

        # Histogram matching standalone option
        st.markdown("---")
        st.subheader("🔄 Histogram Matching (standalone)")
        if st.button("Apply Histogram Match to Enhanced → Original", use_container_width=True):
            matched = match_histograms(enhanced, img)
            st.session_state.img_enhanced = matched
            st.success("Histogram matching applied!")
            st.image(display_img(matched, lo_pct, hi_pct, log),
                     caption="Histogram Matched", use_container_width=True, clamp=True)

        # Download
        buf = io.BytesIO()
        buf.write(enhanced.astype(np.float32).tobytes())
        buf.seek(0)
        st.download_button(
            "⬇️ Download Enhanced (.img float32)",
            data=buf.getvalue(),
            file_name=f"enhanced_{enh_method.replace(' ', '_')}.img",
            mime="application/octet-stream"
        )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: COMPARE & EVALUATE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Compare & Evaluate":

    st.markdown("## 📊 Comparative Evaluation")

    if st.session_state.img_raw is None:
        st.warning("⚠️ Please load an image first.")
        st.stop()

    ref_img = st.session_state.img_clean if st.session_state.get("img_clean") is not None else st.session_state.img_raw
    img_raw = st.session_state.img_raw
    denoised = st.session_state.img_denoised
    enhanced = st.session_state.img_enhanced

    st.sidebar.markdown("---")
    st.sidebar.subheader("🖼️ Images to Compare")

    # Determine available images
    available = {"Original (Ref)": ref_img}
    if img_raw is not None and not np.array_equal(img_raw, ref_img):
        available["Input (Noisy)"] = img_raw
    if denoised is not None:
        available[f"Denoised ({st.session_state.denoise_method})"] = denoised
    if enhanced is not None:
        available["Enhanced"] = enhanced

    selected_names = st.sidebar.multiselect(
        "Select images to compare (2–4)",
        list(available.keys()),
        default=list(available.keys())[:min(4, len(available))]
    )

    log, lo_pct, hi_pct = sidebar_display_controls("compare")

    # Separate clip settings per image
    st.sidebar.markdown("---")
    per_image_clip = st.sidebar.toggle("Independent clipping per image", value=False)

    if len(selected_names) < 2:
        st.info("Select at least 2 images to compare.")
        st.stop()

    selected_imgs = {n: available[n] for n in selected_names}

    # ── 4-panel comparison ──
    st.markdown("### 🖼️ Visual Comparison")

    n = len(selected_names)
    cols = st.columns(min(n, 4))

    clip_settings = {}
    for i, name in enumerate(selected_names):
        with cols[i % 4]:
            if per_image_clip:
                lop = st.slider(f"Lo% ({name[:8]})", 0.0, 50.0, 1.0, 0.5,
                                key=f"lo_{i}")
                hip = st.slider(f"Hi% ({name[:8]})", 50.0, 100.0, 99.0, 0.5,
                                key=f"hi_{i}")
                clip_settings[name] = (lop, hip)
            else:
                clip_settings[name] = (lo_pct, hi_pct)

    cols2 = st.columns(min(n, 4))
    for i, name in enumerate(selected_names):
        lo_c, hi_c = clip_settings[name]
        with cols2[i % 4]:
            u8 = display_img(selected_imgs[name], lo_c, hi_c, log)
            st.image(u8, caption=name, use_container_width=True, clamp=True)

    # ── Metrics comparison table ──
    st.markdown("---")
    st.markdown("### 📋 Quantitative Metrics Comparison")

    # NESZ patch selection
    with st.expander("⚙️ NESZ Patch Configuration"):
        st.markdown("Define homogeneous patches for NESZ computation `[[y0,y1],[x0,x1]]`")
        use_auto_nesz = st.checkbox("Auto-detect from corners", value=True)
        nesz_patches = None
        if not use_auto_nesz:
            ps = st.number_input("Patch size", 16, 256, 64)
            H_img, W_img = ref_img.shape
            nesz_patches = [
                [[0, ps], [0, ps]],
                [[0, ps], [W_img - ps, W_img]],
                [[H_img - ps, H_img], [0, ps]],
                [[H_img - ps, H_img], [W_img - ps, W_img]]
            ]

    with st.spinner("Computing metrics..."):
        results = {}
        for name, im in selected_imgs.items():
            results[name] = compute_metrics(ref_img, im, nesz_patches)

    # Display as table
    import pandas as pd
    df = pd.DataFrame(results).T
    df = df.round(4)

    # Color the best values
    st.dataframe(df.style.highlight_max(
        subset=["ENL", "SNR (dB)", "PSNR (dB)", "SSIM", "EPI", "ESI"],
        color="#d4edda"
    ).highlight_min(
        subset=["CV", "NESZ (dB)"],
        color="#d4edda"
    ), use_container_width=True)

    st.markdown("""
<div class="info-box">
🟢 Green = best value for that metric &nbsp;|&nbsp; 
Higher ENL, SNR, PSNR, SSIM, EPI, ESI = better &nbsp;|&nbsp;
Lower CV, NESZ = better speckle suppression
</div>
""", unsafe_allow_html=True)

    # ── Radar/bar comparison charts ──
    st.markdown("### 📊 Metric Charts")

    chart_metrics = ["ENL", "CV", "SNR (dB)", "SSIM", "EPI", "ESI"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, len(selected_names)))

    for ax_i, metric in enumerate(chart_metrics):
        vals = [results[n][metric] for n in selected_names]
        bars = axes[ax_i].bar(range(len(selected_names)), vals,
                              color=colors, edgecolor='white', linewidth=1.5)
        axes[ax_i].set_xticks(range(len(selected_names)))
        axes[ax_i].set_xticklabels([n[:12] for n in selected_names],
                                   rotation=20, ha='right', fontsize=9)
        axes[ax_i].set_title(metric, fontweight='bold')
        axes[ax_i].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            axes[ax_i].text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() * 1.01, f"{val:.3f}",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Difference maps ──
    if len(selected_names) >= 2:
        st.markdown("---")
        st.markdown("### 🗺️ Difference Maps (vs Original)")
        diff_names = [n for n in selected_names if n not in ["Original (Ref)", "Input (Noisy)"]]
        if diff_names:
            dcols = st.columns(len(diff_names))
            for i, name in enumerate(diff_names):
                diff = np.abs(ref_img - selected_imgs[name])
                with dcols[i]:
                    fig_d, ax_d = plt.subplots(figsize=(4, 4))
                    im_d = ax_d.imshow(diff, cmap='hot', aspect='auto')
                    plt.colorbar(im_d, ax=ax_d, fraction=0.046)
                    ax_d.set_title(f"|Ref - {name[:14]}|", fontsize=9)
                    ax_d.axis('off')
                    st.pyplot(fig_d)
                    plt.close(fig_d)

    # ── CSV Export ──
    st.markdown("---")
    csv_buf = df.to_csv().encode()
    st.download_button(
        "⬇️ Download Metrics CSV",
        data=csv_buf,
        file_name="sar_metrics_comparison.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
🛰️ <b>SAR Denoising Prototype</b><br>
Jani Pruthak Maulik<br>
SAC-ISRO Internship 2025-26<br>
L.D. College of Engineering, GTU
</small>
""", unsafe_allow_html=True)
