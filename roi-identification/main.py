"""
roi-identification/main.py
Hyperspectral ROI Identification for Lunar Surface (Chandrayaan-2 style)

Original logic from the hyperspectral.ipynb of the 2024 winning entry.
This version is self-contained for Colab:

- Synthetic hyperspectral data generator (so it runs without your private .qub files)
- Real data loader (you provide the path in Colab)
- Band visualization, reflectance curves, PCA + KMeans for ROI

Run in Colab:
    !pip install numpy matplotlib scikit-learn
    from roi_identification.main import run_roi_demo
    run_roi_demo()
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# =============================================================================
# SYNTHETIC DATA (for demo / when real data is unavailable)
# =============================================================================

def generate_synthetic_hyperspectral(height=800, width=120, bands=128, seed=42):
    """
    Create fake lunar hyperspectral cube.
    Simulates different mineral/terrain signatures + some noise.
    Shape: (height, width, bands)
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((height, width, bands), dtype=np.float32)

    # Base reflectance + terrain types
    for y in range(height):
        for x in range(width):
            # Different "regions"
            if y < height * 0.3:           # Highland-like
                base = 0.25 + 0.08 * np.sin(np.linspace(0, 6, bands))
            elif y < height * 0.65:        # Mare-like
                base = 0.12 + 0.04 * np.cos(np.linspace(0, 8, bands))
            else:                          # Crater rim / special
                base = 0.18 + 0.12 * np.sin(np.linspace(1, 10, bands))

            # Add absorption features (fake water / mineral bands)
            if 30 < x < 55 and 200 < y < 380:
                base[18:26] -= 0.18   # fake absorption feature
            if 70 < x < 100 and y > 550:
                base[55:62] -= 0.15

            noise = rng.normal(0, 0.015, bands)
            data[y, x] = np.clip(base + noise, 0.01, 0.95)

    # Add some "crater" like spatial structure
    for _ in range(6):
        cy = rng.integers(50, height-50)
        cx = rng.integers(10, width-10)
        r = rng.integers(8, 22)
        for yy in range(max(0, cy-r), min(height, cy+r)):
            for xx in range(max(0, cx-r//2), min(width, cx+r//2)):
                if (yy-cy)**2 + ((xx-cx)*1.4)**2 < r**2:
                    data[yy, xx] *= 0.65   # darker in crater
    return data


# =============================================================================
# REAL DATA LOADER (Chandrayaan-2 IIRS style)
# =============================================================================

def load_real_iiirs(qub_path, hdr_path):
    """
    Load real ENVI .qub + .hdr using the spectral library.
    You must have the files (usually on Google Drive in original notebooks).
    """
    try:
        import spectral
    except ImportError:
        print("Install spectral: !pip install spectral")
        raise

    print(f"Loading real IIRS data from {qub_path}")
    data = spectral.io.envi.open(hdr_path, qub_path)
    cube = data.load()
    print(f"Loaded shape: {cube.shape}")
    return np.array(cube)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def plot_bands(cube, band_indices=(10, 40, 80, 110), title="Hyperspectral Bands"):
    fig, axes = plt.subplots(1, len(band_indices), figsize=(14, 4))
    for ax, b in zip(axes, band_indices):
        if b < cube.shape[2]:
            ax.imshow(cube[:, :, b], cmap="gray")
            ax.set_title(f"Band {b}")
            ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_reflectance(cube, y, x):
    """Plot spectrum at a pixel."""
    if y < cube.shape[0] and x < cube.shape[1]:
        spectrum = cube[y, x, :]
        plt.figure(figsize=(8, 4))
        plt.plot(spectrum)
        plt.xlabel("Band index")
        plt.ylabel("Reflectance (normalized)")
        plt.title(f"Reflectance at pixel ({x}, {y})")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Invalid coordinates")


def compute_roi(cube, n_clusters=5, n_components=8):
    """
    Simple ROI detection using PCA + KMeans on spectra.
    Returns cluster map and top ROIs.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)

    # Standardize
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)

    # PCA
    pca = PCA(n_components=min(n_components, b))
    reduced = pca.fit_transform(flat_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced)

    cluster_map = labels.reshape(h, w)
    return cluster_map, pca, kmeans


def visualize_roi(cluster_map):
    plt.figure(figsize=(10, 6))
    plt.imshow(cluster_map, cmap="tab10")
    plt.colorbar(label="Cluster / ROI label")
    plt.title("ROI Clusters from Hyperspectral Data (PCA + KMeans)")
    plt.axis("off")
    plt.show()


# =============================================================================
# MAIN DEMO (Colab friendly)
# =============================================================================

def run_roi_demo(use_real=False, qub_path=None, hdr_path=None):
    print("=== ROI Identification Demo (Lunar Hyperspectral) ===")

    if use_real and qub_path and hdr_path and os.path.exists(qub_path):
        cube = load_real_iiirs(qub_path, hdr_path)
    else:
        print("Using synthetic hyperspectral cube (fast demo)")
        cube = generate_synthetic_hyperspectral(height=600, width=110, bands=96)

    print(f"Cube shape: {cube.shape}")

    # Show some bands
    plot_bands(cube, band_indices=[5, 25, 50, 75])

    # Example reflectance
    plot_reflectance(cube, y=cube.shape[0]//2, x=cube.shape[1]//3)

    # ROI clustering
    print("Running PCA + KMeans for ROIs...")
    cluster_map, pca, kmeans = compute_roi(cube, n_clusters=5, n_components=6)
    print(f"Explained variance (first 6 PCs): {pca.explained_variance_ratio_[:6]}")
    visualize_roi(cluster_map)

    # Simple "interesting" ROI = cluster with unusual mean spectrum
    unique, counts = np.unique(cluster_map, return_counts=True)
    print("Cluster sizes:", dict(zip(unique, counts)))

    print("ROI demo finished.")
    return cluster_map, cube


if __name__ == "__main__":
    run_roi_demo()
