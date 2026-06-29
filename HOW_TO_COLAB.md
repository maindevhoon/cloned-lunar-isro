# How to Upload the Entire Lunar Project to Google Colab

This guide shows the fastest and cleanest ways to get the **full project** (all `main.py` files + notebooks + analysis) running in Google Colab.

## Recommended Method: Zip + Upload (Easiest for "Entire Thing")

### Step 1: On Your Mac (Local Machine)

Open **Terminal** and run these commands:

```bash
# Go to the parent folder
cd /Users/dev/Documents/Development/lunar-2024-isro

# Create a clean zip of the entire project
zip -r lunar-2024-project.zip lunar
```

This will create `lunar-2024-project.zip` containing everything with the correct folder structure.

### Step 2: Upload to Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a **New Notebook**.
2. In the left sidebar, click the **Files** icon (folder icon).
3. Click the **Upload** button (or drag & drop).
4. Select `lunar-2024-project.zip`.

### Step 3: Extract and Set Up in Colab

Run these cells in order:

```python
# 1. Unzip the project
!unzip -q lunar-2024-project.zip

# 2. Move into the project folder
%cd lunar

# 3. Install dependencies (this covers everything you need)
!pip install -q -r requirements.txt
```

### Step 4: Run the Code

```python
# Run everything with one command (recommended)
import colab_runner
colab_runner.run_everything()
```

Or run individual parts:

```python
# RL + Physics (includes the "realistic physics" demo)
%run path-feedback-rl/main.py
run_all_demos()

# Crater detection
%run obstacle-detection/main.py
run_crater_demo()

# Hyperspectral ROI
%run roi-identification/main.py
run_roi_demo()

# Multi-rover formation
%run path-formation/main.py
run_formation_demo()
```

---

## Alternative: Upload via Google Drive (Good if you have big datasets)

If you have the original large datasets (crater images, IIRS .qub files) on Drive:

1. Upload the **unzipped** `lunar` folder to your Google Drive.
2. In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

# Then copy or cd into it
%cd /content/drive/MyDrive/lunar          # adjust path as needed
```

---

## Alternative: GitHub (Best for repeated use / team)

1. Push this folder to a GitHub repo.
2. In Colab, just do:

```python
!git clone https://github.com/YOUR_USERNAME/lunar-2024-isro.git
%cd lunar-2024-isro/lunar
!pip install -q -r requirements.txt
```

---

## Opening the Original Notebooks

- You can also upload the `.ipynb` files directly:
  - `obstacle-detection/crater.ipynb`
  - `path-feedback-rl/rl.ipynb`
  - `roi-identification/hyperspectral.ipynb`

- Or open them from the unzipped folder using **File → Open notebook**.

**Warning**: The original notebooks expect large data on Google Drive (the crater dataset + IIRS files). The new `main.py` files use **synthetic data** so they run immediately.

---

## Quick One-Cell Starter (Copy-Paste)

Paste this into a fresh Colab notebook:

```python
# === LUNAR 2024 → COLAB STARTER ===
!unzip -q lunar-2024-project.zip   # only if you uploaded the zip
%cd lunar
!pip install -q -r requirements.txt

import colab_runner
colab_runner.run_everything()
```

---

## Notes

- The `main.py` files are self-contained and use synthetic data for demos.
- For the **real "Isaac Sim" physics** part, you will later need to move development to a machine with NVIDIA Isaac Sim installed (it does **not** run in Colab).
- The `colab_runner.py` has fallback standalone demos so it works even with import issues.

You're ready to run the entire reconstructed project in Colab!