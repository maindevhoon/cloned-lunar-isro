"""
colab_runner.py
One-file entry point for the reconstructed Lunar 2024 ISRO project.

Best usage after cloning from GitHub (recommended):

    !git clone https://github.com/maindevhoon/cloned-lunar-isro.git
    %cd cloned-lunar-isro
    !pip install -q -r requirements.txt

    import colab_runner
    colab_runner.run_everything()

This runs standalone demos for:
- ROI identification (hyperspectral)
- Crater / obstacle detection
- RL + simple physics path planning
- Multi-rover path formation

The main.py files contain richer versions if you want to %run them individually.
"""

# =============================================================================
# QUICK INSTALL (run this first in Colab)
# =============================================================================
INSTALL_CMD = """
!pip install -q gymnasium torch numpy matplotlib opencv-python-headless scikit-learn scipy pillow
"""

print("Recommended Colab install:")
print(INSTALL_CMD)

# =============================================================================
# HOW TO USE IN COLAB (easiest)
# =============================================================================
"""
BEST WAY IN COLAB:

1. Upload the four main.py files + this colab_runner.py to your Colab environment
   (or copy the content of each main.py into separate cells if preferred).

2. Run the cell with the pip install above.

3. Then:

   %run roi-identification/main.py
   %run obstacle-detection/main.py
   %run path-feedback-rl/main.py
   %run path-formation/main.py
   %run colab_runner.py

   run_everything()

Alternative: paste the important functions you need from each main.py directly.
"""

# Standalone minimal demos so it works even without proper package structure

def _run_roi_standalone():
    print("\n[ROI] Running standalone hyperspectral demo...")
    # (light version of the roi code)
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    h, w, b = 400, 80, 64
    cube = np.random.rand(h, w, b) * 0.3 + 0.15
    flat = cube.reshape(-1, b)
    reduced = PCA(5).fit_transform(StandardScaler().fit_transform(flat))
    labels = KMeans(4, n_init=10, random_state=42).fit_predict(reduced)
    plt.imshow(labels.reshape(h, w), cmap='tab10'); plt.title("Synthetic ROI clusters"); plt.show()
    print("ROI standalone done.")

def _run_crater_standalone():
    print("\n[CRATERS] Running standalone crater detector...")
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    img = (np.random.rand(400,400)*80 + 140).astype(np.uint8)
    for _ in range(12):
        cx,cy,r = np.random.randint(30,370,3)
        cv2.circle(img, (cx,cy), r, 30, -1)
        cv2.circle(img, (cx,cy), r+3, 210, 2)
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    _, th = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        a = cv2.contourArea(c)
        if 80 < a < 3000:
            x,y,ww,hh = cv2.boundingRect(c)
            dets.append(((x,y,ww,hh), 0.8))
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x,y,w,h),c in dets:
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
    plt.imshow(vis[:,:,::-1]); plt.title(f"Detected ~{len(dets)} craters"); plt.show()
    print("Crater standalone done.")

def _run_rl_standalone():
    print("\n[RL + PHYSICS] Running lightweight RL + physics demo...")
    import numpy as np
    import matplotlib.pyplot as plt
    # tiny physics sim
    pos = np.array([5., 5.])
    vel = np.array([0.3, 0.1])
    path = [pos.copy()]
    for _ in range(80):
        action = np.random.randint(0,5)
        ax = ay = 0
        if action==1: ay=0.4
        elif action==2: ay=-0.4
        elif action==3: ax=-0.4
        elif action==4: ax=0.4
        vel = (vel + [ax,ay]) * 0.9
        pos = np.clip(pos + vel, [0,0], [90,25])
        path.append(pos.copy())
    p = np.array(path)
    plt.plot(p[:,0], p[:,1]); plt.scatter([p[0,0]],[p[0,1]],c='g'); plt.scatter([p[-1,0]],[p[-1,1]],c='r',marker='*')
    plt.title("Simple physics path (demo)"); plt.show()
    print("RL + Physics standalone done.")

def _run_formation_standalone():
    print("\n[FORMATION] Running multi-rover formation demo...")
    import numpy as np
    import matplotlib.pyplot as plt
    base = np.linspace([5,5], [95,30], 30)
    for off in [0, 4, -4, 8]:
        f = base + np.array([0, off])
        plt.plot(f[:,0], f[:,1], alpha=0.8)
    plt.title("Simple formation paths"); plt.show()
    print("Formation standalone done.")

def run_everything():
    print("\n" + "="*55)
    print("LUNAR 2024 - COLAB RUNNER (standalone mode)")
    print("="*55)
    _run_roi_standalone()
    _run_crater_standalone()
    _run_rl_standalone()
    _run_formation_standalone()
    print("\nAll done! Upload the full main.py files for the richer versions.")
    print("For real physics → NVIDIA Isaac Sim")


def run_everything():
    print("\n" + "="*55)
    print("LUNAR 2024 - COLAB RUNNER (standalone mode)")
    print("="*55)
    _run_roi_standalone()
    _run_crater_standalone()
    _run_rl_standalone()
    _run_formation_standalone()
    print("\nAll done! For richer versions, you can also %run the individual main.py files.")
    print("For real physics, wheel-terrain interaction, sensors etc. → use NVIDIA Isaac Sim")


if __name__ == "__main__":
    run_everything()
