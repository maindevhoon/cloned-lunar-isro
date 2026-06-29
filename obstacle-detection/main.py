"""
obstacle-detection/main.py
Crater / Hazard Detection for Lunar Rovers

Reconstructed from the original crater.ipynb of the ISRO BAH 2024 winner.

Features:
- Synthetic crater image generator (runs without the Roboflow dataset)
- Simple but effective crater detector using OpenCV contours + circularity
- Hook for your trained YOLO / best.pt model
- Visualization of detected hazards

Colab usage:
    !pip install numpy matplotlib opencv-python-headless scikit-image
    from obstacle_detection.main import run_crater_demo
    run_crater_demo()
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# =============================================================================
# SYNTHETIC CRATER IMAGE GENERATOR
# =============================================================================

def generate_synthetic_crater_image(height=512, width=512, num_craters=18, seed=123):
    """
    Create a grayscale lunar surface image with circular craters of varying size.
    """
    rng = np.random.default_rng(seed)
    img = np.ones((height, width), dtype=np.float32) * 0.65

    # Add some background texture / illumination gradient
    for i in range(height):
        img[i] *= 0.85 + 0.15 * (i / height)

    # Add craters
    for _ in range(num_craters):
        cx = rng.integers(30, width - 30)
        cy = rng.integers(30, height - 30)
        r = rng.integers(8, 38)

        # Dark crater floor + bright rim
        for y in range(max(0, cy - r - 6), min(height, cy + r + 6)):
            for x in range(max(0, cx - r - 6), min(width, cx + r + 6)):
                dist = np.hypot(x - cx, y - cy)
                if dist < r:
                    # crater interior
                    img[y, x] = 0.18 + rng.normal(0, 0.03)
                elif dist < r + 4:
                    # rim
                    img[y, x] = min(0.95, img[y, x] + 0.28)

    # Add some rocks / small hazards
    for _ in range(35):
        rx = rng.integers(10, width-10)
        ry = rng.integers(10, height-10)
        rs = rng.integers(2, 5)
        cv2.circle(img, (rx, ry), rs, 0.35, -1)

    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8


# =============================================================================
# CRATER DETECTOR (Lightweight CV version)
# =============================================================================

def detect_craters_cv(image, min_area=120, max_area=4500, min_circularity=0.55):
    """
    Detect circular dark regions as craters using contour analysis.
    Returns list of (x, y, w, h, confidence)
    """
    # Preprocess
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Adaptive threshold to find dark regions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > min_circularity:
            x, y, w, h = cv2.boundingRect(cnt)
            # fake confidence based on how circular it is
            conf = min(0.98, 0.6 + (circularity - 0.55) * 1.2)
            detections.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": round(float(conf), 3),
                "area": int(area)
            })

    # Sort by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def detect_craters_yolo(image, model_path="best.pt"):
    """
    Placeholder for the original best.pt YOLO model.
    In the 2024 notebook they used a trained YOLO model.
    You can load Ultralytics YOLO here if you have the weights.
    """
    print(f"[INFO] YOLO detection not executed (model_path={model_path}).")
    print("To use: install ultralytics + put best.pt in the folder, then implement this function.")
    # Example skeleton:
    # from ultralytics import YOLO
    # model = YOLO(model_path)
    # results = model(image)
    # ... parse boxes ...
    return []


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_detections(image, detections, title="Detected Craters / Hazards"):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"{conf:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title + f"  ({len(detections)} detections)")
    plt.axis("off")
    plt.show()

    return vis


# =============================================================================
# DEMO
# =============================================================================

def run_crater_demo():
    print("=== Obstacle / Crater Detection Demo ===")
    img = generate_synthetic_crater_image(480, 480, num_craters=14)
    print(f"Generated image shape: {img.shape}")

    # CV detector
    dets = detect_craters_cv(img)
    print(f"CV detector found {len(dets)} craters")
    for d in dets[:5]:
        print("  ", d)

    visualize_detections(img, dets)

    # YOLO placeholder
    yolo_dets = detect_craters_yolo(img)
    print(f"YOLO placeholder returned {len(yolo_dets)} detections")

    print("Crater detection demo finished.")
    return dets


if __name__ == "__main__":
    run_crater_demo()
