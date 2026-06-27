"""
laptop_detector_v2.py
=====================
Improved detection using three physical signals instead of just brightness:
  1. Edge density   — crumpled paper has many crease edges, cans are smooth
  2. Texture variance — paper creases = high local variance, metal = uniform
  3. Specular highlights — metal has bright glare spots, paper does not

Each signal votes CAN or PAPER, results are combined into a weighted score.
Much more robust to lighting changes and deformed/crumpled objects.

Requirements:
    pip install opencv-python pyserial numpy

Usage:
    python laptop_detector_v2.py

Setup:
    - Update SERIAL_PORT to match your Pico's COM port
    - Run calibrate_v2.py first to tune the thresholds for your lighting
"""

import cv2
import numpy as np
import serial
import time

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SERIAL_PORT = "COM3"       # Change to your Pico's port
BAUD_RATE   = 9600
CAMERA_ID   = 0

# Detection zone (x, y, width, height) — run calibrate_v2.py to tune these
ZONE_X, ZONE_Y, ZONE_W, ZONE_H = 200, 150, 240, 180

# Minimum pixel area inside zone that counts as "object present"
MIN_OBJECT_AREA = 3000

# ── Thresholds (tune with calibrate_v2.py) ────────────────────────────────────
# Edge density: ratio of edge pixels to total pixels (0.0 – 1.0)
# Paper (crumpled) has many creases → high edge ratio (~0.15–0.35)
# Can (even crushed) has smooth surfaces → low edge ratio (~0.03–0.12)
EDGE_DENSITY_THRESHOLD = 0.13   # above = paper, below = can

# Texture variance: mean of local std deviation across small patches
# Paper creases = high variance (~30–80), metal uniform = low (~5–25)
TEXTURE_VARIANCE_THRESHOLD = 28  # above = paper, below = can

# Specular highlight ratio: fraction of pixels that are very bright (near-white)
# Metal glare spots push this high (~0.04–0.15), paper rarely exceeds ~0.02
SPECULAR_THRESHOLD = 0.035       # above = can, below = paper

# Confirmation: how many consecutive agreeing frames before sending command
CONFIRM_FRAMES = 8

# Cooldown between sends (seconds)
COOLDOWN_SECONDS = 2.5
# ───────────────────────────────────────────────────────────────────────────────


def is_object_present(roi_grey):
    """Check if something meaningful is in the detection zone."""
    _, thresh = cv2.threshold(roi_grey, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(cv2.contourArea(c) for c in contours) >= MIN_OBJECT_AREA


def edge_density_score(roi_grey):
    """
    Run Canny edge detection and return the fraction of pixels that are edges.
    Crumpled paper → many crease lines → HIGH score → PAPER
    Can surface    → smooth/uniform   → LOW score  → CAN
    """
    blurred = cv2.GaussianBlur(roi_grey, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    density = np.count_nonzero(edges) / edges.size
    vote    = 'P' if density > EDGE_DENSITY_THRESHOLD else 'C'
    return density, vote


def texture_variance_score(roi_grey):
    """
    Divide the ROI into small patches and measure standard deviation in each.
    Average them — high mean std = rough/creased = PAPER, low = smooth = CAN.
    """
    patch = 8  # patch size in pixels
    stds  = []
    for y in range(0, roi_grey.shape[0] - patch, patch):
        for x in range(0, roi_grey.shape[1] - patch, patch):
            tile = roi_grey[y:y+patch, x:x+patch]
            stds.append(np.std(tile))
    mean_std = np.mean(stds) if stds else 0
    vote     = 'P' if mean_std > TEXTURE_VARIANCE_THRESHOLD else 'C'
    return mean_std, vote

def specular_score(roi_grey):
    """
    Count the fraction of pixels with very high brightness (specular highlights).
    Metal reflects light as a small number of very bright spots.
    Paper diffuses light — no extreme highlights.
    High specular fraction → CAN, low → PAPER.
    """
    bright_pixels = np.count_nonzero(roi_grey > 230)
    ratio         = bright_pixels / roi_grey.size
    vote          = 'C' if ratio > SPECULAR_THRESHOLD else 'P'
    return ratio, vote

def classify(roi):
    """
    Combine the three signals into a weighted majority vote.
    Returns ('C' or 'P', debug_dict) or (None, debug_dict) if nothing present.
    """
    if roi is None or roi.size == 0:
        return None, {}

    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if not is_object_present(grey):
        return None, {}

    edge_val,     edge_vote     = edge_density_score(grey)
    texture_val,  texture_vote  = texture_variance_score(grey)
    specular_val, specular_vote = specular_score(grey)

    # Weighted vote — edge density is the most reliable signal
    weights = {'edge': 2, 'texture': 2, 'specular': 1}
    can_score   = (weights['edge']     if edge_vote     == 'C' else 0) + \
                  (weights['texture']  if texture_vote  == 'C' else 0) + \
                  (weights['specular'] if specular_vote == 'C' else 0)
    paper_score = (weights['edge']     if edge_vote     == 'P' else 0) + \
                  (weights['texture']  if texture_vote  == 'P' else 0) + \
                  (weights['specular'] if specular_vote == 'P' else 0)

    result = 'C' if can_score > paper_score else 'P'

    debug = {
        'edge':     (edge_val,     edge_vote),
        'texture':  (texture_val,  texture_vote),
        'specular': (specular_val, specular_vote),
        'can_score':   can_score,
        'paper_score': paper_score,
    }
    return result, debug


def draw_debug(frame, debug, zone):
    """Render signal values and votes onto the frame."""
    zx, zy, zw, zh = zone
    if not debug:
        return

    y = zy + zh + 22
    lines = [
        (f"Edges:    {debug['edge'][0]:.3f}  ({debug['edge'][1]})",
         (0, 180, 255) if debug['edge'][1] == 'C' else (0, 200, 80)),
        (f"Texture:  {debug['texture'][0]:.1f}  ({debug['texture'][1]})",
         (0, 180, 255) if debug['texture'][1] == 'C' else (0, 200, 80)),
        (f"Specular: {debug['specular'][0]:.3f}  ({debug['specular'][1]})",
         (0, 180, 255) if debug['specular'][1] == 'C' else (0, 200, 80)),
    ]
    for text, colour in lines:
        cv2.putText(frame, text, (zx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1)
        y += 18

    score_text = (f"Score  CAN:{debug['can_score']}  "
                  f"PAPER:{debug['paper_score']}")
    cv2.putText(frame, score_text, (zx, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[Serial] Connected on {SERIAL_PORT}")
        time.sleep(1.5)
    except serial.SerialException as e:
        print(f"[Serial] Could not open port: {e}")
        print("[Serial] Running in NO-SERIAL mode")
        ser = None

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[Camera] Could not open camera {CAMERA_ID}")
        return

    print("[Camera] Running. Press 'q' to quit, 's' to save snapshot.")

    confirm_buffer = []
    last_sent_time = 0
    zone = (ZONE_X, ZONE_Y, ZONE_W, ZONE_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        zx, zy, zw, zh = zone

        # Detection zone rectangle
        cv2.rectangle(frame, (zx, zy), (zx+zw, zy+zh), (0, 255, 0), 2)
        cv2.putText(frame, "DETECTION ZONE", (zx, zy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        roi    = frame[zy:zy+zh, zx:zx+zw]
        result, debug = classify(roi)

        now         = time.time()
        in_cooldown = (now - last_sent_time) < COOLDOWN_SECONDS

        if not in_cooldown:
            if result is not None:
                confirm_buffer.append(result)
                if len(confirm_buffer) > CONFIRM_FRAMES:
                    confirm_buffer.pop(0)

                if (len(confirm_buffer) == CONFIRM_FRAMES and
                        len(set(confirm_buffer)) == 1):
                    label = confirm_buffer[0]
                    name  = "CAN" if label == 'C' else "PAPER"
                    print(f"[Detect] {name} → sending '{label}'")
                    if ser and ser.is_open:
                        ser.write(label.encode())
                    last_sent_time = now
                    confirm_buffer.clear()
            else:
                confirm_buffer.clear()

        # Status overlay
        if in_cooldown:
            status, colour = "COOLDOWN", (0, 165, 255)
        elif result == 'C':
            status, colour = f"CAN  ({len(confirm_buffer)}/{CONFIRM_FRAMES})", (0, 80, 255)
        elif result == 'P':
            status, colour = f"PAPER ({len(confirm_buffer)}/{CONFIRM_FRAMES})", (0, 200, 80)
        else:
            status, colour = "Waiting for object...", (180, 180, 180)

        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

        draw_debug(frame, debug, zone)

        cv2.imshow("Rubbish Sorter v2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[Saved] {fname}")

    cap.release()
    if ser and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
