
import cv2
import numpy as np
import os
import sys
import argparse
import time

SERIAL_PORT     = "COM3"
BAUD_RATE       = 9600
CAMERA_ID       = 0
ZONE_X, ZONE_Y, ZONE_W, ZONE_H = 400, 100, 400, 400
MODEL_PATH      = "sorter_model.h5"
DATA_DIR        = "training_data"
IMG_SIZE        = (150, 150)
CONFIRM_FRAMES  = 8
COOLDOWN        = 2.5
CONFIDENCE_MIN  = 0.0   # minimum confidence before sending command

CLASSES = ["CAN", "PAPER"]


def collect():
    """
    Press 'c' to capture a CAN image, 'p' for PAPER, 'q' to quit.
    Aim for 80-150 images per class.
    Vary angle, distance, rotation, and lighting as you capture.
    """
    for cls in CLASSES:
        os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_ID)
    counts = {c: len(os.listdir(os.path.join(DATA_DIR, c))) for c in CLASSES}
    print("\n── Collecting training images ──")
    print("  'c' = capture CAN image")
    print("  'p' = capture PAPER image")
    print("  'q' = quit and move on to training\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[ZONE_Y:ZONE_Y+ZONE_H, ZONE_X:ZONE_X+ZONE_W]
        cv2.rectangle(frame, (ZONE_X, ZONE_Y),
                      (ZONE_X+ZONE_W, ZONE_Y+ZONE_H), (0,255,0), 2)

        info = f"CAN:{counts['CAN']}  PAPER:{counts['PAPER']}  (need ~100 each)"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.putText(frame, "c=CAN  p=PAPER  q=done", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        cv2.imshow("Collect Training Images", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img = cv2.resize(roi, IMG_SIZE)
            path = os.path.join(DATA_DIR, "CAN", f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, img)
            counts['CAN'] += 1
            print(f"  Saved CAN image #{counts['CAN']}")

        elif key == ord('p'):
            img = cv2.resize(roi, IMG_SIZE)
            path = os.path.join(DATA_DIR, "PAPER", f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, img)
            counts['PAPER'] += 1
            print(f"  Saved PAPER image #{counts['PAPER']}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. CAN:{counts['CAN']}  PAPER:{counts['PAPER']}")
    print("Now run:  python train_and_detect.py --train")


def train():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    print("\n── Loading images ──")
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        files  = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        print(f"  {cls}: {len(files)} images")
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"\n── Building model ──")
    model = models.Sequential([
        # Data augmentation (only during training)
        layers.RandomFlip("horizontal", input_shape=(*IMG_SIZE, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Conv block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Conv block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Conv block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')   # binary: 0=CAN, 1=PAPER
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print("\n── Training ──")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=4, restore_best_weights=True
            )
        ]
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\n── Best validation accuracy: {val_acc*100:.1f}% ──")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Now run:  python train_and_detect.py --detect")


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — DETECT
# ═══════════════════════════════════════════════════════════════════

def detect():
    import tensorflow as tf
    import serial

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model not found: {MODEL_PATH}")
        print("Run --train first.")
        return

    print(f"[Model] Loading {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[Model] Ready.")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[Serial] Connected on {SERIAL_PORT}")
        time.sleep(1.5)
    except Exception as e:
        print(f"[Serial] {e} — running without serial")
        ser = None

    cap = cv2.VideoCapture(CAMERA_ID)
    confirm_buffer = []
    last_sent      = 0

    print("[Camera] Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame, (ZONE_X, ZONE_Y),
                      (ZONE_X+ZONE_W, ZONE_Y+ZONE_H), (0,255,0), 2)

        roi = frame[ZONE_Y:ZONE_Y+ZONE_H, ZONE_X:ZONE_X+ZONE_W]

        # Preprocess for model
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

        #pred       = model.predict(img, verbose=0)[0][0]
        #confidence = pred if pred > 0.5 else 1 - pred
        #label      = 'P' if pred > 0.5 else 'C'
        #name       = 'PAPER' if label == 'P' else 'CAN'

        pred       = model.predict(img, verbose=0)[0][0]
        # CAN = low pred value (near 0), PAPER = high pred value (near 1)
        can_confidence = 1 - pred
        if can_confidence >= 0.80:
            label      = 'C'
            confidence = can_confidence
            name       = 'CAN'
        else:
            label      = 'P'
            confidence = pred
            name       = 'PAPER'

        now         = time.time()
        in_cooldown = (now - last_sent) < COOLDOWN

        if not in_cooldown and confidence >= CONFIDENCE_MIN:
            confirm_buffer.append(label)
            if len(confirm_buffer) > CONFIRM_FRAMES:
                confirm_buffer.pop(0)

            if (len(confirm_buffer) == CONFIRM_FRAMES and
                    len(set(confirm_buffer)) == 1):
                print(f"[Detect] {name} ({confidence*100:.0f}%) → sending '{label}'")
                if ser and ser.is_open:
                    ser.write(label.encode())
                last_sent = now
                confirm_buffer.clear()
        elif confidence < CONFIDENCE_MIN:
            confirm_buffer.clear()

        # UI
        colour = (0,80,255) if label == 'C' else (0,200,80)
        if in_cooldown:
            status = "COOLDOWN"
            colour = (0,165,255)
        elif confidence < CONFIDENCE_MIN:
            status = f"Uncertain ({confidence*100:.0f}%)"
            colour = (180,180,180)
        else:
            status = f"{name} {confidence*100:.0f}%  ({len(confirm_buffer)}/{CONFIRM_FRAMES})"

        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        cv2.imshow("Rubbish Sorter — Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if ser and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Collect training images")
    parser.add_argument("--train",   action="store_true", help="Train the model")
    parser.add_argument("--detect",  action="store_true", help="Run live detection")
    args = parser.parse_args()

    if args.collect:
        collect()
    elif args.train:
        train()
    elif args.detect:
        detect()
    else:
        print("Usage:")
        print("  python train_and_detect.py --collect   (capture training images)")
        print("  python train_and_detect.py --train     (train the model)")
        print("  python train_and_detect.py --detect    (run live detection)")
