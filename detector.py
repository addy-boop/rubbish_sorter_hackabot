import cv2
from ultralytics import YOLO
import time

#ser = serial.Serial('/dev/tty.usbmodem11401', 9600)  # update this
model = YOLO('best1.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

frame_count = 0
results = []
last_sent = 0
COOLDOWN = 1

while True:
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % 3 == 0:
        results = model(frame, verbose=False, imgsz=320)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label + " " + str(round(conf, 2)), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print("Detected: " + label)

            now = time.time()
            if now - last_sent > COOLDOWN:
                if label == "can":
                    #ser.write(b'C')
                    print("Sent C to Pico")
                else:
                    #ser.write(b'P')
                    print("Sent P to Pico")
                last_sent = now

    cv2.imshow("YOLOv8 Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#ser.close()
cv2.destroyAllWindows()