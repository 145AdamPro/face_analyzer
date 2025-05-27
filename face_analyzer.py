import cv2
from deepface import DeepFace
import logging
import time

logging.getLogger("tensorflow").setLevel(logging.ERROR)

analyze_every_n_frames = 1
detector_backend_choice = "mtcnn"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    print(f"Loading models using {detector_backend_choice} backend, this might take a moment...")
    dummy_frame = cv2.imread("placeholder.jpg") # Can be replaced BUT useles
    if dummy_frame is None:
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    DeepFace.analyze(dummy_frame, actions = ["emotion"], detector_backend = detector_backend_choice, enforce_detection=False)
    print("Models loaded successfully.") 
except Exception as e:
    print(f"Could not pre-load models: {e}")

frame_count = 0
last_results = [] 
fps = 0
start_time = time.time()
frame_processed_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    frame_count += 1
    current_results = []

    if frame_count % analyze_every_n_frames == 0:
        try:
            results = DeepFace.analyze(frame, actions = ["emotion"], detector_backend = detector_backend_choice, enforce_detection=False, silent=True)

            if isinstance(results, list):
                last_results = results 
                current_results = results
            else:
                last_results = []
                current_results = []

        except Exception as e:
            last_results = []
            current_results = []
    else:
        current_results = last_results

    if isinstance(current_results, list):
        for result in current_results:
            if isinstance(result, dict) and "region" in result:
                region = result.get("region")
                if isinstance(region, dict) and all(k in region for k in ["x", "y", "w", "h"]):
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    age = result.get("age", "N/A")
                    emotion = result.get("dominant_emotion", "N/A")
                    text = f"Age: {age}, Emotion: {emotion}"
                    text_y = y - 10 if y - 10 > 10 else y + h + 20
                    cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print("Warning: Invalid region data in result", result) 
            elif isinstance(result, dict):
                 print("Warning: Result dictionary missing 'region' key", result) 


    frame_processed_count += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time >= 1.0:
        fps = frame_processed_count / elapsed_time
        start_time = time.time()
        frame_processed_count = 0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow(f"Face Analysis (Optimized V2 - Backend: {detector_backend_choice})", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Application exited.")

