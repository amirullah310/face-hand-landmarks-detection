# Face and Hand Landmarks Detection using Python - Mediapipe, OpenCV
# Catatan: mediapipe >= 0.10.30 tidak lagi menyediakan mp.solutions.holistic,
# sehingga program ini memakai Holistic Landmarker dari MediaPipe Tasks API
# (fungsinya sama dengan model Holistic pada tutorial).

# STEP-1: Import Libraries
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import hand_landmarker

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "holistic_landmarker.task")

# Model Tasks API berupa file .task, unduh sekali jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model holistic_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# STEP-2: Grabbing the Holistic Model from Mediapipe and Initializing the Model
# min_*_detection_confidence ~ min_detection_confidence,
# min_*_landmarks_confidence  ~ min_tracking_confidence
holistic_model = vision.HolisticLandmarker.create_from_options(
    vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = vision.drawing_utils
HandLandmark = hand_landmarker.HandLandmark

# Koneksi antar landmark untuk wajah dan tangan
FACEMESH_CONTOURS = (
    vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)
HAND_CONNECTIONS = (
    vision.HandLandmarksConnections.HAND_CONNECTIONS)

# Code to access landmarks
for landmark in HandLandmark:
    print(f"HandLandmark.{landmark.name}", landmark.value)

print(HandLandmark.WRIST.value)

# STEP-3: Detecting Face and Hand landmarks from the image
# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Kamera tidak bisa dibuka. Pastikan webcam aktif "
          "dan tidak dipakai aplikasi lain.")

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
startTime = time.time()

# Tangkap layar otomatis saat wajah + kedua tangan terdeteksi
# selama beberapa frame berturut-turut (agar posisinya stabil)
OUTPUT_PATH = os.path.join(BASE_DIR, "output.png")
STABLE_FRAMES = 15
stableCount = 0

while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()
    if not ret:
        break

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # Mode VIDEO membutuhkan timestamp (ms) yang terus naik
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    timestamp_ms = int((time.time() - startTime) * 1000)
    results = holistic_model.detect_for_video(mp_image, timestamp_ms)

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
    )

    # Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS
    )

    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS
    )

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

    # Simpan hasil jika wajah dan kedua tangan terdeteksi stabil
    if (results.face_landmarks and results.left_hand_landmarks
            and results.right_hand_landmarks):
        stableCount += 1
    else:
        stableCount = 0
    if stableCount >= STABLE_FRAMES:
        cv2.imwrite(OUTPUT_PATH, image)
        print("Hasil disimpan ke", OUTPUT_PATH)
        break

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
holistic_model.close()
