
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.framework.formats import landmark_pb2

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas_width = 640
canvas_height = 720
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

drawing = False
points = deque(maxlen=5)  # store recent positions for smoothing

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cam_view = frame[:, canvas_width:].copy()

    # Add instructions to cam_view
    cv2.putText(cam_view, "DRAW: Pinch index & thumb", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(cam_view, "CLEAR: Pinch thumb & pinky", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_array = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w) - canvas_width
                y = int(lm.y * h)
                landmark_array.append((x, y))

            # Get finger positions
            ix, iy = landmark_array[8]  # Index finger
            tx, ty = landmark_array[4]   # Thumb
            px, py = landmark_array[20]  # Pinky

            if 0 <= ix < canvas_width:
                # Calculate distances
                draw_distance = np.hypot(ix - tx, iy - ty)    # Index-Thumb
                clear_distance = np.hypot(tx - px, ty - py)    # Thumb-Pinky
                
                # Drawing logic (index + thumb pinch)
                if draw_distance < 50:
                    drawing = True
                else:
                    drawing = False
                    points.clear()
                
                # Clear canvas logic (thumb + pinky pinch)
                if clear_distance < 50:
                    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    # Visual feedback for clear gesture
                    cv2.circle(cam_view, (tx, ty), 10, (0, 0, 255), -1)
                    cv2.circle(cam_view, (px, py), 10, (0, 0, 255), -1)
                    cv2.line(cam_view, (tx, ty), (px, py), (0, 0, 255), 3)

                # Drawing action
                if drawing:
                    points.append((ix, iy))
                    if len(points) > 1:
                        for i in range(1, len(points)):
                            cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 2)

                # Visual feedback for drawing gesture
                cv2.circle(cam_view, (ix, iy), 8, (0, 255, 255), -1)
                cv2.circle(cam_view, (tx, ty), 8, (255, 0, 255), -1)
                if drawing:
                    cv2.line(cam_view, (ix, iy), (tx, ty), (0, 255, 0), 2)

            # Draw hand landmarks (shifted to cam_view)
            offset_landmarks = []
            for lm in hand_landmarks.landmark:
                lm.x = (lm.x * w - canvas_width) / canvas_width
                lm.y = (lm.y * h) / canvas_height
                offset_landmarks.append(lm)

            hand_landmarks_shifted = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_shifted.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in offset_landmarks
            ])

            mp_draw.draw_landmarks(cam_view, hand_landmarks_shifted, mp_hands.HAND_CONNECTIONS)

    # Combine board and cam
    combined = np.hstack((canvas, cam_view))
    cv2.imshow("Virtual Drawing Board", combined)

    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty("Virtual Drawing Board", cv2.WND_PROP_VISIBLE) < 1 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()