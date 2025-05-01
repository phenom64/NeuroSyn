import cv2
import mediapipe as mp
import time
import numpy as np

# --- Constants for **LEFT** Arm Landmarks (from MediaPipe Pose model) ---
# We are switching to LEFT arm indices as they seem to be tracked more reliably in your case
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
LEFT_SHOULDER = 11 # Changed from 12
LEFT_ELBOW = 13    # Changed from 14
LEFT_WRIST = 15    # Changed from 16
LEFT_PINKY = 17    # Changed from 18 (Tip of pinky finger knuckle)
LEFT_INDEX = 19    # Changed from 20 (Tip of index finger knuckle)
# We might not strictly need 17 & 19 for arm pose, but they help orient the wrist

# Define connections for the **LEFT** arm
LEFT_ARM_CONNECTIONS = [ # Renamed from RIGHT_ARM_CONNECTIONS
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_PINKY), # Connect wrist to pinky
    (LEFT_WRIST, LEFT_INDEX), # Connect wrist to index
]

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Camera Setup ---
cap = None
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Successfully opened camera index: {i}")
        break
    else:
        cap.release()
        print(f"Failed to open camera index: {i}")

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

print("Initializing MediaPipe Pose and Hands...")

# Initialize Pose model
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Initialize Hands model - Still looking for only one hand
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
print("Models initialized. Starting video stream...")

prev_time = 0

# --- Main Loop ---
try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        current_time = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        image.flags.writeable = False
        results_pose = pose.process(image)
        results_hands = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Draw **LEFT** Arm ---
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            # Custom drawing using LEFT arm indices
            for connection in LEFT_ARM_CONNECTIONS: # Use updated connection list
                start_idx = connection[0]
                end_idx = connection[1]
                # Check visibility before drawing
                if (landmarks[start_idx].visibility > 0.5 and
                    landmarks[end_idx].visibility > 0.5):
                    start_point = (int(landmarks[start_idx].x * image_width),
                                   int(landmarks[start_idx].y * image_height))
                    end_point = (int(landmarks[end_idx].x * image_width),
                                 int(landmarks[end_idx].y * image_height))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2) # Green lines

            # Draw key LEFT arm landmarks as circles
            arm_indices_to_draw = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_PINKY, LEFT_INDEX] # Use updated indices
            for idx in arm_indices_to_draw:
                 # Check visibility
                 if landmarks[idx].visibility > 0.5:
                     point = (int(landmarks[idx].x * image_width),
                              int(landmarks[idx].y * image_height))
                     cv2.circle(image, point, 5, (0, 0, 255), -1) # Red circles

        # --- Draw Right Hand ---
        # This logic remains the same - we still want the RIGHT hand landmarks drawn
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())


        # Calculate and display FPS
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Pose (Left Arm) + Right Hand Test', image) # Updated window title slightly

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("Exit key pressed. Closing...")
            break
finally:
    print("Releasing resources...")
    if pose:
        pose.close()
    if hands:
        hands.close()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete.")