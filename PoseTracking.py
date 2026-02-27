import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles        
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks,
            vision.PoseLandmarksConnections.POSE_LANDMARKS,
            pose_landmark_style,
            pose_connection_style
        )

    return annotated_image

model_path = 'models/pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)
cap=cv2.VideoCapture(0)
with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
    while True:
        success, frame =cap.read()
        if not success:
            break
        # Converti BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
          # Timestamp richiesto dalla modalità VIDEO
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp)
        frame=draw_landmarks_on_image(frame, result)
        if result.pose_landmarks:
             for landmark in result.pose_landmarks[0]:
                h, w, _ = frame.shape
                x ,y = int(landmark.x * w),  int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()

