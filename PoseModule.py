import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles  
import statistics

class PoseDetector():
    def __init__(self):
        model_path="models/pose_landmarker_full.task"
        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options= BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )

        self.landmarker = PoseLandmarker.create_from_options(self.options) #creazione detector 
        self.hip_x_history=[]


    # -------------------------------------------------
    # Processa un frame e restituisce il risultato
    # -------------------------------------------------
    def findPose(self, frame, draw):
        #opencv usa BGR, mediapipe RGB, quindi facciamo la conversione
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        timestamp = int(time.time() * 1000) #timestamp crescente -> va bene per la webcam in tempo reale
        self.result = self.landmarker.detect_for_video(mp_image, timestamp)
        if self.result.pose_landmarks:
            if(draw):
                frame=self.draw_landmarks(frame)
        return frame
    
    def findPosition(self, frame,draw):
        self.lmList = [] 
        if self.result.pose_landmarks:
            for id, landmark in enumerate(self.result.pose_landmarks[0]):
                h, w, _ = frame.shape
                x ,y = int(landmark.x * w),  int(landmark.y * h)
                self.lmList.append([id,x,y])
                if draw:
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), cv2.FILLED) #il colore è in BGR con cv2
        return self.lmList
    def findAngle(self, frame, p1,p2,p3, draw):
        #metodo per trovare l'angolo con 3 punti
        if len(self.lmList)<=max(p1,p2,p3): return 0
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]
        angle= math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        #facciamo in modo che restituisca un angolo compreso tra 0° e 180°
        if angle <0:
            angle+=360
        if angle>180:
            angle=360-angle
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
            
    def detectWalking(self):
        hip_x=(self.lmList[23][1]+self.lmList[24][1]) /2
        self.hip_x_history.append(hip_x)
        if len(self.hip_x_history) >10:
            self.hip_x_history.pop(0)
        if len(self.hip_x_history)<2: return 0 
        movement=statistics.stdev(self.hip_x_history) #misura quanto si è mosso il corpo
        return movement
    
        
    # -------------------------------------------------
    # Disegna landmarks sul frame, utili per le connessioni tra le articolazioni
    # -------------------------------------------------
    def draw_landmarks(self, frame):
        if not self.result.pose_landmarks:
            return frame
        annotated_image = np.copy(frame)
        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(
            color=(0, 255, 0),
            thickness=2
        )
        for pose_landmarks in self.result.pose_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                pose_landmark_style,
                pose_connection_style
            )

        return annotated_image

        