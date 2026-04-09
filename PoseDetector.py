# PoseModule.py (versione migliorata)
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np
import math
import json
import statistics
from collections import deque
import time

class PoseDetector:
    def __init__(self, model_path="models/pose_landmarker_full.task"):
        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.landmarker = PoseLandmarker.create_from_options(self.options)
        
        # Risultati
        self.result = None
        self.lmList = []
        
        # Storico per smoothing
        self.history_size = 10
        self.hip_x_history = deque(maxlen=self.history_size)
        self.hip_y_history = deque(maxlen=self.history_size)
        self.movement_history = deque(maxlen=self.history_size)
        self.angle_history = deque(maxlen=5)
        self.hip_z_history = deque(maxlen=self.history_size)
        
        # Calibrazione distanza (pixel → metri)
        self.pixels_per_meter = None
        self.distance_calibrating = False
        self._calib_start_hip_x = None
        
        # Metriche qualità
        self.min_confidence_threshold = 0.75
        self.tracking_quality = 0.0
        
        # Frame info
        self.frame_width = 0
        self.frame_height = 0
        self.frame_count = 0
        self.fps = 30
        
        # Detector
        self.last_knee_angle = 0
        self.last_movement = 0
        self.hip_x = 0
        self.hip_y = 0
        self.hip_z=0
        self.hip_y_velocity=0
        self.ankle_diff_history=deque(maxlen=15)
        self.left_ankle_y = 0
        self.right_ankle_y = 0
        self.left_ankle_x = 0
        self.right_ankle_x = 0
        self.shoulder_mid_x = 0
        self.shoulder_mid_y = 0
        #storico pixel
        self.hip_x_raw_history = deque(maxlen=self.history_size)
        self.hip_y_raw_history = deque(maxlen=self.history_size)
        self.ankle_y_history_left = deque(maxlen=self.history_size)
        self.ankle_y_history_right = deque(maxlen=self.history_size)
        self.step_activity = 0
        self.posture_history = deque(maxlen=15)
        self.knee_angle_history = deque(maxlen=15)
        self.hip_position_window = deque(maxlen=60)  # ~2 secondi a 30fps
        self.total_displacement=0

   
    # =========================================
    # CALIBRAZIONE DISTANZA (pixel → metri)
    # =========================================
    def start_distance_calibration(self):
        if not self.distance_calibrating:
            self.distance_calibrating = True
            self._calib_start_hip_x = self.hip_x
            self._calib_start_hip_y = self.hip_y
            print("=" * 50)
            print("CALIBRAZIONE DISTANZA AVVIATA")
            print("La persona cammini per 2 metri in linea retta,")
            print("lateralmente rispetto alla telecamera.")
            print("Premi di nuovo [W] quando ha raggiunto i 2 metri.")
            print("=" * 50)
            return False
        else:
            dx = self.hip_x - self._calib_start_hip_x
            dy = self.hip_y - self._calib_start_hip_y
            pixels = math.sqrt(dx**2 + dy**2)
            self.distance_calibrating = False
            
            if pixels < 50:
                print("Troppo poco spostamento, riprova.")
                return False
            
            self.pixels_per_meter = pixels / 2.0
            print("=" * 50)
            print(f"CALIBRAZIONE COMPLETATA")
            print(f"  2 metri = {pixels:.0f} pixel")
            print(f"  1 metro = {self.pixels_per_meter:.1f} pixel")
            print("=" * 50)
            
            self.save_distance_calibration()
            return True
    def save_distance_calibration(self, path="distance_calibration.json"):
        data = {"pixels_per_meter": self.pixels_per_meter}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Calibrazione distanza salvata in {path}")
    
    def load_distance_calibration(self, path="distance_calibration.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.pixels_per_meter = data["pixels_per_meter"]
            print(f"Calibrazione distanza caricata: 1m = {self.pixels_per_meter:.1f} px")
            return True
        except FileNotFoundError:
            print("Nessuna calibrazione distanza trovata. Premi [W] per calibrare.")
            return False
    
    def is_distance_calibrated(self):
        return self.pixels_per_meter is not None
    
    def get_pixels_for_meters(self, meters):
        """Converte metri in pixel. Utile per il TUG test."""
        if self.pixels_per_meter is None:
            return None
        return self.pixels_per_meter * meters
    
    # =========================================
    # VELOCITÀ IN M/S
    # =========================================
    def get_speed_ms(self):
        """Velocità in metri/secondo. Richiede calibrazione distanza."""
        if not self.pixels_per_meter or len(self.hip_x_raw_history) < 2:
            return None
        dx = self.hip_x_raw_history[-1] - self.hip_x_raw_history[-2]
        dy = self.hip_y_raw_history[-1] - self.hip_y_raw_history[-2]
        px_per_frame = math.sqrt(dx**2 + dy**2)
        meters_per_frame = px_per_frame / self.pixels_per_meter
        return meters_per_frame * self.fps
    
    # =========================================
    # RILEVAMENTO POSE
    # =========================================
    def findPose(self, frame, draw=True, timestamp_ms=None):
        self.frame_height, self.frame_width, _ = frame.shape
        self.frame_count += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        self.result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        self._update_tracking_quality()
        if draw and self.result.pose_landmarks:
            frame = self._draw_landmarks(frame)
        
        return frame
    
    def findPosition(self, frame, draw=False, filtered=True):
        self.lmList = []
        if not self.result.pose_landmarks:
            return self.lmList
        landmarks = self.result.pose_landmarks[0]
        h, w = self.frame_height, self.frame_width
        for id, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            confidence = lm.visibility if hasattr(lm, 'visibility') else 1.0 
            self.lmList.append([id, x, y, confidence])
            if draw and confidence > self.min_confidence_threshold:
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                cv2.circle(frame, (x, y), 5, color, cv2.FILLED)
        return  self.lmList
    
    # =========================================
    # CALCOLO ANGOLI
    # =========================================
    def findAngle(self, frame, p1, p2, p3, draw=True, filtered=True):
        lmList_to_use = self.lmList
        if len(lmList_to_use) <= max(p1, p2, p3):
            return 0
        
        c1 = lmList_to_use[p1][3]
        c2 = lmList_to_use[p2][3]
        c3 = lmList_to_use[p3][3]
        
        if min(c1, c2, c3) < self.min_confidence_threshold:
            if self.angle_history:
                return self.angle_history[-1]
            return 0
        
        x1, y1 = lmList_to_use[p1][1:3]
        x2, y2 = lmList_to_use[p2][1:3]
        x3, y3 = lmList_to_use[p3][1:3]
        
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle
        
        if filtered:
            self.angle_history.append(angle)
            angle = statistics.mean(self.angle_history)
        
        # DISEGNO: usa sempre coordinate PIXEL (self.lmList), non normalizzate
        if draw and frame is not None and len(self.lmList) > max(p1, p2, p3):
            px1, py1 = self.lmList[p1][1:3]
            px2, py2 = self.lmList[p2][1:3]
            px3, py3 = self.lmList[p3][1:3]
            self._draw_angle(frame, px1, py1, px2, py2, px3, py3, angle)
        
        return angle
        
    def _draw_angle(self, frame, x1, y1, x2, y2, x3, y3, angle):
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
        
        for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
            cv2.circle(frame, (x, y), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x, y), 15, (0, 0, 255), 2)
        
        cv2.putText(frame, f"{int(angle)}°", (x2 - 50, y2 + 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    # =========================================
    # RILEVAMENTO MOVIMENTO
    # =========================================
    def detectWalking(self):
        """
        Rileva il movimento della persona.
        
        - hip_x, hip_y, raw_history: sempre in PIXEL (per TUG e velocità m/s)
        - hip_x_history, hip_y_history: normalizzati se calibrato (per confronto movimento)
        """
        # Serve sempre lmList in pixel per hip_x/hip_y
        if len(self.lmList) < 25:
            return 0
        
        # =========================================
        # COORDINATE PIXEL (per TUG, velocità m/s)
        # =========================================
        l_hip_px = self.lmList[23]
        r_hip_px = self.lmList[24]    
        hip_x_px = (l_hip_px[1] + r_hip_px[1]) / 2
        hip_y_px = (l_hip_px[2] + r_hip_px[2]) / 2
        self.hip_position_window.append((hip_x_px, hip_y_px))
        # Oscillazione caviglie (rileva passi in qualsiasi direzione)
        self.ankle_y_history_left.append(self.left_ankle_y)
        self.ankle_y_history_right.append(self.right_ankle_y)
        self.hip_x = hip_x_px  # Sempre pixel
        self.hip_y = hip_y_px  # Sempre pixel
        self.left_ankle_y = self.lmList[27][2]
        self.right_ankle_y = self.lmList[28][2]
        self.left_ankle_x=self.lmList[27][1]
        self.right_ankle_x=self.lmList[28][1]
        self.shoulder_mid_x = (self.lmList[11][1] + self.lmList[12][1]) / 2
        self.shoulder_mid_y = (self.lmList[11][2] + self.lmList[12][2]) / 2
        self.hip_x_raw_history.append(hip_x_px)
        self.hip_y_raw_history.append(hip_y_px)
        if len(self.ankle_y_history_left) >= 10:
            left_range = max(self.ankle_y_history_left) - min(self.ankle_y_history_left)
            right_range = max(self.ankle_y_history_right) - min(self.ankle_y_history_right)
            self.step_activity = max(left_range, right_range)
        else:
            self.step_activity = 0
        # Velocità verticale (per detect sit-to-stand)
        if len(self.hip_y_raw_history) >= 2:
            self.hip_y_velocity = self.hip_y_raw_history[-1] - self.hip_y_raw_history[-2]
        #calcolo movimento:

        self.hip_x_history.append(hip_x_px)
        self.hip_y_history.append(hip_y_px)
        
        if len(self.hip_x_history) < 2:
            return 0

        dx = self.hip_x_history[-1] - self.hip_x_history[-2]
        dy = self.hip_y_history[-1] - self.hip_y_history[-2]

        if abs(self.hip_y_velocity) > 3:
            dy = 0 # la y viene azzerata se la velocity è alta(l'anca sale o scende per seduta o alzata, non per cammino
        """
        Quando cammini, il corpo si sposta principalmente in orizzontale (dx grande).
        L'anca oscilla anche un po' su e giù ad ogni passo (dy piccolo). 
        va ridotto il contributo della y
        """
        instant_movement = math.sqrt(dx**2 + (dy * 0.75)**2) 

        self.movement_history.append(instant_movement)
        return statistics.mean(self.movement_history)

  
    # RILEVAMENTO POSTURA
    # =========================================
    def detect_posture(self, current_test=None):
        # Usa lista normalizzata se calibrato
        lmList_to_use =  self.lmList
        if len(lmList_to_use) < 29:
            return 'UNKNOWN'
        
        movement = self.detectWalking()  # usa normalized se calibrato
        self.last_movement = movement
        # Soglie movimento in base al test
        if current_test == "TUG":
            movement_threshold = 2
        elif current_test == "STS":
            movement_threshold = 15
        else:
            movement_threshold = 5
        
        # Landmark principali
        nose = lmList_to_use[0]
        l_hip = lmList_to_use[23]
        r_hip = lmList_to_use[24]
        l_knee = lmList_to_use[25]
        r_knee = lmList_to_use[26]
        l_ankle = lmList_to_use[27]
        r_ankle = lmList_to_use[28]
        
        key_points = [nose, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
        avg_confidence = statistics.mean([p[3] for p in key_points])
        
        if avg_confidence < self.min_confidence_threshold or self.tracking_quality<0.75:
            return 'UNKNOWN'
        

        # Angoli ginocchio
        conf_r = min(self.lmList[24][3], self.lmList[26][3], self.lmList[28][3])
        conf_l = min(self.lmList[23][3], self.lmList[25][3], self.lmList[27][3])

        knee_angle_r = self.findAngle(None, 24, 26, 28, draw=False)
        knee_angle_l = self.findAngle(None, 23, 25, 27, draw=False)

        if conf_r > conf_l:
            knee_angle = knee_angle_r
        elif conf_l > conf_r:
            knee_angle = knee_angle_l
        elif knee_angle_r > 0 and knee_angle_l > 0:
            knee_angle = (knee_angle_r + knee_angle_l) / 2
        else:
            knee_angle = 0
        self.knee_angle_history.append(knee_angle)
        self.last_knee_angle = knee_angle
        recently_sitting = self.posture_history.count('SITTING') > 3
        stepping = self.step_activity > 30 and knee_angle>140
        # Classificazione postura
        was_sitting = self.posture_history[-1] == 'SITTING' if self.posture_history else False

        if was_sitting:
            sit_threshold = 150  # più alto per restare seduto
        else:
            sit_threshold = 140  # più basso per diventare seduto

        if knee_angle < sit_threshold and (movement < movement_threshold or knee_angle<120):
            posture = 'SITTING'
        elif (movement > movement_threshold or stepping) and not recently_sitting:
            posture = 'WALKING'
        else:
            posture = 'STANDING'
                
        self.posture_history.append(posture)
        return posture
    # =========================================
    # METRICHE QUALITÀ
    # =========================================
    def _update_tracking_quality(self):
        if not self.result.pose_landmarks:
            self.tracking_quality = 0.0
            return
        
        landmarks = self.result.pose_landmarks[0]
        key_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        confidences = []
        
        for idx in key_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                conf = lm.visibility if hasattr(lm, 'visibility') else 0.5
                confidences.append(conf)
        
        self.tracking_quality = statistics.mean(confidences) if confidences else 0.0
    
    # =========================================
    # DISEGNO
    # =========================================
    def _draw_landmarks(self, frame):
        if not self.result.pose_landmarks:
            return frame
        
        annotated = np.copy(frame)
        
        for pose_landmarks in self.result.pose_landmarks:
            drawing_utils.draw_landmarks(
                annotated,
                pose_landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_styles.get_default_pose_landmarks_style(),
                drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        self._draw_info_overlay(annotated)
        return annotated
    
    def _draw_info_overlay(self, frame):
        y_offset = 30
        
        quality_color = (0, 255, 0) if self.tracking_quality > 0.7 else (0, 255, 255) if self.tracking_quality > 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Quality: {self.tracking_quality:.0%}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
    
    # =========================================
    # CONFIGURAZIONE
    # =========================================
    def set_fps(self, fps):
        self.fps = fps
    
    def set_confidence_threshold(self, threshold):
        self.min_confidence_threshold = threshold
    
    def set_history_size(self, size):
        self.history_size = size
        self.hip_x_history = deque(maxlen=size)
        self.hip_y_history = deque(maxlen=size)
        self.movement_history = deque(maxlen=size)
        self.hip_x_raw_history = deque(maxlen=size)
        self.hip_y_raw_history = deque(maxlen=size)
