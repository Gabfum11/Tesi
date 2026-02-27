import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model_path = "models/hand_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.VIDEO)
cap=cv2.VideoCapture(0)
with HandLandmarker.create_from_options(options) as landmarker:
    while True: #ciclo per avviare la webcam
        success, frame =cap.read()
        if not success:
            break
        # Converti BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
         # Timestamp richiesto dalla modalità VIDEO
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        #rilevamento mani
        result = landmarker.detect_for_video(mp_image, timestamp)
        if result.hand_landmarks: #controllo se è stata rilevata almeno una mano , result.handlandmarks è una lista di mani
            for hand_landmarks in result.hand_landmarks: #ogni hand_landmarks contiene 21 punti nella mano
                for landmark in hand_landmarks: 
                    h, w, _ = frame.shape #frame.shape = (altezza, larghezza, numero colori)
                    #mediapipe restituisce coordinate tra 0 e 1 (landmark.x e landmark.y)
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, #immagine
                                (x, y), #posizione
                                5, #raggio del cerchio
                                (0, 255, 0), #colore verde
                                -1 #cerchio pieno
                                ) #si sta disegnando n puntino verde su ogni landmark 
        cv2.imshow("Image", frame)
        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()