import cv2
import PoseModule as pm
import Person as ps
person=ps.Person()
cap=cv2.VideoCapture(0)
detector=pm.PoseDetector()
while True:
    success, frame =cap.read()
    if not success:
        break
    frame=detector.findPose(frame, draw=True)
    lmList=detector.findPosition(frame, draw=False)
    if len(lmList)!=0:
        knee_angle=detector.findAngle(frame,24,26,28, draw=True)
        movement= detector.detectWalking()
        cv2.putText(frame,
            f"Movement: {int(movement)}",
            (50,150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2)
        state=person.update(knee_angle, movement)
        cv2.putText(frame, 
                    f"State: {state}",
                    (50, 100), #posizione del testo
                    cv2.FONT_HERSHEY_SIMPLEX, #font
                    1.5, #grandezza testo
                    (0, 255, 0),
                    3) #spessore lettere
        
    # cv2.circle(frame, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED) #monitoraggio del gomito
    cv2.imshow("Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()