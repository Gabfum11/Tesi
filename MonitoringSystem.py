# monitoring_system.py
import cv2
import time
import numpy as np
import mss
from datetime import datetime
import PoseDetector as pm
from TugTest import TUGTest
from SitToStandTest import SitToStandTest
from Daily_monitor import DailyMonitor
from Database_manager import DatabaseManager
from Contextanalyzer import ContextAnalyzer
from c500 import CameraController
import os
from dotenv import load_dotenv

class MonitoringSystem:
    def __init__(self, monitor_index=1, display_width=1280):
        self.display_width = display_width
        self.video_source = f"monitor_{monitor_index}"

        # Telecamera
        self.cam_ctrl = CameraController()

        # Screen capture con mss
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]
        self.frame_width = self.monitor["width"]
        self.frame_height = self.monitor["height"]
        self.is_file = False

        # Componenti
        self.detector = pm.PoseDetector()
        self.real_fps = 30
        self.detector.set_fps(self.real_fps)
        self.detector.set_confidence_threshold(0.6)
        self.detector.set_history_size(10)
        self.daily_monitor = DailyMonitor()
        self.db = DatabaseManager()

        # Test
        self.tug_test = TUGTest()
        self.sts_test = SitToStandTest()
        self.current_test = None
        self.current_test_id = None

        # Calibrazione distanza (ora nel detector)
        self.detector.load_distance_calibration()

        # Analisi contestuale (Vision LLM)
        load_dotenv()
        # Recupera la chiave
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("API Key non trovata! Verifica il file .env")
        self.context = ContextAnalyzer(api_key)

        # Calcola dimensioni display
        self.needs_resize = self.frame_width > self.display_width
        if self.needs_resize:
            aspect = self.frame_height / self.frame_width
            self.display_height = int(self.display_width * aspect)
            print(f"Schermo: {self.frame_width}x{self.frame_height} → Display: {self.display_width}x{self.display_height}")
        else:
            self.display_height = self.frame_height
            print(f"Schermo: {self.frame_width}x{self.frame_height}")

        print(f"FPS target: {self.real_fps}")
        print(f"Modalità: Screen capture (monitor {monitor_index})")

    def run(self):
        while True:
            # Cattura schermo con mss
            sct_img = self.sct.grab(self.monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Ridimensiona
            if self.needs_resize:
                frame = cv2.resize(frame, (self.display_width, self.display_height))

            # MediaPipe
            frame = self.detector.findPose(frame, draw=True)
            lmList = self.detector.findPosition(frame, draw=False)
            if self.detector.tracking_quality < 0.6:
                pose_detected = False
            else:
                pose_detected = len(lmList) > 0

            if pose_detected:
                state = self.detector.detect_posture(self.current_test)
                knee_angle = self.detector.last_knee_angle
                movement = self.detector.last_movement

                self.detector.findAngle(frame, 24, 26, 28, draw=True, filtered=False)
                self.detector.findAngle(frame, 23, 25, 27, draw=True, filtered=False)

                # Update daily monitor
                self.daily_monitor.update(state, movement, pose_detected, self.detector.hip_y_velocity)

                # Tracking episodi di cammino (metodo separato, parametri obbligatori)
                # Recupera posizione Y caviglie per analisi andatura
              
                self.daily_monitor.track_walking(
                    state,
                    self.detector.hip_x,
                    self.detector.hip_y,
                    self.detector.pixels_per_meter,
                    self.detector.left_ankle_y,
                    self.detector.right_ankle_y
                    
                )

                self._update_tests(state, knee_angle, movement, frame)

                cv2.putText(frame, f"Stato: {state}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Mov: {movement:.1f}  VelY: {self.detector.hip_y_velocity:.1f}",
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Steps: {self.detector.step_activity:.1f}",
                            (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Mostra velocità in m/s se calibrato
                speed_ms = self.detector.get_speed_ms()
                if speed_ms is not None:
                    cv2.putText(frame, f"Speed: {speed_ms:.2f} m/s",
                                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                self.daily_monitor.update("UNKNOWN", 0, False, 0)

            # === ANALISI CONTESTUALE (Vision LLM) ===
            # Periodico
            snapshot = self.context.update(frame)
            if snapshot:
                self._save_snapshot(snapshot)

            self._draw_overlay(frame)
            self.current_frame = frame
            cv2.imshow("Monitoring", frame)

            key = cv2.waitKeyEx(1)
            self._handle_input(key)

            if key == ord('q'):
                break

        self._shutdown()

    def _draw_overlay(self, display):
        h, w = display.shape[:2]
        cv2.putText(display, "S=TUG  T=STS  W=CalibDist  Q=Esci",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        dist_text = "Dist: OK" if self.detector.is_distance_calibrated() else "Dist: No (W)"
        dist_color = (0, 255, 0) if self.detector.is_distance_calibrated() else (0, 0, 255)
        cv2.putText(display, dist_text, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)

    def _update_tests(self, state, knee_angle, movement, frame):
        if self.current_test == "TUG":
            phase = self.tug_test.update(state, self.detector.hip_x, self.detector.hip_y, movement, knee_angle)
            cv2.putText(frame, f"TUG: {phase}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Hip X: {self.detector.hip_x:.0f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if phase == "FINISHED":
                self._save_tug_result()
        elif self.current_test == "STS":
            self.sts_test.update(state, knee_angle)
            cv2.putText(frame, f"STS Reps: {self.sts_test.reps}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.sts_test.reps >= 5:
                self._save_sts_result()

    def _handle_input(self, key):
        if key == ord('s'):
            self._start_tug()
        elif key == ord('t'):
            self._start_sts()
        elif key == ord('w'):
            self.detector.start_distance_calibration()
        elif key == 2490368:
            self.cam_ctrl.up()
        elif key == 2621440:
            self.cam_ctrl.down()
        elif key == 2424832:
            self.cam_ctrl.left()
        elif key == 2555904:
            self.cam_ctrl.right()

    def _start_tug(self):
        self.current_test = "TUG"
        self.tug_test.start(self.detector.hip_x, self.detector.hip_y, self.detector.pixels_per_meter)
        self.current_test_id = self.db.create_test_session(
            summary_date=self.daily_monitor.date,
            test_type="TUG",
            start_time=datetime.now().isoformat(),
            video_source=self.video_source
        )

    def _start_sts(self):
        self.current_test = "STS"
        self.sts_test.start()
        self.current_test_id = self.db.create_test_session(
            summary_date=self.daily_monitor.date,
            test_type="STS",
            start_time=datetime.now().isoformat(),
            video_source=self.video_source
        )

    def _save_tug_result(self):
        result = self.tug_test.get_result()
        if result and self.current_test_id:
            self.db.complete_test_session(self.current_test_id, datetime.now().isoformat())
            self.db.save_tug_result(self.current_test_id, result)
            print(f"TUG salvato: {result['total_time']}s, Risk: {result['fall_risk_level']}")
        self.current_test = None
        self.current_test_id = None

    def _save_sts_result(self):
        result = self.sts_test.get_result()
        if result and self.current_test_id:
            self.db.complete_test_session(self.current_test_id, datetime.now().isoformat())
            self.db.save_sts_result(self.current_test_id, result)
            print(f"STS salvato: {result['reps_completed']} reps in {result['total_time']}s")
        self.current_test = None
        self.current_test_id = None

    def _save_snapshot(self, snapshot):
        """Salva uno snapshot contestuale nel DB."""
        snapshot['date'] = self.daily_monitor.date
        self.db.save_context_snapshot(snapshot)

    def _shutdown(self):
        summary = self.daily_monitor.get_summary()
        self.db.save_daily_summary(summary)
        print(f"Daily summary salvato per {summary['date']}")
        print(f"Durata: {summary['total_monitoring_min']:.1f} secondi")

        # Salva episodi di cammino
        for ep in self.daily_monitor.walking_episodes:
            self.db.save_walking_episode(ep)
        print(f"Salvati {len(self.daily_monitor.walking_episodes)} episodi di cammino")

        # Salva eventi con timestamp
        for ev in self.daily_monitor.events:
            self.db.save_activity_event(ev)
        print(f"Salvati {len(self.daily_monitor.events)} eventi")

        # Salva time slot
        for slot in self.daily_monitor.completed_slots:
            self.db.save_time_slot(slot)
        print(f"Salvati {len(self.daily_monitor.completed_slots)} slot temporali")

        # Salva snapshot contestuali non ancora salvati (periodici)
        for snap in self.context.get_snapshots():
            snap['date'] = self.daily_monitor.date
            # Evita duplicati (quelli già salvati in tempo reale)
            if 'saved' not in snap:
                self.db.save_context_snapshot(snap)
        print(f"Totale snapshot contestuali: {len(self.context.get_snapshots())}")

        cv2.destroyAllWindows()
        print("Sistema chiuso, dati salvati.")


if __name__ == "__main__":
    system = MonitoringSystem(
        monitor_index=1,
        display_width=1280
    )
    start = time.time()
    system.run()
    elapsed = time.time() - start
    print(f"Tempo di processamento: {elapsed:.1f} secondi")