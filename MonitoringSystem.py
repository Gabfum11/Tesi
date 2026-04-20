# monitoring_system.py
import cv2
import time
import numpy as np
import mss
import json
from datetime import datetime
import PoseDetector as pm
from TugTest import TUGTest
from SitToStandTest import SitToStandTest
from Daily_monitor import DailyMonitor
from Database_manager import DatabaseManager
from Dataaggregator import DataAggregator
from c500 import CameraController
from collections import deque
from EventFrameBuffer import EventFrameBuffer

class MonitoringSystem:
    def __init__(self, monitor_index=1, display_width=1280):
        # Screen capture con mss
        self.sct = mss.mss()
        full_monitor = self.sct.monitors[1]
        screen_w = full_monitor["width"]
        screen_h = full_monitor["height"]
        self.display_width = display_width
        self.video_source = f"monitor_{monitor_index}"

        # Telecamera
        self.cam_ctrl = CameraController()
        
        self.monitor = {"top": 130, "left": 100, "width": 870, "height": 520}  # Monitor 1: area di interesse (da calibrare)
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
        self.aggregator = DataAggregator(self.db.db_path)

        # Test
        self.tug_test = TUGTest()
        self.sts_test = SitToStandTest()
        self.current_test = None
        self.current_test_id = None

        # Calibrazione distanza (ora nel detector)
        #self.detector.load_distance_calibration()


       
        # Recupera la chiave
        
        self.event_buffer = EventFrameBuffer(model="qwen2.5vl:7b")  # Buffer per eventi contestuali
        self.daily_monitor.event_buffer = self.event_buffer
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
        cv2.namedWindow("Monitoring", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Monitoring", self.monitor["width"], self.monitor["height"] // 2)
        cv2.moveWindow("Monitoring", self.monitor["width"], 0)

        # Inizializza contatore FPS una sola volta
        self._frame_times = deque(maxlen=60)
        self._last_fps_print = 0

        while True:
            # Cattura schermo
            sct_img = self.sct.grab(self.monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if self.needs_resize:
                frame = cv2.resize(frame, (self.display_width, self.display_height))

            # MediaPipe
            frame = self.detector.findPose(frame, draw=True)
            lmList = self.detector.findPosition(frame, draw=False)
            pose_detected = (self.detector.tracking_quality >= 0.75) and (len(lmList) > 0)

            if pose_detected:
                state = self.detector.detect_posture(self.current_test)
                knee_angle = self.detector.last_knee_angle
                movement = self.detector.last_movement

                self.detector.findAngle(frame, 24, 26, 28, draw=True, filtered=False)
                self.detector.findAngle(frame, 23, 25, 27, draw=True, filtered=False)

                self.daily_monitor.update(state, movement, pose_detected, self.detector.hip_y_velocity)

                self.daily_monitor.track_walking(
                    state,
                    self.detector.hip_x,
                    self.detector.hip_y,
                    self.detector.pixels_per_meter,
                    self.detector.left_ankle_y,
                    self.detector.right_ankle_y,
                    self.detector.left_ankle_x,
                    self.detector.right_ankle_x,
                    self.detector.shoulder_mid_x,
                    self.detector.shoulder_mid_y,
                    fps=self.detector.fps
                )

                # Context UNA VOLTA
                context = self._build_vlm_context(state)

                

                self._update_tests(state, knee_angle, movement, frame)

                # Overlay
                cv2.putText(frame, f"Stato: {state}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Mov: {movement:.1f}  VelY: {self.detector.hip_y_velocity:.1f}",
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Steps: {self.detector.step_activity:.1f}",
                            (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                speed_ms = self.detector.get_speed_ms()
                if speed_ms is not None:
                    cv2.putText(frame, f"Speed: {speed_ms:.2f} m/s",
                                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                self.daily_monitor.update("UNKNOWN", 0, False, 0)
                context = {'state': 'UNKNOWN', 'state_duration_sec': 0}
            self.event_buffer.update(frame, context)
            # Raccogli risultati VLM completati
            for result in self.event_buffer.get_results():
                self._save_event_snapshot(result)
          

            self._draw_overlay(frame)
            self.current_frame = frame
            cv2.imshow("Monitoring", frame)

            # FPS rolling
            now = time.time()
            self._frame_times.append(now)
            if len(self._frame_times) >= 30:
                span = self._frame_times[-1] - self._frame_times[0]
                if span > 0:
                    real_fps = (len(self._frame_times) - 1) / span
                    self.detector.set_fps(real_fps)
                    self.daily_monitor.gait_analyzer.set_fps(real_fps)

                    if now - self._last_fps_print > 5:
                        print(f"[FPS] {real_fps:.1f}")
                        self._last_fps_print = now

            # Input tastiera OGNI frame
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
    def _build_vlm_context(self, state):
        """Costruisce il contesto corrente per il ContextAnalyzer."""
        # Durata nello stato corrente
        if state == self.daily_monitor.prev_state:
            state_duration = time.time() - self.daily_monitor.prev_time
        else:
            state_duration = 0

        # Ultima simmetria disponibile
        last_symmetry = None
        if self.daily_monitor.walking_episodes:
            last_ep = self.daily_monitor.walking_episodes[-1]
            last_symmetry = last_ep.get('symmetry')

        # Ultimo sit-to-stand
        last_sts_time = None
        if self.daily_monitor.sit_to_stand_times:
            last_sts_time = self.daily_monitor.sit_to_stand_times[-1]

        # Inattivita' corrente
        inactivity = time.time() - self.daily_monitor.current_inactivity_start

        return {
            'state': state,
            'state_duration_sec': state_duration,
            'movement': self.detector.last_movement,
            'speed_ms': self.detector.get_speed_ms(),
            'knee_angle': self.detector.last_knee_angle,
            'symmetry': last_symmetry,
            'last_sit_to_stand_time': last_sts_time,
            'current_test': self.current_test,
            'inactivity_sec': inactivity
    }
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

   
       

    def _shutdown(self):
        # ── 1. Salva tutti i dati giornalieri nel DB ──
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
        
        for result in self.event_buffer.get_all_results():
            self._save_event_snapshot(result)
       

        # ── 2. Aggregazione e analisi trend ──
        print("\n" + "=" * 50)
        print("RIEPILOGO AGGREGATO")
        print("=" * 50)

        weekly = self.aggregator.get_weekly_summary()
        trend = self.aggregator.get_trend_analysis()

        # Stampa riepilogo settimanale
        if weekly["days_monitored"] > 0:
            avg = weekly["daily_averages"]
            print(f"\n--- Settimana ({weekly['period_start']} → {weekly['period_end']}) ---")
            print(f"  Giorni monitorati: {weekly['days_monitored']}")
            print(f"  Cammino medio/giorno: {avg.get('avg_time_walking_min') or 0:.1f} min")
            print(f"  Alzate medie/giorno:  {avg.get('avg_num_sit_to_stand') or 0:.1f}")
            print(f"  Indipendenza media:   {avg.get('avg_independence_score') or 0:.0f}%")

            gait = weekly.get("gait", {})
            if gait.get("avg_speed_ms"):
                print(f"  Velocità cammino:     {gait['avg_speed_ms']:.2f} m/s")
            if gait.get("avg_cadence"):
                print(f"  Cadenza:              {gait['avg_cadence']:.0f} passi/min")
            if gait.get("avg_symmetry"):
                print(f"  Simmetria:            {gait['avg_symmetry']:.0f}%")

        # Stampa trend
        if trend.get("status") == "ok":
            print(f"\n--- Trend vs settimana precedente ---")
            for key, entry in trend["trends"].items():
                if entry["previous"] > 0:
                    symbol = "↑" if entry["change_pct"] > 0 else "↓" if entry["change_pct"] < 0 else "="
                    interp = entry["interpretation"].upper()
                    print(f"  {entry['label']}: {entry['current']:.1f} "
                          f"{symbol} {abs(entry['change_pct']):.0f}% [{interp}]")
        else:
            print(f"\n  Trend: {trend.get('message', 'dati insufficienti')}")

        # Stampa alert
        alerts = trend.get("alerts", []) or weekly.get("alerts", [])
        if alerts:
            print(f"\n--- Alert ({len(alerts)}) ---")
            for a in alerts:
                tag = "[WARN]" if a["level"] == "warning" else "[CRIT]" if a["level"] == "critical" else "[OK]" if a["level"] == "positive" else "[INFO]"
                print(f"  {tag} {a['message']}")

        print("=" * 50)

        cv2.destroyAllWindows()
        print("Sistema chiuso, dati salvati.")
    def _save_event_snapshot(self, result):
        """Salva un risultato dell'analisi VLM evento nel DB."""
        snapshot_data = {
            'date': self.daily_monitor.date,
            'timestamp': result.get('timestamp'),
            'description': result.get('description'),
            'parsed_json': None,
            'context_json': json.dumps(result.get('context')) if result.get('context') else None,
            'state': result.get('context', {}).get('event', 'event_analysis')
        }
        self.db.save_context_snapshot(snapshot_data)

if __name__ == "__main__":
    system = MonitoringSystem(
        monitor_index=1,
        display_width=1280
    )
    start = time.time()
    system.run()
    elapsed = time.time() - start
    print(f"Tempo di processamento: {elapsed:.1f} secondi")