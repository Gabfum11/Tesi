import time
import math
import json
from datetime import date, datetime
from collections import deque
from Gaitanalyzer import GaitAnalyzer


class DailyMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resetta per una nuova giornata"""
        self.date = date.today().isoformat()
        self.start_time = time.time()

        # Contatori frame
        self.frames_total = 0
        self.frames_with_pose = 0
        self.warmup_frames = 20
        self.frame_counter = 0

        # Tempo per stato (in secondi)
        self.time_sitting = 0
        self.time_standing = 0
        self.time_walking = 0

        # Transizioni
        self.num_sit_to_stand = 0
        self.num_stand_to_sit = 0
        self.num_walkings = 0

        # Sit-to-stand timing
        self.sit_to_stand_times = []
        self.sit_to_stand_start = None
        self.slow_transition_threshold = 3.0  # secondi
        self.num_slow_transitions = 0

        # Rilevamento alzata
        self.knee_angle_increasing = []
        self._prev_knee = 0
        self.rising_frames = 0
        self.recent_velocities = deque(maxlen=60)  # ~2 secondi
        self.rising_in_progress = False
        self.stable_standing_frames = 0

        # Inattività
        self.longest_inactivity = 0
        self.current_inactivity_start = time.time()

        # Stato precedente
        self.prev_state = "SITTING"
        self.prev_time = time.time()

        # === EPISODI DI CAMMINO ===
        self.walking_episodes = []
        self.current_walk_start = None
        self.current_walk_distance_px = 0
        self.current_walk_speeds = []
        self.prev_hip_x = None
        self.prev_hip_y = None

        # === EVENTI CON TIMESTAMP ===
        self.events = []

        # === ANALISI ANDATURA ===
        self.gait_analyzer = GaitAnalyzer(fps=10)

        # === TIME SLOT (ogni 30 minuti) ===
        self.slot_duration = 1800  # 30 minuti in secondi
        self.slot_start = time.time()
        self.slot_sitting = 0
        self.slot_standing = 0
        self.slot_walking = 0
        self.slot_sit_to_stand = 0
        self.completed_slots = []
        #frame
        self._no_tracking_frames = 0
        self._had_tracking_gap = False

    def _log_event(self, event_type, duration=None, details=None):
        """Registra un evento con timestamp"""
        self.events.append({
            'date': self.date,
            'event_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'event_type': event_type,
            'duration_sec': round(duration, 2) if duration else None,
            'details': json.dumps(details) if details else None
        })

    def _find_rising_start(self, end_time):
        if not self.recent_velocities:
            return None
        
        # Trova il picco di velocità negativa (massima salita)
        peak_idx = 0
        peak_vel = 0
        for i, (t, vel) in enumerate(self.recent_velocities):
            if vel < peak_vel:
                peak_vel = vel
                peak_idx = i
        
        # Dal picco, risali all'indietro fino a trovare calma stabile
        start_idx = peak_idx
        calm_streak = 0
        
        for i in range(peak_idx, -1, -1):
            t, vel = self.recent_velocities[i]
            if abs(vel) < 2:
                calm_streak += 1
                if calm_streak >= 5:
                    start_idx = i + calm_streak
                    break
            else:
                calm_streak = 0
                start_idx = i
        
        return self.recent_velocities[start_idx][0]
    def update(self, state, movement, pose_detected, hip_y_velocity):
        """Chiamata ogni frame"""
        current_time = time.time()
        dt = current_time - self.prev_time

        self.frames_total += 1
        if pose_detected:
            self.frames_with_pose += 1
            self._no_tracking_frames = 0

        if not pose_detected:
            self._no_tracking_frames += 1
            
            # Chiudi episodio di cammino se era aperto
            if self._no_tracking_frames > 5 and self.current_walk_start is not None:
                self.track_walking("STANDING", None, None, None, 0, 0, None, None,None,None)
            
            # Resetta stato dopo troppi frame senza tracking
            if self._no_tracking_frames > 10:
                self.prev_state = "UNKNOWN"
            self.prev_time = current_time
            return

        # Accumula tempo per stato (totale giornaliero)
        if state == "SITTING":
            self.time_sitting += dt
        elif state == "STANDING":
            self.time_standing += dt
        elif state == "WALKING":
            self.time_walking += dt

        # Accumula tempo per slot corrente
        if state == "SITTING":
            self.slot_sitting += dt
        elif state == "STANDING":
            self.slot_standing += dt
        elif state == "WALKING":
            self.slot_walking += dt

        # Controlla se lo slot è finito
        if current_time - self.slot_start >= self.slot_duration:
            self.completed_slots.append({
                'date': self.date,
                'slot_start': datetime.fromtimestamp(self.slot_start).isoformat(),
                'slot_end': datetime.now().isoformat(),
                'time_sitting_sec': round(self.slot_sitting, 1),
                'time_standing_sec': round(self.slot_standing, 1),
                'time_walking_sec': round(self.slot_walking, 1),
                'num_sit_to_stand': self.slot_sit_to_stand
            })
            # Reset per il nuovo slot
            self.slot_start = current_time
            self.slot_sitting = 0
            self.slot_standing = 0
            self.slot_walking = 0
            self.slot_sit_to_stand = 0

        # Salta i primi frame (warmup)
        self.frame_counter += 1
        if self.frame_counter <= self.warmup_frames:
            self.prev_state = state
            self.prev_time = current_time
            return

        # Rileva cambi di stato
        if state != self.prev_state:
            self._handle_transition(self.prev_state, state, current_time)

        # === ALZATA ===
        self.recent_velocities.append((current_time, hip_y_velocity))
        if self.prev_state == "SITTING" and state in ("STANDING", "WALKING") and not self.rising_in_progress:
            self.rising_in_progress = True
            self.stable_standing_frames = 0

        if self.rising_in_progress:
            if state == "SITTING":
                self.rising_in_progress = False
                self.stable_standing_frames = 0
            elif abs(hip_y_velocity) < 2:
                self.stable_standing_frames += 1
            else:
                self.stable_standing_frames = 0

            if self.stable_standing_frames >= 5:
                stable_start_time = self.recent_velocities[-5][0] if len(self.recent_velocities) >= 5 else current_time
                start_time = self._find_rising_start(current_time)
                if start_time:
                    duration = stable_start_time - start_time
                    if duration > 0.3:  # ignora misurazioni troppo corte
                        self.sit_to_stand_times.append(duration)
                        if duration > self.slow_transition_threshold:
                            self.num_slow_transitions += 1
                        print(f"[SIT-TO-STAND] Durata: {duration:.2f}s")
                        self._log_event("SIT_TO_STAND_COMPLETE", duration=duration, details={
                            'duration_sec': round(duration, 2),
                            'slow': duration > self.slow_transition_threshold
                        })
                self.rising_in_progress = False
                self.stable_standing_frames = 0
                self.recent_velocities.clear()

        # Traccia inattività
        if movement > 2:
            inactivity = current_time - self.current_inactivity_start
            if inactivity > self.longest_inactivity:
                self.longest_inactivity = inactivity
            self.current_inactivity_start = current_time

        self.prev_state = state
        self.prev_time = current_time

    def track_walking(self, state, hip_x, hip_y, pixels_per_meter,
                      left_ankle_y, right_ankle_y, left_ankle_x,right_ankle_x, shoulder_mid_x, shoulder_mid_y):
        """Tracking episodi di cammino + analisi andatura."""
        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time else 0
        # --- Calcolo FPS reale stabile ---
        if not hasattr(self, '_fps_frame_count'):
            self._fps_frame_count = 0
            self._fps_start_time = current_time

        self._fps_frame_count += 1
        if self._fps_frame_count % 50 == 0:
            elapsed = current_time - self._fps_start_time
            if elapsed > 0:
                avg_fps = self._fps_frame_count / elapsed
                self.gait_analyzer.set_fps(avg_fps)

        if state == "WALKING":
            # Inizio nuovo episodio
            if self.current_walk_start is None:
                self.current_walk_start = current_time
                self.current_walk_distance_px = 0
                self.current_walk_speeds = []
                self.prev_hip_x = hip_x
                self.prev_hip_y = hip_y
                self.gait_analyzer.start()

            # Accumula distanza
            if hip_x is not None and self.prev_hip_x is not None:
                dx = hip_x - self.prev_hip_x
                dy = hip_y - self.prev_hip_y
                step_px = math.sqrt(dx**2 + dy**2)
                if step_px < 50:  # filtra salti tracking
                    self.current_walk_distance_px += step_px

                    # Velocità istantanea
                    if pixels_per_meter and dt > 0:
                        speed = (step_px / pixels_per_meter) / dt
                        if speed < 5:  # filtra valori assurdi
                            self.current_walk_speeds.append(speed)
            

            # Alimenta il GaitAnalyzer con le posizioni delle caviglie
            if left_ankle_y is not None and right_ankle_y is not None:
                #print(f"[DEBUG] Ankles Y: L={left_ankle_y:.1f}, R={right_ankle_y:.1f}")
                self.gait_analyzer.update(left_ankle_y, right_ankle_y, current_time,hip_x,hip_y, left_ankle_x,right_ankle_x, shoulder_mid_x, shoulder_mid_y)

            self.prev_hip_x = hip_x
            self.prev_hip_y = hip_y

        elif self.current_walk_start is not None:
            # Fine episodio: stato non è più WALKING
            duration = current_time - self.current_walk_start
            if duration > 1.0:  # ignora episodi < 1 secondo
                distance_m = (self.current_walk_distance_px / pixels_per_meter) if pixels_per_meter else None
                avg_speed = (sum(self.current_walk_speeds) / len(self.current_walk_speeds)) if self.current_walk_speeds else None

                # Analisi andatura
                gait = self.gait_analyzer.stop()
                #print(f"[GAIT DEBUG] frames: {self.gait_analyzer.frame_count}, result: {gait}")
                episode = {
                    'date': self.date,
                    'start_time': datetime.fromtimestamp(self.current_walk_start).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_sec': round(duration, 2),
                    'distance_m': round(distance_m, 2) if distance_m else None,
                    'avg_speed_ms': round(avg_speed, 3) if avg_speed else None,
                    'cadence': gait['cadence'] if gait else None,
                    'symmetry': gait['symmetry'] if gait else None,
                    'regularity': gait['regularity'] if gait else None,
                    'total_steps': gait['total_steps'] if gait else None,
                    'reliability_score': gait['reliability_score'] if gait else None,
                    'signal_used': gait['signal_used'] if gait else None
                }
                self.walking_episodes.append(episode)

                gait_str = ""
                if gait:
                   gait_str = ( f"  Cadenza: {gait['cadence']}p/min"
                                f"  Simm: {gait['symmetry']}%"
                                f"  Reg: {gait['regularity']}%"
                                f"  Passi: {gait['total_steps']}"
                                f"  Rel: {gait['reliability']}({gait['reliability_score']})"
                                f"  Sig: {gait['signal_used']}"
                                f"  SwayL: {gait['trunk_sway_lateral']}"
                                f"  SwayV: {gait['trunk_sway_vertical']}")
                self._log_event("WALK_END", duration=duration, details={
                    'duration_sec': round(duration, 2),
                    'distance_m': round(distance_m, 2) if distance_m else None,
                    'avg_speed_ms': round(avg_speed, 3) if avg_speed else None,
                    'gait': gait
                })
                print(f"[WALK] Durata: {duration:.1f}s"
                      + (f"  Speed: {avg_speed:.2f}m/s" if avg_speed else "")
                      + gait_str)
            else:
                self.gait_analyzer.stop()

            self.current_walk_start = None
            self.prev_hip_x = None
            self.prev_hip_y = None

    def _handle_transition(self, old_state, new_state, current_time):
        """Gestisce le transizioni di stato"""
        print(f"[TRANSIZIONE] {old_state} -> {new_state}")

        if old_state == "SITTING" and new_state != "SITTING":
            self.num_sit_to_stand += 1
            self.slot_sit_to_stand += 1
            self._log_event("SIT_TO_STAND")

        if old_state in ("STANDING", "WALKING") and new_state == "SITTING":
            self.num_stand_to_sit += 1
            self._log_event("STAND_TO_SIT")

        if new_state == "WALKING" and old_state != "WALKING":
            self.num_walkings += 1
            self._log_event("WALK_START")

    def get_summary(self):
        """Restituisce il riepilogo da salvare nel DB"""
        total_time = time.time() - self.start_time

        avg_sit_to_stand = 0
        max_sit_to_stand = 0
        if self.sit_to_stand_times:
            avg_sit_to_stand = sum(self.sit_to_stand_times) / len(self.sit_to_stand_times)
            max_sit_to_stand = max(self.sit_to_stand_times)

        active_time = self.time_standing + self.time_walking
        total_tracked = self.time_sitting + self.time_standing + self.time_walking
        independence_score = (active_time / total_tracked * 100) if total_tracked > 0 else 0

        final_inactivity = time.time() - self.current_inactivity_start
        if final_inactivity > self.longest_inactivity:
            self.longest_inactivity = final_inactivity

        # Chiudi l'ultimo slot parziale
        if self.slot_sitting + self.slot_standing + self.slot_walking > 0:
            self.completed_slots.append({
                'date': self.date,
                'slot_start': datetime.fromtimestamp(self.slot_start).isoformat(),
                'slot_end': datetime.now().isoformat(),
                'time_sitting_sec': round(self.slot_sitting, 1),
                'time_standing_sec': round(self.slot_standing, 1),
                'time_walking_sec': round(self.slot_walking, 1),
                'num_sit_to_stand': self.slot_sit_to_stand
            })

        return {
            'date': self.date,
            'total_monitoring_min': total_time,
            'time_sitting_min': self.time_sitting,
            'time_standing_min': self.time_standing,
            'time_walking_min': self.time_walking,
            'num_walkings': self.num_walkings,
            'num_sit_to_stand': self.num_sit_to_stand,
            'num_stand_to_sit': self.num_stand_to_sit,
            'avg_sit_to_stand_time': avg_sit_to_stand,
            'max_sit_to_stand_time': max_sit_to_stand,
            'num_slow_transitions': self.num_slow_transitions,
            'longest_inactivity_min': self.longest_inactivity,
            'independence_score': independence_score
        }