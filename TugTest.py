import time
import math


class TUGTest:
    def __init__(self):
        self.active = False
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.phase = None
        self.start_hip_x = None
        self.start_hip_y = None
        self.prev_hip_x = None
        self.prev_hip_y = None
        self.max_distance = 0
        self.distance_walk = 500  # Default, verrà sovrascritto da pixels_per_meter
        
        # Spostamento cumulativo
        self.cumulative_distance = 0
        self.forward_distance = 0
        self.return_distance = 0
        self.turn_detected = False
        
        # Per rilevare inversione di direzione
        self.recent_deltas_x = []
        self.recent_deltas_y = []
        
        # Campi per DB
        self.phase_times = {
            'sit_to_stand': None,
            'walk_forward': None,
            'turn': None,
            'walk_back': None
        }
        self.phase_start_time = None
        self.movement_values = []
        self.knee_angles = []
    
    def start(self, hip_x, hip_y, pixels_per_meter=None):
        self.reset()
        
        # Usa calibrazione dal detector se disponibile
        if pixels_per_meter:
            self.distance_walk = pixels_per_meter * 3  # 3 metri in pixel
        
        self.active = True
        self.phase = "SIT_TO_STAND"
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.start_hip_x = hip_x
        self.start_hip_y = hip_y
        self.prev_hip_x = hip_x
        self.prev_hip_y = hip_y
    
    def update(self, state, hip_x, hip_y, movement=0, knee_angle=0):
        if not self.active:
            return self.phase
        
        # Traccia metriche
        if movement > 0:
            self.movement_values.append(movement)
        if knee_angle > 0:
            self.knee_angles.append(knee_angle)
        
        # Calcola spostamento dal frame precedente (X e Y)
        if self.prev_hip_x is not None and self.prev_hip_y is not None:
            delta_x = hip_x - self.prev_hip_x
            delta_y = hip_y - self.prev_hip_y
            delta = math.sqrt(delta_x**2 + delta_y**2)
            
            # Ignora micro-movimenti (jitter del tracking)
            if delta > 2:
                # Filtra salti della telecamera
                if delta > 50:
                    self.prev_hip_x = hip_x
                    self.prev_hip_y = hip_y
                    return self.phase
                
                self.cumulative_distance += delta
                
                self.recent_deltas_x.append(delta_x)
                self.recent_deltas_y.append(delta_y)
                if len(self.recent_deltas_x) > 15:
                    self.recent_deltas_x.pop(0)
                    self.recent_deltas_y.pop(0)
                
                if self.phase == "WALK_FORWARD":
                    self.forward_distance += delta
                elif self.phase == "WALK_BACK":
                    self.return_distance += delta
        
        self.prev_hip_x = hip_x
        self.prev_hip_y = hip_y
        
        # === TRANSIZIONI DI FASE ===
        
        # SIT_TO_STAND → WALK_FORWARD
        if self.phase == "SIT_TO_STAND" and state == "WALKING":
            self.phase_times['sit_to_stand'] = time.time() - self.phase_start_time
            self.phase = "WALK_FORWARD"
            self.phase_start_time = time.time()
            self.forward_distance = 0
        
        # WALK_FORWARD → TURN (basato su distanza calibrata)
        elif self.phase == "WALK_FORWARD" and self.forward_distance > self.distance_walk:
            self.phase_times['walk_forward'] = time.time() - self.phase_start_time
            self.phase = "TURN"
            self.phase_start_time = time.time()
            self.recent_deltas_x = []
            self.recent_deltas_y = []
        
        # TURN → WALK_BACK (rileva inversione di direzione)
        elif self.phase == "TURN" and self._detect_direction_change():
            self.phase_times['turn'] = time.time() - self.phase_start_time
            self.phase = "WALK_BACK"
            self.phase_start_time = time.time()
            self.return_distance = 0
        
        # WALK_BACK → FINISHED (stessa distanza dell'andata)
        elif self.phase == "WALK_BACK":
            if self.return_distance > self.distance_walk and (state == "SITTING" or knee_angle < 120):
                self.phase_times['walk_back'] = time.time() - self.phase_start_time
                self.phase = "FINISHED"
                self.end_time = time.time()
                self.active = False
        
        return self.phase
    
    def _detect_direction_change(self):
        """Rileva quando la persona ha invertito la direzione."""
        if len(self.recent_deltas_x) < 10:
            return False
        
        # Usa la componente dominante (X o Y) per rilevare l'inversione
        recent_avg_x = sum(self.recent_deltas_x[-10:]) / 10
        early_avg_x = sum(self.recent_deltas_x[:5]) / 5
        
        recent_avg_y = sum(self.recent_deltas_y[-10:]) / 10
        early_avg_y = sum(self.recent_deltas_y[:5]) / 5
        
        # Inversione su X
        if early_avg_x != 0 and recent_avg_x != 0:
            if (early_avg_x > 1 and recent_avg_x < -1) or (early_avg_x < -1 and recent_avg_x > 1):
                return True
        
        # Inversione su Y
        if early_avg_y != 0 and recent_avg_y != 0:
            if (early_avg_y > 1 and recent_avg_y < -1) or (early_avg_y < -1 and recent_avg_y > 1):
                return True
        
        return False
    
    def get_result(self):
        if not self.end_time:
            return None
        
        total_time = self.end_time - self.start_time
        avg_movement = sum(self.movement_values) / len(self.movement_values) if self.movement_values else 0
        avg_knee = sum(self.knee_angles) / len(self.knee_angles) if self.knee_angles else 0
        
        if total_time < 10:
            fall_risk = "LOW"
        elif total_time < 14:
            fall_risk = "MEDIUM"
        else:
            fall_risk = "HIGH"
        
        return {
            'total_time': round(total_time, 2),
            'sit_to_stand_time': round(self.phase_times['sit_to_stand'] or 0, 2),
            'walk_forward_time': round(self.phase_times['walk_forward'] or 0, 2),
            'turn_time': round(self.phase_times['turn'] or 0, 2),
            'walk_back_time': round(self.phase_times['walk_back'] or 0, 2),
            'forward_distance': round(self.forward_distance, 0),
            'return_distance': round(self.return_distance, 0),
            'total_distance': round(self.cumulative_distance, 0),
            'avg_movement': round(avg_movement, 2),
            'avg_knee_angle': round(avg_knee, 2),
            'fall_risk_level': fall_risk
        }
    
    def get_time(self):
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None