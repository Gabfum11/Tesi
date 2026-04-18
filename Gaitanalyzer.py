"""
GaitAnalyzer - Analisi dell'andatura multi-segnale.

Tre segnali per il rilevamento passi, usati in ordine di priorita':
1. Caviglie Y (vista laterale) — il piu' preciso, separa dx/sx
2. Distanza inter-caviglia (qualsiasi vista) — robusto, non separa dx/sx
3. Oscillazione verticale anca (qualsiasi vista) — fallback universale

Il sistema sceglie automaticamente il segnale migliore in base
alla qualita' dei dati raccolti.
"""

import math
import statistics
from collections import deque


class GaitAnalyzer:
    def __init__(self, fps):
        self.fps = fps
        self.reset()

    def reset(self):
        """Reset per un nuovo episodio di cammino."""
        # Segnale primario: caviglie Y (vista laterale)
        self.left_ankle_y = []
        self.right_ankle_y = []
        # Segnale alternativo: posizione X caviglie (per distanza inter-caviglia)
        self.left_ankle_x = []
        self.right_ankle_x = []
        # Segnale fallback: posizione anca
        self.hip_x = []
        self.hip_y = []
        # Oscillazione tronco: punto medio spalle
        self.shoulder_mid_x = []
        self.shoulder_mid_y = []

        self.timestamps = []
        self.frame_count = 0
        self.active = False
        self.signal_used = None

    def start(self):
        """Inizia a raccogliere dati per un episodio."""
        self.reset()
        self.active = True

    # Soglie durata per validita' analisi
    MIN_DURATION_SEC = 3.0
    RELIABLE_DURATION_SEC = 5.0
    HIGH_RELIABLE_SEC = 10.0
    MIN_STEPS_RELIABLE = 6
    MIN_STEPS_HIGH = 10

    # Soglia per considerare un segnale "piatto" (inutilizzabile)
    FLAT_SIGNAL_THRESHOLD = 15

    def stop(self):
        """Ferma la raccolta e restituisce le metriche."""
        self.active = False
        if len(self.timestamps) < 2:
            return None
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration < self.MIN_DURATION_SEC:
            return None
        return self.analyze(duration)

    def update(self, left_ankle_y, right_ankle_y, timestamp=None,
               hip_x=None, hip_y=None,
               left_ankle_x=None, right_ankle_x=None,
               shoulder_mid_x=None, shoulder_mid_y=None):
        """
        Chiamata ogni frame durante il cammino.

        Parametri obbligatori (retrocompatibili):
            left_ankle_y, right_ankle_y

        Parametri opzionali (per multi-segnale e oscillazione tronco):
            hip_x, hip_y, left_ankle_x, right_ankle_x,
            shoulder_mid_x, shoulder_mid_y
        """
        if not self.active:
            return

        self.left_ankle_y.append(left_ankle_y)
        self.right_ankle_y.append(right_ankle_y)
        self.timestamps.append(timestamp or self.frame_count / self.fps)

        if hip_x is not None:
            self.hip_x.append(hip_x)
        if hip_y is not None:
            self.hip_y.append(hip_y)
        if left_ankle_x is not None:
            self.left_ankle_x.append(left_ankle_x)
        if right_ankle_x is not None:
            self.right_ankle_x.append(right_ankle_x)
        if shoulder_mid_x is not None:
            self.shoulder_mid_x.append(shoulder_mid_x)
        if shoulder_mid_y is not None:
            self.shoulder_mid_y.append(shoulder_mid_y)

        self.frame_count += 1

    # =========================================
    # ANALISI: SELEZIONE AUTOMATICA SEGNALE
    # =========================================
    def analyze(self, duration_sec):
        """
        Prova i tre segnali in ordine di priorita':
        1. Caviglie Y (separa dx/sx, il piu' informativo)
        2. Distanza inter-caviglia (robusto, non separa)
        3. Hip Y (fallback universale, non separa)
        """
        if len(self.left_ankle_y) < self.fps:
            return None

        result = self._try_ankle_y(duration_sec)
        if result is None:
            result = self._try_inter_ankle_distance(duration_sec)
        if result is None:
            result = self._try_hip_y(duration_sec)

        return result

    # =========================================
    # SEGNALE 1: CAVIGLIE Y (vista laterale)
    # =========================================
    def _try_ankle_y(self, duration_sec):
        """
        Massimi locali nella Y delle caviglie.
        Funziona con vista laterale. Separa dx/sx.
        """
        left_smooth = self._smooth(self.left_ankle_y)
        right_smooth = self._smooth(self.right_ankle_y)

        left_smooth, right_smooth = self._trim_flat_tail(
            left_smooth, right_smooth,
            window=max(3, int(self.fps * 0.4)), threshold=2
        )

        min_frames = max(10, int(self.fps * 1.0))
        if len(left_smooth) < min_frames:
            return None

        left_range = max(left_smooth) - min(left_smooth)
        right_range = max(right_smooth) - min(right_smooth)

        if left_range < self.FLAT_SIGNAL_THRESHOLD and right_range < self.FLAT_SIGNAL_THRESHOLD:
            return None

        left_steps = self._find_steps(left_smooth)
        right_steps = self._find_steps(right_smooth)

        if len(left_steps) < 2 and len(right_steps) < 2:
            return None
        self.signal_used = "ankle_y"

        return self._build_result_bilateral(
            left_smooth, right_smooth,
            left_steps, right_steps,
            duration_sec
        )

    # =========================================
    # SEGNALE 2: DISTANZA INTER-CAVIGLIA
    # =========================================
    def _try_inter_ankle_distance(self, duration_sec):
        """
        Distanza euclidea tra caviglia sx e dx.
        Oscilla ad ogni passo da qualsiasi angolazione.
        Non separa dx/sx.
        """
        if not self.left_ankle_x or not self.right_ankle_x:
            return None
        if len(self.left_ankle_x) != len(self.left_ankle_y):
            return None

        dist_signal = []
        for i in range(len(self.left_ankle_x)):
            dx = self.left_ankle_x[i] - self.right_ankle_x[i]
            dy = self.left_ankle_y[i] - self.right_ankle_y[i]
            dist_signal.append(math.sqrt(dx * dx + dy * dy))

        smooth = self._smooth(dist_signal)
        smooth = self._trim_single_signal(smooth)

        if len(smooth) < max(10, int(self.fps * 1.0)):
            return None

        if max(smooth) - min(smooth) < self.FLAT_SIGNAL_THRESHOLD:
            return None

        steps = self._find_steps(smooth)

        if len(steps) < 4:
            return None

        self.signal_used = "inter_ankle_distance"

        return self._build_result_unified(smooth, steps, duration_sec)

    # =========================================
    # SEGNALE 3: HIP Y (fallback universale)
    # =========================================
    def _try_hip_y(self,duration_sec):
        """
        Oscillazione verticale dell'anca (2-4 cm per passo).
        Visibile da qualsiasi angolazione.
        Segnale piu' piccolo ma universale.
        """
        if not self.hip_y:
            return None

        smooth = self._smooth(self.hip_y, window=max(3, int(self.fps * 0.4)))
        smooth = self._trim_single_signal(smooth)

        if len(smooth) < max(10, int(self.fps * 1.0)):
            return None

        if max(smooth) - min(smooth) < 2:
            return None

        steps = self._find_steps(smooth, min_distance=max(3, int(self.fps * 0.35)))

        if len(steps) < 4:
            return None

        self.signal_used = "hip_y"

        return self._build_result_unified(smooth, steps, duration_sec)

    # =========================================
    # RISULTATO BILATERALE (dx/sx separati)
    # =========================================
    def _build_result_bilateral(self, left_smooth, right_smooth,
                                 left_steps, right_steps, duration_sec):
        total_steps = len(left_steps) + len(right_steps)
        cadence = (total_steps / duration_sec) * 60

        left_intervals = self._step_intervals(left_steps)
        right_intervals = self._step_intervals(right_steps)

        symmetry_result = self._calc_symmetry(
            left_smooth, right_smooth,
            left_steps, right_steps,
            left_intervals, right_intervals
        )

        regularity = self._calc_regularity(left_steps, right_steps)
        reliability = self._calc_reliability(duration_sec, total_steps,
                                              left_steps, right_steps)
        trunk_sway = self._calc_trunk_sway()

        return {
            'cadence': round(cadence, 1),
            'symmetry': round(symmetry_result['combined'], 1),
            'symmetry_temporal': round(symmetry_result['temporal'], 1),
            'symmetry_amplitude': round(symmetry_result['amplitude'], 1),
            'symmetry_count': round(symmetry_result['count'], 1),
            'regularity': round(regularity, 1),
            'trunk_sway_lateral': trunk_sway['lateral'],
            'trunk_sway_vertical': trunk_sway['vertical'],
            'total_steps': total_steps,
            'left_steps': len(left_steps),
            'right_steps': len(right_steps),
            'avg_step_time_left': round(statistics.mean(left_intervals), 3) if left_intervals else None,
            'avg_step_time_right': round(statistics.mean(right_intervals), 3) if right_intervals else None,
            'duration_sec': round(duration_sec, 2),
            'reliability': reliability['level'],
            'reliability_score': reliability['score'],
            'reliability_details': reliability['details'],
            'signal_used': self.signal_used
        }

    # =========================================
    # RISULTATO UNIFICATO (senza dx/sx)
    # =========================================
    def _build_result_unified(self, signal, steps, duration_sec):
        """
        Quando non possiamo separare dx/sx.
        Cadenza e regolarita' ok. Simmetria = None.
        """
        total_steps = len(steps)
        cadence = (total_steps / duration_sec) * 60

        intervals = self._step_intervals(steps)
        intervals = [dt for dt in intervals if 0.25 <= dt <= 1.5]
        if len(intervals) >= 2:
            mean_int = statistics.mean(intervals)
            if mean_int > 0:
                cv = statistics.stdev(intervals) / mean_int
                regularity = max(0, (1 - cv) * 100)
            else:
                regularity = 0
        else:
            regularity = 100.0

        reliability = self._calc_reliability(duration_sec, total_steps, steps, [])

        # Penalizza: segnale alternativo = meno informativo
        adj_score = max(0, reliability['score'] - 15)
        if adj_score >= 80:
            adj_level = "high"
        elif adj_score >= 50:
            adj_level = "medium"
        else:
            adj_level = "low"

        details = reliability['details']
        if "analisi completa" in details:
            details = f"segnale: {self.signal_used}"
        else:
            details += f"; segnale: {self.signal_used}"

        trunk_sway = self._calc_trunk_sway()

        return {
            'cadence': round(cadence, 1),
            'symmetry': None,
            'symmetry_temporal': None,
            'symmetry_amplitude': None,
            'symmetry_count': None,
            'regularity': round(regularity, 1),
            'trunk_sway_lateral': trunk_sway['lateral'],
            'trunk_sway_vertical': trunk_sway['vertical'],
            'total_steps': total_steps,
            'left_steps': None,
            'right_steps': None,
            'avg_step_time_left': None,
            'avg_step_time_right': None,
            'duration_sec': round(duration_sec, 2),
            'reliability': adj_level,
            'reliability_score': adj_score,
            'reliability_details': details,
            'signal_used': self.signal_used
        }

    # =========================================
    # TRIM
    # =========================================
    def _trim_single_signal(self, signal, window=None, threshold=2):
        if window is None:
            window = max(3, int(self.fps * 0.4))
        end = len(signal)
        for i in range(len(signal) - 1, window, -1):
            seg = signal[i - window:i]
            if max(seg) - min(seg) < threshold:
                end = i - window
            else:
                break
        return signal[:end]

    def _trim_flat_tail(self, left, right, window=5, threshold=2):
        end = len(left)
        for i in range(len(left) - 1, window, -1):
            seg_l = left[i - window:i]
            seg_r = right[i - window:i]
            if (max(seg_l) - min(seg_l) < threshold and
                    max(seg_r) - min(seg_r) < threshold):
                end = i - window
            else:
                break
        return left[:end], right[:end]

    # =========================================
    # SIMMETRIA
    # =========================================
    def _calc_symmetry(self, left, right, left_steps, right_steps,
                       left_intervals, right_intervals):
        temporal = self._symmetry_temporal(left_intervals, right_intervals)
        amplitude = self._symmetry_amplitude(left, right, left_steps, right_steps)
        count = self._symmetry_count(len(left_steps), len(right_steps))

        components = []
        if temporal is not None:
            components.append(('temporal', temporal, 0.50))
        if amplitude is not None:
            components.append(('amplitude', amplitude, 0.30))
        if count is not None:
            components.append(('count', count, 0.20))

        if not components:
            return {'combined': 100.0, 'temporal': 100.0,
                    'amplitude': 100.0, 'count': 100.0}

        total_weight = sum(w for _, _, w in components)
        combined = sum(val * (w / total_weight) for _, val, w in components)

        return {
            'combined': combined,
            'temporal': temporal if temporal is not None else 100.0,
            'amplitude': amplitude if amplitude is not None else 100.0,
            'count': count if count is not None else 100.0
        }

    def _symmetry_temporal(self, left_intervals, right_intervals):
        if not left_intervals or not right_intervals:
            return None
        avg_left = statistics.mean(left_intervals)
        avg_right = statistics.mean(right_intervals)
        if max(avg_left, avg_right) == 0:
            return 100.0
        return (min(avg_left, avg_right) / max(avg_left, avg_right)) * 100

    def _symmetry_amplitude(self, left, right, left_steps, right_steps):
        left_amp = self._avg_amplitude(left, left_steps)
        right_amp = self._avg_amplitude(right, right_steps)
        if left_amp == 0 and right_amp == 0:
            return 100.0
        if max(left_amp, right_amp) == 0:
            return 0.0
        return (min(left_amp, right_amp) / max(left_amp, right_amp)) * 100

    def _symmetry_count(self, n_left, n_right):
        if n_left == 0 and n_right == 0:
            return None
        if max(n_left, n_right) == 0:
            return 0.0
        return (min(n_left, n_right) / max(n_left, n_right)) * 100

    # =========================================
    # OSCILLAZIONE TRONCO
    # =========================================
    def _calc_trunk_sway(self):
        """
        Calcola l'oscillazione del tronco durante il cammino.
        
        Usa il punto medio delle spalle (landmark 11 e 12).
        
        Laterale (X): la deviazione standard della posizione X
          delle spalle rispetto alla traiettoria di cammino.
          Non usa la std grezza (che includerebbe lo spostamento
          lineare), ma la std dei residui dopo aver rimosso
          il trend lineare (la persona si sposta camminando).
          
        Verticale (Y): stessa logica sulla Y.
          L'oscillazione verticale e' naturale (2-4cm per passo),
          ma se eccessiva indica instabilita'.
        
        I valori sono in pixel. Se pixels_per_meter e' disponibile
        nel sistema, possono essere convertiti in cm a valle.
        
        Interpretazione clinica:
          - Laterale alta → problemi di equilibrio, rischio caduta
          - Verticale alta → andatura "a papera", debolezza muscolare
          - Entrambe alte → instabilita' generale
        
        Ritorna {'lateral': float|None, 'vertical': float|None}
        """
        result = {'lateral': None, 'vertical': None}
        
        min_samples = max(8, int(self.fps * 1.0))
        
        if len(self.shoulder_mid_x) < min_samples:
            return result
        
        result['lateral'] = self._detrended_std(self.shoulder_mid_x)
        
        if len(self.shoulder_mid_y) >= min_samples:
            result['vertical'] = self._detrended_std(self.shoulder_mid_y)
        
        return result
    
    def _detrended_std(self, signal):
        """
        Deviazione standard dopo aver rimosso il trend con media mobile.
        
        Usa una media mobile lenta (finestra ~2 secondi) come trend.
        I residui catturano solo le oscillazioni rapide passo-passo.
        
        Funziona con percorsi curvi (la persona gira intorno ai mobili)
        dove la regressione lineare fallirebbe.
        
        Esempio: persona cammina in curva per 10 secondi.
          - std con trend lineare: 150px (la curva non e' una retta)
          - std con media mobile: 4px (solo l'oscillazione reale)
        """
        n = len(signal)
        if n < 3:
            return None
        
        # Finestra di circa 2 secondi: abbastanza lenta da seguire
        # la traiettoria, abbastanza veloce da non mangiare l'oscillazione
        window = max(5, int(self.fps * 2.0))
        # Se la finestra e' piu' grande del segnale, usa tutto
        if window >= n:
            window = n - 1 if n > 1 else 1
        
        # Calcola la media mobile (trend lento)
        half = window // 2
        trend = []
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            trend.append(sum(signal[start:end]) / (end - start))
        
        # Residui: segnale - trend
        residuals = [signal[i] - trend[i] for i in range(n)]
        
        if len(residuals) < 2:
            return 0.0
        
        return round(statistics.stdev(residuals), 2)

    # =========================================
    # AFFIDABILITA'
    # =========================================
    def _calc_reliability(self, duration_sec, total_steps,
                          left_steps, right_steps):
        score = 0
        details = []

        if duration_sec >= self.HIGH_RELIABLE_SEC:
            score += 40
        elif duration_sec >= self.RELIABLE_DURATION_SEC:
            t = (duration_sec - self.RELIABLE_DURATION_SEC)
            t /= (self.HIGH_RELIABLE_SEC - self.RELIABLE_DURATION_SEC)
            score += 20 + int(t * 20)
        else:
            t = (duration_sec - self.MIN_DURATION_SEC)
            t /= (self.RELIABLE_DURATION_SEC - self.MIN_DURATION_SEC)
            score += 5 + int(max(0, t) * 14)
            details.append(f"durata breve ({duration_sec:.1f}s)")

        if total_steps >= self.MIN_STEPS_HIGH:
            score += 35
        elif total_steps >= self.MIN_STEPS_RELIABLE:
            t = (total_steps - self.MIN_STEPS_RELIABLE)
            t /= (self.MIN_STEPS_HIGH - self.MIN_STEPS_RELIABLE)
            score += 15 + int(t * 20)
        else:
            score += max(5, total_steps * 2)
            details.append(f"pochi passi ({total_steps})")

        n_left = len(left_steps) if isinstance(left_steps, list) else 0
        n_right = len(right_steps) if isinstance(right_steps, list) else 0
        min_side = min(n_left, n_right)

        if min_side >= 4:
            score += 25
        elif min_side >= 2:
            score += 10 + (min_side - 2) * 5
        elif min_side >= 1:
            score += 5
            details.append("una gamba ha pochi passi rilevati")
        else:
            if n_left + n_right > 0:
                details.append("passi non separabili dx/sx")
            else:
                details.append("passi rilevati solo su una gamba")

        score = min(100, max(0, score))

        # Penalita' per FPS basso: sotto 10fps il segnale e' intrinsecamente meno affidabile
        if self.fps < 10:
            fps_penalty = int((10 - self.fps) * 2)  # 8fps = -4, 6fps = -8
            score = max(0, score - fps_penalty)
            details.append(f"fps basso ({self.fps:.0f})")

        if score >= 80:
            level = "high"
        elif score >= 50:
            level = "medium"
        else:
            level = "low"

        if not details:
            details.append("analisi completa")

        return {'level': level, 'score': score, 'details': "; ".join(details)}

    # =========================================
    # UTILITA' SEGNALE
    # =========================================
    def _smooth(self, signal, window=None):
        if window is None:
            # A basso FPS serve smooth piu' aggressivo per compensare il rumore
            # A 8fps: max(3, 3) = 3 — poco ma e' il massimo ragionevole
            # A 14fps: max(3, 5) = 5
            # A 30fps: max(3, 10) = 10
            window = max(3, int(self.fps * 0.35))
        if len(signal) < window:
            return signal[:]
        smoothed = []
        half = window // 2
        for i in range(len(signal)):
            start = max(0, i - half)
            end = min(len(signal), i + half + 1)
            smoothed.append(sum(signal[start:end]) / (end - start))
        return smoothed

    def _find_steps(self, signal, min_distance=None):
        if min_distance is None:
            # A 8fps: max(3, 3) = 3 frames = 375ms
            # A 14fps: max(3, 5) = 5 frames = 357ms
            # A 30fps: max(3, 10) = 10 frames = 333ms
            min_distance = max(3, int(self.fps * 0.45))
        if len(signal) < 3:
            return []

        sig_range = max(signal) - min(signal)
        if sig_range < 2:
            return []

        threshold = min(signal) + sig_range * 0.3

        # A basso FPS (sotto 12) controlla solo ±1 frame
        # A FPS normale controlla ±2 frame
        look_back = 2 if self.fps >= 12 else 1
        margin = look_back

        steps = []
        last_step = -min_distance

        for i in range(margin, len(signal) - margin):
            is_peak = True
            for offset in range(1, look_back + 1):
                if signal[i] <= signal[i - offset] or signal[i] <= signal[i + offset]:
                    is_peak = False
                    break

            if (is_peak and
                signal[i] > threshold and
                i - last_step >= min_distance):
                steps.append(i)
                last_step = i

        return steps

    def _avg_amplitude(self, signal, steps):
        if len(steps) < 2:
            return 0
        min_segment = 2 if self.fps < 12 else 3
        amplitudes = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            if end - start < min_segment:
                continue
            segment = signal[start:end]
            amplitudes.append(max(segment) - min(segment))
        return statistics.mean(amplitudes) if amplitudes else 0

    def _calc_regularity(self, left_steps, right_steps):
        all_steps = sorted(
            [(s, 'L') for s in left_steps] +
            [(s, 'R') for s in right_steps]
        )
        if len(all_steps) < 3:
            return 100.0

        # Intervalli tra passi consecutivi
        raw_intervals = []
        for i in range(1, len(all_steps)):
            idx_prev = all_steps[i-1][0]
            idx_curr = all_steps[i][0]
            dt = self.timestamps[idx_curr] - self.timestamps[idx_prev]
            raw_intervals.append(dt)
        # Filtra: un passo non puo' durare meno di 0.25s ne' piu' di 1.5s
        # A 8fps, 0.25s = 2 frame (sotto e' rumore)
        # 1.5s = pausa tra passi, non un passo
        intervals = [dt for dt in raw_intervals if 0.25 <= dt <= 1.5]

        if len(intervals) < 2:
            return 100.0

        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.0

        cv = statistics.stdev(intervals) / mean_interval
        return max(0, (1 - cv) * 100)

    def _step_intervals(self, steps):
        if len(steps) < 2:
            return []
        # Usa i timestamp reali invece di dividere per fps
        return [self.timestamps[steps[i]] - self.timestamps[steps[i-1]] 
                for i in range(1, len(steps))]

    def set_fps(self, fps):
        self.fps = fps