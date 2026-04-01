"""
GaitAnalyzer - Analisi dell'andatura da oscillazioni verticali delle caviglie.

Raccoglie la posizione Y dei landmark 27 (caviglia sinistra) e 28 (caviglia destra)
durante gli episodi di cammino, rileva i passi come minimi locali, e calcola:
- Cadenza (passi/minuto)
- Simmetria destra/sinistra (0-100%)
- Regolarità del passo (0-100%)
"""

import statistics
from collections import deque


class GaitAnalyzer:
    def __init__(self, fps):
        self.fps = fps
        self.reset()

    def reset(self):
        """Reset per un nuovo episodio di cammino."""
        self.left_ankle_y = []   # posizioni Y caviglia sinistra
        self.right_ankle_y = []  # posizioni Y caviglia destra
        self.timestamps = []
        self.frame_count = 0
        self.active = False

    def start(self):
        """Inizia a raccogliere dati per un episodio."""
        self.reset()
        self.active = True

    def stop(self):
        """Ferma la raccolta e restituisce le metriche."""
        self.active = False
        if self.frame_count < 15:  # meno di 1 secondo, troppo poco
            return None
        return self.analyze()

    def update(self, left_ankle_y, right_ankle_y, timestamp=None):
        """Chiamata ogni frame durante il cammino."""
        if not self.active:
            return

        self.left_ankle_y.append(left_ankle_y)
        self.right_ankle_y.append(right_ankle_y)
        self.timestamps.append(timestamp or self.frame_count / self.fps)
        self.frame_count += 1

    def analyze(self):
        """Analizza i dati raccolti e restituisce le metriche."""
        if len(self.left_ankle_y) < self.fps:
            return None

        # Smooth del segnale per ridurre il rumore del tracking
        left_smooth = self._smooth(self.left_ankle_y, window=5)
        right_smooth = self._smooth(self.right_ankle_y, window=5)
        #left_range = max(left_smooth) - min(left_smooth)
        #right_range = max(right_smooth) - min(right_smooth)
        left_smooth, right_smooth = self._trim_flat_tail(left_smooth, right_smooth, window=5,threshold=2)
        if len(left_smooth) < 15:
            return None
        #if left_range < 15 and right_range < 15:
            #print(f"[GAIT] Range troppo basso (L:{left_range:.0f} R:{right_range:.0f}), dati non affidabili")
            #return None

        # Trova i passi (minimi locali = quando il piede tocca terra)
        left_steps = self._find_steps(left_smooth)
        right_steps = self._find_steps(right_smooth)
         # DEBUG: stampa il segnale
        print(f"[GAIT] Left range: {max(left_smooth)-min(left_smooth):.1f}px")
        print(f"[GAIT] Right range: {max(right_smooth)-min(right_smooth):.1f}px")
        print(f"[GAIT] Left values: {[round(v,1) for v in left_smooth[-20:]]}")
        print(f"[GAIT] Right values: {[round(v,1) for v in right_smooth[-20:]]}")
        print(f"[GAIT] Steps L:{left_steps} R:{right_steps}")

        if len(left_steps) < 2 and len(right_steps) < 2:
            return None

        duration_sec = len(left_smooth) / self.fps

        # === CADENZA (passi/minuto) ===
        total_steps = len(left_steps) + len(right_steps)
        cadence = (total_steps / duration_sec) * 60

        # === SIMMETRIA (confronto ampiezza oscillazione dx/sx) ===
        symmetry = self._calc_symmetry(left_smooth, right_smooth,
                                        left_steps, right_steps)

        # === REGOLARITÀ (costanza degli intervalli tra passi) ===
        regularity = self._calc_regularity(left_steps, right_steps)

        # Intervalli medi per gamba
        left_intervals = self._step_intervals(left_steps)
        right_intervals = self._step_intervals(right_steps)

        return {
            'cadence': round(cadence, 1),
            'symmetry': round(symmetry, 1),
            'regularity': round(regularity, 1),
            'total_steps': total_steps,
            'left_steps': len(left_steps),
            'right_steps': len(right_steps),
            'avg_step_time_left': round(statistics.mean(left_intervals), 3) if left_intervals else None,
            'avg_step_time_right': round(statistics.mean(right_intervals), 3) if right_intervals else None,
            'duration_sec': round(duration_sec, 2)
        }

    def _smooth(self, signal, window=5):
        """Media mobile per ridurre il rumore."""
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
        """
        Trova i minimi locali nel segnale Y della caviglia.
        Ogni minimo = il piede tocca terra (punto più basso dell'oscillazione).
        
        In coordinate video, Y cresce verso il basso,
        quindi il piede a terra = valore Y MASSIMO (picco, non valle).
        Cerchiamo i massimi locali.
        
        min_distance: distanza minima in frame tra due passi consecutivi
                      (a 30fps, 8 frame = ~0.27s, evita di contare rimbalzi)
        """
        if min_distance is None:
            min_distance = max(2, int(self.fps * 0.35))
        if len(signal) < 3:
            return []

        # Calcola la soglia: i picchi devono essere abbastanza pronunciati
        sig_range = max(signal) - min(signal)
        if sig_range < 3:  # segnale piatto, non sta camminando davvero
            return []

        threshold = min(signal) + sig_range * 0.3  # almeno 30% dell'ampiezza

        steps = []
        last_step = -min_distance

        for i in range(2, len(signal) - 2):
            # Massimo locale (piede a terra in coordinate video)
            if (signal[i] > signal[i-1] and
                signal[i] > signal[i+1] and
                signal[i] > signal[i-2] and
                signal[i] > signal[i+2] and
                signal[i] > threshold and
                i - last_step >= min_distance):
                steps.append(i)
                last_step = i

        return steps

    def _calc_symmetry(self, left, right, left_steps, right_steps):
        """
        Simmetria: confronta l'ampiezza media di oscillazione
        tra gamba sinistra e destra. 100% = perfettamente simmetrica.
        """
        left_amp = self._avg_amplitude(left, left_steps)
        right_amp = self._avg_amplitude(right, right_steps)

        if left_amp == 0 and right_amp == 0:
            return 100.0

        if max(left_amp, right_amp) == 0:
            return 0.0

        return (min(left_amp, right_amp) / max(left_amp, right_amp)) * 100

    def _avg_amplitude(self, signal, steps):
        """Calcola l'ampiezza media di oscillazione intorno ai passi."""
        if len(steps) < 2:
            return 0

        amplitudes = []
        for i in range(len(steps) - 1):
            # Tra due passi consecutivi, trova il minimo (piede in aria)
            start = steps[i]
            end = steps[i + 1]
            if end - start < 3:
                continue
            segment = signal[start:end]
            peak = max(segment)
            valley = min(segment)
            amplitudes.append(peak - valley)

        return statistics.mean(amplitudes) if amplitudes else 0

    def _calc_regularity(self, left_steps, right_steps):
        """
        Regolarità: quanto sono costanti gli intervalli tra passi.
        Usa il coefficiente di variazione (CV).
        100% = perfettamente regolare, 0% = caotico.
        """
        # Unisci tutti i passi e ordinali
        all_steps = sorted(
            [(s, 'L') for s in left_steps] +
            [(s, 'R') for s in right_steps]
        )

        if len(all_steps) < 3:
            return 100.0

        # Intervalli tra passi consecutivi (qualsiasi gamba)
        intervals = []
        for i in range(1, len(all_steps)):
            dt = (all_steps[i][0] - all_steps[i-1][0]) / self.fps
            if dt > 0.1:  # ignora passi troppo vicini
                intervals.append(dt)

        if len(intervals) < 2:
            return 100.0

        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.0

        std_interval = statistics.stdev(intervals)
        cv = std_interval / mean_interval  # coefficiente di variazione

        # Converti in percentuale: CV=0 → 100%, CV>=1 → 0%
        regularity = max(0, (1 - cv) * 100)
        return regularity

    def _step_intervals(self, steps):
        """Calcola gli intervalli in secondi tra passi consecutivi di una gamba."""
        if len(steps) < 2:
            return []
        return [(steps[i] - steps[i-1]) / self.fps for i in range(1, len(steps))]

    def set_fps(self, fps):
        self.fps = fps
    def _trim_flat_tail(self, left, right, window=5, threshold=2):
        """Rimuove i frame finali dove il segnale è piatto (persona ferma)."""
        end = len(left)
        for i in range(len(left) - 1, window, -1):
            segment_l = left[i - window:i]
            segment_r = right[i - window:i]
            range_l = max(segment_l) - min(segment_l)
            range_r = max(segment_r) - min(segment_r)
            if range_l < threshold and range_r < threshold:
                end = i - window
            else:
                break
        return left[:end], right[:end]