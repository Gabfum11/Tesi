"""
EventFrameBuffer — Buffer circolare di frame con analisi VLM per eventi.

Mantiene un buffer di frame a bassa frequenza (~2fps).
Quando un evento si completa (alzata, cammino, anomalia),
estrae i frame rilevanti e li invia a Ollama per una
descrizione narrativa in stile cartella clinica.

L'analisi avviene in un thread separato per non bloccare il loop principale.
"""

import cv2
import time
import base64
import json
import requests
import threading
from collections import deque
from datetime import datetime


class EventFrameBuffer:
    def __init__(self, model="qwen2.5vl:7b", ollama_url="http://localhost:11434",
                 buffer_seconds=30, capture_fps=2, max_frames_per_request=5,
                 frame_max_size=512):
        """
        Args:
            model: modello Ollama con supporto vision
            ollama_url: URL base di Ollama
            buffer_seconds: quanti secondi di frame mantenere
            capture_fps: frame catturati al secondo (non serve più di 2)
            max_frames_per_request: max frame inviati a Ollama per evento
            frame_max_size: lato massimo in pixel per le immagini inviate a Ollama
        """
        self.model = model
        self.ollama_url = ollama_url
        self.capture_fps = capture_fps
        self.max_frames = max_frames_per_request
        self.frame_max_size = frame_max_size

        # Buffer circolare: (timestamp, frame_jpg_bytes)
        buffer_size = buffer_seconds * capture_fps
        self._buffer = deque(maxlen=buffer_size)
        self._last_capture_time = 0
        self._capture_interval = 1.0 / capture_fps

        # Risultati analisi (thread-safe)
        self._results = deque(maxlen=50)
        self._lock = threading.Lock()

        # Evita analisi sovrapposte
        self._analyzing = False

        # Osservazione periodica (sostituisce il ContextAnalyzer)
        self._periodic_interval = 900  # ogni 15 minuti
        self._last_periodic = time.time()

    # =========================================
    # CATTURA FRAME
    # =========================================
    def update(self, frame, context=None):
        """Chiamare ogni frame del loop principale. Salva a ~2fps.
        Gestisce anche l'osservazione periodica.
        
        Args:
            frame: frame corrente
            context: dict con stato corrente (state, movement, ecc.)
        """
        now = time.time()
        
        # Cattura frame a frequenza ridotta
        if now - self._last_capture_time >= self._capture_interval:
            # Ridimensiona per ridurre il peso (800x600 → 512x384)
            h, w = frame.shape[:2]
            if max(h, w) > self.frame_max_size:
                scale = self.frame_max_size / max(h, w)
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small = frame
            _, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            self._buffer.append((now, jpg.tobytes()))
            self._last_capture_time = now

        # Osservazione periodica
        if now - self._last_periodic >= self._periodic_interval:
            self._last_periodic = now
            state = context.get('state', 'UNKNOWN') if context else 'UNKNOWN'
            self._periodic_observation(state, context)

    # =========================================
    # ESTRAZIONE FRAME PER FINESTRA TEMPORALE
    # =========================================
    def _get_frames(self, start_time, end_time, max_frames=None):
        """Estrae i frame nel range temporale, uniformemente distribuiti."""
        if max_frames is None:
            max_frames = self.max_frames

        # Filtra per finestra temporale
        candidates = [(t, jpg) for t, jpg in self._buffer
                       if start_time <= t <= end_time]

        if not candidates:
            return []

        # Sottocampiona uniformemente se troppi
        if len(candidates) > max_frames:
            step = len(candidates) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            candidates = [candidates[i] for i in indices]

        return candidates

    def _frames_to_base64(self, frames):
        """Converte lista di (timestamp, jpg_bytes) in lista di base64."""
        return [base64.b64encode(jpg).decode('utf-8') for _, jpg in frames]

    # =========================================
    # CHIAMATA OLLAMA
    # =========================================
    def _call_ollama(self, prompt, frames_b64):
        """Invia prompt + immagini a Ollama e ritorna la risposta."""
        try:
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": prompt + " /no_think",
                    "images": frames_b64
                }],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 250
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                msg = data.get("message", {})
                content = msg.get("content", "")
                thinking = msg.get("thinking", "")
                
                # Qwen3-VL ignora think:False e mette <think>...</think> nel content
                if content and "<think>" in content:
                    import re
                    # Caso 1: tag chiuso — prendi quello che c'è dopo
                    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    if cleaned:
                        return cleaned
                    # Caso 2: tag non chiuso — il modello ha esaurito i token nel ragionamento
                    # Non c'è risposta utile, ritorna None
                    print(f"[EVENT VLM] Modello ha esaurito i token nel ragionamento, risposta scartata")
                    return None
                
                if content:
                    return content
                elif thinking:
                    return thinking
                else:
                    print(f"[EVENT VLM] Risposta vuota: {str(data)[:300]}")
                    return None
            else:
                print(f"[EVENT VLM] Errore Ollama: {response.status_code}")
                return None

        except Exception as e:
            print(f"[EVENT VLM] Errore connessione: {e}")
            return None

    # =========================================
    # ANALISI IN BACKGROUND
    # =========================================
    def _analyze_async(self, event_type, prompt, start_time, end_time, context):
        """Lancia l'analisi in un thread separato."""
        if self._analyzing:
            print(f"[EVENT VLM] Analisi già in corso, skip {event_type}")
            return

        frames = self._get_frames(start_time, end_time)
        if not frames:
            print(f"[EVENT VLM] Nessun frame per {event_type}")
            return

        self._analyzing = True
        thread = threading.Thread(
            target=self._analyze_worker,
            args=(event_type, prompt, frames, context),
            daemon=True
        )
        thread.start()

    def _analyze_worker(self, event_type, prompt, frames, context):
        """Worker che gira nel thread di analisi."""
        try:
            frames_b64 = self._frames_to_base64(frames)
            n_frames = len(frames_b64)
            print(f"\n{'='*60}")
            print(f"[EVENT VLM] Invio a {self.model}")
            print(f"  Evento:  {event_type}")
            print(f"  Frame:   {n_frames}")
            print(f"  Context: {context}")
            print(f"  Prompt:  {prompt[:200]}...")
            print(f"{'='*60}")

            description = self._call_ollama(prompt, frames_b64)

            if description:
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'event_type': event_type,
                    'description': description.strip(),
                    'context': context,
                    'n_frames_analyzed': n_frames
                }

                with self._lock:
                    self._results.append(result)

                print(f"\n{'─'*60}")
                print(f"[EVENT VLM] Risposta per {event_type}:")
                print(f"{description.strip()}")
                print(f"{'─'*60}\n")
            else:
                print(f"[EVENT VLM] Nessuna risposta da Ollama per {event_type}")
        except Exception as e:
            print(f"[EVENT VLM] Errore worker: {e}")
        finally:
            self._analyzing = False

    # =========================================
    # TRIGGER EVENTI
    # =========================================
    def on_sit_to_stand(self, duration, rising_start_time):
        """Trigger dopo un'alzata completata."""
        # Finestra: da 2s prima dell'inizio a 1s dopo la fine
        start = rising_start_time - 2.0
        end = time.time() + 0.5

        label = "lenta" if duration > 3.0 else "normale"
        prompt = (
            f"Osserva questa sequenza di frame che mostra una persona che si alza da seduta. "
            f"Dati sensore: durata alzata {duration:.1f} secondi ({label}).\n\n"
            f"Descrivi brevemente come si è alzata: ha usato le mani per appoggiarsi? "
            f"Ha fatto fatica? Una volta in piedi era stabile?\n"
            f"Scrivi 2-3 frasi come se annotassi in una cartella clinica di monitoraggio domiciliare."
        )

        context = {
            'event': 'sit_to_stand',
            'duration_sec': round(duration, 2),
            'label': label
        }

        self._analyze_async("sit_to_stand", prompt, start, end, context)

    def on_walking_episode(self, episode):
        """Trigger alla fine di un episodio di cammino."""
        duration = episode.get('duration_sec', 0)
        if duration < 4.0:
            return  # episodi troppo corti non valgono l'analisi con il VLM

        # Ricostruisci i tempi dalla stringa ISO
        try:
            start_dt = datetime.fromisoformat(episode['start_time'])
            start = start_dt.timestamp() - 1.0
        except (KeyError, ValueError):
            start = time.time() - duration - 1.0
        end = time.time() + 0.5

        # Costruisci la descrizione dei dati sensore
        sensor_parts = [f"durata {duration:.1f}s"]
        if episode.get('avg_speed_ms'):
            sensor_parts.append(f"velocità {episode['avg_speed_ms']:.2f}m/s")
        if episode.get('symmetry') is not None:
            sensor_parts.append(f"simmetria {episode['symmetry']}%")
        if episode.get('cadence'):
            sensor_parts.append(f"cadenza {episode['cadence']} passi/min")
        sensor_str = ", ".join(sensor_parts)

        prompt = (
            f"Osserva questa sequenza di frame che mostra una persona che cammina. "
            f"Dati sensore: {sensor_str}.\n\n"
            f"Descrivi brevemente l'andatura: era fluida o incerta? "
            f"Si appoggiava a qualcosa? Zoppicava? Usava ausili?\n"
            f"Scrivi 2-3 frasi come se annotassi in una cartella clinica di monitoraggio domiciliare."
        )

        context = {
            'event': 'walking_episode',
            'duration_sec': duration,
            'speed_ms': episode.get('avg_speed_ms'),
            'symmetry': episode.get('symmetry'),
            'cadence': episode.get('cadence')
        }

        self._analyze_async("walking_episode", prompt, start, end, context)

    def on_rapid_transitions(self, n_transitions, time_window):
        """Trigger quando ci sono troppe transizioni SITTING↔STANDING in poco tempo."""
        start = time.time() - time_window - 1.0
        end = time.time() + 0.5

        prompt = (
            f"Osserva questa sequenza di frame. Il sistema ha rilevato {n_transitions} "
            f"transizioni seduto/in piedi in {time_window:.0f} secondi.\n\n"
            f"Descrivi cosa sta succedendo: la persona sta provando ad alzarsi senza riuscirci? "
            f"Ricade? Mostra segni di disagio o frustrazione?\n"
            f"Scrivi 2-3 frasi come se annotassi in una cartella clinica di monitoraggio domiciliare."
        )

        context = {
            'event': 'rapid_transitions',
            'n_transitions': n_transitions,
            'time_window_sec': round(time_window, 1)
        }

        self._analyze_async("rapid_transitions", prompt, start, end, context)

    def on_tracking_lost(self, gap_duration, last_state, last_movement):
        """Trigger dopo una perdita di tracking prolungata."""
        start = time.time() - gap_duration - 2.0
        end = time.time() + 0.5

        prompt = (
            f"Osserva questa sequenza di frame. Il sistema di rilevamento ha perso il "
            f"tracciamento della persona per {gap_duration:.1f} secondi. "
            f"Ultimo stato noto: {last_state}.\n\n"
            f"Descrivi cosa vedi: la persona è visibile? È in una posizione anomala? "
            f"È caduta? C'è qualcuno che la aiuta?\n"
            f"Scrivi 2-3 frasi come se annotassi in una cartella clinica di monitoraggio domiciliare."
        )

        context = {
            'event': 'tracking_lost',
            'gap_duration_sec': round(gap_duration, 1),
            'last_state': last_state,
            'last_movement': round(last_movement, 1)
        }

        self._analyze_async("tracking_lost", prompt, start, end, context)

    def on_prolonged_inactivity(self, inactivity_minutes, state):
        """Trigger dopo inattività prolungata."""
        # Per l'inattività, serve solo il frame corrente + qualcuno recente
        start = time.time() - 10.0
        end = time.time() + 0.5

        prompt = (
            f"Osserva questo frame. La persona è ferma da {inactivity_minutes:.0f} minuti "
            f"nello stato {state}.\n\n"
            f"Descrivi cosa vedi: è sveglia? Dorme? La postura è naturale? "
            f"Ci sono segni di malessere?\n"
            f"Scrivi 2-3 frasi come se annotassi in una cartella clinica di monitoraggio domiciliare."
        )

        context = {
            'event': 'prolonged_inactivity',
            'inactivity_min': round(inactivity_minutes, 1),
            'state': state
        }

        self._analyze_async("prolonged_inactivity", prompt, start, end, context)

    def _periodic_observation(self, state, context):
        """Osservazione periodica — sostituisce il ContextAnalyzer periodico."""
        start = time.time() - 5.0
        end = time.time() + 0.5

        # Costruisci un riepilogo del contesto
        ctx_parts = [f"stato attuale: {state}"]
        if context:
            if context.get('movement'):
                ctx_parts.append(f"movimento: {context['movement']:.1f}")
            if context.get('inactivity_sec') and context['inactivity_sec'] > 60:
                ctx_parts.append(f"fermo da {context['inactivity_sec']/60:.0f} minuti")
        ctx_str = ", ".join(ctx_parts)

        prompt = (
            f"Osserva questo frame di monitoraggio domiciliare. "
            f"Dati sensore: {ctx_str}.\n\n"
            f"Descrivi brevemente cosa sta facendo la persona, la sua postura "
            f"e l'ambiente circostante. Segnala qualsiasi cosa rilevante.\n"
            f"Scrivi 2-3 frasi come se annotassi in una cartella clinica di monitoraggio domiciliare."
        )

        obs_context = {
            'event': 'periodic_observation',
            'state': state
        }
        if context:
            obs_context.update(context)

        self._analyze_async("periodic_observation", prompt, start, end, obs_context)

    # =========================================
    # RACCOLTA RISULTATI
    # =========================================
    def get_results(self):
        """Ritorna e svuota i risultati delle analisi completate (thread-safe)."""
        with self._lock:
            results = list(self._results)
            self._results.clear()
        return results

    def get_all_results(self):
        """Ritorna tutti i risultati senza svuotare (per shutdown)."""
        with self._lock:
            return list(self._results)