import cv2
import base64
import time
import requests
from datetime import datetime


class ContextAnalyzer:
    def __init__(self, api_key, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.enabled = bool(api_key)

        self.interval_sec = 300  # ogni 5 minuti
        self.last_analysis_time = 0
        self.snapshots = []

        self.prompt = (
            "Osserva questa immagine di monitoraggio domiciliare. "
            "NON descrivere la postura della persona (in piedi, seduta, ecc). "
            "Rispondi SOLO in questo formato:\n"
            "STANZA: [cucina/soggiorno/camera/bagno/corridoio/altro]\n"
            "ATTIVITA: [mangiare/cucinare/guardare TV/leggere/dormire/"
            "usare telefono/conversare/igiene personale/nessuna attività evidente]\n"
            "AUSILI: [nessuno/bastone/deambulatore/sedia a rotelle/"
            "si appoggia a mobili]\n"
            #"COMPAGNIA: [sola/con altra persona]\n"
            "RISCHI: [nessuno/ostacoli a terra/illuminazione scarsa/"
            "pavimento bagnato/tappeto/cavi]\n"
            "NOTE: [solo se noti qualcosa di rilevante, altrimenti 'nulla']"
        )

        if self.enabled:
            print(f"[CONTEXT] Attivo, intervallo {self.interval_sec // 60} min")

    def update(self, frame):
        """Chiamata ogni frame. Manda al LLM solo quando il timer scade."""
        if not self.enabled:
            return None

        if time.time() - self.last_analysis_time < self.interval_sec:
            return None

        self.last_analysis_time = time.time()

        try:
            img_b64 = self._frame_to_base64(frame)
            response = self._call_api(img_b64)

            snapshot = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': response
            }
            self.snapshots.append(snapshot)
            print(f"[CONTEXT] {response[:120]}...")
            return snapshot

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("[CONTEXT] Rate limit, salto")
            else:
                print(f"[CONTEXT] Errore HTTP: {e}")
            return None
        except requests.exceptions.ConnectionError:
            print("[CONTEXT] Nessuna connessione, salto")
            return None
        except Exception as e:
            print(f"[CONTEXT] Errore: {e}")
            return None

    def _call_api(self, img_base64):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }}
                ]
            }],
            "max_tokens": 200,
            "temperature": 0.2,
        }
        response = requests.post(self.url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _frame_to_base64(self, frame, quality=80):
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")

    def get_snapshots(self):
        return self.snapshots

    def set_interval(self, seconds):
        self.interval_sec = seconds