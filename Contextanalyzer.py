import cv2
import base64
import time
import json
import requests
from datetime import datetime


class ContextAnalyzer:
    def __init__(self, api_key, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.enabled = bool(api_key)

        self.interval_sec = 120  # ogni 2 minuti
        self.last_analysis_time = 0
        self.snapshots = []

        if self.enabled:
            print(f"[CONTEXT] Attivo, intervallo {self.interval_sec // 60} min")

    def update(self, frame, context=None):
        """
        Chiamata ogni frame. Manda al VLM solo quando il timer scade.
        
        context: dizionario opzionale con lo stato corrente del sistema.
        Se fornito, il prompt viene adattato alla situazione.
        
        Campi attesi in context:
            state: "SITTING" | "STANDING" | "WALKING" | "UNKNOWN"
            state_duration_sec: da quanto tempo e' in questo stato
            movement: valore di movimento corrente
            speed_ms: velocita' in m/s (None se non calibrato)
            knee_angle: angolo ginocchio corrente
            symmetry: simmetria ultimo episodio (None se non disponibile)
            last_sit_to_stand_time: durata ultima alzata in secondi (None)
            current_test: "TUG" | "STS" | None
            inactivity_sec: secondi di inattivita' corrente
        """
        if not self.enabled:
            return None

        if time.time() - self.last_analysis_time < self.interval_sec:
            return None

        self.last_analysis_time = time.time()

        try:
            prompt = self._build_prompt(context)
            img_b64 = self._frame_to_base64(frame)
            response = self._call_api(img_b64, prompt)

            # Prova a parsare come JSON
            parsed = self._parse_response(response)

            snapshot = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': response,
                'parsed': parsed,
                'context': context
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

    # =========================================
    # COSTRUZIONE PROMPT CONTESTUALE
    # =========================================
    def _build_prompt(self, context):
        """
        Costruisce un prompt specifico in base allo stato corrente.
        Se context e' None, usa il prompt generico.
        """
        if not context:
            return self._prompt_generic()

        state = context.get("state", "UNKNOWN")
        duration = context.get("state_duration_sec", 0)
        test = context.get("current_test")

        if test:
            return self._prompt_test(test, context)
        elif state == "WALKING":
            return self._prompt_walking(context)
        elif state == "SITTING":
            return self._prompt_sitting(duration, context)
        elif state == "STANDING":
            return self._prompt_standing(duration, context)
        else:
            return self._prompt_generic()

    def _prompt_walking(self, ctx):
        """Prompt durante il cammino."""
        parts = [
            "La persona sta camminando.",
            "Osserva l'immagine e rispondi SOLO in questo formato JSON:"
        ]

        # Aggiungi contesto metrico se disponibile
        hints = []
        sym = ctx.get("symmetry")
        if sym is not None and sym < 75:
            hints.append(
                f"Il sistema rileva asimmetria nel passo ({sym:.0f}%). "
                "Osserva se una gamba sembra piu' debole o rigida."
            )
        speed = ctx.get("speed_ms")
        if speed is not None and speed < 0.6:
            hints.append(
                f"La velocita' e' bassa ({speed:.2f} m/s). "
                "Osserva se il passo sembra incerto o trascinato."
            )
        if hints:
            parts.append(" ".join(hints))

        parts.append("""{
  "ausilio": "nessuno | bastone | deambulatore | appoggio_muro |altro,cosa?",
  "postura_tronco": "eretta | inclinata_avanti | inclinata_laterale",
  "ostacoli": ["lista di ostacoli visibili oppure vuota"],
  "illuminazione": "adeguata | scarsa",
  "passo": "sicuro | incerto | trascinato",
  "note": "eventuali osservazioni rilevanti oppure null"
}""")
        return "\n".join(parts)

    def _prompt_sitting(self, duration_sec, ctx):
        """Prompt durante la seduta. Piu' dettagliato se prolungata."""
        if duration_sec < 1800:  # meno di 30 minuti
            parts = [
                "La persona e' seduta.",
                "Osserva l'immagine e rispondi SOLO in questo formato JSON:",
                """{
  "attivita": "descrizione breve di cosa sta facendo",
  "postura": "corretta | scivolata | inclinata",
  "ausili_visibili": ["lista oppure vuota"],
  "note": null
}"""
            ]
        else:
            # Seduta prolungata
            ore = duration_sec / 3600
            parts = [
                f"La persona e' seduta da {ore:.1f} ore.",
                "Osserva attentamente l'immagine e rispondi SOLO in JSON:"
            ]
            if duration_sec > 7200:
                parts.append(
                    "ATTENZIONE: seduta prolungata oltre 2 ore. "
                    "Valuta se ci sono segni di disagio o malessere."
                )
            parts.append("""{
  "stato_coscienza": "sveglio_attivo | sveglio_inattivo | assopito",
  "attivita": "descrizione breve di cosa sta facendo",
  "postura": "corretta | scivolata | inclinata",
  "segni_disagio": "nessuno | descrizione",
  "ausili_visibili": ["lista oppure vuota"],
  "note": null
}""")
        return "\n".join(parts)

    def _prompt_standing(self, duration_sec, ctx):
        """Prompt durante la stazione eretta."""
        parts = [
            "La persona e' in piedi.",
            "Osserva l'immagine e rispondi SOLO in questo formato JSON:"
        ]

        # Se c'e' stata un'alzata recente lenta
        sts_time = ctx.get("last_sit_to_stand_time")
        if sts_time and sts_time > 3.0:
            parts.insert(1,
                f"Si e' alzata da poco con difficolta' ({sts_time:.1f}s). "
                "Osserva se si sta appoggiando a qualcosa per mantenersi in equilibrio."
            )

        parts.append("""{
  "equilibrio": "stabile | instabile | appoggiato",
  "appoggio_a": "nulla | mobile | muro | persona | altro",
  "attivita": "descrizione breve",
  "rischi_ambientali": ["lista oppure vuota"],
  "note": null
}""")
        return "\n".join(parts)

    def _prompt_test(self, test_type, ctx):
        """Prompt durante un test clinico (TUG o STS)."""
        if test_type == "TUG":
            return """La persona sta eseguendo il test TUG (Timed Up and Go).
Osserva l'immagine e rispondi SOLO in questo formato JSON:
{
  "uso_braccia": "si | no",
  "appoggio": "nessuno | braccioli | tavolo | accompagnatore",
  "espressione": "neutrale | sforzo | dolore",
  "stabilita": "stabile | oscillante | instabile",
  "note": null
}"""
        elif test_type == "STS":
            return """La persona sta eseguendo il test Sit-to-Stand.
Osserva l'immagine e rispondi SOLO in questo formato JSON:
{
  "uso_braccia": "si_per_spingersi | si_per_equilibrio | no",
  "espressione": "neutrale | sforzo | dolore",
  "tentativi_multipli": "si | no",
  "stabilita_in_piedi": "stabile | oscillante | instabile",
  "note": null
}"""
        return self._prompt_generic()

    def _prompt_generic(self):
        """Prompt generico quando non c'e' contesto."""
        return """Osserva questa immagine di monitoraggio domiciliare.
Rispondi SOLO in questo formato JSON:
{
  "attivita": "descrizione breve di cosa sta facendo la persona",
  "ausili_visibili": ["lista oppure vuota"],
  "rischi_ambientali": ["lista oppure vuota"],
  "note": null
}"""

    # =========================================
    # PARSING RISPOSTA
    # =========================================
    def _parse_response(self, response):
        """
        Prova a parsare la risposta come JSON.
        Se fallisce, restituisce None (la risposta grezza e' comunque salvata).
        """
        text = response.strip()

        # Rimuovi eventuale markdown ```json ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Trova il primo { e l'ultimo }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    # =========================================
    # CHIAMATA API
    # =========================================
    def _call_api(self, img_base64, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }}
                ]
            }],
            "max_tokens": 300,
            "temperature": 0.1,
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