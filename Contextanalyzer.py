"""
ContextAnalyzer v2 — Analisi contestuale con Ollama (locale).

Rispetto alla v1:
  - Il LLM osserva liberamente la scena, non solo campi rigidi
  - Ha memoria: riceve le ultime osservazioni per notare cambiamenti
  - Trigger adattivi: si attiva su eventi significativi, non solo a timer
  - Sintesi periodiche: ogni 30 min produce un riassunto del blocco temporale
  - Timeline narrativa: ogni snapshot diventa un tassello della giornata

Uso:
    analyzer = ContextAnalyzer(model="gemma3:4b")
    
    # Ogni frame:
    snapshot = analyzer.update(frame, context)
    
    # Su eventi significativi:
    snapshot = analyzer.trigger(frame, context, reason="alzata_lenta")
    
    # I blocchi della timeline:
    analyzer.get_timeline_blocks()
"""

import cv2
import base64
import time
import json
import requests
from datetime import datetime
from collections import deque


class ContextAnalyzer:
    def __init__(self, model="gemma3:4b", ollama_url="http://localhost:11434"):
        self.model = model
        self.url = f"{ollama_url}/api/chat"
        self.ollama_url = ollama_url
        self.enabled = True

        # ── Timer ──
        self.interval_sec = 120  # osservazione periodica ogni 2 min
        self.last_analysis_time = 0
        self.synthesis_interval = 1800  # sintesi ogni 30 min
        self.last_synthesis_time = time.time()

        # ── Memoria ──
        self.recent_observations = deque(maxlen=10)  # ultime 10 osservazioni
        self.snapshots = []  # tutti gli snapshot della giornata
        self.timeline_blocks = []  # sintesi ogni 30 min
        self.current_block_snapshots = []  # snapshot del blocco corrente

        # ── Trigger adattivi ──
        self.trigger_cooldown = 30  # min 30s tra trigger consecutivi
        self.last_trigger_time = 0
        self.prev_state = None
        self.long_inactivity_alerted = False

        # ── Verifica Ollama ──
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(model in m for m in models):
                print(f"[CONTEXT] Modello '{model}' non trovato. Disponibili: {models}")
                print(f"[CONTEXT] Esegui: ollama pull {model}")
                self.enabled = False
            else:
                print(f"[CONTEXT] Ollama OK — {model}")
                print(f"[CONTEXT] Osservazione ogni {self.interval_sec}s, sintesi ogni {self.synthesis_interval // 60} min")
        except requests.exceptions.ConnectionError:
            print("[CONTEXT] Ollama non raggiungibile. Avvialo con: ollama serve")
            self.enabled = False

    # =========================================
    # UPDATE PERIODICO (chiamato ogni frame)
    # =========================================
    def update(self, frame, context=None):
        """
        Chiamata ogni frame. Analizza la scena quando:
        1. Il timer periodico scade (ogni interval_sec)
        2. Il blocco di sintesi scade (ogni synthesis_interval)
        """
        if not self.enabled:
            return None

        now = time.time()
        snapshot = None

        # Osservazione periodica
        if now - self.last_analysis_time >= self.interval_sec:
            self.last_analysis_time = now
            snapshot = self._observe(frame, context, trigger_reason=None)

        # Sintesi periodica
        if now - self.last_synthesis_time >= self.synthesis_interval:
            self._synthesize_block()
            self.last_synthesis_time = now

        return snapshot

    # =========================================
    # TRIGGER SU EVENTO (chiamato dal monitor)
    # =========================================
    def trigger(self, frame, context, reason):
        """
        Trigger su evento significativo. Chiamato dal MonitoringSystem
        quando succede qualcosa di clinicamente rilevante.

        reason: stringa che descrive l'evento, es:
            "alzata_lenta"         — sit-to-stand > 3s
            "inattivita_lunga"     — inattivita' > 60 min
            "simmetria_bassa"      — simmetria < 70% in un episodio
            "velocita_bassa"       — velocita' < 0.5 m/s
            "caduta_possibile"     — tracking perso improvvisamente
            "cambio_stato"         — transizione dopo lunga inattivita'
        """
        if not self.enabled:
            return None

        now = time.time()
        if now - self.last_trigger_time < self.trigger_cooldown:
            return None

        self.last_trigger_time = now
        return self._observe(frame, context, trigger_reason=reason)

    # =========================================
    # OSSERVAZIONE CORE
    # =========================================
    def _observe(self, frame, context, trigger_reason=None):
        """Esegue un'osservazione: prompt + chiamata LLM + salvataggio."""
        try:
            prompt = self._build_prompt(context, trigger_reason)
            img_b64 = self._frame_to_base64(frame)
            response = self._call_api(img_b64, prompt)
            print(f"[DEBUG RAW] {response}")
            # Parsing: estrai JSON strutturato + testo libero
            structured, free_text = self._parse_response_v2(response)

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            snapshot = {
                'timestamp': timestamp,
                'trigger': trigger_reason,
                'description': free_text or response,
                'structured': structured,
                'context': context,
                'raw_response': response
            }

            # Aggiorna memoria
            memory_entry = f"[{timestamp[-8:-3]}] {free_text or response[:150]}"
            if trigger_reason:
                memory_entry = f"[{timestamp[-8:-3]}] ⚡{trigger_reason}: {free_text or response[:120]}"
            self.recent_observations.append(memory_entry)

            # Salva
            self.snapshots.append(snapshot)
            self.current_block_snapshots.append(snapshot)

            label = f"⚡{trigger_reason}" if trigger_reason else "Osservazione periodica"
            print(f"[CONTEXT] {label} {(free_text or response)}")

            return snapshot

        except requests.exceptions.ConnectionError:
            print("[CONTEXT] Ollama non raggiungibile, salto")
            return None
        except Exception as e:
            print(f"[CONTEXT] Errore: {e}")
            return None

    # =========================================
    # COSTRUZIONE PROMPT
    # =========================================
    def _build_prompt(self, context, trigger_reason):
        """
        Prompt a due livelli:
        1. Parte strutturata minima (3 campi per aggregazione)
        2. Osservazione libera (il LLM descrive cosa vede)
        
        Con memoria delle osservazioni precedenti.
        """
        parts = []

        # ── Ruolo ──
        parts.append(
            "Sei un osservatore clinico in un sistema di monitoraggio domiciliare. "
            "Il tuo compito e' osservare l'immagine e descrivere cosa vedi, "
            "con attenzione agli aspetti rilevanti per la salute e la sicurezza della persona."
        )

        # ── Contesto numerico ──
        if context:
            state = context.get("state", "UNKNOWN")
            duration = context.get("state_duration_sec", 0)

            state_desc = {
                "SITTING": "seduta",
                "STANDING": "in piedi",
                "WALKING": "camminando"
            }.get(state, "stato non chiaro")

            parts.append(f"\nSTATO ATTUALE: la persona e' {state_desc}")

            if duration > 60:
                parts.append(f"da {duration / 60:.0f} minuti.")

            hints = []
            speed = context.get("speed_ms")
            if speed is not None and state == "WALKING":
                hints.append(f"velocita' {speed:.2f} m/s")
            sym = context.get("symmetry")
            if sym is not None and sym < 80:
                hints.append(f"simmetria passo {sym:.0f}%")
            knee = context.get("knee_angle")
            if knee and knee < 140:
                hints.append(f"angolo ginocchio {knee:.0f}°")
            sts = context.get("last_sit_to_stand_time")
            if sts and sts > 2.5:
                hints.append(f"ultima alzata in {sts:.1f}s")
            inact = context.get("inactivity_sec", 0)
            if inact > 1800:
                hints.append(f"inattiva da {inact / 60:.0f} minuti")

            if hints:
                parts.append(f"DATI SENSORI: {', '.join(hints)}.")

        # ── Motivo trigger ──
        if trigger_reason:
            trigger_descriptions = {
                "alzata_lenta": "La persona si e' appena alzata con difficolta' (tempo lungo). "
                                "Osserva se usa le braccia, se e' stabile, se mostra segni di sforzo.",
                "inattivita_lunga": "La persona e' ferma da molto tempo. "
                                    "Osserva se e' sveglia, se mostra segni di disagio, la postura.",
                "simmetria_bassa": "Il sistema ha rilevato un'andatura asimmetrica. "
                                   "Osserva se una gamba sembra piu' debole, se zoppica, se trascina un piede.",
                "velocita_bassa": "La persona cammina molto lentamente. "
                                  "Osserva se il passo e' incerto, se si appoggia a qualcosa.",
                "caduta_possibile": "Il tracking della persona e' stato perso improvvisamente. "
                                    "Osserva se la persona e' caduta, si e' accasciata, o e' uscita dal campo visivo.",
                "cambio_stato": "La persona ha appena cambiato attivita'. "
                                "Descrivi cosa sta facendo ora e come si e' mossa."
            }
            desc = trigger_descriptions.get(trigger_reason,
                f"Evento rilevato: {trigger_reason}. Osserva con attenzione.")
            parts.append(f"\n⚡ ATTENZIONE: {desc}")

        # ── Memoria: cosa hai visto prima ──
        if self.recent_observations:
            parts.append("\nLE TUE OSSERVAZIONI PRECEDENTI (in ordine cronologico):")
            for obs in self.recent_observations:
                parts.append(f"  {obs}")
            parts.append(
                "Tieni conto di queste osservazioni: nota se qualcosa e' cambiato, "
                "peggiorato, migliorato rispetto a prima."
            )

        # ── Istruzioni di risposta ──
        parts.append("""
RISPONDI con:
1. Un blocco JSON con questi 3 campi (per il database):
```json
{
  "ausilio": "nessuno | bastone | deambulatore | appoggio_muro | appoggio_mobile | aiuto_persona | altro",
  "rischio": "basso | medio | alto",
  "postura": "buona | accettabile | scorretta"
}
```

2. Subito dopo il JSON, scrivi una OSSERVAZIONE LIBERA (2-4 frasi).
   Descrivi cosa vedi nell'immagine: cosa fa la persona, come si muove,
   l'ambiente intorno, qualsiasi dettaglio clinicamente rilevante.
   Se noti cambiamenti rispetto alle osservazioni precedenti, segnalali.
   Scrivi in italiano, tono clinico ma chiaro.""")

        return "\n".join(parts)

    # =========================================
    # SINTESI PERIODICA (ogni 30 min)
    # =========================================
    def _synthesize_block(self):
        """
        Ogni 30 minuti prende gli snapshot del blocco e chiede al LLM
        di produrre un paragrafo riassuntivo. Questo diventa un tassello
        della timeline della giornata.
        """
        if not self.current_block_snapshots:
            return

        # Raccogli le osservazioni del blocco
        observations = []
        for snap in self.current_block_snapshots:
            ts = snap['timestamp'][-8:-3]  # HH:MM
            desc = snap.get('description', '')[:200]
            trigger = snap.get('trigger')
            if trigger:
                observations.append(f"  [{ts}] ⚡{trigger}: {desc}")
            else:
                observations.append(f"  [{ts}] {desc}")

        time_start = self.current_block_snapshots[0]['timestamp']
        time_end = self.current_block_snapshots[-1]['timestamp']
        n_obs = len(self.current_block_snapshots)

        prompt = f"""Sei un osservatore clinico. Ecco le osservazioni degli ultimi 30 minuti
di monitoraggio domiciliare ({time_start[-8:]} → {time_end[-8:]}, {n_obs} osservazioni):

{chr(10).join(observations)}

Scrivi un PARAGRAFO RIASSUNTIVO (3-5 frasi) di questo periodo.
Evidenzia: attivita' prevalente, eventuali criticita', cambiamenti osservati,
uso di ausili, livello di autonomia.
Scrivi in italiano, tono clinico. Non usare elenchi puntati."""

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300}
            }
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            synthesis = response.json()["message"]["content"].strip()

            block = {
                'time_start': time_start,
                'time_end': time_end,
                'num_observations': n_obs,
                'num_triggers': len([s for s in self.current_block_snapshots if s.get('trigger')]),
                'trigger_types': [s['trigger'] for s in self.current_block_snapshots if s.get('trigger')],
                'synthesis': synthesis,
                'snapshots': self.current_block_snapshots.copy()
            }
            self.timeline_blocks.append(block)
            print(f"[CONTEXT] 📝 Sintesi {time_start[-8:]}→{time_end[-8:]}: {synthesis[:80]}...")

        except Exception as e:
            print(f"[CONTEXT] Errore sintesi: {e}")

        # Reset blocco corrente
        self.current_block_snapshots = []

    # =========================================
    # SINTESI GIORNALIERA (per il report)
    # =========================================
    def generate_daily_narrative(self, daily_summary=None, walking_episodes=None):
        """
        Chiamata a fine giornata dal DailyReportGenerator.
        Prende tutti i blocchi della timeline e produce una sintesi
        narrativa dell'intera giornata.
        """
        if not self.timeline_blocks and not self.current_block_snapshots:
            return self._fallback_daily_narrative(daily_summary)

        # Chiudi l'ultimo blocco aperto
        if self.current_block_snapshots:
            self._synthesize_block()

        # Raccogli le sintesi dei blocchi
        block_summaries = []
        for b in self.timeline_blocks:
            time_range = f"{b['time_start'][-8:-3]}–{b['time_end'][-8:-3]}"
            triggers = ""
            if b['trigger_types']:
                triggers = f" (eventi: {', '.join(b['trigger_types'])})"
            block_summaries.append(f"  [{time_range}]{triggers} {b['synthesis']}")

        # Contesto numerico
        context_str = ""
        if daily_summary:
            s = daily_summary
            sit_min = (s.get('time_sitting_min') or 0) / 60
            walk_min = (s.get('time_walking_min') or 0) / 60
            n_sts = s.get('num_sit_to_stand') or 0
            n_slow = s.get('num_slow_transitions') or 0
            avg_sts = s.get('avg_sit_to_stand_time') or 0
            context_str = (
                f"\nDATI NUMERICI: seduta {sit_min:.0f} min, cammino {walk_min:.1f} min, "
                f"{n_sts} alzate (media {avg_sts:.1f}s, {n_slow} lente >3s)"
            )

        walk_str = ""
        if walking_episodes:
            speeds = [e.get('avg_speed_ms') for e in walking_episodes if e.get('avg_speed_ms')]
            syms = [e.get('symmetry') for e in walking_episodes if e.get('symmetry')]
            if speeds:
                walk_str += f", velocita' media {sum(speeds)/len(speeds):.2f} m/s"
            if syms:
                walk_str += f", simmetria media {sum(syms)/len(syms):.0f}%"

        prompt = f"""Sei un assistente clinico specializzato in monitoraggio geriatrico.
Ecco i riassunti di ogni blocco di 30 minuti della giornata di oggi:

{chr(10).join(block_summaries)}
{context_str}{walk_str}

Scrivi un DIARIO CLINICO GIORNALIERO (max 250 parole) destinato a un
fisioterapista o medico.

Istruzioni:
- Descrivi l'andamento cronologico della giornata in prosa fluida
- Evidenzia criticita' cliniche: affaticamento, instabilita', sedentarieta'
- Nota i pattern: peggioramento serale, miglioramento dopo riposo, etc.
- Integra le osservazioni qualitative (ausili, postura, ambiente) con i numeri
- Concludi con cosa monitorare nei prossimi giorni
- NON usare elenchi puntati, scrivi in italiano, tono clinico"""

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 800}
            }
            response = requests.post(self.url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()

        except Exception as e:
            print(f"[CONTEXT] Errore narrativa giornaliera: {e}")
            return self._fallback_daily_narrative(daily_summary)

    def _fallback_daily_narrative(self, summary):
        """Narrativa minima senza LLM."""
        if not summary:
            return "Nessun dato disponibile per questa giornata."

        sit = (summary.get("time_sitting_min") or 0) / 60
        walk = (summary.get("time_walking_min") or 0) / 60
        n_sts = summary.get("num_sit_to_stand") or 0
        return (
            f"Giornata di monitoraggio: {sit:.0f} minuti seduta, "
            f"{walk:.1f} minuti di cammino, {n_sts} alzate dalla sedia."
        )

    # =========================================
    # PARSING RISPOSTA (v2: JSON + testo libero)
    # =========================================
    def _parse_response_v2(self, response):
        """
        Estrae:
        1. Il blocco JSON strutturato (ausilio, rischio, postura)
        2. Il testo libero che segue
        """
        text = response.strip()

        # Rimuovi markdown fences
        clean = text.replace("```json", "").replace("```", "")

        structured = None
        free_text = text  # fallback: tutto il testo

        # Cerca il JSON
        json_start = clean.find("{")
        json_end = clean.find("}", json_start) if json_start >= 0 else -1

        if json_start >= 0 and json_end > json_start:
            json_str = clean[json_start:json_end + 1]
            try:
                structured = json.loads(json_str)
            except json.JSONDecodeError:
                pass

            # Il testo libero e' tutto cio' che viene dopo il JSON
            after_json = clean[json_end + 1:].strip()
            if after_json:
                free_text = after_json
            elif json_start > 0:
                # Testo prima del JSON
                before_json = clean[:json_start].strip()
                if before_json:
                    free_text = before_json

        return structured, free_text

    # =========================================
    # CHIAMATA API OLLAMA
    # =========================================
    def _call_api(self, img_base64, prompt):
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [img_base64]
            }],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 400
            } , "think": False
        }
        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _frame_to_base64(self, frame, quality=80):
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")

    # =========================================
    # ACCESSORI
    # =========================================
    def get_snapshots(self):
        return self.snapshots

    def get_timeline_blocks(self):
        """Ritorna i blocchi di sintesi per la timeline del report."""
        return self.timeline_blocks

    def set_interval(self, seconds):
        self.interval_sec = seconds

    def set_synthesis_interval(self, seconds):
        self.synthesis_interval = seconds

    def should_trigger(self, context):
        """
        Logica di trigger chiamabile dal MonitoringSystem.
        Ritorna (should_trigger: bool, reason: str) in base al contesto.

        Uso nel loop principale:
            should, reason = self.context.should_trigger(context)
            if should:
                self.context.trigger(frame, context, reason)
        """
        if not context or not self.enabled:
            return False, None

        now = time.time()
        if now - self.last_trigger_time < self.trigger_cooldown:
            return False, None

        state = context.get("state", "UNKNOWN")

        # Alzata lenta
        sts_time = context.get("last_sit_to_stand_time")
        if sts_time and sts_time > 3.0 and state in ("STANDING", "WALKING"):
            return True, "alzata_lenta"

        # Inattivita' lunga (>60 min) — triggera una volta sola
        inactivity = context.get("inactivity_sec", 0)
        if inactivity > 3600 and not self.long_inactivity_alerted:
            self.long_inactivity_alerted = True
            return True, "inattivita_lunga"
        if inactivity < 60:
            self.long_inactivity_alerted = False

        # Simmetria bassa su episodio appena concluso
        symmetry = context.get("symmetry")
        if symmetry is not None and symmetry < 70:
            return True, "simmetria_bassa"

        # Velocita' molto bassa
        speed = context.get("speed_ms")
        if speed is not None and speed < 0.5 and state == "WALKING":
            return True, "velocita_bassa"

        # Cambio stato dopo lunga inattivita'
        if state != self.prev_state:
            if self.prev_state == "SITTING" and inactivity > 1800:
                self.prev_state = state
                return True, "cambio_stato"
            self.prev_state = state

        return False, None