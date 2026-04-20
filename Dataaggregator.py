"""
DataAggregator - Aggregazione temporale e analisi trend per il monitoraggio.

Fornisce riepiloghi settimanali, mensili e annuali a partire dai dati
giornalieri salvati nel database. Rileva trend, genera alert e calcola
variazioni rispetto ai periodi precedenti.

NOTA: gli episodi di cammino con reliability_score < 50 vengono
esclusi dalle aggregazioni cliniche. Restano salvati nel DB per
eventuali analisi future.

Uso:
    from DataAggregator import DataAggregator
    aggregator = DataAggregator(db_path="monitoring.db")
    weekly = aggregator.get_weekly_summary("2026-03-30")
    trend = aggregator.get_trend_analysis("2026-03-30")
"""

import sqlite3
import statistics
from datetime import date, datetime, timedelta
from collections import defaultdict

# Soglia minima di affidabilità per includere un episodio di cammino
# nelle aggregazioni cliniche. Gli episodi sotto questa soglia vengono
# salvati nel DB ma esclusi da medie, trend e alert.
MIN_RELIABILITY_SCORE = 50


class DataAggregator:

    def __init__(self, db_path="monitoring.db"):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ================================================================
    #  RIEPILOGO SETTIMANALE
    # ================================================================
    def get_weekly_summary(self, reference_date=None):
        ref = _parse_date(reference_date)
        monday = ref - timedelta(days=ref.weekday())
        sunday = monday + timedelta(days=6)
        return self._aggregate_period(start=monday, end=sunday, period_label="weekly")

    # ================================================================
    #  RIEPILOGO MENSILE
    # ================================================================
    def get_monthly_summary(self, year, month):
        first_day = date(year, month, 1)
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        summary = self._aggregate_period(start=first_day, end=last_day, period_label="monthly")
        summary["weekly_breakdown"] = self._weekly_breakdown(first_day, last_day)
        return summary

    # ================================================================
    #  RIEPILOGO ANNUALE
    # ================================================================
    def get_yearly_summary(self, year):
        first_day = date(year, 1, 1)
        last_day = date(year, 12, 31)

        summary = self._aggregate_period(start=first_day, end=last_day, period_label="yearly")

        monthly = []
        for m in range(1, 13):
            m_start = date(year, m, 1)
            if m == 12:
                m_end = date(year, 12, 31)
            else:
                m_end = date(year, m + 1, 1) - timedelta(days=1)
            row = self._aggregate_period(m_start, m_end, period_label="month")
            if row["days_monitored"] > 0:
                row["month"] = m
                monthly.append(row)

        summary["monthly_breakdown"] = monthly
        return summary

    # ================================================================
    #  ANALISI TREND
    # ================================================================
    def get_trend_analysis(self, reference_date=None, period="week"):
        ref = _parse_date(reference_date)

        if period == "week":
            monday_curr = ref - timedelta(days=ref.weekday())
            sunday_curr = monday_curr + timedelta(days=6)
            monday_prev = monday_curr - timedelta(days=7)
            sunday_prev = monday_curr - timedelta(days=1)
        elif period == "month":
            first_curr = date(ref.year, ref.month, 1)
            if ref.month == 12:
                last_curr = date(ref.year + 1, 1, 1) - timedelta(days=1)
            else:
                last_curr = date(ref.year, ref.month + 1, 1) - timedelta(days=1)
            if ref.month == 1:
                first_prev = date(ref.year - 1, 12, 1)
                last_prev = date(ref.year - 1, 12, 31)
            else:
                first_prev = date(ref.year, ref.month - 1, 1)
                last_prev = first_curr - timedelta(days=1)
            monday_curr, sunday_curr = first_curr, last_curr
            monday_prev, sunday_prev = first_prev, last_prev
        else:
            raise ValueError("period deve essere 'week' o 'month'")

        current = self._aggregate_period(monday_curr, sunday_curr, "current")
        previous = self._aggregate_period(monday_prev, sunday_prev, "previous")

        if current["days_monitored"] == 0 or previous["days_monitored"] == 0:
            return {
                "status": "insufficient_data",
                "current_days": current["days_monitored"],
                "previous_days": previous["days_monitored"],
                "message": "Dati insufficienti per il confronto trend."
            }

        metrics = [
            ("avg_time_walking_min", "Tempo cammino/giorno", "min", True),
            ("avg_time_sitting_min", "Tempo seduto/giorno", "min", False),
            ("avg_num_sit_to_stand", "Alzate/giorno", None, True),
            ("avg_sit_to_stand_time", "Tempo medio alzata", "s", False),
            ("avg_independence_score", "Indice indipendenza", "%", True),
            ("avg_num_walkings", "Episodi cammino/giorno", None, True),
            ("avg_longest_inactivity_min", "Inattività massima/giorno", "min", False),
        ]

        gait_metrics = [
            ("avg_speed_ms", "Velocità cammino", "m/s", True),
            ("avg_cadence", "Cadenza", "passi/min", None),
            ("avg_symmetry", "Simmetria passo", "%", True),
            ("avg_regularity", "Regolarità passo", "%", True),
        ]

        trends = {}
        ca = current["daily_averages"]
        pa = previous["daily_averages"]

        for key, label, unit, higher_is_better in metrics:
            curr_val = ca.get(key, 0)
            prev_val = pa.get(key, 0)
            trends[key] = _build_trend_entry(label, unit, curr_val, prev_val, higher_is_better)

        cg = current.get("gait", {})
        pg = previous.get("gait", {})
        for key, label, unit, higher_is_better in gait_metrics:
            curr_val = cg.get(key, 0) or 0
            prev_val = pg.get(key, 0) or 0
            trends[key] = _build_trend_entry(label, unit, curr_val, prev_val, higher_is_better)

        return {
            "status": "ok",
            "period": period,
            "current": {
                "start": monday_curr.isoformat(),
                "end": sunday_curr.isoformat(),
                "days": current["days_monitored"]
            },
            "previous": {
                "start": monday_prev.isoformat(),
                "end": sunday_prev.isoformat(),
                "days": previous["days_monitored"]
            },
            "trends": trends,
            "alerts": self._generate_alerts(current, previous, trends)
        }

    # ================================================================
    #  STORICO METRICHE (per grafici)
    # ================================================================
    def get_metric_history(self, metric, days=30, reference_date=None):
        ref = _parse_date(reference_date)
        start = ref - timedelta(days=days - 1)

        conn = self._connect()
        cursor = conn.cursor()

        gait_map = {
            "gait_speed": "avg_speed_ms",
            "gait_cadence": "cadence",
            "gait_symmetry": "symmetry",
            "gait_regularity": "regularity",
        }

        if metric in gait_map:
            col = gait_map[metric]
            cursor.execute(f"""
                SELECT summary_date AS date,
                       SUM({col} * COALESCE(reliability_score, total_steps, 1))
                         / SUM(COALESCE(reliability_score, total_steps, 1)) AS value,
                       COUNT(*) AS episodes,
                       SUM(total_steps) AS total_steps
                FROM walking_episode
                WHERE summary_date BETWEEN ? AND ?
                  AND {col} IS NOT NULL
                  AND (reliability_score IS NULL OR reliability_score >= ?)
                GROUP BY summary_date
                ORDER BY summary_date
            """, (start.isoformat(), ref.isoformat(), MIN_RELIABILITY_SCORE))
        else:
            cursor.execute(f"""
                SELECT date, {metric} AS value
                FROM daily_summary
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            """, (start.isoformat(), ref.isoformat()))

        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    # ================================================================
    #  EPISODI CAMMINO AGGREGATI
    # ================================================================
    def get_walking_summary(self, start_date, end_date):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM walking_episode
            WHERE summary_date BETWEEN ? AND ?
              AND (reliability_score IS NULL OR reliability_score >= ?)
            ORDER BY start_time
        """, (start_date, end_date, MIN_RELIABILITY_SCORE))

        episodes = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if not episodes:
            return {"total_episodes": 0}

        durations = [e["duration_sec"] for e in episodes if e["duration_sec"]]
        distances = [e["distance_m"] for e in episodes if e["distance_m"]]
        speeds = [e["avg_speed_ms"] for e in episodes if e["avg_speed_ms"]]
        steps_list = [e["total_steps"] for e in episodes if e["total_steps"]]

        total_steps = sum(steps_list) if steps_list else 0
        total_duration = sum(durations) if durations else 0
        pooled_cadence = (total_steps / total_duration * 60) if total_duration > 0 and total_steps > 0 else None

        total_distance = sum(distances) if distances else None
        pooled_speed = (total_distance / total_duration) if total_distance and total_duration > 0 else None

        has_reliability = "reliability_score" in episodes[0] if episodes else False

        def _get_weight(ep):
            if has_reliability and ep.get("reliability_score"):
                return ep["reliability_score"]
            return ep.get("total_steps") or 1

        weighted_cadence = _weighted_mean(
            [(e["cadence"], _get_weight(e)) for e in episodes if e.get("cadence")]
        )
        weighted_symmetry = _weighted_mean(
            [(e["symmetry"], _get_weight(e)) for e in episodes if e.get("symmetry")]
        )
        weighted_regularity = _weighted_mean(
            [(e["regularity"], _get_weight(e)) for e in episodes if e.get("regularity")]
        )
        weighted_speed = _weighted_mean(
            [(e["avg_speed_ms"], _get_weight(e)) for e in episodes if e.get("avg_speed_ms")]
        )

        return {
            "total_episodes": len(episodes),
            "total_duration_sec": total_duration,
            "total_distance_m": total_distance,
            "total_steps": total_steps,
            "pooled_cadence": round(pooled_cadence, 1) if pooled_cadence else None,
            "pooled_speed_ms": round(pooled_speed, 3) if pooled_speed else None,
            "avg_cadence": weighted_cadence,
            "avg_speed_ms": weighted_speed,
            "avg_symmetry": weighted_symmetry,
            "avg_regularity": weighted_regularity,
            "max_speed_ms": round(max(speeds), 3) if speeds else None,
            "min_speed_ms": round(min(speeds), 3) if speeds else None,
            "longest_walk_sec": max(durations) if durations else None,
            "shortest_walk_sec": min(durations) if durations else None,
        }

    # ================================================================
    #  TUG / STS STORICO
    # ================================================================
    def get_test_history(self, test_type, limit=20):
        conn = self._connect()
        cursor = conn.cursor()

        if test_type == "TUG":
            cursor.execute("""
                SELECT ts.summary_date, ts.start_time,
                       tr.total_time, tr.walk_forward_time, tr.turn_time,
                       tr.walk_back_time, tr.avg_movement,
                       tr.avg_knee_angle, tr.fall_risk_level
                FROM test_session ts
                JOIN tug_result tr ON ts.test_id = tr.test_id
                WHERE ts.test_type = 'TUG' AND ts.completed = 1
                ORDER BY ts.start_time DESC
                LIMIT ?
            """, (limit,))
        elif test_type == "STS":
            cursor.execute("""
                SELECT ts.summary_date, ts.start_time,
                       sr.total_time, sr.reps_completed,
                       sr.avg_rep_time, sr.avg_knee_angle
                FROM test_session ts
                JOIN sts_result sr ON ts.test_id = sr.test_id
                WHERE ts.test_type = 'STS' AND ts.completed = 1
                ORDER BY ts.start_time DESC
                LIMIT ?
            """, (limit,))
        else:
            conn.close()
            return []

        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    # ================================================================
    #  CONTEXT SNAPSHOT AGGREGATI
    # ================================================================
    def get_context_snapshots(self, start_date, end_date):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT summary_date, timestamp, description
            FROM context_snapshot
            WHERE summary_date BETWEEN ? AND ?
            ORDER BY timestamp
        """, (start_date, end_date))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    # ================================================================
    #  DISTRIBUZIONE ORARIA ATTIVITÀ
    # ================================================================
    def get_hourly_distribution(self, target_date):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT slot_start, slot_end,
                   time_sitting_sec, time_standing_sec,
                   time_walking_sec, num_sit_to_stand
            FROM time_slot
            WHERE summary_date = ?
            ORDER BY slot_start
        """, (target_date,))
        slots = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if not slots:
            return []

        hourly = defaultdict(lambda: {
            "sitting": 0, "standing": 0, "walking": 0, "sit_to_stand": 0
        })
        for s in slots:
            try:
                hour = datetime.fromisoformat(s["slot_start"]).hour
            except (ValueError, TypeError):
                continue
            hourly[hour]["sitting"] += s["time_sitting_sec"] or 0
            hourly[hour]["standing"] += s["time_standing_sec"] or 0
            hourly[hour]["walking"] += s["time_walking_sec"] or 0
            hourly[hour]["sit_to_stand"] += s["num_sit_to_stand"] or 0

        return [
            {"hour": h, **hourly[h]}
            for h in sorted(hourly.keys())
        ]

    # ================================================================
    #  AGGREGAZIONE PERIODO GENERICO
    # ================================================================
    def _aggregate_period(self, start, end, period_label="period"):
        conn = self._connect()
        cursor = conn.cursor()

        start_str = start.isoformat()
        end_str = end.isoformat()

        cursor.execute("""
            SELECT * FROM daily_summary
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """, (start_str, end_str))
        days = [dict(r) for r in cursor.fetchall()]

        # Solo episodi affidabili per le aggregazioni cliniche
        cursor.execute("""
            SELECT * FROM walking_episode
            WHERE summary_date BETWEEN ? AND ?
              AND (reliability_score IS NULL OR reliability_score >= ?)
        """, (start_str, end_str, MIN_RELIABILITY_SCORE))
        walks = [dict(r) for r in cursor.fetchall()]

        cursor.execute("""
            SELECT duration_sec FROM activity_event
            WHERE summary_date BETWEEN ? AND ?
              AND event_type = 'SIT_TO_STAND_COMPLETE'
              AND duration_sec IS NOT NULL
        """, (start_str, end_str))
        sts_durations = [r["duration_sec"] for r in cursor.fetchall()]

        conn.close()

        n = len(days)
        if n == 0:
            return {
                "period": period_label,
                "period_start": start_str,
                "period_end": end_str,
                "days_monitored": 0,
                "totals": {},
                "daily_averages": {},
                "ranges": {},
                "gait": {},
                "sit_to_stand_detail": {},
                "alerts": []
            }

        totals = {
            "total_monitoring_min": sum(d["total_monitoring_min"] or 0 for d in days),
            "time_sitting_min": sum(d["time_sitting_min"] or 0 for d in days),
            "time_standing_min": sum(d["time_standing_min"] or 0 for d in days),
            "time_walking_min": sum(d["time_walking_min"] or 0 for d in days),
            "num_walkings": sum(d["num_walkings"] or 0 for d in days),
            "num_sit_to_stand": sum(d["num_sit_to_stand"] or 0 for d in days),
            "num_stand_to_sit": sum(d["num_stand_to_sit"] or 0 for d in days),
            "num_slow_transitions": sum(d["num_slow_transitions"] or 0 for d in days),
        }

        daily_averages = {
            "avg_time_sitting_min": _safe_mean([d["time_sitting_min"] for d in days]),
            "avg_time_standing_min": _safe_mean([d["time_standing_min"] for d in days]),
            "avg_time_walking_min": _safe_mean([d["time_walking_min"] for d in days]),
            "avg_num_walkings": _safe_mean([d["num_walkings"] for d in days]),
            "avg_num_sit_to_stand": _safe_mean([d["num_sit_to_stand"] for d in days]),
            "avg_sit_to_stand_time": _safe_mean(
                [d["avg_sit_to_stand_time"] for d in days if d["avg_sit_to_stand_time"]]
            ),
            "avg_independence_score": _safe_mean(
                [d["independence_score"] for d in days if d["independence_score"]]
            ),
            "avg_longest_inactivity_min": _safe_mean(
                [d["longest_inactivity_min"] for d in days if d["longest_inactivity_min"]]
            ),
        }

        ranges = {}
        for key in ["time_walking_min", "time_sitting_min", "num_sit_to_stand",
                     "independence_score", "longest_inactivity_min"]:
            values = [d[key] for d in days if d[key] is not None]
            if values:
                ranges[key] = {
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "best_day": days[_argmax(values)]["date"] if key != "time_sitting_min"
                               else days[_argmin(values)]["date"],
                    "worst_day": days[_argmin(values)]["date"] if key != "time_sitting_min"
                                else days[_argmax(values)]["date"],
                }

        gait = {}
        if walks:
            steps_list = [w["total_steps"] for w in walks if w["total_steps"]]
            durations_w = [w["duration_sec"] for w in walks if w["duration_sec"]]
            speeds = [w["avg_speed_ms"] for w in walks if w["avg_speed_ms"]]
            distances_w = [w["distance_m"] for w in walks if w["distance_m"]]

            total_steps_all = sum(steps_list) if steps_list else 0
            total_dur = sum(durations_w) if durations_w else 0

            has_rel = "reliability_score" in walks[0]
            def _w(ep):
                if has_rel and ep.get("reliability_score"):
                    return ep["reliability_score"]
                return ep.get("total_steps") or 1

            gait = {
                "total_episodes": len(walks),
                "total_steps": total_steps_all,
                "total_walk_duration_sec": total_dur,
                "pooled_cadence": round(total_steps_all / total_dur * 60, 1) if total_dur > 0 and total_steps_all > 0 else None,
                "pooled_speed_ms": round(sum(distances_w) / total_dur, 3) if distances_w and total_dur > 0 else None,
                "avg_speed_ms": _weighted_mean(
                    [(w["avg_speed_ms"], _w(w)) for w in walks if w.get("avg_speed_ms")]
                ),
                "avg_cadence": _weighted_mean(
                    [(w["cadence"], _w(w)) for w in walks if w.get("cadence")]
                ),
                "avg_symmetry": _weighted_mean(
                    [(w["symmetry"], _w(w)) for w in walks if w.get("symmetry")]
                ),
                "avg_regularity": _weighted_mean(
                    [(w["regularity"], _w(w)) for w in walks if w.get("regularity")]
                ),
                "max_speed_ms": round(max(speeds), 3) if speeds else None,
                "min_speed_ms": round(min(speeds), 3) if speeds else None,
            }

        sts_detail = {}
        if sts_durations:
            sts_detail = {
                "count": len(sts_durations),
                "avg_time": round(statistics.mean(sts_durations), 2),
                "median_time": round(statistics.median(sts_durations), 2),
                "max_time": round(max(sts_durations), 2),
                "min_time": round(min(sts_durations), 2),
                "std_dev": round(statistics.stdev(sts_durations), 2) if len(sts_durations) > 1 else 0,
                "num_slow": sum(1 for t in sts_durations if t > 3.0),
                "pct_slow": round(sum(1 for t in sts_durations if t > 3.0) / len(sts_durations) * 100, 1),
            }

        return {
            "period": period_label,
            "period_start": start_str,
            "period_end": end_str,
            "days_monitored": n,
            "totals": totals,
            "daily_averages": daily_averages,
            "ranges": ranges,
            "gait": gait,
            "sit_to_stand_detail": sts_detail,
            "alerts": []
        }

    # ================================================================
    #  BREAKDOWN SETTIMANALE
    # ================================================================
    def _weekly_breakdown(self, month_start, month_end):
        weeks = []
        current = month_start
        current = current - timedelta(days=current.weekday())

        while current <= month_end:
            week_end = current + timedelta(days=6)
            effective_start = max(current, month_start)
            effective_end = min(week_end, month_end)

            week = self._aggregate_period(effective_start, effective_end, "week")
            if week["days_monitored"] > 0:
                week["week_start"] = effective_start.isoformat()
                week["week_end"] = effective_end.isoformat()
                weeks.append(week)

            current += timedelta(days=7)

        return weeks

    # ================================================================
    #  GENERAZIONE ALERT
    # ================================================================
    def _generate_alerts(self, current, previous, trends):
        alerts = []
        ca = current["daily_averages"]
        gait = current.get("gait", {})
        sts = current.get("sit_to_stand_detail", {})

        avg_speed = gait.get("avg_speed_ms")
        if avg_speed is not None and avg_speed < 0.8:
            level = "warning" if avg_speed >= 0.6 else "critical"
            alerts.append({
                "level": level,
                "category": "gait",
                "metric": "avg_speed_ms",
                "value": avg_speed,
                "threshold": 0.8,
                "message": f"Velocità media cammino {avg_speed:.2f} m/s "
                           f"(soglia fragilità: 0.8 m/s)"
            })

        avg_symmetry = gait.get("avg_symmetry")
        if avg_symmetry is not None and avg_symmetry < 75:
            alerts.append({
                "level": "warning",
                "category": "gait",
                "metric": "avg_symmetry",
                "value": avg_symmetry,
                "threshold": 75,
                "message": f"Simmetria passo {avg_symmetry:.0f}% — possibile asimmetria significativa"
            })

        avg_sts_time = sts.get("avg_time")
        if avg_sts_time is not None and avg_sts_time > 3.0:
            alerts.append({
                "level": "warning",
                "category": "transitions",
                "metric": "avg_sit_to_stand_time",
                "value": avg_sts_time,
                "threshold": 3.0,
                "message": f"Tempo medio alzata {avg_sts_time:.1f}s — possibile debolezza arti inferiori"
            })

        avg_inactivity = ca.get("avg_longest_inactivity_min")
        if avg_inactivity is not None and avg_inactivity > 120:
            alerts.append({
                "level": "info",
                "category": "activity",
                "metric": "longest_inactivity",
                "value": avg_inactivity,
                "threshold": 120,
                "message": f"Periodi di inattività prolungata (media {avg_inactivity:.0f} min/giorno)"
            })

        significant_decline = 15
        for key, entry in trends.items():
            if (entry.get("interpretation") == "peggioramento"
                    and abs(entry.get("change_pct", 0)) > significant_decline):
                alerts.append({
                    "level": "warning",
                    "category": "trend",
                    "metric": key,
                    "value": entry.get("current"),
                    "previous": entry.get("previous"),
                    "change_pct": entry.get("change_pct"),
                    "message": f"{entry['label']}: {entry['change_pct']:+.1f}% rispetto al periodo precedente"
                })

        for key, entry in trends.items():
            if (entry.get("interpretation") == "miglioramento"
                    and abs(entry.get("change_pct", 0)) > 10):
                alerts.append({
                    "level": "positive",
                    "category": "trend",
                    "metric": key,
                    "value": entry.get("current"),
                    "previous": entry.get("previous"),
                    "change_pct": entry.get("change_pct"),
                    "message": f"{entry['label']}: {entry['change_pct']:+.1f}% — miglioramento"
                })

        return alerts


# ====================================================================
#  FUNZIONI HELPER
# ====================================================================

def _parse_date(d):
    if d is None:
        return date.today()
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


def _safe_mean(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.mean(clean), 2)


def _weighted_mean(pairs):
    clean = [(v, w) for v, w in pairs if v is not None and w is not None and w > 0]
    if not clean:
        return None
    total_weight = sum(w for _, w in clean)
    if total_weight == 0:
        return None
    result = sum(v * w for v, w in clean) / total_weight
    return round(result, 2)


def _argmin(values):
    return min(range(len(values)), key=lambda i: values[i])


def _argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def _build_trend_entry(label, unit, curr_val, prev_val, higher_is_better):
    curr_val = curr_val if curr_val is not None else 0
    prev_val = prev_val if prev_val is not None else 0

    entry = {
        "label": label,
        "unit": unit,
        "current": round(curr_val, 2),
        "previous": round(prev_val, 2),
        "change_pct": 0,
        "interpretation": "stabile"
    }

    if prev_val != 0:
        pct = ((curr_val - prev_val) / abs(prev_val)) * 100
        entry["change_pct"] = round(pct, 1)

        if abs(pct) < 5:
            entry["interpretation"] = "stabile"
        elif higher_is_better is True:
            entry["interpretation"] = "miglioramento" if pct > 0 else "peggioramento"
        elif higher_is_better is False:
            entry["interpretation"] = "miglioramento" if pct < 0 else "peggioramento"
        else:
            entry["interpretation"] = "variazione"

    return entry