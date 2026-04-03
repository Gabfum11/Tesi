import sqlite3
from datetime import date, datetime
import json

class DatabaseManager:
    def __init__(self, db_path="monitoring.db"):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        return self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def create_tables(self):
        cursor = self.connect()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_monitoring_min REAL,
                time_sitting_min REAL,
                time_standing_min REAL,
                time_walking_min REAL,
                num_walkings INTEGER,
                num_sit_to_stand INTEGER,
                num_stand_to_sit INTEGER,
                avg_sit_to_stand_time REAL,
                max_sit_to_stand_time REAL,
                num_slow_transitions INTEGER,
                longest_inactivity_min REAL,
                independence_score REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_session (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                test_type TEXT,
                start_time TEXT,
                end_time TEXT,
                completed INTEGER,
                video_source TEXT,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tug_result (
                test_id INTEGER PRIMARY KEY,
                total_time REAL,
                walk_forward_time REAL,
                turn_time REAL,
                walk_back_time REAL,
                avg_movement REAL,
                avg_knee_angle REAL,
                fall_risk_level TEXT,
                FOREIGN KEY(test_id) REFERENCES test_session(test_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sts_result (
                test_id INTEGER PRIMARY KEY,
                total_time REAL,
                reps_completed INTEGER,
                avg_rep_time REAL,
                avg_knee_angle REAL,
                FOREIGN KEY(test_id) REFERENCES test_session(test_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS walking_episode (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_sec REAL,
                distance_m REAL,
                avg_speed_ms REAL,
                cadence REAL,
                symmetry REAL,
                regularity REAL,
                total_steps INTEGER,
                reliability_score INTEGER,
                signal_used TEXT,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_event (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                event_time TEXT,
                event_type TEXT,
                duration_sec REAL,
                details TEXT,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_slot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                slot_start TEXT,
                slot_end TEXT,
                time_sitting_sec REAL,
                time_standing_sec REAL,
                time_walking_sec REAL,
                num_sit_to_stand INTEGER,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_snapshot (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                timestamp TEXT,
                description TEXT,
                parsed_json TEXT,
                context_json TEXT,
                state TEXT,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        self.close()

    # === DAILY SUMMARY ===
    def save_daily_summary(self, summary_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT OR REPLACE INTO daily_summary VALUES (
                :date, :total_monitoring_min,
                :time_sitting_min, :time_standing_min, :time_walking_min,
                :num_walkings, :num_sit_to_stand, :num_stand_to_sit,
                :avg_sit_to_stand_time, :max_sit_to_stand_time,
                :num_slow_transitions,
                :longest_inactivity_min, :independence_score
            )
        ''', summary_data)
        self.close()

    # === TEST SESSION ===
    def create_test_session(self, summary_date, test_type, start_time, video_source=None):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO test_session (summary_date, test_type, start_time, completed, video_source)
            VALUES (?, ?, ?, 0, ?)
        ''', (summary_date, test_type, start_time, video_source))
        test_id = cursor.lastrowid
        self.close()
        return test_id

    def complete_test_session(self, test_id, end_time):
        cursor = self.connect()
        cursor.execute('''
            UPDATE test_session SET end_time = ?, completed = 1 WHERE test_id = ?
        ''', (end_time, test_id))
        self.close()

    # === TUG RESULT ===
    def save_tug_result(self, test_id, tug_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO tug_result VALUES (
                :test_id, :total_time, :walk_forward_time, :turn_time,
                :walk_back_time, :avg_movement, :avg_knee_angle, :fall_risk_level
            )
        ''', {**tug_data, 'test_id': test_id})
        self.close()

    # === STS RESULT ===
    def save_sts_result(self, test_id, sts_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO sts_result VALUES (
                :test_id, :total_time, :reps_completed, :avg_rep_time, :avg_knee_angle
            )
        ''', {**sts_data, 'test_id': test_id})
        self.close()

    # === WALKING EPISODE ===
    def save_walking_episode(self, episode_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO walking_episode
            (summary_date, start_time, end_time, duration_sec, distance_m, avg_speed_ms,
             cadence, symmetry, regularity, total_steps)
            VALUES (:date, :start_time, :end_time, :duration_sec, :distance_m, :avg_speed_ms,
                    :cadence, :symmetry, :regularity, :total_steps)
        ''', episode_data)
        self.close()

    # === ACTIVITY EVENT ===
    def save_activity_event(self, event_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO activity_event
            (summary_date, event_time, event_type, duration_sec, details)
            VALUES (:date, :event_time, :event_type, :duration_sec, :details)
        ''', event_data)
        self.close()

    # === TIME SLOT ===
    def save_time_slot(self, slot_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO time_slot
            (summary_date, slot_start, slot_end, time_sitting_sec, time_standing_sec,
             time_walking_sec, num_sit_to_stand)
            VALUES (:date, :slot_start, :slot_end, :time_sitting_sec, :time_standing_sec,
                    :time_walking_sec, :num_sit_to_stand)
        ''', slot_data)
        self.close()

    # === CONTEXT SNAPSHOT ===
    def save_context_snapshot(self, snapshot_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO context_snapshot
            (summary_date, timestamp, description, parsed_json, context_json, state)
            VALUES (:date, :timestamp, :description, :parsed_json, :context_json, :state)
        ''', {
            'date': snapshot_data.get('date'),
            'timestamp': snapshot_data.get('timestamp'),
            'description': snapshot_data.get('description'),
            'parsed_json': json.dumps(snapshot_data.get('parsed')) if snapshot_data.get('parsed') else None,
            'context_json': json.dumps(snapshot_data.get('context')) if snapshot_data.get('context') else None,
            'state': snapshot_data.get('context', {}).get('state') if snapshot_data.get('context') else None
        })
        self.close()