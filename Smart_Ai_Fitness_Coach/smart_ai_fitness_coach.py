import os
import csv
import time
from datetime import datetime
from random import choice

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

try:
    import pyttsx3
except:
    pyttsx3 = None

try:
    import speech_recognition as sr
except:
    sr = None

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


class Config:
    camera_index = 0
    reps_per_set = 15
    rest_seconds = 20
    logfile = "workout_log.csv"
    angle_smooth_alpha = 0.35
    encouragements = (
        "You're doing great!", "Keep pushing!", "Feel the burn!",
        "Stay strong!", "Almost there!", "Crushing it!"
    )
    motion_threshold = 0.02  # sensitivity for detecting motion

CFG = Config()


class Voice:
    def __init__(self):
        self.enabled = pyttsx3 is not None
        if self.enabled:
            self.engine = pyttsx3.init()

    def say(self, text):
        if self.enabled:
            self.engine.say(text)
            self.engine.runAndWait()
        print(f"[Coach]: {text}")



class Listener:
    def __init__(self):
        self.enabled = sr is not None
        if self.enabled:
            self.recog = sr.Recognizer()

    def ask(self, voice: Voice, question, cast=str):
        voice.say(question)
        if self.enabled:
            with sr.Microphone() as source:
                try:
                    audio = self.recog.listen(source, timeout=6, phrase_time_limit=6)
                    text = self.recog.recognize_google(audio)
                    print("Heard:", text)
                    return cast(text)
                except:
                    pass
        return cast(input(f"{question}: "))



def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ema(prev, cur, alpha):
    if prev is None:
        return cur
    return alpha * cur + (1 - alpha) * prev

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    cosang = clamp(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

class PoseCounter:
    def __init__(self, voice: Voice):
        self.voice = voice
        self.cap = cv2.VideoCapture(CFG.camera_index)
        self.pose = mp_pose.Pose()
        self.stage = None
        self.counter = 0
        self.angle_smooth = None
        self.active_time = 0.0
        self.last_motion_time = None
        self.prev_landmarks = None

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _motion_detected(self, lm):
        if self.prev_landmarks is None:
            self.prev_landmarks = [(p.x, p.y) for p in lm]
            return False
        prev = np.array(self.prev_landmarks)
        curr = np.array([(p.x, p.y) for p in lm])
        disp = np.mean(np.linalg.norm(curr - prev, axis=1))
        self.prev_landmarks = curr
        return disp > CFG.motion_threshold

    def _pushup_logic(self, lm):
        L = mp_pose.PoseLandmark
        S, E, W = L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST
        shoulder = (lm[S.value].x, lm[S.value].y)
        elbow = (lm[E.value].x, lm[E.value].y)
        wrist = (lm[W.value].x, lm[W.value].y)
        angle = calc_angle(shoulder, elbow, wrist)
        self.angle_smooth = ema(self.angle_smooth, angle, CFG.angle_smooth_alpha)
        ang = self.angle_smooth
        if ang < 90:
            self.stage = "down"
        if ang > 150 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.voice.say(f"{self.counter}")
            if self.counter % 5 == 0:
                self.voice.say(choice(CFG.encouragements))

    def _situp_logic(self, lm):
        L = mp_pose.PoseLandmark
        H, K, S = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_SHOULDER
        hip = (lm[H.value].x, lm[H.value].y)
        knee = (lm[K.value].x, lm[K.value].y)
        shoulder = (lm[S.value].x, lm[S.value].y)
        angle = calc_angle(knee, hip, shoulder)
        self.angle_smooth = ema(self.angle_smooth, angle, CFG.angle_smooth_alpha)
        ang = self.angle_smooth
        if ang > 150:
            self.stage = "down"
        if ang < 90 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.voice.say(f"{self.counter}")
            if self.counter % 5 == 0:
                self.voice.say(choice(CFG.encouragements))

    def run_exercise(self, name, reps_target):
        self.stage = None
        self.counter = 0
        self.angle_smooth = None
        self.active_time = 0.0
        self.last_motion_time = None

        self.voice.say(f"Start {name}")

        while self.cap.isOpened() and self.counter < reps_target:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            lm = results.pose_landmarks.landmark if results.pose_landmarks else None

            if lm:
                if self._motion_detected(lm):
                    if self.last_motion_time is None:
                        self.last_motion_time = time.time()
                else:
                    if self.last_motion_time is not None:
                        self.active_time += time.time() - self.last_motion_time
                        self.last_motion_time = None

                if name == "pushups":
                    self._pushup_logic(lm)
                elif name == "situps":
                    self._situp_logic(lm)

            # active time
            active_duration = self.active_time
            if self.last_motion_time:
                active_duration += time.time() - self.last_motion_time

        
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 140), (30, 30, 30), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, f"Exercise: {name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 230, 90), 2)
            cv2.putText(frame, f"Reps: {self.counter}/{reps_target}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 255, 180), 2)
            cv2.putText(frame, f"Active Time: {active_duration:.1f}s",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

            # progress bar
            progress = int((self.counter / reps_target) * (frame.shape[1] - 40))
            cv2.rectangle(frame, (20, frame.shape[0] - 40),
                          (20 + progress, frame.shape[0] - 10),
                          (0, 255, 0), -1)
            cv2.rectangle(frame, (20, frame.shape[0] - 40),
                          (frame.shape[1] - 20, frame.shape[0] - 10),
                          (255, 255, 255), 2)

            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow(name, frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(name)

        if self.last_motion_time:
            self.active_time += time.time() - self.last_motion_time
            self.last_motion_time = None

        return self.counter, self.active_time / 60.0

def log_workout(path, name, exercise, reps, duration_min):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            name, exercise, reps, round(duration_min, 2)
        ])

def show_dashboard(name, path):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, header=None,
                     names=["Timestamp", "Name", "Exercise", "Reps", "DurationMin"])
    if df.empty or name not in df["Name"].unique():
        return
    u = df[df["Name"] == name].copy()
    u["Timestamp"] = pd.to_datetime(u["Timestamp"])
    u.set_index("Timestamp", inplace=True)
    daily = u.resample("D").sum(numeric_only=True)
    daily[["Reps"]].plot(kind="bar", title=f"Daily Reps - {name}", color="green")
    plt.ylabel("Reps")
    plt.show()

def main():
    voice = Voice()
    listener = Listener()

    name = listener.ask(voice, "What is your name?")
    age = listener.ask(voice, "What is your age?", int)
    height = listener.ask(voice, "What is your height in centimeters?", float)
    weight = listener.ask(voice, "What is your weight in kilograms?", float)

    chosen = listener.ask(voice, "Which exercise do you want? Pushups or Situps?", str).lower()
    if chosen not in ["pushups", "situps"]:
        chosen = "pushups"

    sets = listener.ask(voice, "How many sets do you want?", int)

    pc = PoseCounter(voice)

    for s in range(1, sets + 1):
        voice.say(f"{chosen} set {s}")
        reps, minutes = pc.run_exercise(chosen, CFG.reps_per_set)
        log_workout(CFG.logfile, name, chosen, reps, minutes)
        if s < sets:
            voice.say(f"Rest {CFG.rest_seconds} seconds")
            time.sleep(CFG.rest_seconds)

    pc.close()
    show_dashboard(name, CFG.logfile)
    voice.say("Workout complete.")

if __name__ == "__main__":
    main()
