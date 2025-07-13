"""
Music Gesture Controller

This script captures video from the webcam and detects hand landmarks using MediaPipe.
It allows you to control music playback (volume, speed, pitch, track change) with hand gestures:

- The volume is unlocked when both hands are at the same level..
- Pinching with the right hand (thumb to index finger) controls playback speed(you can control the lock  with three other finger).
- Pinching with the left hand (thumb to index finger) controls pitch (in semitones)(you can control the lock  with three other finger).
- Making a fist with the right or left hand skips to the next or previous track respectively.

Dependencies:
- OpenCV (cv2)
- MediaPipe
- numpy
- pyo (Server, SfPlayer, Follower)

Usage:
1. Install dependencies: `pip install opencv-python mediapipe numpy pyo`
2. Place your .mp3 or .wav files in a folder named "music" next to this script.
3. Run: `python music_controller.py` and follow on-screen gestures.
"""

import cv2
import mediapipe as mp
import os
import math
import time
import numpy as np
from collections import deque
from pyo import Server, SfPlayer, Follower

# === CONSTANTS ===
MUSIC_FOLDER           = "music"
SONG_CHANGE_CD         = 0.5           # Minimum seconds between track changes
dB_MIN, dB_MAX         = 0, 100        # dB range for volume mapping
MIN_PINCH_DIST         = 0.02          # Minimum normalized distance for pinch detection
MAX_PINCH_DIST         = 0.50          # Maximum normalized distance for pinch detection
VOLUME_SMOOTH_STEP     = 4             # dB per frame smoothing step
SPEED_MIN, SPEED_MAX   = 0.2, 3.0      # Playback speed range
PITCH_MIN, PITCH_MAX   = -12, 12       # Pitch semitone range
SPEED_SMOOTH_STEP      = 0.1           # Speed smoothing step
PITCH_SMOOTH_STEP      = 0.5           # Pitch smoothing step
NODE_RADIUS            = 8             # Radius for drawing landmark circles
WAVE_BAR_COUNT         = 20            # Number of bars in the audio visualizer
BAR_LENGTH_FACTOR      = 2.0           # Max bar length relative to thumb distance

class MusicController:
    def __init__(self):
        # Start the audio server
        self.server = Server().boot()
        self.server.start()

        # Load all .mp3/.wav files from the music folder
        self.files = self._load_files(MUSIC_FOLDER)
        if not self.files:
            print(f"No audio files found in '{MUSIC_FOLDER}'")
            exit(1)

        # Initial playback state
        self.current_index    = 0
        self.current_vol_db   = 50.0
        self.target_vol_db    = 50.0
        self.vol_lock         = False

        self.current_speed    = 1.0
        self.target_speed     = 1.0
        self.speed_lock       = False

        self.current_pitch    = 0.0
        self.target_pitch     = 0.0
        self.pitch_lock       = False

        self.last_song_time   = time.monotonic()

        # Queue to store previous amplitude values for visualizer
        self.amp_history = deque([0.0] * WAVE_BAR_COUNT, maxlen=WAVE_BAR_COUNT)

        # Create the initial audio player
        self.player = SfPlayer(
            self.files[self.current_index],
            speed=self.current_speed * (2 ** (self.current_pitch / 12)),
            loop=True,
            mul=self._db_to_mul(self.current_vol_db)
        ).out()

        # Amplitude follower for real-time volume levels
        self.env = Follower(self.player, freq=50)

        # Initialize MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7
        )

    def _load_files(self, folder):
        # Return a sorted list of audio file paths
        return sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".mp3", ".wav"))
        )

    def _db_to_mul(self, db):
        # Convert dB value to multiplier between 0 and 1
        return max(0.0, min(1.0, (db - dB_MIN) / (dB_MAX - dB_MIN)))

    def _dist(self, a, b):
        # Euclidean distance between two normalized landmarks
        return math.hypot(a.x - b.x, a.y - b.y)

    def _change_song(self, delta):
        # Stop current track and move to next/previous
        self.player.stop()
        self.current_index = (self.current_index + delta) % len(self.files)
        self.player = SfPlayer(
            self.files[self.current_index],
            speed=self.current_speed * (2 ** (self.current_pitch / 12)),
            loop=True,
            mul=self._db_to_mul(self.current_vol_db)
        ).out()
        self.env.setInput(self.player)
        self.last_song_time = time.monotonic()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            now = time.monotonic()

            song_next = song_prev = False
            thumbs = []

            if res.multi_hand_landmarks:
                # Collect thumb tip landmarks for both hands
                thumbs = [lm.landmark[4] for lm in res.multi_hand_landmarks]

                # Volume lock: both thumbs at approximately same height
                if len(thumbs) == 2:
                    y1, y2 = thumbs[0].y, thumbs[1].y
                    self.vol_lock = abs(y1 - y2) <= 0.05

                # Speed & Pitch locks: index, middle & pinky all extended
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    mid_up   = lm.landmark[12].y < lm.landmark[10].y
                    ring_up  = lm.landmark[16].y < lm.landmark[14].y
                    pinky_up = lm.landmark[20].y < lm.landmark[18].y
                    if mid_up and ring_up and pinky_up:
                        if label == 'Right': self.speed_lock = True
                        if label == 'Left':  self.pitch_lock = True
                    elif not mid_up and not ring_up and not pinky_up:
                        if label == 'Right': self.speed_lock = False
                        if label == 'Left':  self.pitch_lock = False

                # Track change: right fist = next, left fist = previous
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    fist = all(
                        lm.landmark[t].y > lm.landmark[p].y
                        for t, p in zip([8,12,16,20], [6,10,14,18])
                    )
                    ang = math.degrees(math.atan2(
                        lm.landmark[4].y - lm.landmark[3].y,
                        lm.landmark[4].x - lm.landmark[3].x
                    ))
                    if fist and label == 'Right' and -40 <= ang <= 40:
                        song_next = True
                    if fist and label == 'Left' and (ang >= 140 or ang <= -140):
                        song_prev = True

                if song_next and now - self.last_song_time > SONG_CHANGE_CD:
                    self._change_song(1)
                if song_prev and now - self.last_song_time > SONG_CHANGE_CD:
                    self._change_song(-1)

                # Speed Control (right hand pinch)
                if self.speed_lock:
                    for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                        if hd.classification[0].label == 'Right':
                            d = self._dist(lm.landmark[4], lm.landmark[8])
                            norm = max(0, min(1, (d - MIN_PINCH_DIST) / (MAX_PINCH_DIST - MIN_PINCH_DIST)))
                            self.target_speed = norm * (SPEED_MAX - SPEED_MIN) + SPEED_MIN
                            x1, y1 = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
                            x2, y2 = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (255,255,255), 1)
                            cv2.putText(frame, f"Speed: {self.current_speed:.2f}x",
                                        (x2+5, y2-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255), 1)
                            cv2.circle(frame, (x1, y1), NODE_RADIUS, (255,255,255), 1)
                            cv2.circle(frame, (x2, y2), NODE_RADIUS, (255,255,255), 1)

                # Pitch Control (left hand pinch)
                if self.pitch_lock:
                    for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                        if hd.classification[0].label == 'Left':
                            d2 = self._dist(lm.landmark[4], lm.landmark[8])
                            norm2 = max(0, min(1, (d2 - MIN_PINCH_DIST) / (MAX_PINCH_DIST - MIN_PINCH_DIST)))
                            self.target_pitch = norm2 * (PITCH_MAX - PITCH_MIN) + PITCH_MIN
                            x1, y1 = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
                            x2, y2 = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (255,255,255), 1)
                            cv2.putText(frame, f"Pitch: {self.current_pitch:.1f} st",
                                        (x2+5, y2-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255), 1)
                            cv2.circle(frame, (x1, y1), NODE_RADIUS, (255,255,255), 1)
                            cv2.circle(frame, (x2, y2), NODE_RADIUS, (255,255,255), 1)

                # Volume Control & Visualizer (thumb pinch)
                if self.vol_lock and len(thumbs) == 2:
                    p1, p2 = thumbs
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.circle(frame, (x1, y1), NODE_RADIUS, (255,255,255), 1)
                    cv2.circle(frame, (x2, y2), NODE_RADIUS, (255,255,255), 1)

                    dx, dy = x2 - x1, y2 - y1
                    length = math.hypot(dx, dy) or 1
                    ux, uy = dx / length, dy / length
                    perp_x, perp_y = -uy, ux
                    max_bar_len = length * BAR_LENGTH_FACTOR

                    amp = self.env.get()
                    self.amp_history.append(amp)
                    for i, a in enumerate(self.amp_history):
                        t = i / (WAVE_BAR_COUNT - 1)
                        bx = x1 + dx * t
                        by = y1 + dy * t
                        bar_len = a * max_bar_len
                        sx = int(bx - perp_x * (bar_len / 2))
                        sy = int(by - perp_y * (bar_len / 2))
                        ex = int(bx + perp_x * (bar_len / 2))
                        ey = int(by + perp_y * (bar_len / 2))
                        cv2.line(frame, (sx, sy), (ex, ey), (255,255,255), 2)

                    d3 = self._dist(p1, p2)
                    norm3 = max(0.0, min(1.0, (d3 - MIN_PINCH_DIST) / (MAX_PINCH_DIST - MIN_PINCH_DIST)))
                    self.target_vol_db = dB_MIN + norm3 * (dB_MAX - dB_MIN)

                    # Display volume above visualizer
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    offset = 20
                    tx = int(mx - perp_x * offset)
                    ty = int(my - perp_y * offset)
                    vol_10 = int((self.current_vol_db - dB_MIN) / (dB_MAX - dB_MIN) * 10)
                    cv2.putText(
                        frame,
                        f"Volume: {vol_10}",
                        (tx, ty),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.5,
                        (255,255,255),
                        1
                    )
                else:
                    self.target_vol_db = self.current_vol_db

            # Smooth transitions
            if abs(self.target_speed - self.current_speed) > SPEED_SMOOTH_STEP:
                self.current_speed += SPEED_SMOOTH_STEP * (1 if self.target_speed > self.current_speed else -1)
            else:
                self.current_speed = self.target_speed

            if abs(self.target_pitch - self.current_pitch) > PITCH_SMOOTH_STEP:
                self.current_pitch += PITCH_SMOOTH_STEP * (1 if self.target_pitch > self.current_pitch else -1)
            else:
                self.current_pitch = self.target_pitch

            if abs(self.target_vol_db - self.current_vol_db) > VOLUME_SMOOTH_STEP:
                self.current_vol_db += VOLUME_SMOOTH_STEP * (1 if self.target_vol_db > self.current_vol_db else -1)
            else:
                self.current_vol_db = self.target_vol_db

            # Update player parameters
            self.player.setSpeed(self.current_speed * (2 ** (self.current_pitch / 12)))
            self.player.setMul(self._db_to_mul(self.current_vol_db))

            cv2.imshow('Music Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.server.stop()
        self.server.shutdown()

if __name__ == "__main__":
    MusicController().run()
