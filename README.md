# 🎵 HandMusicControl

Control your music using nothing but your hands!

This Python project uses **MediaPipe** to track hand gestures via webcam and **Pyo** for real-time audio playback and control. Change songs, adjust volume, and modify pitch/speed — all through intuitive gestures.

---

## 🚀 Features

- ✅ Volume control with **thumb–index finger pinch**
- ✅ Song switching via **fist + thumb direction gesture**
- ✅ Speed and pitch control via **finger poses with open fingers**
- ✅ Real-time **audio visualizer**
- ✅ Gesture locking for stable volume control
- ✅ Cooldown system to prevent mis-triggers

---

## 🔧 Technologies

- MediaPipe
- OpenCV
- Pyo
- Python 3.10+

---

## 📦 Installation

```bash
git clone https://github.com/Adrimars/HandMusicControl.git
cd HandMusicControl
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt


# Add your .wav or .mp3 music files to the music/ folder.
# Open issues or fork and pull request to contribute.
