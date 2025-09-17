# Smart-AI-Fitness-Coach
An AI-powered personal fitness coach using OpenCV + Mediapipe + Speech Recognition + Voice Feedback. Counts push-ups and sit-ups in real-time, tracks active workout time, gives spoken encouragement, and shows progress with a live HUD and daily dashboard.

---

## Features

- Voice interaction: asks your profile (name, age, height, weight) using speech recognition with fallback to typing.
- Real-time HUD overlay showing:
  - Exercise name
  - Rep counter
  - Active workout time (motion-based)
  - Progress bar
- Voice feedback:
  - Announces reps as you complete them
  - Motivational encouragements
  - Guides you through sets and rest periods
- Workout logs stored in workout_log.csv.
- Daily dashboard visualization with matplotlib.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SAITEJAVARMA801/smart-ai-fitness-coach.git
  
2. Install dependencies

              pip3 install opencv-python
              pip3 install mediapipe
              pip3 install numpy
              pip3 install matplotlib
              pip3 install pandas
              pip3 install SpeechRecognition
              pip3 install pyttsx3
              pip3 install pyaudio
              pip3 install pyyaml
              pip3 install requests
              pip3 install setuptools wheel
              pip3 install sounddevice
              pip3 install pydub

3. Run the smart_ai_fitness_coach.py

              python3 smart_ai_fitness_coach.py
