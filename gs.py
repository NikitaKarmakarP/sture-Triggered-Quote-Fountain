import cv2
import mediapipe as mp
import pygame
import time
import random

# === Setup ===
# Quotes to display
quotes = [
    "Believe in yourself and all that you are.",
    "Push yourself, because no one else will.",
    "Sometimes later becomes never. Do it now.",
    "Dream it. Wish it. Do it.",
    "Great things never come from comfort zones.",
    "Success doesn’t just find you. Go get it.",
    "Your only limit is your mind."
]

# Load calm background music
pygame.mixer.init()
pygame.mixer.music.load("meditation_music.mp3")  # Use your own calm mp3 file here

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# State variables
quote_displayed = False
fade_count = 0
selected_quote = ""

# Function to detect namaste pose (both hands index fingertips close)
def is_namaste(landmarks):
    if len(landmarks) != 2:
        return False
    hand1 = landmarks[0].landmark[8]  # Index finger tip
    hand2 = landmarks[1].landmark[8]
    distance = ((hand1.x - hand2.x) ** 2 + (hand1.y - hand2.y) ** 2) ** 0.5
    return distance < 0.07

# === Main loop ===
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    multi_hand_landmarks = results.multi_hand_landmarks

    if multi_hand_landmarks:
        for lm in multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        if is_namaste(multi_hand_landmarks) and not quote_displayed:
            selected_quote = random.choice(quotes)
            quote_displayed = True
            fade_count = 0
            pygame.mixer.music.play(-1)

    if quote_displayed:
        fade_count += 5
        alpha = min(fade_count, 255)

        # Create fade-in overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 100), (w - 50, 200), (0, 0, 0), -1)
        blended = cv2.addWeighted(overlay, alpha / 255.0, frame, 1 - alpha / 255.0, 0)
        frame = blended

        # Put quote text
        cv2.putText(frame, selected_quote, (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        if fade_count > 255:
            quote_displayed = False
            pygame.mixer.music.stop()

    cv2.imshow("Quote Fountain", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()
