import os
import sounddevice as sd
import queue
import vosk
import requests
import json
from gtts import gTTS
import tempfile
import pygame
import threading
from time import sleep
import face_recognition
import cv2
from datetime import datetime
import multiprocessing
from multiprocessing import Process, Queue, Manager

# Global variables
model_path = "vosk-model-en-us-0.22"
API_KEY = "AIzaSyDd5x5Xc6potNayl0BDkhxe7B2YN2iuvyc"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
SYSTEM_PROMPT = "You are a Helpful Robot called JOJO in a Program which is made for speaking with other people so it is given to use shortend answers however when the user wants a specific answer u can make a 20+ word text. However do not use any sort of text enhancer for example markdown because your prompt is directly spoken out and do not use any - and *. The User Prompt is:"

# Control flags and settings
conversation_active = False
last_conversation_end = None
CONVERSATION_COOLDOWN = 30  # Auf 30 Sekunden erhöht
activation_detected = False
user_cooldowns = {}  # Neues Dictionary für Benutzer-spezifische Cooldowns

# Initializations
vosk_model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()
pygame.mixer.init()

# Chat history
chat_history = []

# Face recognition variables
KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []
current_user = None

last_detection_time = {}  # Speichert die letzte Erkennungszeit pro Benutzer
last_greeting_time = {}   # Speichert die letzte Begrüßungszeit pro Benutzer
tracking_status = {}      # Speichert den aktuellen Tracking-Status pro Benutzer

def load_known_faces():
    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    print("Faces loaded!")

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_filename = fp.name

        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            sleep(0.1)

    finally:
        pygame.mixer.music.stop()
        os.remove(temp_filename)

def callback(indata, frames, time, status):
    audio_queue.put(bytes(indata))

def record_command():
    global conversation_active
    recognizer.Reset()  # Reset the recognizer before each new recording
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening...")
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()  # Convert to lowercase
                if text:
                    if "stop" in text:
                        conversation_active = False
                        return None
                    return text

def query_gemini(prompt):
    global chat_history
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    
    data = {"contents": [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}, *chat_history]}
    
    response = requests.post(GEMINI_URL, json=data)
    if response.status_code == 200:
        ai_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        chat_history.append({"role": "model", "parts": [{"text": ai_response}]})
        return ai_response
    else:
        return "Sorry, I couldn't process that."

def get_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good day"
    else:
        return "Good evening"

def listen_for_start():
    global conversation_active, last_detection_time
    while True:
        if not conversation_active and len(last_detection_time) > 0:  # Nur lauschen wenn Personen erkannt sind
            user_input = record_command()
            if user_input and "start" in user_input.lower():
                print(f"start detected! Starting conversation with present users: {', '.join(last_detection_time.keys())}")
                conversation_thread = threading.Thread(target=conversation_loop)
                conversation_thread.start()

def conversation_loop():
    global conversation_active, last_detection_time
    conversation_active = True
    
    present_users = ", ".join(last_detection_time.keys())
    speak(f"Starting conversation with {present_users}. How can I help you today?")
    
    while conversation_active:
        # Stoppe Konversation wenn keine Personen mehr erkannt sind
        if len(last_detection_time) == 0:
            speak("I don't see anyone anymore. Ending conversation.")
            conversation_active = False
            break
            
        user_input = record_command()
        if user_input is None or "start" in user_input.lower():  # Stop command detected
            speak("Goodbye! Let me know if you need anything else.")
            conversation_active = False
            break
            
        print("User:", user_input)
        response = query_gemini(user_input)
        print("AI:", response)
        speak(response)

def face_detection_process(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:  # Poison pill to stop the process
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        result_queue.put((face_locations, face_encodings))

def main_loop():
    global conversation_active, last_detection_time, last_greeting_time, tracking_status
    
    # Initialisiere die Queues für die Prozess-Kommunikation
    frame_queue = Queue()
    result_queue = Queue()
    
    # Starte den Face-Detection Prozess
    num_processes = max(1, multiprocessing.cpu_count() - 1)  # Nutze alle Kerne außer einen
    detection_processes = []
    for _ in range(num_processes):
        p = Process(target=face_detection_process, args=(frame_queue, result_queue))
        p.daemon = True
        p.start()
        detection_processes.append(p)
    
    video_capture = cv2.VideoCapture(0)
    FORGET_TIMEOUT = 20
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Sende Frame an den Detection-Prozess
            frame_queue.put(frame)
            
            # Hole Ergebnisse vom Detection-Prozess
            face_locations, face_encodings = result_queue.get()
            
            current_time = datetime.now()
            
            # Prüfe für jeden bekannten Benutzer, ob er "vergessen" werden soll
            for user in list(last_detection_time.keys()):
                if (current_time - last_detection_time[user]).total_seconds() > FORGET_TIMEOUT:
                    print(f"Forgotten: {user}")
                    last_detection_time.pop(user)
                    last_greeting_time.pop(user, None)
                    tracking_status.pop(user, None)
                    
                    if len(last_detection_time) == 0 and conversation_active:
                        print("All users forgotten, conversation will end")
                elif (current_time - last_detection_time[user]).total_seconds() > 2:  # 2 Sekunden ohne Erkennung
                    if tracking_status.get(user) != "lost":
                        print(f"Lost tracking: {user}")
                        tracking_status[user] = "lost"

            # Liste der aktuell erkannten Benutzer für diesen Frame
            current_frame_users = set()

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                
                if True in matches:
                    match_index = matches.index(True)
                    detected_user = known_names[match_index]
                    current_frame_users.add(detected_user)
                    current_time = datetime.now()

                    # Aktualisiere die letzte Erkennungszeit
                    last_detection_time[detected_user] = current_time

                    # Begrüße nur, wenn der Benutzer nicht in last_greeting_time ist
                    if detected_user not in last_greeting_time:
                        print(f"Found: {detected_user}")
                        speak(f"Hello {detected_user}, how can I assist you today?")
                        last_greeting_time[detected_user] = current_time
                        tracking_status[detected_user] = "tracking"
                    elif tracking_status.get(detected_user) == "lost":
                        print(f"Found: {detected_user}")
                        tracking_status[detected_user] = "tracking"
                    elif tracking_status.get(detected_user) != "tracking":
                        print(f"Tracking: {detected_user}")
                        tracking_status[detected_user] = "tracking"

            # Display the frame with names
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add name label if recognized
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        for _ in detection_processes:
            frame_queue.put(None)  # Send poison pills
        for p in detection_processes:
            p.join()
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    try:
        multiprocessing.freeze_support()
        load_known_faces()
        
        # Starte den start-Listener in einem separaten Thread
        start_listener_thread = threading.Thread(target=listen_for_start, daemon=True)
        start_listener_thread.start()
        
        main_loop()
        
    except KeyboardInterrupt:
        pygame.mixer.quit()
        print("\nProgram terminated.")

