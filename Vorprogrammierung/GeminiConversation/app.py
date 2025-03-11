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

# Globale Variablen
model_path = "vosk-model-en-us-0.22"
activation_word = "jojo"
API_KEY = "AIzaSyDd5x5Xc6potNayl0BDkhxe7B2YN2iuvyc"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
SYSTEM_PROMPT = "You are a Helpfull Roboter in a Programm which is made for speaking with other people. However do not use any sort of text enhancer for exaple markdown because your promp is diractly spoken out and not use any - and *. The User Prompt is:"

# Initialisierungen
vosk_model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()

# Pygame für unterbrechbare Soundwiedergabe
pygame.mixer.init()

# Flags für Steuerung
activation_detected = False
stop_playback = False
is_speaking = False

# Chatverlauf
chat_history = []

def speak(text):
    global is_speaking, stop_playback
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_filename = fp.name

        is_speaking = True
        stop_playback = False
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy() and not stop_playback:
            sleep(0.1)

    finally:
        is_speaking = False
        pygame.mixer.music.stop()
        os.remove(temp_filename)

def callback(indata, frames, time, status):
    audio_queue.put(bytes(indata))

def listen_for_activation():
    global activation_detected, stop_playback
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                if activation_word in text:
                    if is_speaking:
                        stop_playback = True
                        activation_detected = False
                    else:
                        activation_detected = True
                    return

def record_command():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening...")
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
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

def main_loop():
    global activation_detected, stop_playback
    while True:
        print("Waiting for activation word...")
        listen_for_activation()
        
        if stop_playback:
            stop_playback = False
            activation_detected = True
        
        if activation_detected:
            speak("Yes?")
            activation_detected = False
            
            user_input = record_command()
            print("User:", user_input)
            
            response = query_gemini(user_input)
            print("AI:", response)
            
            speak_thread = threading.Thread(target=speak, args=(response,))
            speak_thread.start()
            
            # Überwache während der Wiedergabe auf Unterbrechung
            listen_thread = threading.Thread(target=listen_for_activation)
            listen_thread.start()
            
            speak_thread.join()
            listen_thread.join()

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        pygame.mixer.quit()
        print("\nProgramm beendet.")




