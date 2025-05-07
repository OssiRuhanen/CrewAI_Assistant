import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agent_assistant.main import process_with_crewai

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speaking rate
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Set Finnish voice if available
voices = engine.getProperty('voices')
finnish_voice = None
for voice in voices:
    if "finnish" in voice.name.lower():
        finnish_voice = voice
        break
if finnish_voice:
    engine.setProperty('voice', finnish_voice.id)

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300  # Adjust based on your microphone
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8  # Shorter pause threshold for more responsive recognition

# Create a queue for text-to-speech
tts_queue = queue.Queue()

def text_to_speech_worker():
    """Worker function for text-to-speech queue."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

def speak(text):
    """Add text to the speech queue."""
    tts_queue.put(text)

def save_conversation(text):
    """Save conversation to history file."""
    history_file = os.path.join(project_root, "src", "agent_assistant", "knowledge", "conversation_history.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    # Create file with header if it doesn't exist
    if not os.path.exists(history_file):
        with open(history_file, 'w', encoding='utf-8') as f:
            f.write("# Conversation History\n")
            f.write("# Format: [Timestamp] Speaker: Message\n\n")
    
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {text}\n")

def process_command(text):
    """Process voice commands."""
    text = text.lower()
    
    # Check for specific commands
    if "aja" in text and any(agent_word in text for agent_word in ["agentit", "agendit", "agent", "agend"]):
        # Process with CrewAI for task management
        result = process_with_crewai("Käy läpi keskusteluhistoria ja päivitä tehtävät")
        if result:
            speak(result)
            save_conversation(f"Avustaja: {result}")
    elif any(agent_word in text for agent_word in ["agentit", "agendit", "agent", "agend"]):
        # Switch to CrewAI mode
        speak("Agenttitila käytössä.")
        save_conversation("Avustaja: Agenttitila käytössä.")
    elif "puhe" in text:
        # Switch to direct OpenAI mode
        speak("Puhetila käytössä.")
        save_conversation("Avustaja: Puhetila käytössä.")
    elif "lopeta" in text:
        speak("Lopetetaan keskustelu")
        return False
    else:
        # Process with CrewAI
        result = process_with_crewai(text)
        if result:
            speak(result)
            save_conversation(f"Avustaja: {result}")
    
    return True

def main():
    # Start text-to-speech worker thread
    tts_thread = threading.Thread(target=text_to_speech_worker, daemon=True)
    tts_thread.start()
    
    print("Voice chat käynnistetty. Puhu nyt...")
    speak("Voice chat käynnistetty. Puhu nyt...")
    
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                try:
                    print("\nKuuntelen...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    try:
                        text = recognizer.recognize_google(audio, language="fi-FI")
                        print(f"Sinä: {text}")
                        save_conversation(f"Sinä: {text}")
                        
                        if not process_command(text):
                            break
                            
                    except sr.UnknownValueError:
                        print("En ymmärtänyt mitä sanoit")
                    except sr.RequestError as e:
                        print(f"Virhe Google Speech Recognition -palvelussa: {e}")
                        
                except sr.WaitTimeoutError:
                    continue
                    
    except KeyboardInterrupt:
        print("\nLopetetaan...")
    finally:
        # Stop text-to-speech worker
        tts_queue.put(None)
        tts_thread.join()
        print("Voice chat lopetettu.")

if __name__ == "__main__":
    main() 