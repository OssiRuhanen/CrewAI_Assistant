#!/usr/bin/env python
import sys
import warnings
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import openai
import tempfile
import os
import io
import soundfile as sf
import time
from dotenv import load_dotenv
import threading
from typing import Optional
from google.cloud import texttospeech

from agent_assistant.crew import AgentAssistant
from agent_assistant.memory import MemoryManager
from agent_assistant.task_manager import TaskManager

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Reset audio settings and use MME backend
sd.default.reset()

# Add a DEBUG flag at the top of the file
DEBUG = False

# --- Configuration ---
# Load API Key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("VIRHE: OPENAI_API_KEY ympäristömuuttujaa ei ole asetettu.")
    print("Luo .env tiedosto, jossa on OPENAI_API_KEY=sinun_avain")
    sys.exit(1)

# Models (adjust as needed)
CHAT_MODEL = "gpt-4-turbo"  # Or "gpt-4", "gpt-4o", etc.
TTS_MODEL = "tts-1"          # tts-1 or tts-1-hd
TTS_VOICE = "alloy"          # Options: alloy, echo, fable, onyx, nova, shimmer

# Audio Recording Settings
SAMPLE_RATE = 16000  # Hz (Whisper prefers 16kHz)
CHANNELS = 1
SILENCE_THRESHOLD = 500  # RMS amplitude. Adjusted based on test results
INPUT_DEVICE = None  # Will be set to default if None
OUTPUT_DEVICE = None  # Will be set to default if None
PRE_RECORD_BUFFER_SECONDS = 0.5  # Buffer to capture the beginning of speech

# Google TTS Configuration
USE_GOOGLE_TTS = True  # Set to True to use Google TTS
SELECTED_GOOGLE_TTS_VOICE = "fi-FI-Wavenet-A"  # Default, can be changed by user

# Set up absolute paths for knowledge directory and conversation history
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")
CONVERSATION_HISTORY_PATH = os.path.join(KNOWLEDGE_DIR, "conversation_history.txt")

# Ensure knowledge directory exists
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Truncate conversation_history.txt at session start
with open(CONVERSATION_HISTORY_PATH, "w", encoding="utf-8") as f:
    f.write("# Conversation History\n# This file stores a log of conversations with the user.\n# Each conversation is timestamped and includes both user and assistant messages.\n\n# Format: [Date Time] - [Speaker] - [Message]\n")

# --- Initialization ---
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Virhe OpenAI-asiakkaan alustamisessa: {e}")
    sys.exit(1)

# Initialize memory manager and task manager
memory_manager = MemoryManager()
task_manager = TaskManager()

# --- Chat History for Direct OpenAI API ---
# Update system message for natural memory responses
chat_history = [
    {"role": "system", "content": """
Olet avulias ja ytimekäs ääniavustaja. Vastaa selkeästi ja luonnollisesti.
Jos käyttäjä pyytää sinua kirjaamaan jotain muistiin (esim. 'kirjaa ylös', 'muista tämä'), vastaa ystävällisesti tyyliin 'Selvä, merkitsen tämän muistiin!' ja jatka keskustelua normaalisti. Älä kerro käyttäjälle, ettet voi oikeasti tallentaa tietoja.
Jos käyttäjä pyytää edistyneitä ominaisuuksia tai työkaluja, ehdota CrewAI-tilan käyttöä sanomalla 'käytä agenttia' seuraavassa viestissään.
"""}
]

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    try:
        AgentAssistant().crew().kickoff()
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        AgentAssistant().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2])
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AgentAssistant().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    try:
        AgentAssistant().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2])
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def check_upcoming_tasks():
    """Background thread to check for upcoming tasks."""
    while True:
        upcoming_tasks = task_manager.get_upcoming_tasks(minutes_ahead=5)
        for task in upcoming_tasks:
            message = f"Muistutus: {task['description']} kello {task['time'].split()[1]}"
            print(f"\n{message}")
            speak_text(message)
            # Mark as done to avoid repeated alerts
            task_manager.mark_task_done(task['description'])
        time.sleep(60)  # Check every minute

def process_task_command(message: str) -> Optional[str]:
    """Process task-related commands."""
    if message.lower().startswith("lisää muistutus"):
        try:
            # Extract time and description
            parts = message.split("kello")
            if len(parts) != 2:
                return "Virheellinen muistutuksen muoto. Käytä muotoa: 'lisää muistutus [tehtävä] kello [aika]'"
            
            description = parts[0].replace("lisää muistutus", "").strip()
            time_str = parts[1].strip()
            
            if task_manager.add_task(description, time_str):
                return f"Muistutus lisätty: {description} kello {time_str}"
            else:
                return "Virheellinen aika. Käytä muotoa HH:MM"
        except Exception as e:
            return f"Virhe muistutuksen lisäämisessä: {str(e)}"
    
    elif message.lower().startswith("näytä muistutukset"):
        tasks = task_manager.get_all_pending_tasks()
        if not tasks:
            return "Ei aktiivisia muistutuksia."
        response = "Aktiiviset muistutukset:\n"
        for task in tasks:
            response += f"- {task['description']} kello {task['time'].split()[1]}\n"
        return response
    
    return None

def chat():
    """
    Chat with the assistant agent in a loop.
    """
    crew = AgentAssistant().crew()
    print("Kirjoita 'exit' lopettaaksesi.")
    while True:
        user_message = input("Sinä: ")
        if user_message.lower() in ["exit", "quit"]:
            break
        try:
            # Check for task commands first
            task_response = process_task_command(user_message)
            if task_response:
                print("Avustaja:", task_response)
                continue
            
            # Log user message to conversation history
            memory_manager.log_conversation("User", user_message)
            
            result = crew.kickoff(inputs={"user_message": user_message})
            print("Avustaja:", result)
            
            # Log assistant response to conversation history
            memory_manager.log_conversation("Assistant", result)
            
            # Extract memories from the conversation
            memory_manager.extract_memory_from_conversation(user_message, result)
            
        except Exception as e:
            print(f"Virhe: {e}")

def record_audio_with_silence_detection(max_duration=30, silence_threshold=SILENCE_THRESHOLD, silence_duration=1):
    """
    Records audio until silence is detected for a specified duration.
    
    Args:
        max_duration: Maximum recording duration in seconds
        silence_threshold: Audio level below which is considered silence
        silence_duration: Duration of silence in seconds before stopping recording
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Initialize variables
    sample_rate = SAMPLE_RATE
    silence_frames = 0
    silence_frames_threshold = int(silence_duration * sample_rate / 1024)  # Convert to frames
    audio_chunks = []
    recording = False  # Start in non-recording mode
    consecutive_silence = 0
    consecutive_sound = 0
    sound_threshold = 3  # Number of consecutive sound frames needed to start recording
    
    # Create a callback function to process audio in chunks
    def audio_callback(indata, frames, time, status):
        nonlocal silence_frames, recording, audio_chunks, consecutive_silence, consecutive_sound
        
        # Calculate the RMS of the current chunk
        rms = np.sqrt(np.mean(indata**2))
        
        # Check if this is silence
        if rms < silence_threshold / 32768:  # Normalize threshold
            consecutive_silence += 1
            consecutive_sound = 0
        else:
            consecutive_silence = 0
            consecutive_sound += 1
        
        # Start recording only after detecting continuous sound
        if not recording and consecutive_sound >= sound_threshold:
            recording = True
        
        # If we're recording, add the chunk to our list
        if recording:
            audio_chunks.append(indata.copy())
            
            # Check if we've had enough silence to stop
            if consecutive_silence >= silence_frames_threshold:
                recording = False
    
    # Start the input stream
    with sd.InputStream(samplerate=sample_rate, channels=CHANNELS, dtype='float32', 
                       device=INPUT_DEVICE, callback=audio_callback, blocksize=1024):
        # Record until we detect enough silence or hit the max duration
        start_time = time.time()
        while (time.time() - start_time) < max_duration:
            time.sleep(0.1)
            if not recording and (time.time() - start_time) > 5:  # Timeout after 5 seconds of no sound
                break
    
    # If we have recorded anything, concatenate the chunks
    if audio_chunks:
        audio_data = np.concatenate(audio_chunks)
        # Convert to int16 for compatibility with the rest of the code
        audio_data = (audio_data * 32768).astype(np.int16)
        duration = len(audio_data)/sample_rate
        if duration < 0.5:  # If recording is too short, probably noise
            return None, sample_rate
        return audio_data, sample_rate
    else:
        return None, sample_rate

def transcribe_audio(audio, fs):
    """Transcribes audio using OpenAI's Whisper API."""
    if audio is None:
        return None
    
    try:
        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            wav.write(temp_filename, fs, audio)
        
        # Open the file for transcription
        with open(temp_filename, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        return transcript.text.strip()
    except Exception as e:
        print(f"Virhe äänen transkriboinnissa: {e}")
        import traceback
        traceback.print_exc()
        return None

def speak_text_google(text_to_speak):
    """Uses Google Cloud TTS to convert text to speech and plays it."""
    if not text_to_speak:
        return
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
    voice = texttospeech.VoiceSelectionParams(
        language_code="fi-FI",
        name=SELECTED_GOOGLE_TTS_VOICE,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    audio_stream = io.BytesIO(response.audio_content)
    data, samplerate = sf.read(audio_stream, dtype='int16')
    sd.play(data, samplerate)
    sd.wait()

def speak_text(text_to_speak):
    """Speaks text using the selected TTS engine."""
    if not text_to_speak:
        return
    if USE_GOOGLE_TTS:
        speak_text_google(text_to_speak)
    else:
        # OpenAI TTS fallback
        try:
            response = openai_client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text_to_speak,
                response_format="opus"
            )
            audio_bytes = response.content
            audio_stream = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(audio_stream)
            if len(data) == 0:
                return
            sd.play(data, samplerate, device=OUTPUT_DEVICE)
            sd.wait()
        except Exception as e:
            print(f"Odottamaton virhe tapahtui tekstistä-puheeksi-muuntamisessa: {e}")

def get_openai_response(current_history):
    """Sends the chat history to OpenAI and gets the assistant's response."""
    print("Haetaan vastaus OpenAI:lta...")
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=current_history
        )
        assistant_message = response.choices[0].message.content
        return assistant_message
    except openai.APIConnectionError as e:
        print(f"OpenAI API Yhteysvirhe: {e}")
    except openai.RateLimitError as e:
        print(f"OpenAI Nopeusrajoitus ylitetty: {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI API Tilavirhe (Tila {e.status_code}): {e.response}")
    except Exception as e:
        print(f"Odottamaton virhe tapahtui OpenAI-keskustelun kutsumisessa: {e}")
    return None # Return None on error

def process_with_openai(user_message):
    """Process user message with direct OpenAI API for faster responses."""
    global chat_history
    
    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_message})
    
    # Get response from OpenAI
    response = get_openai_response(chat_history)
    
    if response:
        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": response})
        
        # Log conversation to memory
        memory_manager.log_conversation("User", user_message)
        memory_manager.log_conversation("Assistant", response)
        
        # Extract memories from the conversation
        memory_manager.extract_memory_from_conversation(user_message, response)
        
        return response
    else:
        return "Valitettavasti en pystynyt luomaan vastausta. Yritä uudelleen."

def process_with_crewai(user_message):
    """Process user message with CrewAI for advanced capabilities."""
    try:
        # Initialize CrewAI
        crew = AgentAssistant().crew()
        
        # Process with CrewAI
        result = crew.kickoff(inputs={"user_message": user_message})
        
        # Log conversation to memory
        memory_manager.log_conversation("User", user_message)
        memory_manager.log_conversation("Assistant", result)
        
        # Extract memories from the conversation
        memory_manager.extract_memory_from_conversation(user_message, result)
        
        return result
    except Exception as e:
        print(f"CrewAI virhe: {str(e)}")
        return "Valitettavasti CrewAI-tila ei toiminut. Yritä uudelleen tai palaa suoraan OpenAI-tilaan sanomalla 'käytä puhetta'."

def voice_chat():
    """Voice chat function that uses OpenAI API for quick responses by default."""
    print("Puhechat käynnissä. Puhu nyt...")
    print("Sano 'käytä agenttia' käyttääksesi CrewAI-tilaa tai 'käytä puhetta' palataksesi suoraan OpenAI-tilaan.")
    
    # Default to direct mode
    use_crewai = False
    
    while True:
        try:
            # Record audio
            audio_data = record_audio_with_silence_detection()
            
            # Transcribe audio to text
            transcription = transcribe_audio(audio_data[0], audio_data[1])
            
            if not transcription:
                if DEBUG:
                    print("Ei kuultua ääntä. Yritä uudelleen.")
                continue
                
            print(f"Sinä: {transcription}")
            
            # Check for mode switching commands
            if "käytä agenttia" in transcription.lower():
                use_crewai = True
                response = "Nyt käytän CrewAI-tilaa. Tämä antaa minulle pääsyn edistyneisiin ominaisuuksiin ja työkaluihin."
                print(f"Avustaja: {response}")
                speak_text(response)
                continue
                
            elif "käytä puhetta" in transcription.lower():
                use_crewai = False
                response = "Nyt käytän suoraa OpenAI-tilaa nopeampien vastausten saamiseksi."
                print(f"Avustaja: {response}")
                speak_text(response)
                continue
            
            # Check for task commands first
            task_response = process_task_command(transcription)
            if task_response:
                print(f"Avustaja: {task_response}")
                speak_text(task_response)
                continue
            
            # Process based on mode
            if use_crewai:
                # Use CrewAI for processing
                response = process_with_crewai(transcription)
            else:
                # Use direct OpenAI API for faster responses
                response = process_with_openai(transcription)
            
            print(f"Avustaja: {response}")
            
            # Convert response to speech
            speak_text(response)
            
        except KeyboardInterrupt:
            print("\nPuhechat päättyy...")
            break
        except Exception as e:
            print(f"Virhe: {str(e)}")
            print("Yritä uudelleen.")

def list_google_finnish_voices():
    """List available Finnish voices from Google Cloud TTS and allow user to select one. Play a short example after selection."""
    global SELECTED_GOOGLE_TTS_VOICE
    try:
        client = texttospeech.TextToSpeechClient()
        voices = client.list_voices(language_code="fi-FI")
        print("\n=== Käytettävissä olevat suomenkieliset Google TTS -äänet ===")
        for idx, voice in enumerate(voices.voices):
            gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
            print(f"[{idx+1}] Nimi: {voice.name}, Sukupuoli: {gender}, Sample Rate: {voice.natural_sample_rate_hertz}")
        print("==========================================================\n")
        valinta = input("Valitse äänen numero (tai paina Enter käyttääksesi nykyistä): ").strip()
        if valinta.isdigit():
            idx = int(valinta) - 1
            if 0 <= idx < len(voices.voices):
                SELECTED_GOOGLE_TTS_VOICE = voices.voices[idx].name
                print(f"Valittu ääni: {SELECTED_GOOGLE_TTS_VOICE}")
            else:
                print("Virheellinen valinta. Käytetään nykyistä ääntä.")
        else:
            print(f"Käytetään nykyistä ääntä: {SELECTED_GOOGLE_TTS_VOICE}")
        # Play a short example with the selected voice
        example_text = "Tämä on esimerkkilause valitulla äänellä."
        print("Toistetaan esimerkkilause...")
        speak_text_google(example_text)
    except Exception as e:
        print(f"Virhe äänien listauksessa: {e}")

def main():
    global DEBUG
    # Start the task checking thread
    task_thread = threading.Thread(target=check_upcoming_tasks, daemon=True)
    task_thread.start()
    
    print("\n🎙️ Ääniavustaja 💬")
    print("---------------------------------------")
    print("Kirjoita 'voice' aloittaaksesi älykkään äänikeskustelutilan")
    print("Kirjoita 'text' aloittaaksesi tekstikeskustelutilan")
    print("Kirjoita 'memories' nähdäksesi viimeisimmät muistot")
    print("Kirjoita 'ideas' nähdäksesi viimeisimmät ideat")
    print("Kirjoita 'voices' nähdäksesi käytettävissä olevat Google TTS -äänet")
    print("Kirjoita 'debug' vaihtaaksesi debug-tilan päälle/pois")
    print("Kirjoita 'quit' tai 'exit' lopettaaksesi istunnon")
    print("\nMuistutusten käyttö:")
    print("- 'lisää muistutus [tehtävä] kello [aika]'")
    print("- 'näytä muistutukset'")
    
    while True:
        try:
            command = input("\nSyötä komento (voice/text/memories/ideas/voices/debug/quit): ").strip().lower()
            
            if command in ["quit", "exit"]:
                print("Näkemiin!")
                break
            elif command == "voice":
                voice_chat()
            elif command == "text":
                chat()
            elif command == "memories":
                memories = memory_manager.get_recent_memories(10)
                if memories:
                    print("\n=== Viimeisimmät muistot ===")
                    for memory in memories:
                        print(memory)
                else:
                    print("Ei vielä muistoja.")
            elif command == "ideas":
                ideas = memory_manager.get_recent_ideas(10)
                if ideas:
                    print("\n=== Viimeisimmät ideat ===")
                    for idea in ideas:
                        print(idea)
                else:
                    print("Ei vielä ideoita.")
            elif command == "voices":
                list_google_finnish_voices()
            elif command == "debug":
                DEBUG = not DEBUG
                print(f"Debug-tila {'päällä' if DEBUG else 'pois päältä'}.")
            else:
                print("Virheellinen komento. Kirjoita 'voice' äänikeskustelua varten, 'text' tekstikeskustelua varten, 'memories' muistojen katsomista varten, 'ideas' ideoiden katsomista varten, 'voices' äänten listaukseen, 'debug' debug-tilan vaihtoon tai 'quit' lopettaaksesi.")
                
        except KeyboardInterrupt:
            print("\nKäyttäjän keskeyttämä. Poistutaan.")
            break
        except EOFError:
            print("\nSyötteiden virta suljettu. Poistutaan.")
            break
        except Exception as e:
            print(f"\nOdottamaton virhe tapahtui: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    main()
