#!/usr/bin/env python
import sys
import warnings
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav # Keep for potential future use? Though sf is preferred now
import openai
import tempfile
import os
import io
import soundfile as sf
import time
from dotenv import load_dotenv
import threading # Import threading
import queue
import traceback

# Assume AgentAssistant and MemoryManager are in the correct path
try:
    from agent_assistant.crew import AgentAssistant
    from agent_assistant.memory import MemoryManager
except ImportError:
    print("VIRHE: Varmista, että agent_assistant-moduuli on asennettu ja PYTHONPATH on oikein.")
    # Mock classes if running without the module for testing basic voice functionality
    class AgentAssistant:
        def crew(self):
            class MockCrew:
                def kickoff(self, inputs=None):
                    print(f"[Mock Crew] Käsittelyssä: {inputs.get('user_message', '')}")
                    time.sleep(1)
                    return f"Mock vastaus viestiin: {inputs.get('user_message', '')}"
            return MockCrew()
    class MemoryManager:
        def log_conversation(self, role, message): pass
        def extract_memory_from_conversation(self, user_msg, assistant_msg): pass
        def get_recent_memories(self, count): return ["Mock muisto 1", "Mock muisto 2"]
        def get_recent_ideas(self, count): return ["Mock idea 1"]

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("VIRHE: OPENAI_API_KEY ympäristömuuttujaa ei ole asetettu.")
    print("Luo .env tiedosto, jossa on OPENAI_API_KEY=sinun_avain")
    sys.exit(1)

CHAT_MODEL = "gpt-4o"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 500 # Adjust based on microphone sensitivity
INPUT_DEVICE = None
OUTPUT_DEVICE = None
SILENCE_DURATION_SECONDS = 1.0
MAX_RECORD_DURATION_SECONDS = 30

# --- Initialization ---
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Virhe OpenAI-asiakkaan alustamisessa: {e}")
    sys.exit(1)

memory_manager = MemoryManager()

chat_history = [
    {"role": "system", "content": """Olet avulias ja ytimekäs ääniavustaja. Vastaa selkeästi ja luonnollisesti suomeksi.
    Jos käyttäjä pyytää edistyneitä ominaisuuksia tai työkaluja, ehdota CrewAI-tilan käyttöä sanomalla 'käytä agenttia' seuraavassa viestissään.
    Älä mainitse komentoja kuten 'pysäytä' vastauksissasi.
    """}
]

# Global flag/thread reference to manage playback thread
current_playback_thread = None
playback_lock = threading.Lock() # To manage access to the thread variable

# --- CrewAI functions (kept as is) ---
# ... (run, train, replay, test, chat functions remain the same) ...
def run():
    """Run the crew."""
    try: AgentAssistant().crew().kickoff()
    except Exception as e: raise Exception(f"An error occurred while running the crew: {e}")
def train():
    """Train the crew."""
    try: AgentAssistant().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2])
    except Exception as e: raise Exception(f"An error occurred while training the crew: {e}")
# etc. for replay, test, chat

# --- Audio Processing Functions ---

def record_audio_with_silence_detection(
    max_duration=MAX_RECORD_DURATION_SECONDS,
    silence_threshold=SILENCE_THRESHOLD,
    silence_duration=SILENCE_DURATION_SECONDS
    ):
    """Records audio until silence or max duration. Uses a queue."""
    print("Kuuntelen... ", end="", flush=True) # More concise prompt

    sample_rate = SAMPLE_RATE
    block_size = 1024
    silence_blocks_threshold = int(silence_duration * sample_rate / block_size)
    max_blocks = int(max_duration * sample_rate / block_size)
    normalized_threshold = silence_threshold / 32768.0

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    has_speech = threading.Event()
    recording_active = True # Flag to control callback activity

    def audio_callback(indata, frames, time_info, status):
        nonlocal recording_active
        if not recording_active:
            return # Stop processing if recording should be inactive
        if status:
            print(f"Stream status: {status}", file=sys.stderr)

        rms = np.sqrt(np.mean(indata**2))

        if has_speech.is_set() or rms >= normalized_threshold:
            if not has_speech.is_set():
                print(" Ääntä havaittu.", flush=True) # Indicate speech start
                has_speech.set()
            audio_queue.put(indata.copy())
        # No pre-buffering for simplicity here

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype='float32',
        device=INPUT_DEVICE,
        callback=audio_callback,
        blocksize=block_size
    )

    recorded_blocks = 0
    silence_counter = 0
    recorded_data = []

    with stream:
        start_time = time.time()
        while recorded_blocks < max_blocks:
            try:
                chunk = audio_queue.get(timeout=0.1)
                recorded_data.append(chunk)
                recorded_blocks += 1

                if has_speech.is_set():
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms < normalized_threshold:
                        silence_counter += 1
                        if silence_counter >= silence_blocks_threshold:
                            print(" Hiljaisuus havaittu.", flush=True)
                            break
                    else:
                        silence_counter = 0
            except queue.Empty:
                current_time = time.time()
                if current_time - start_time > max_duration:
                     print(" Maksimikesto saavutettu.", flush=True)
                     break
                # Check for silence timeout even if queue is empty
                if has_speech.is_set() and silence_counter > 0:
                    silence_counter += int(0.1 * sample_rate / block_size) # Approx blocks in timeout
                    if silence_counter >= silence_blocks_threshold:
                        print(" Hiljaisuus havaittu (timeout).", flush=True)
                        break
                continue

    recording_active = False # Signal callback to stop processing
    print("Tallennus päättyi.")

    if recorded_data:
        audio_data = np.concatenate(recorded_data, axis=0)
        return audio_data, sample_rate
    else:
        return None, sample_rate

def transcribe_audio(audio_data, sample_rate):
    """Transcribes audio using OpenAI Whisper API via in-memory buffer."""
    if audio_data is None or len(audio_data) == 0:
        print("Ei äänidataa transkriboitavaksi.")
        return None
    try:
        print("Transkriboidaan...", end="", flush=True)
        buffer = io.BytesIO()
        # Ensure data is float32 before writing, if needed (should be from record func)
        if audio_data.dtype != np.float32:
             audio_data = audio_data.astype(np.float32) / 32767.0 # Example conversion if needed

        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        file_tuple = ("audio.wav", buffer, "audio/wav")

        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=file_tuple
        )
        print(" Valmis.")
        return transcript.text.strip()
    except Exception as e:
        print(f" Virhe transkriptiossa: {e}")
        return None

# --- Modified TTS and Playback ---

def generate_tts_audio(text_to_speak):
    """Generates TTS audio data using OpenAI, returns data and samplerate."""
    if not text_to_speak:
        print("Varoitus: Ei tekstiä puhuttavaksi.")
        return None, None

    print("Luodaan puhetta...", end="", flush=True)
    try:
        response = openai_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text_to_speak,
            response_format="opus" # Using opus for efficiency
        )
        audio_bytes = response.content
        audio_stream = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_stream, dtype='float32')
        print(" Valmis.")
        return data, samplerate
    except sf.SoundFileError as e:
        print(f" Virhe äänidatan lukemisessa (libsndfile/ffmpeg): {e}")
    except sd.PortAudioError as e:
         print(f"\nVirhe äänilaitteessa (TTS Generation?): {e}")
    except openai.APIConnectionError as e:
        print(f" OpenAI API Yhteysvirhe (TTS): {e}")
    except openai.RateLimitError as e:
        print(f" OpenAI Nopeusrajoitus ylitetty (TTS): {e}")
    except openai.APIStatusError as e:
        print(f" OpenAI API Tilavirhe (TTS) (Tila {e.status_code}): {e.response}")
    except Exception as e:
        print(f" Odottamaton virhe TTS-luonnissa: {e}")
    return None, None # Return None on error

def play_audio_task(audio_data, samplerate):
    """Task function to play audio, meant to be run in a thread."""
    global current_playback_thread
    try:
        print("Toistetaan vastaus...")
        sd.play(audio_data, samplerate, device=OUTPUT_DEVICE)
        sd.wait()
        print("Toisto valmis.")
    except sd.PortAudioError as e:
        print(f"\nVirhe äänilaitteessa (Playback Thread): {e}")
    except Exception as e:
        print(f"\nVirhe toistothreadissa: {e}")
    finally:
        # Clean up: Signal that playback is finished
        with playback_lock:
            if threading.current_thread() == current_playback_thread:
                 current_playback_thread = None


# --- Core Logic Functions ---

def get_openai_response(current_history):
    """Gets response from OpenAI Chat API."""
    print("Hakee vastausta OpenAI:lta...", end="", flush=True)
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=current_history
        )
        assistant_message = response.choices[0].message.content
        print(" Valmis.")
        return assistant_message
    except Exception as e:
        print(f" Virhe OpenAI-kutsussa: {e}")
        return None

def process_with_openai(user_message):
    """Processes message using direct OpenAI API."""
    global chat_history
    chat_history.append({"role": "user", "content": user_message})
    assistant_response = get_openai_response(chat_history)
    if assistant_response:
        chat_history.append({"role": "assistant", "content": assistant_response})
        memory_manager.log_conversation("User", user_message)
        memory_manager.log_conversation("Assistant", assistant_response)
        memory_manager.extract_memory_from_conversation(user_message, assistant_response)
        return assistant_response
    else:
        error_message = "Pahoittelut, en saanut vastausta OpenAI:lta."
        chat_history.append({"role": "assistant", "content": error_message}) # Add error to history too
        return error_message

def process_with_crewai(user_message):
    """Processes message using CrewAI."""
    try:
        print("Käynnistetään CrewAI...", end="", flush=True)
        crew = AgentAssistant().crew()
        result = crew.kickoff(inputs={"user_message": user_message})
        print(" Valmis.")
        memory_manager.log_conversation("User (CrewAI)", user_message)
        memory_manager.log_conversation("Assistant (CrewAI)", result)
        memory_manager.extract_memory_from_conversation(user_message, result)
        return result
    except Exception as e:
        print(f" CrewAI virhe: {str(e)}")
        return "Valitettavasti CrewAI-tila kohtasi virheen."

# --- Main Voice Chat Loop ---

def stop_current_playback():
    """Stops any active playback managed by sounddevice."""
    global current_playback_thread
    was_playing = False
    with playback_lock:
        if current_playback_thread and current_playback_thread.is_alive():
            print("Pysäytetään aktiivinen toisto...")
            sd.stop() # The core function to stop sounddevice streams
            was_playing = True
            # Note: We don't forcefully kill the thread, sd.stop() should interrupt sd.wait()
            # letting the thread exit gracefully via its finally block.
            # We might set current_playback_thread to None here or let the thread do it.
            # Let the thread clean itself up for now.
        elif sd.get_stream().active: # Check if maybe sd.stop didn't catch it last time
             print("Streami aktiivinen, yritetään pysäyttää...")
             sd.stop()
             was_playing = True

    if was_playing:
        print("Toisto pysäytetty.")
        time.sleep(0.1) # Short pause to allow stream closure
    # else:
    #     print("Ei aktiivista toistoa pysäytettäväksi.") # Optional debug msg


def voice_chat():
    """Main voice chat loop with interruptible TTS."""
    global current_playback_thread

    print("\n--- ÄÄNICHAT KÄYNNISSÄ ---")
    print("Puhu. Ole hiljaa lopettaaksesi tallennuksen.")
    print("Komennot: 'käytä agenttia', 'käytä puhetta', 'pysäytä' (toiston aikana).")
    print("Paina Ctrl+C lopettaaksesi.")
    print("-" * 30)

    use_crewai = False
    current_mode = "Suora OpenAI"

    while True:
        try:
            print(f"\n[{current_mode}]")

            # *** Crucial: Stop any previous playback before listening ***
            # This prevents issues if user speaks immediately after assistant finishes
            # or if a previous stop command didn't fully clear the thread variable.
            stop_current_playback()

            # 1. Record audio
            audio_data, sample_rate = record_audio_with_silence_detection()

            if audio_data is None:
                print("Ei havaittu puhetta.")
                continue

            # 2. Transcribe
            transcription = transcribe_audio(audio_data, sample_rate)

            if not transcription:
                print("Transkriptio epäonnistui.")
                continue

            print(f"Sinä: {transcription}")
            normalized_transcription = transcription.lower().strip()

            # 3. Handle commands FIRST
            if "pysäytä" in normalized_transcription:
                print("Pysäytyskomento vastaanotettu.")
                stop_current_playback()
                continue # Go back to listening

            elif "käytä agenttia" in normalized_transcription:
                 if not use_crewai:
                     stop_current_playback() # Stop assistant if it was talking
                     use_crewai = True
                     current_mode = "CrewAI"
                     response = "Siirrytään CrewAI-tilaan."
                     print(f"Avustaja: {response}")
                     # Play confirmation using the new threaded method
                     tts_data, tts_sr = generate_tts_audio(response)
                     if tts_data is not None:
                         with playback_lock:
                            current_playback_thread = threading.Thread(target=play_audio_task, args=(tts_data, tts_sr), daemon=True)
                            current_playback_thread.start()
                 else: print("Olet jo CrewAI-tilassa.")
                 continue

            elif "käytä puhetta" in normalized_transcription:
                 if use_crewai:
                     stop_current_playback() # Stop assistant if it was talking
                     use_crewai = False
                     current_mode = "Suora OpenAI"
                     response = "Siirrytään suoraan OpenAI-tilaan."
                     print(f"Avustaja: {response}")
                     tts_data, tts_sr = generate_tts_audio(response)
                     if tts_data is not None:
                         with playback_lock:
                             current_playback_thread = threading.Thread(target=play_audio_task, args=(tts_data, tts_sr), daemon=True)
                             current_playback_thread.start()
                 else: print("Olet jo suorassa OpenAI-tilassa.")
                 continue

            # 4. Process transcription (only if not a command handled above)
            stop_current_playback() # Stop any prior response before generating new one

            assistant_response = None
            if use_crewai:
                assistant_response = process_with_crewai(transcription)
            else:
                assistant_response = process_with_openai(transcription)

            # 5. Generate and Play TTS in a thread
            if assistant_response:
                print(f"Avustaja: {assistant_response}")
                tts_data, tts_sr = generate_tts_audio(assistant_response)
                if tts_data is not None:
                    with playback_lock:
                        # Make sure no other thread is assigned already
                        if current_playback_thread is None or not current_playback_thread.is_alive():
                            current_playback_thread = threading.Thread(target=play_audio_task, args=(tts_data, tts_sr), daemon=True)
                            current_playback_thread.start()
                        else:
                            print("Varoitus: Edellinen toisto vielä aktiivinen? Ei aloitettu uutta.")
            else:
                # Handle processing failure
                error_msg = "Pahoittelut, en voinut käsitellä pyyntöäsi."
                print(f"Avustaja: {error_msg}")
                # Optionally play the error message
                tts_data, tts_sr = generate_tts_audio(error_msg)
                if tts_data is not None:
                    with playback_lock:
                        if current_playback_thread is None or not current_playback_thread.is_alive():
                             current_playback_thread = threading.Thread(target=play_audio_task, args=(tts_data, tts_sr), daemon=True)
                             current_playback_thread.start()

            # Loop continues immediately to record next input

        except KeyboardInterrupt:
            print("\nLopetetaan äänichat...")
            stop_current_playback()
            break
        except Exception as e:
            print(f"\nOdottamaton virhe voice_chat-loopissa: {e}")
            traceback.print_exc()
            stop_current_playback() # Try to stop audio on error
            time.sleep(1) # Pause before retrying


# --- Main Entry Point ---
def main():
    # ... (main menu logic remains the same) ...
    print("\n🎙️ Ääniavustaja 💬")
    print("---------------------------------------")
    print("Kirjoita 'voice' aloittaaksesi älykkään äänikeskustelutilan")
    print("Kirjoita 'text' aloittaaksesi tekstikeskustelutilan (CrewAI)")
    print("Kirjoita 'memories' nähdäksesi viimeisimmät muistot")
    print("Kirjoita 'ideas' nähdäksesi viimeisimmät ideat")
    print("Kirjoita 'quit' tai 'exit' lopettaaksesi istunnon")

    while True:
        try:
            command = input("\nSyötä komento (voice/text/memories/ideas/quit): ").strip().lower()

            if command in ["quit", "exit"]:
                print("Näkemiin!")
                break
            elif command == "voice":
                voice_chat()
            elif command == "text":
                # Assuming 'chat()' uses CrewAI as implemented before
                chat() # Replace with direct call if chat() doesn't exist/work
            elif command == "memories":
                memories = memory_manager.get_recent_memories(10)
                print("\n=== Viimeisimmät muistot ===" if memories else "Ei muistoja.")
                for i, memory in enumerate(memories): print(f"{i+1}. {memory}")
            elif command == "ideas":
                ideas = memory_manager.get_recent_ideas(10)
                print("\n=== Viimeisimmät ideat ===" if ideas else "Ei ideoita.")
                for i, idea in enumerate(ideas): print(f"{i+1}. {idea}")
            else:
                print("Virheellinen komento.")

        except KeyboardInterrupt: print("\nLopetetaan..."); break
        except EOFError: print("\nSyötevirta suljettu. Lopetetaan."); break
        except Exception as e:
            print(f"\nOdottamaton virhe pääloopissa: {e}")
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    # Set default devices if needed
    try:
        if INPUT_DEVICE is None:
            INPUT_DEVICE = sd.default.device[0]
            print(f"Using default input device: {sd.query_devices(INPUT_DEVICE)['name']}")
        if OUTPUT_DEVICE is None:
            OUTPUT_DEVICE = sd.default.device[1]
            print(f"Using default output device: {sd.query_devices(OUTPUT_DEVICE)['name']}")
    except Exception as e:
        print(f"Virhe äänilaitteiden asettamisessa: {e}")
        print("Varmista, että äänilaitteet ovat kytketty ja toimivat.")
        sys.exit(1)

    main()