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
from crewai import Agent, Task, Crew, Process
from agent_assistant.config import load_config
from agent_assistant.tools.task_tools import AddTaskTool, ConversationTool
from agent_assistant.tools.memory_tools import MemoryTool
from agent_assistant.crew import AgentAssistant
from agent_assistant.memory import MemoryManager
from agent_assistant.task_manager import TaskManager
from agent_assistant.config import KNOWLEDGE_DIR, CONFIG_DIR
from agent_assistant.performance import measure_performance, log_performance_metrics
from datetime import datetime
import keyboard
from PIL import ImageGrab
import base64
from langdetect import detect

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Reset audio settings and use MME backend
sd.default.reset()

# Add a DEBUG flag at the top of the file
DEBUG = False

# --- Configuration ---
# Load API Key from .env file
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("VIRHE: OPENAI_API_KEY ympäristömuuttujaa ei ole asetettu.")
    print("Luo .env tiedosto, jossa on OPENAI_API_KEY=sinun_avain")
    sys.exit(1)

# Models (adjust as needed)
CHAT_MODEL = "gpt-4-turbo"  # Or "gpt-4", "gpt-4o", etc.
TTS_MODEL = "tts-1-hd"          # tts-1 or tts-1-hd
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
CONVERSATION_HISTORY_PATH = os.path.join(KNOWLEDGE_DIR, "conversation_history.txt")

# Ensure knowledge directory exists
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Initialize conversation history file if it doesn't exist
if not os.path.exists(CONVERSATION_HISTORY_PATH):
    with open(CONVERSATION_HISTORY_PATH, "w", encoding="utf-8") as f:
        f.write("# Conversation History\n# This file stores a log of conversations with the user.\n# Each conversation is timestamped and includes both user and assistant messages.\n\n# Format: [Date Time] - [Speaker] - [Message]\n")

# --- Initialization ---
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Virhe OpenAI-asiakkaan alustamisessa: {e}")
    sys.exit(1)

# Initialize memory manager and task manager
memory_manager = MemoryManager(knowledge_dir=KNOWLEDGE_DIR)
TASKS_FILE = os.path.join(KNOWLEDGE_DIR, "tasks.json")
task_manager = TaskManager(storage_file=TASKS_FILE)

# --- Chat History for Direct OpenAI API ---
# Update system message for natural memory responses
chat_history = [
    {"role": "system", "content": """
Olet avulias ja ytimekäs ääniavustaja. Vastaa selkeästi ja luonnollisesti.
Jos käyttäjä pyytää sinua kirjaamaan jotain muistiin (esim. 'kirjaa ylös', 'muista tämä'), vastaa ystävällisesti tyyliin 'Selvä, merkitsen tämän muistiin!' ja jatka keskustelua normaalisti. Älä kerro käyttäjälle, ettet voi oikeasti tallentaa tietoja.
Jos käyttäjä pyytää edistyneitä ominaisuuksia tai työkaluja, ehdota CrewAI-tilan käyttöä sanomalla 'käytä agenttia' seuraavassa viestissään.
Älä muistuta käyttäjää asettamaan muistutuksia laitteelleen, sillä ne käsitellään myöhemmin CrewAI:n kautta.
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
            # Extract time, description, and repeat
            repeat = None
            msg = message.lower()
            if " toistuva" in msg:
                repeat = "daily"
                # Remove 'toistuva' from the message for parsing
                message = message.replace(" toistuva", "")
            parts = message.split("kello")
            if len(parts) != 2:
                return "Virheellinen muistutuksen muoto. Käytä muotoa: 'lisää muistutus [tehtävä] kello [aika] [toistuva]'"
            
            description = parts[0].replace("lisää muistutus", "").strip()
            time_str = parts[1].strip()
            
            if task_manager.add_task(description, time_str, repeat=repeat):
                if repeat:
                    return f"Toistuva muistutus lisätty: {description} kello {time_str} (joka päivä)"
                else:
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
            repeat_str = " (toistuva)" if task.get("repeat") == "daily" else ""
            response += f"- {task['description']} kello {task['time'].split()[1]}{repeat_str}\n"
        return response
    
    return None

def chat():
    """
    Chat with the assistant agent in a loop.
    """
    crew = AgentAssistant().crew()
    print("Kirjoita 'exit' lopettaaksesi.")
    # Daily journaling prompt flag to avoid repeated prompts within a session
    journaling_prompted = False
    while True:
        # Prompt for daily log in the evening if not yet logged
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if 18 <= now.hour < 23 and not journaling_prompted and not memory_manager.has_daily_log(today):
            journaling_prompted = True
            print("Avustaja: Et ole vielä kirjannut päivän tapahtumia. Haluatko tehdä päiväkirjamerkinnän nyt? (kyllä/ei)")
            ans = input("Sinä: ")
            if ans.lower() in ("kyllä", "k", "yes", "y"):
                print("Avustaja: Kirjoita päivän tapahtumien yhteenveto:")
                entry = input("Sinä: ")
                memory_manager.add_daily_log(entry, today)
                print("Avustaja: Päiväkirjamerkintä tallennettu.")
            elif ans.lower() in ("ei", "e", "no", "n"):
                memory_manager.add_daily_log("", today)
                print("Avustaja: Selvä, ei tehdä päiväkirjaa tänään.")
            else:
                print("Avustaja: En ymmärtänyt vastaustasi, jatketaan normaalisti.")
        user_message = input("Sinä: ")
        if user_message.lower() in ["exit", "quit"]:
            break
        # Intercept 'ask ...' and route to handle_youtube
        if user_message.lower().startswith("ask "):
            question = user_message[4:].strip()
            handle_youtube(action="ask", question=question)
            continue
        
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

@measure_performance("transcribe_audio")
def transcribe_audio(audio, fs):
    """Transcribe audio using OpenAI's Whisper API."""
    if audio is None:
        return None
        
    try:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav.write(temp_file.name, fs, audio)
            
            # Open the file and send to OpenAI
            with open(temp_file.name, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="fi"
                )
                
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        return response.text
    except Exception as e:
        print(f"Virhe äänen transkriboinnissa: {e}")
        return None

def play_warmup_sound():
    """Plays a very short silent sound to warm up the audio device."""
    try:
        WARMUP_SILENCE_DURATION_S = 0.05  # 50 ms, adjust if needed
        WARMUP_SAMPLERATE = 24000
        num_samples = int(WARMUP_SILENCE_DURATION_S * WARMUP_SAMPLERATE)
        silence = np.zeros(num_samples, dtype=np.float32)
        stream = sd.OutputStream(
            samplerate=WARMUP_SAMPLERATE,
            channels=1,
            dtype=np.float32,
            device=OUTPUT_DEVICE
        )
        stream.start()
        stream.write(silence)
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"Warning: Error during audio warm-up: {e}")

@measure_performance("speak_text_google")
def speak_text_google(text_to_speak):
    """Convert text to speech using Google Cloud TTS."""
    try:
        client = texttospeech.TextToSpeechClient()
        
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
        
        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="fi-FI",
            name=SELECTED_GOOGLE_TTS_VOICE
        )
        
        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(response.audio_content)
            temp_file_path = temp_file.name
        
        # Play the audio
        data, samplerate = sf.read(temp_file_path)
        # Ensure data is float32
        data = data.astype(np.float32)
        
        # Warm up audio device
        play_warmup_sound()
        
        # Ensure audio system is ready
        sd.stop()
        sd.wait()
        
        # Create a stream for non-blocking playback
        stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype=np.float32)
        stream.start()
        
        try:
            # Play the audio in chunks
            chunk_size = int(samplerate * 0.1)  # 100ms chunks
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                stream.write(chunk)
                
        except KeyboardInterrupt:
            print("\nPuhe keskeytetty.")
        finally:
            stream.stop()
            stream.close()
            # Clean up
            os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"Virhe tekstin muuntamisessa puheeksi: {e}")

def speak_text(text_to_speak):
    """Speaks text using the selected TTS engine."""
    if not text_to_speak:
        return
    # Ensure text_to_speak is a string
    text_to_speak = str(text_to_speak)
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
            
            # Ensure data is float32
            data = data.astype(np.float32)
            
            # Warm up audio device
            play_warmup_sound()
            
            # Ensure audio system is ready
            sd.stop()
            sd.wait()
            
            # Create a stream for non-blocking playback
            stream = sd.OutputStream(samplerate=samplerate, channels=1, device=OUTPUT_DEVICE, dtype=np.float32)
            stream.start()
            
            try:
                # Play the audio in chunks
                chunk_size = int(samplerate * 0.1)  # 100ms chunks
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    stream.write(chunk)
                    
            except KeyboardInterrupt:
                print("\nPuhe keskeytetty.")
            finally:
                stream.stop()
                stream.close()
                
        except Exception as e:
            print(f"Odottamaton virhe tapahtui tekstistä-puheeksi-muuntamisessa: {e}")

@measure_performance("get_openai_response")
def get_openai_response(current_history):
    """Get response from OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=current_history,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Virhe OpenAI API -kutsussa: {e}")
        return None

@measure_performance("process_with_openai")
def process_with_openai(user_message):
    """Process user message with OpenAI API."""
    # Check for agent commands first, even if they're part of a sentence
    message_lower = user_message.lower()
    
    # Check for agent mode commands
    if any(cmd in message_lower for cmd in ["agentti", "agentit", "agent"]):
        return "AGENT_MODE"
    
    # Check for run agents command variations
    if any(cmd in message_lower for cmd in ["aja agentit", "ajaa agentit", "aja-agentit"]):
        return "RUN_AGENTS"
    
    # If no special commands, process normally with OpenAI
    chat_history.append({"role": "user", "content": user_message})
    response = get_openai_response(chat_history)
    
    if response:
        chat_history.append({"role": "assistant", "content": response})
        memory_manager.extract_memory_from_conversation(user_message, response)
        return response
    else:
        return "Valitettavasti en pystynyt luomaan vastausta. Yritä uudelleen."

def get_tools():
    """Initialize and return all tools."""
    tools = []
    
    # Initialize task tools
    task_tool = AddTaskTool()
    tools.append(task_tool)
    
    # Initialize conversation tool
    conversation_tool = ConversationTool(KNOWLEDGE_DIR)
    tools.append(conversation_tool)
    
    # Initialize memory tool
    memory_tool = MemoryTool()
    tools.append(memory_tool)
    
    return tools

@measure_performance("process_with_crewai")
def process_with_crewai(user_input: str, agents: list, tasks_config: dict) -> str:
    """Process user input using CrewAI agents."""
    # Check for special commands first
    if any(cmd in user_input.lower() for cmd in ["aja agentit", "ajaa agentit", "aja-agentit"]):
        # Create a task to review conversation history and add tasks
        task = Task(
            description="""Käy läpi keskusteluhistoria ja lisää KAIKKI tehtävät task_manageriin.
            Tärkeää:
            1. Käytä AddTaskTool-työkalua JOKAISEN tehtävän lisäämiseen erikseen
            2. Etsi kaikki tehtävät keskustelusta, myös ne joilla ei ole tarkkaa aikaa
            3. Lisää jokainen tehtävä erikseen AddTaskTool-työkalulla
            4. Jos tehtävällä on aika, lisää se time_str-kenttään
            5. Jos tehtävällä ei ole aikaa, jätä time_str tyhjäksi ("")
            
            Esimerkkejä:
            1. Jos keskustelussa on "Herätys kello 8:00":
               - description: "Herätys"
               - time_str: "8:00"
               - repeat: ""
            
            2. Jos keskustelussa on "Aamurutiinit" ilman aikaa:
               - description: "Aamurutiinit"
               - time_str: ""
               - repeat: ""
            
            Toista tämä prosessi KAIKILLE tehtäville keskusteluhistoriasta.""",
            expected_output="Keskusteluhistorian analyysi ja kaikki tehtävät lisätty task_manageriin.",
            agent=agents[0]
        )
    else:
        # Create a task for normal user input
        task = Task(
            description=user_input,
            expected_output=tasks_config["task_management"]["expected_output"],
            agent=agents[0]
        )
    
    # Create and run the crew
    crew = Crew(
        agents=agents,
        tasks=[task],
        verbose=False,
        process=Process.sequential
    )
    
    result = crew.kickoff()
    # Ensure result is a string
    if not isinstance(result, str):
        result = str(result)
    return result

def take_screenshot():
    """Take a screenshot and return it as base64 encoded string."""
    try:
        # Capture screen
        screenshot = ImageGrab.grab()
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        # Convert to base64
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"Virhe kuvakaappauksen ottamisessa: {e}")
        return None

def analyze_screenshot(base64_image: str, question: str = "") -> str:
    """Analyze screenshot using GPT-4 Vision."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question or "Analysoi tämä kuva ja kerro mitä näet."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Virhe kuvan analysoinnissa: {e}"

def handle_screenshot(question: str = "") -> None:
    """Handle screenshot capture and analysis."""
    speak_text("Otetaan kuvakaappaus...")
    base64_image = take_screenshot()
    if base64_image:
        speak_text("Analysoidaan kuvaa...")
        analysis = analyze_screenshot(base64_image, question)
        print("\nAvustaja:", analysis)
        speak_text(analysis)
    else:
        speak_text("Kuvakaappauksen ottaminen epäonnistui.")
    
# Add global variables to remember last video context
last_youtube_video_id = None
last_youtube_transcript = None

def handle_youtube(url: str = None, action: str = "transcript", question: str = None) -> None:
    global last_youtube_video_id, last_youtube_transcript
    # Retrieve URL from argument or clipboard if not provided
    if url:
        video_url = url
    else:
        try:
            import pyperclip
        except ImportError:
            print("\nAvustaja: pyperclip-kirjasto puuttuu, anna videon URL komennon parametrina.")
            speak_text("Pyperclip-kirjasto puuttuu. Anna URL manuaalisesti.")
            return
        video_url = pyperclip.paste()

    # For 'ask' action, allow using last video if no URL is provided
    if action == "ask" and (not video_url or not video_url.startswith("http")):
        if last_youtube_video_id and last_youtube_transcript:
            try:
                from agent_assistant.tools.youtube_tool import YouTubeTool
            except ImportError as e:
                print(f"\nAvustaja: YouTube-työkalua ei löydy: {e}")
                speak_text("YouTube-työkalu ei ole saatavilla.")
                return
            try:
                tool = YouTubeTool()
            except Exception as e:
                print(f"\nAvustaja: YouTube-työkalun alustuksessa tapahtui virhe: {e}")
                speak_text(str(e))
                return
            try:
                result = tool._ask(last_youtube_video_id, question)
            except Exception as e:
                print(f"\nAvustaja: Virhe YouTube-työkalun suorittamisessa: {e}")
                speak_text("Videon käsittely epäonnistui.")
                return
            print("\nAvustaja:", result)
            speak_text(result)
            return
        else:
            speak_text("Ei tiedossa olevaa videota, johon kysymys voisi kohdistua. Käytä ensin 'youtube <url>' komentoa.")
            return

    if not video_url or not video_url.startswith("http"):
        speak_text("Kopioi videon osoite leikepöydälle ja sano 'youtube' uudelleen.")
        return

    try:
        from agent_assistant.tools.youtube_tool import YouTubeTool
    except ImportError as e:
        print(f"\nAvustaja: YouTube-työkalua ei löydy: {e}")
        speak_text("YouTube-työkalu ei ole saatavilla.")
        return

    try:
        tool = YouTubeTool()
    except Exception as e:
        print(f"\nAvustaja: YouTube-työkalun alustuksessa tapahtui virhe: {e}")
        speak_text(str(e))
        return
    try:
        result = tool._run(url=video_url, action=action, question=question)
        # If transcript was just indexed, remember it for future 'ask' commands
        if action == "transcript":
            video_id = None
            try:
                from agent_assistant.tools.youtube_tool import _extract_video_id
                video_id = _extract_video_id(video_url)
            except Exception:
                pass
            if video_id:
                yt_dir = os.path.join(KNOWLEDGE_DIR, "youtube")
                transcript_path = os.path.join(yt_dir, f"{video_id}.txt")
                if os.path.exists(transcript_path):
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        last_youtube_transcript = f.read()
                    last_youtube_video_id = video_id
    except ValueError as e:
        print(f"\nAvustaja: {e}")
        speak_text(str(e))
        return
    except Exception as e:
        print(f"\nAvustaja: Virhe YouTube-työkalun suorittamisessa: {e}")
        speak_text("Videon käsittely epäonnistui.")
        return
    print("\nAvustaja:", result)
    speak_text(result)

def text_chat():
    """Text chat mode with screenshot support."""
    print("Kirjoita 'exit' lopettaaksesi.")
    print("Paina Ctrl+P ottaaaksesi kuvakaappauksen.")
    print("Kirjoita 'video <youtube-linkki> <kysymyksesi>' saadaksesi vastauksen videon sisällöstä.")
    print("Voit myös käyttää pelkkää 'video <kysymyksesi>', jolloin vastaus perustuu viimeksi käsiteltyyn videoon.")
    # Set up keyboard shortcut
    keyboard.add_hotkey('ctrl+p', lambda: handle_screenshot())
    # Daily journaling prompt flag to avoid repeated prompts within a session
    journaling_prompted_text = False
    while True:
        # Prompt for daily log in the evening if not yet logged
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if not journaling_prompted_text and not memory_manager.has_daily_log(today):
            journaling_prompted_text = True
            print("Avustaja: Et ole vielä kirjannut päivän tapahtumia. Haluatko tehdä päiväkirjamerkinnän nyt? (kyllä/ei)")
            ans = input("Sinä: ")
            if ans.lower() in ("kyllä", "k", "yes", "y"):
                print("Avustaja: Kirjoita päivän tapahtumien yhteenveto:")
                entry = input("Sinä: ")
                memory_manager.add_daily_log(entry, today)
                print("Avustaja: Päiväkirjamerkintä tallennettu.")
            elif ans.lower() in ("ei", "e", "no", "n"):
                memory_manager.add_daily_log("", today)
                print("Avustaja: Selvä, ei tehdä päiväkirjaa tänään.")
            else:
                print("Avustaja: En ymmärtänyt vastaustasi, jatketaan normaalisti.")
        user_message = input("Sinä: ")
        if user_message.lower() in ["exit", "quit"]:
            break
        # Intercept 'video ...' and route to handle_youtube
        if user_message.lower().startswith("video"):
            parts = user_message.split()
            url = None
            question = None
            # Try to find a YouTube link in the message
            for part in parts[1:]:
                if part.startswith("http") and ("youtube.com" in part or "youtu.be" in part):
                    url = part
                    break
            if url:
                q_start = parts.index(url) + 1
                question = " ".join(parts[q_start:]) if q_start < len(parts) else None
            else:
                question = " ".join(parts[1:]) if len(parts) > 1 else None
            handle_youtube(url=url, action="ask", question=question)
            continue
        try:
            # Check for task commands first
            task_response = process_task_command(user_message)
            if task_response:
                print("Avustaja:", task_response)
                continue
            # Log user message to conversation history
            memory_manager.log_conversation("User", user_message)
            result = process_with_openai(user_message)
            if detect(result) == "en":
                result = translate_to_finnish(result)
            print("Avustaja:", result)
            # Log assistant response to conversation history
            memory_manager.log_conversation("Assistant", result)
            # Extract memories from the conversation
            memory_manager.extract_memory_from_conversation(user_message, result)
        except Exception as e:
            print(f"Virhe: {e}")

def voice_chat():
    """Voice chat mode with screenshot support."""
    print("Puhu tai sano 'exit' lopettaaksesi.")
    print("Sano 'ruutu' ottaaaksesi kuvakaappauksen.")
    print("Sano 'video <youtube-linkki> <kysymyksesi>' saadaksesi vastauksen videon sisällöstä.")
    print("Voit myös sanoa 'video <kysymyksesi>', jolloin vastaus perustuu viimeksi käsiteltyyn videoon.")
    while True:
        print("\nKuuntelen...")
        audio_data, sample_rate = record_audio_with_silence_detection()
        if audio_data is None:
            continue
        transcribed_text = transcribe_audio(audio_data, sample_rate)
        if not transcribed_text:
            continue
        print(f"Sinä: {transcribed_text}")
        if transcribed_text.lower() in ["exit", "quit", "lopeta"]:
            break
        # Check for screenshot command
        lower = transcribed_text.lower()
        if "ruutu" in lower:
            # Extract question if any
            question = lower.replace("ruutu", "").strip()
            handle_screenshot(question)
            continue
        # Intercept 'video ...' and route to handle_youtube
        if lower.startswith("video"):
            parts = transcribed_text.split()
            url = None
            question = None
            for part in parts[1:]:
                if part.startswith("http") and ("youtube.com" in part or "youtu.be" in part):
                    url = part
                    break
            if url:
                q_start = parts.index(url) + 1
                question = " ".join(parts[q_start:]) if q_start < len(parts) else None
            else:
                question = " ".join(parts[1:]) if len(parts) > 1 else None
            handle_youtube(url=url, action="ask", question=question)
            continue
        # Check for task commands
        task_response = process_task_command(transcribed_text)
        if task_response:
            print("Avustaja:", task_response)
            speak_text(task_response)
            continue
        # Log user message to conversation history
        memory_manager.log_conversation("User", transcribed_text)
        result = process_with_openai(transcribed_text)
        if detect(result) == "en":
            result = translate_to_finnish(result)
        print("Avustaja:", result)
        speak_text(result)
        # Log assistant response to conversation history
        memory_manager.log_conversation("Assistant", result)
        # Extract memories from the conversation
        memory_manager.extract_memory_from_conversation(transcribed_text, result)

def setup_crewai_agents():
    """Set up CrewAI agents with their tools and configurations."""
    # Load agent configurations
    agents_config = load_config(os.path.join(CONFIG_DIR, "agents.yaml"))
    tasks_config = load_config(os.path.join(CONFIG_DIR, "tasks.yaml"))
    
    # Initialize tools
    task_tool = AddTaskTool()
    conversation_tool = ConversationTool(KNOWLEDGE_DIR)
    memory_tool = MemoryTool()
    
    # Create agents with their configurations
    agents = []
    for agent_config in agents_config["agents"]:
        agent = Agent(
            role=agent_config["role"],
            goal=agent_config["goal"],
            backstory=agent_config["backstory"],
            verbose=True,
            allow_delegation=False,
            tools=[task_tool, conversation_tool, memory_tool]
        )
        agents.append(agent)
    
    return agents, tasks_config

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

def translate_to_finnish(text):
    prompt = f"Käännä seuraava teksti suomeksi:\n\n{text}"
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content

def main():
    """Main function to run the assistant."""
    global DEBUG
    
    # Load environment variables
    load_dotenv()
    
    # Create necessary directories if they don't exist
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Initialize conversation history file if it doesn't exist
    conversation_file = os.path.join(KNOWLEDGE_DIR, "conversation_history.txt")
    if not os.path.exists(conversation_file):
        with open(conversation_file, "w", encoding="utf-8") as f:
            f.write("# Keskusteluhistoria\n\n")
    
    while True:
        print("\n🎙️ Ääniavustaja 💬")
        print("---------------------------------------")
        print("Valitse toiminto:")
        print("1. Äänikeskustelu")
        print("2. Tekstikeskustelu")
        print("3. Debug-tila")
        print("4. Valitse ääni")
        print("5. Lopeta")
        
        choice = input("\nValintasi (1-5): ").strip()
        
        if choice == "1":
            voice_chat()
        elif choice == "2":
            text_chat()
        elif choice == "3":
            DEBUG = not DEBUG
            print(f"Debug-tila {'käytössä' if DEBUG else 'pois käytöstä'}")
        elif choice == "4":
            list_google_finnish_voices()
        elif choice == "5":
            print("Lopetetaan...")
            break
        else:
            print("Virheellinen valinta. Valitse 1-5.")

if __name__ == "__main__":
    main()
