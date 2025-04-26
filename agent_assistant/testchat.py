import openai
import os
import sys
import sounddevice as sd
import numpy as np
import whisper
import time
import tempfile
from scipy.io.wavfile import write as write_wav
import io
import soundfile as sf
from dotenv import load_dotenv

# Reset audio settings and use MME backend
sd.default.reset()

# --- Configuration ---

# Load API Key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with OPENAI_API_KEY=your_key")
    sys.exit(1)

# Models (adjust as needed)
WHISPER_MODEL = "base.en" # Tiny, base, small, medium, large. Use '.en' for English-only models.
CHAT_MODEL = "gpt-3.5-turbo" # Or "gpt-4", "gpt-4o", etc.
TTS_MODEL = "tts-1"          # tts-1 or tts-1-hd
TTS_VOICE = "alloy"          # Options: alloy, echo, fable, onyx, nova, shimmer

# Audio Recording Settings
SAMPLE_RATE = 16000  # Hz (Whisper prefers 16kHz)
CHANNELS = 1
SILENCE_THRESHOLD = 500  # RMS amplitude. Adjusted based on test results
INPUT_DEVICE = 1  # Jabra headset microphone
OUTPUT_DEVICE = 3  # Jabra headset earphone

# --- Initialization ---
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

print(f"Loading Whisper model ({WHISPER_MODEL})... This might take a moment.")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL)
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model '{WHISPER_MODEL}': {e}")
    print("Ensure the model name is correct and you have internet access if downloading.")
    print("Also ensure ffmpeg is installed and in your system's PATH.")
    sys.exit(1)

# --- Chat History ---
chat_history = [
    {"role": "system", "content": "You are a helpful and concise voice assistant. Respond clearly and naturally."}
]

# --- Core Functions ---

def record_audio_with_silence_detection(max_duration=30, silence_threshold=500, silence_duration=2):
    """
    Records audio until silence is detected for a specified duration.
    
    Args:
        max_duration: Maximum recording duration in seconds
        silence_threshold: Audio level below which is considered silence
        silence_duration: Duration of silence in seconds before stopping recording
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    print("Listening... (speak now, pause for 2 seconds to stop)")
    
    # Initialize variables
    sample_rate = SAMPLE_RATE
    silence_frames = 0
    silence_frames_threshold = int(silence_duration * sample_rate / 1024)  # Convert to frames
    audio_chunks = []
    recording = True
    
    # Create a callback function to process audio in chunks
    def audio_callback(indata, frames, time, status):
        nonlocal silence_frames, recording, audio_chunks
        if status:
            print(f"Status: {status}")
        
        # Calculate the RMS of the current chunk
        rms = np.sqrt(np.mean(indata**2))
        
        # Check if this is silence
        if rms < silence_threshold / 32768:  # Normalize threshold
            silence_frames += 1
        else:
            silence_frames = 0
        
        # If we've had enough silence, stop recording
        if silence_frames >= silence_frames_threshold:
            recording = False
        
        # Add the chunk to our list
        audio_chunks.append(indata.copy())
    
    # Start the input stream
    with sd.InputStream(samplerate=sample_rate, channels=CHANNELS, dtype='float32', 
                       device=INPUT_DEVICE, callback=audio_callback, blocksize=1024):
        # Record until we detect enough silence or hit the max duration
        start_time = time.time()
        while recording and (time.time() - start_time) < max_duration:
            time.sleep(0.1)
    
    # If we have recorded anything, concatenate the chunks
    if audio_chunks:
        audio_data = np.concatenate(audio_chunks)
        # Convert to int16 for compatibility with the rest of the code
        audio_data = (audio_data * 32768).astype(np.int16)
        return audio_data, sample_rate
    else:
        return None, sample_rate

def transcribe_audio(audio, fs):
    """Transcribes audio using OpenAI's Whisper API."""
    if audio is None:
        print("No audio data to transcribe.")
        return None
    
    try:
        print("Saving audio to temporary file...")
        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            write_wav(temp_filename, fs, audio)
        
        print("Sending to OpenAI for transcription...")
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
        print(f"Error transcribing audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_openai_response(current_history):
    """Sends the chat history to OpenAI and gets the assistant's response."""
    print("Getting response from OpenAI...")
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=current_history
        )
        assistant_message = response.choices[0].message.content
        return assistant_message
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Exceeded: {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error (Status {e.status_code}): {e.response}")
    except Exception as e:
        print(f"An unexpected error occurred calling OpenAI chat: {e}")
    return None # Return None on error

def speak_text(text_to_speak):
    """Uses OpenAI TTS to convert text to speech and plays it."""
    if not text_to_speak:
        print("Warning: No text provided to speak.")
        return

    print("Generating speech...")
    try:
        # Use stream=True for potentially faster playback start, but requires more complex handling.
        # For simplicity here, we get the whole audio content first.
        response = openai_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text_to_speak,
            response_format="opus" # opus is efficient, soundfile can handle it
        )

        # Read the audio data from the response content
        audio_bytes = response.content
        audio_stream = io.BytesIO(audio_bytes)

        print("Playing response...")
        # Use soundfile to read the audio data (handles various formats like opus)
        data, samplerate = sf.read(audio_stream)
        
        # Make sure we have the complete audio data
        if len(data) == 0:
            print("Warning: Empty audio data received.")
            return
            
        # Play the audio and wait for it to finish
        sd.play(data, samplerate, device=OUTPUT_DEVICE)
        sd.wait() # Wait for playback to finish

    except sf.SoundFileError as e:
        print(f"Error reading audio stream (is ffmpeg installed?): {e}")
    except sd.PortAudioError as e:
         print(f"\nError with audio output device: {e}")
         print("Please check your speaker/headphone connection and system settings.")
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error (TTS): {e}")
    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Exceeded (TTS): {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error (TTS) (Status {e.status_code}): {e.response}")
    except Exception as e:
        print(f"An unexpected error occurred during text-to-speech: {e}")
        import traceback
        traceback.print_exc()

def voice_chat():
    """Continuously listens for voice input and responds with voice."""
    print("\n=== VOICE CHAT MODE ===")
    print("Listening for voice input... (speak now, pause for 2 seconds to stop)")
    print("Press Ctrl+C to exit voice chat mode")
    
    try:
        while True:
            # Record audio with silence detection
            audio, fs = record_audio_with_silence_detection()
            if audio is None or len(audio) == 0:
                print("No audio recorded. Continuing to listen...")
                continue
                
            # Check if audio level is too low
            max_level = np.abs(audio).max()
            print(f"Audio level: {max_level} (threshold: {SILENCE_THRESHOLD})")
            
            if max_level < SILENCE_THRESHOLD:
                print("No speech detected. Continuing to listen...")
                continue
            
            # Start timing
            start_time = time.time()
            
            # Transcribe audio
            print("Processing audio...")
            transcription = transcribe_audio(audio, fs)
            
            if not transcription:
                print("Could not transcribe audio. Continuing to listen...")
                continue
                
            print(f"Transcribed: {transcription}")
            
            # Process the message with OpenAI
            print("Sending to AI...")
            chat_history.append({"role": "user", "content": transcription})
            assistant_response = get_openai_response(chat_history)
            
            if assistant_response:
                print(f"\nAssistant: {assistant_response}")
                chat_history.append({"role": "assistant", "content": assistant_response})
                
                # Calculate time to response
                response_time = time.time() - start_time
                print(f"Time to response: {response_time:.2f} seconds")
                
                # Speak the response
                speak_text(assistant_response)
                
                # Calculate total time including speech
                total_time = time.time() - start_time
                print(f"Total processing time: {total_time:.2f} seconds")
                
                # Wait a moment before listening again
                print("\nListening for voice input... (speak now, pause for 2 seconds to stop)")
                time.sleep(1)
            else:
                print("Assistant could not generate a response.")
                print("\nListening for voice input... (speak now, pause for 2 seconds to stop)")
                
    except KeyboardInterrupt:
        print("\nExiting voice chat mode.")
        return

def text_chat():
    """Handles text-based chat with the assistant."""
    print("\n=== TEXT CHAT MODE ===")
    print("Type your message and press Enter")
    print("Type 'back' to return to main menu")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == "back":
                print("Returning to main menu...")
                return
                
            if not user_input:
                continue
                
            # Process the message with OpenAI
            print("Sending to AI...")
            chat_history.append({"role": "user", "content": user_input})
            assistant_response = get_openai_response(chat_history)
            
            if assistant_response:
                print(f"\nAssistant: {assistant_response}")
                chat_history.append({"role": "assistant", "content": assistant_response})
                
                # Ask if user wants to hear the response
                hear_response = input("Would you like to hear this response? (y/n): ").strip().lower()
                if hear_response == 'y':
                    speak_text(assistant_response)
            else:
                print("Assistant could not generate a response.")
                
        except KeyboardInterrupt:
            print("\nReturning to main menu...")
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

def main():
    # Print available audio devices
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")
    
    print("\n🎙️ Voice Chat Assistant 💬")
    print("---------------------------------------")
    print(f"Using audio input device {INPUT_DEVICE} (Jabra headset)")
    print(f"Using audio output device {OUTPUT_DEVICE} (Jabra headset)")
    print("Type 'voice' to start voice chat mode")
    print("Type 'text' to start text chat mode")
    print("Type 'quit' or 'exit' to end the session")
    
    while True:
        try:
            command = input("\nEnter command (voice/text/quit): ").strip().lower()
            
            if command in ["quit", "exit"]:
                print("Goodbye!")
                break
            elif command == "voice":
                voice_chat()
            elif command == "text":
                text_chat()
            else:
                print("Invalid command. Type 'voice' for voice chat, 'text' for text chat, or 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except EOFError:
            print("\nInput stream closed. Exiting.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    main()