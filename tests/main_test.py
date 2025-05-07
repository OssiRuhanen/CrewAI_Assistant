#!/usr/bin/env python
import sys
import os
import io
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import openai
from dotenv import load_dotenv

# Attempt to import Google Cloud TTS, but don't make it a hard requirement if not used
try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False
    print("Warning: google-cloud-texttospeech not installed. Google TTS will not be available.")
    print("To install: pip install google-cloud-texttospeech")


# --- Configuration ---
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# For Google TTS, ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set in .env file or environment.")
    sys.exit(1)

# TTS Models
OPENAI_TTS_MODEL = "tts-1"  # or "tts-1-hd"
OPENAI_TTS_VOICE = "alloy"

# Google TTS Configuration
USE_GOOGLE_TTS_BY_DEFAULT = False  # Set to True to default to Google TTS for tests
SELECTED_GOOGLE_TTS_VOICE = "fi-FI-Wavenet-A" # Example Finnish voice

# Audio Settings
OUTPUT_DEVICE = None  # Use default output device. Specify device index or name if needed.

# Warm-up Settings - TRY ADJUSTING THESE
WARMUP_SILENCE_DURATION_S = 0.05  # 50 milliseconds. Try 0.03, 0.07, 0.1 if needed.
WARMUP_SAMPLERATE = 24000         # A common samplerate for the silent warm-up sound.

# Optional: Silence padding if warm-up isn't enough (milliseconds)
# Set to 0 if warm-up works. Try 50-150 if still facing issues.
TTS_AUDIO_PADDING_MS = 0

# --- Initialization ---
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

google_tts_client = None
if GOOGLE_TTS_AVAILABLE and (USE_GOOGLE_TTS_BY_DEFAULT or "google" in sys.argv): # Allow enabling Google via arg
    try:
        google_tts_client = texttospeech.TextToSpeechClient()
        print("Google TTS client initialized.")
    except Exception as e:
        print(f"Error initializing Google TTS client: {e}")
        print("Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
        GOOGLE_TTS_AVAILABLE = False # Mark as unavailable if init fails

# --- Helper Functions ---

def play_warmup_sound():
    """Plays a very short silent sound to warm up the audio device."""
    try:
        # sd.stop() # Consider if stopping here is too aggressive or helpful
        # sd.wait()

        num_samples = int(WARMUP_SILENCE_DURATION_S * WARMUP_SAMPLERATE)
        silence = np.zeros(num_samples, dtype=np.float32)

        # Short blocking play to ensure device engages.
        # Using a stream explicitly to control parameters like device.
        stream = sd.OutputStream(
            samplerate=WARMUP_SAMPLERATE,
            channels=1, # Mono for silence
            dtype=np.float32,
            device=OUTPUT_DEVICE
        )
        stream.start()
        stream.write(silence) # Write the silent samples
        stream.stop()
        stream.close()
        # print(f"Debug: Audio system warmed up with {WARMUP_SILENCE_DURATION_S*1000:.0f}ms silence.")
    except Exception as e:
        print(f"Warning: Error during audio warm-up: {e}")
        # If warm-up fails, the main playback might still have the initial cut-off.

def speak_text_openai(text_to_speak):
    """Converts text to speech using OpenAI TTS and plays it."""
    if not text_to_speak:
        print("No text provided to speak_text_openai.")
        return

    print(f"OpenAI TTS: Requesting speech for: '{text_to_speak}'")
    try:
        api_start_time = time.time()
        response = openai_client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text_to_speak,
            response_format="opus" # opus, mp3, aac, flac. pcm (24kHz) is also an option.
        )
        api_call_duration = time.time() - api_start_time
        print(f"Debug: OpenAI API call took {api_call_duration:.3f}s")

        audio_bytes = response.content
        audio_stream = io.BytesIO(audio_bytes)

        decode_start_time = time.time()
        data, samplerate = sf.read(audio_stream) # soundfile decodes opus (or mp3, etc.)
        decode_duration = time.time() - decode_start_time
        print(f"Debug: soundfile decoding took {decode_duration:.3f}s. Samplerate: {samplerate}Hz, Shape: {data.shape}, Dtype: {data.dtype}")

        # Ensure mono and float32 for sounddevice
        if len(data.shape) > 1 and data.shape[1] > 1: # If stereo
            print("Debug: Audio is stereo, converting to mono.")
            data = np.mean(data, axis=1)
        if data.dtype != np.float32:
            print(f"Debug: Audio data type is {data.dtype}, converting to float32.")
            data = data.astype(np.float32) # sounddevice prefers float32

        if len(data) == 0:
            print("Warning: Received empty audio data from OpenAI TTS.")
            return

        # Optional: Add padding if warm-up alone is not enough
        if TTS_AUDIO_PADDING_MS > 0:
            print(f"Debug: Prepending {TTS_AUDIO_PADDING_MS}ms of silence.")
            num_padding_samples = int(samplerate * (TTS_AUDIO_PADDING_MS / 1000.0))
            padding = np.zeros(num_padding_samples, dtype=np.float32)
            data = np.concatenate((padding, data))

        sd.stop() # Stop any previous playback
        sd.wait() # Wait for it to actually stop

        print(f"Debug: Playing audio (OpenAI)... (Length: {len(data)/samplerate:.3f}s)")
        sd.play(data, samplerate, device=OUTPUT_DEVICE)
        sd.wait() # Wait for playback to finish

    except Exception as e:
        print(f"Error in speak_text_openai: {e}")
        import traceback
        traceback.print_exc()

def speak_text_google(text_to_speak):
    """Converts text to speech using Google Cloud TTS and plays it."""
    if not text_to_speak:
        print("No text provided to speak_text_google.")
        return
    if not GOOGLE_TTS_AVAILABLE or google_tts_client is None:
        print("Google TTS client not available or not initialized. Skipping Google TTS.")
        return

    print(f"Google TTS: Requesting speech for: '{text_to_speak}'")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
        voice = texttospeech.VoiceSelectionParams(
            language_code=SELECTED_GOOGLE_TTS_VOICE.split('-', 2)[0] + '-' + SELECTED_GOOGLE_TTS_VOICE.split('-', 2)[1], # e.g., fi-FI
            name=SELECTED_GOOGLE_TTS_VOICE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3 # Or LINEAR16 for PCM
        )

        api_start_time = time.time()
        response = google_tts_client.synthesize_speech(
            request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
        )
        api_call_duration = time.time() - api_start_time
        print(f"Debug: Google API call took {api_call_duration:.3f}s")

        audio_stream = io.BytesIO(response.audio_content)

        decode_start_time = time.time()
        data, samplerate = sf.read(audio_stream)
        decode_duration = time.time() - decode_start_time
        print(f"Debug: soundfile decoding took {decode_duration:.3f}s. Samplerate: {samplerate}Hz, Shape: {data.shape}, Dtype: {data.dtype}")

        # Ensure mono and float32 for sounddevice
        if len(data.shape) > 1 and data.shape[1] > 1: # If stereo
            print("Debug: Audio is stereo, converting to mono.")
            data = np.mean(data, axis=1)
        if data.dtype != np.float32:
            print(f"Debug: Audio data type is {data.dtype}, converting to float32.")
            data = data.astype(np.float32)

        if len(data) == 0:
            print("Warning: Received empty audio data from Google TTS.")
            return

        # Optional: Add padding if warm-up alone is not enough
        if TTS_AUDIO_PADDING_MS > 0:
            print(f"Debug: Prepending {TTS_AUDIO_PADDING_MS}ms of silence.")
            num_padding_samples = int(samplerate * (TTS_AUDIO_PADDING_MS / 1000.0))
            padding = np.zeros(num_padding_samples, dtype=np.float32)
            data = np.concatenate((padding, data))

        sd.stop() # Stop any previous playback
        sd.wait() # Wait for it to actually stop

        print(f"Debug: Playing audio (Google)... (Length: {len(data)/samplerate:.3f}s)")
        sd.play(data, samplerate, device=OUTPUT_DEVICE)
        sd.wait() # Wait for playback to finish

    except Exception as e:
        print(f"Error in speak_text_google: {e}")
        import traceback
        traceback.print_exc()

def speak_text_master(text_to_speak, use_google_flag=None):
    """Master function to decide which TTS to use and apply warm-up."""
    if not text_to_speak:
        return

    # Determine which TTS to use
    if use_google_flag is None:
        use_google_flag = USE_GOOGLE_TTS_BY_DEFAULT

    provider_name = "Google" if use_google_flag and GOOGLE_TTS_AVAILABLE else "OpenAI"
    print(f"\n--- Speaking: '{text_to_speak}' (Using {provider_name}) ---")

    # 1. Perform warm-up *before* the API call
    warmup_start_time = time.time()
    play_warmup_sound()
    warmup_duration = time.time() - warmup_start_time
    print(f"Debug: play_warmup_sound() took {warmup_duration:.4f}s")

    # 2. Call the appropriate TTS function
    if use_google_flag and GOOGLE_TTS_AVAILABLE and google_tts_client:
        speak_text_google(text_to_speak)
    elif use_google_flag and (not GOOGLE_TTS_AVAILABLE or not google_tts_client):
        print("Google TTS was requested but is not available/initialized. Falling back to OpenAI.")
        speak_text_openai(text_to_speak)
    else:
        speak_text_openai(text_to_speak)
    print("--- Finished speaking ---")


# --- Main Test ---
if __name__ == "__main__":
    print("TTS Test Script for Cut-off Audio Issue")
    print("--------------------------------------")
    print("Make sure your .env file has OPENAI_API_KEY.")
    print("If testing Google TTS, ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
    print(f"Default sounddevice output: {sd.query_devices(kind='output')['name']}")
    print(f"Config: Warm-up: {WARMUP_SILENCE_DURATION_S*1000:.0f}ms, Padding: {TTS_AUDIO_PADDING_MS}ms")
    print("--------------------------------------")

    test_phrases = [
        "Hello, this is a test from the script.",
        "One two three four five.",
        "Testing the very beginning of audio playback carefully.",
        "A short phrase.",
        "This is a slightly longer sentence to see if the effect is consistent across durations."
    ]

    # Determine if Google TTS should be tested based on global flag or command line argument
    test_google_tts_this_run = USE_GOOGLE_TTS_BY_DEFAULT
    if "google" in sys.argv: # Allow easy override: python main_test.py google
        test_google_tts_this_run = True
        if not GOOGLE_TTS_AVAILABLE or not google_tts_client: # Re-check if client actually initialized
             print("Google test requested via command line, but client is not ready. Please check setup.")
             test_google_tts_this_run = False


    # Test OpenAI
    print("\n\n>>> TESTING OPENAI TTS <<<")
    for i, phrase in enumerate(test_phrases):
        print(f"\nTest {i+1} (OpenAI):")
        speak_text_master(phrase, use_google_flag=False)
        time.sleep(0.5) # Small pause between tests

    # Test Google (if enabled and available)
    if test_google_tts_this_run and GOOGLE_TTS_AVAILABLE and google_tts_client:
        print("\n\n>>> TESTING GOOGLE TTS <<<")
        for i, phrase in enumerate(test_phrases):
            print(f"\nTest {i+1} (Google):")
            speak_text_master(phrase, use_google_flag=True)
            time.sleep(0.5)
    elif test_google_tts_this_run and (not GOOGLE_TTS_AVAILABLE or not google_tts_client):
         print("\n\n>>> SKIPPING GOOGLE TTS (Client not available/initialized despite request) <<<")
    else:
        print("\n\n>>> SKIPPING GOOGLE TTS (USE_GOOGLE_TTS_BY_DEFAULT is False or 'google' not in args) <<<")
        print("To test Google TTS, set USE_GOOGLE_TTS_BY_DEFAULT = True at the top, or run: python main_test.py google")

    print("\n--------------------------------------")
    print("Testing complete.")
    print("Listen carefully to the beginning of each phrase.")
    print(f"If still cut off, try adjusting WARMUP_SILENCE_DURATION_S (current: {WARMUP_SILENCE_DURATION_S}s)")
    print(f"and/or TTS_AUDIO_PADDING_MS (current: {TTS_AUDIO_PADDING_MS}ms) at the top of the script.")
    print("--------------------------------------")