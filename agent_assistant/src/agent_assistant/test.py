from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
voices = client.list_voices(language_code="fi-FI")

for voice in voices.voices:
    print(f"Name: {voice.name}, Gender: {texttospeech.SsmlVoiceGender(voice.ssml_gender).name}, Natural Sample Rate: {voice.natural_sample_rate_hertz}")