import sounddevice as sd
import librosa
import numpy as np
import os
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import pyaudio
import wave
import datetime




def record_audio(output_dir, filename, duration=5):
    audio = pyaudio.PyAudio()

    # Create an audio stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)


    frames = []
    for i in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a .wav file
    file_path = os.path.join(output_dir, filename)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
    
    return file_path

def convert_audio_text(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)  # You can use other recognition engines too
        return text
    except sr.UnknownValueError:
        return "Google Web Speech API could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Web Speech API; {e}"

def button():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    audio_file = record_audio(output_dir, "recorded_audio.wav")
    text = convert_audio_text(audio_file)
    
    return text