"""
Voice interaction handler using OpenAI Whisper and TTS.
Supports both voice and text input modes for flexible user interaction.
"""

import pyaudio
import wave
import tempfile
import os
from openai import OpenAI
import pygame
from basics.logger import color_text

# Global variable to store input mode
_input_mode = None


def set_input_mode(mode):
    """
    Set the global input mode.
    
    """
    global _input_mode
    _input_mode = mode


def get_input_mode():
    """
    Get the current input mode.
    
    """
    return _input_mode

class VoiceOpenAI:
    """
    A class for handling voice input/output using OpenAI's Whisper and TTS APIs.
    

    client : OpenAI
        An instance of the OpenAI API client.
    mic : pyaudio.PyAudio
        PyAudio instance for microphone access.
    sample_rate : int
        Audio sampling rate (16000 Hz for Whisper).
    chunk_size : int
        Audio buffer chunk size.
    format : int
        Audio format (16-bit PCM).
    channels : int
        Number of audio channels (1 for mono).
        
    """
    
    def __init__(self):
        """
        Initializes the VoiceOpenAI class.
        
        """
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError(
                "OpenAI API key not found! Please set your API key using one of these methods:\n"
                "1. Set environment variable: export OPENAI_API_KEY='your_api_key_here'\n"
                "2. Add to your shell profile (~/.bashrc, ~/.zshrc, etc.): export OPENAI_API_KEY='your_api_key_here'\n"
                "3. Get your API key from: https://platform.openai.com/api-keys\n"
                "\nAfter setting the API key, restart your terminal or run: source ~/.bashrc"
            )
        
        self.client = OpenAI()
        self.mic = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        pygame.mixer.init()
    
    def record_audio(self, duration=10, silence_threshold=500, silence_duration=2):
        """
        Records audio from microphone with automatic silence detection.
        
        Parameters:
        duration : int, optional
            Maximum recording duration in seconds (default is 10).
        silence_threshold : int, optional
            Amplitude threshold for silence detection (default is 500).
        silence_duration : int, optional
            Duration of silence to stop recording in seconds (default is 2).
            
        """
        stream = self.mic.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print(color_text("🎤 Listening... (speak now)", 'cyan'))
        
        frames = []
        silent_chunks = 0
        started = False
        
        max_chunks = int(self.sample_rate / self.chunk_size * duration)
        silence_chunks_threshold = int(self.sample_rate / self.chunk_size * silence_duration)
        
        for _ in range(max_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
            
            amplitude = sum(abs(int.from_bytes(data[i:i+2], byteorder='little', signed=True)) 
                          for i in range(0, len(data), 2)) / (len(data) / 2)
            
            if amplitude > silence_threshold:
                started = True
                silent_chunks = 0
            elif started:
                silent_chunks += 1
                if silent_chunks > silence_chunks_threshold:
                    break
        
        stream.stop_stream()
        stream.close()
        
        return frames
    
    def save_audio(self, frames, filename):
        """
        Saves recorded audio frames to a WAV file.
        
        Parameters:
        frames : list
            List of audio frames to save.
        filename : str
            Path to the output WAV file.
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.mic.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
    
    def transcribe(self, audio_file):
        """
        Transcribes audio file to text using Whisper API.
        
        Parameters:
        audio_file : str
            Path to the audio file to transcribe.
            
        """
        with open(audio_file, 'rb') as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    
    def listen(self):
        """
        Records and transcribes user speech.
        
        """
        frames = self.record_audio()
        
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        self.save_audio(frames, temp_audio_path)
        
        try:
            text = self.transcribe(temp_audio_path)
            os.unlink(temp_audio_path)
            return text
        except Exception as e:
            os.unlink(temp_audio_path)
            raise e
    
    def speak(self, text):
        """
        Converts text to speech and plays it using TTS API.
        
        Parameters:
        text : str
            Text to convert to speech.
        """
        print(color_text(f"🔊 System: {text}", 'green'))
        
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=text
            ) as response:
                response.stream_to_file(temp_audio_path)
            
            pygame.mixer.music.load(temp_audio_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
        finally:
            try:
                os.unlink(temp_audio_path)
            except:
                pass
    
    def close(self):
        """
        Terminates PyAudio instance and pygame mixer.
        """
        self.mic.terminate()
        pygame.mixer.quit()


def get_user_input_voice():
    """
    Gets user input via voice or text based on global mode.
    
    """
    global _input_mode
    
    # If text mode, use keyboard input
    if _input_mode == 'text':
        return input(color_text("Your adjustment command: ", 'orange'))
    
    # Otherwise use voice input
    voice = VoiceOpenAI()
    
    while True:
        try:
            text = voice.listen()
            
            if not text:
                print(color_text("✗ No speech detected.", 'red'))
                voice.speak("Sorry, I didn't catch that. Please try again.")
                continue
            
            print(color_text(f"✓ Recognized: \"{text}\"", 'green'))
            voice.close()
            return text
            
        except Exception as e:
            print(color_text(f"✗ Error: {e}", 'red'))
            voice.close()
            return input(color_text("Your adjustment command: ", 'orange'))


def listen_for_response_voice(prompt_text):
    """
    Speaks a prompt and listens for user's voice response.
    
    Parameters:
    prompt_text : str
        Text to speak as prompt.
    """
    voice = VoiceOpenAI()
    voice.speak(prompt_text)
    
    while True:
        try:
            text = voice.listen()
            
            if not text:
                voice.speak("Sorry, I didn't catch that. Please say again.")
                continue
            
            print(color_text(f"✓ You said: \"{text}\"", 'green'))
            voice.close()
            return text
            
        except Exception as e:
            print(color_text(f"✗ Error: {e}", 'red'))
            voice.close()
            return input(color_text("Your response: ", 'orange'))


def listen_for_yes_no_voice(prompt_text):
    """
    Speaks a prompt and listens for yes/no response.
    
    Parameters:
    prompt_text : str
        Text to speak as prompt.
        
    Returns:
    bool
        True if user said yes, False if no.
    """
    voice = VoiceOpenAI()
    voice.speak(prompt_text)
    
    while True:
        try:
            text = voice.listen().lower()
            
            print(color_text(f"✓ You said: \"{text}\"", 'green'))
            
            if any(word in text for word in ['yes', 'yeah', 'sure', 'okay', 'yep', 'accept']):
                voice.close()
                return True
            elif any(word in text for word in ['no', 'nope', 'reject', 'cancel']):
                voice.close()
                return False
            else:
                voice.speak("Please say yes or no.")
                
        except Exception as e:
            print(color_text(f"✗ Error: {e}", 'red'))
            voice.close()
            choice = input(color_text("Accept? (y/n): ", 'orange'))
            return choice.lower() == 'y'