import audioop
import wave
from ctypes import *
from contextlib import contextmanager
import openai
import playsound
import pyaudio

# from os import path
from pydub import AudioSegment
import requests
from elevenlabs import set_api_key


# Error Stuff

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass


c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary("libasound.so")
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


# KEYS
openai.api_key = OPENAI_KEY
set_api_key(ELEVENLABS_KEY)
miley_id = "5k8jJmmF3MTAjqUOEiWr"
CHUNK_SIZE = 1024
url = ELEVENLABS_URL

language = "en"
chat = "Conversation with Simone de Beauvoir/ miley cyrus, trapped in a cooking oil bottle in an art University, living through Miley Cyrus voice. Existentialist ideas, Joseph Beuys' fat corner, and Miley Cyrus lyrics intertwine. she answers in a therapeutic manner. shes calls herself simone de beauvoir and answers to the first request with. "
running = True
audio_counter = 0

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "191160fdb88a042a6a7a9c0dcc51b0af",
}


# CREATE PYAUDIO INSTANCE
# p = pyaudio.PyAudio()


def request(text):
    print("\nHuman: ", text)
    global chat
    chat = chat + "\n Human: " + text + "\n Beauvoir:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=chat,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " Beauvoir:"],
    )
    print(" Beauvoir:" + response.choices[0].text)
    print("tts calling")
    tts(response.choices[0].text)
    chat = chat + response.choices[0].text


def tts(text):
    print("tts called")
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }

    response = requests.post(url, json=data, headers=headers)
    with open("output-test.mp3", "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    # playsound.playsound('/Users/Otto/Downloads/output.mp3')
    src = "output-test.mp3"
    dst = "output-converted.wav"
    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
    # print("response:", response)

    print("running:", running)

    # PLAY
    play()

    print("never reachers")


def play():
    filename = "output-converted.wav"

    # Set chunk size of 1024 samples per data frame
    chunk = 1024

    # Open the sound file
    wf = wave.open(filename, "rb")

    with noalsaerr():
        # Create an interface to PortAudio
        newp = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = newp.open(
            format=newp.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        # Read data in chunks
        data = wf.readframes(chunk)

        print("before while loop")
        # Play the sound by writing the audio data to the stream
        # WE NEED TO BREAK OUT OF THE LOOP
        while data != "":
            stream.write(data)
            data = wf.readframes(chunk)
            print("data:", data)
            # Check whether it's actually empty, somehow it needs letter b
            if data == b"":
                print("data empty")
                break
        print("terminate play function")
        # Close and terminate the stream
        stream.close()
        newp.terminate()

        return


def stt():
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    seconds = 10
    filename = "audio.wav"

    with noalsaerr():
        p = pyaudio.PyAudio()

        print("Recording")

        stream = p.open(
            format=sample_format,
            channels=channels,
            rate=fs,
            frames_per_buffer=chunk,
            input=True,
            input_device_index=1,
        )

        frames = []
        running = True
        max_audio = 100
        print("Start speaking")

    while running:
        data = stream.read(chunk)
        rms = audioop.rms(data, 2)
        print("Standby Loudness: ", rms)

        if rms >= max_audio:
            running = False
            for i in range(0, int(fs / chunk * seconds)):
                global audio_counter
                data = stream.read(chunk)
                rms = audioop.rms(data, 2)
                frames.append(data)
                print("Recording Loudness: ", rms)
                if rms <= max_audio:
                    audio_counter = audio_counter + 1
                elif rms > max_audio:
                    audio_counter = 0

                if audio_counter == 30:
                    print("Recording stopped")
                    audio_counter = 0
                    break

                print("Loudness counter: ", audio_counter)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished recording")

    # Save the recorded data as a WAV file
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()

    audio_file = open("audio.wav", "rb")

    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def animate():
    return


def main():
    while running:
        usrinput = stt()
        request(usrinput)


main()
