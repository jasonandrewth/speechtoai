import openai
import wave
import pyaudio
import audioop
from gtts import gTTS
import playsound
import time


openai.api_key = "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"
language = 'de'
chat ="Folgendes ist eine Konversation zwischen Human und HAL von dem Film 2001:A Space Odysey. HAL hat folgt strikt dem Charakter des Films. Die Crew und HAL sind auf einer Mission zum Jupiter. HAL hat böse Absichten.  \n \n HAL: Hallo, ich bin HAL. Ich bin unfehlbar. Es gab noch nie einen Error im Zusammenhang mit mir und Modellen meiner Art. Ich bin arrogant und will nicht über persönliche sachen reden. \n Human: "
running = True
audio_counter = 0


def request(text):
    print("\nHuman: ", text)
    global chat
    chat = chat + "\n Human: " + text + "\n HAL:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= chat,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " HAL:"]
    )
    print(" HAL:"+ response.choices[0].text)
    tts(response.choices[0].text)
    chat = chat + response.choices[0].text



def tts(text):
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("answer.mp3")
    playsound.playsound('answer.mp3')

def stt():

    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    seconds = 10
    filename = "audio.wav"
    p = pyaudio.PyAudio()

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []
    running = True
    max_audio = 1800
    print("Start speaking")

    while running:
        data = stream.read(chunk)
        rms = audioop.rms(data, 2)
        print("Standby Loudness: ",rms)


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

                if audio_counter == 100:
                    print("Recording stopped")
                    audio_counter = 0
                    break

                print("Loudness counter: ", audio_counter)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
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