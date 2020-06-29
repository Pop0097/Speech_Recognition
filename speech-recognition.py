# imports all required modules
# packages for recording audio
import sounddevice as sd
from scipy.io.wavfile import write


# records five audio samples that will be used for the machine learning

print("You will record five 3 SECOND audio files. For each recording, you have to say \"Hey Pop\" into the microphone.\n\n")

duration = 3  # each audio sample will be 5 seconds long
fs = 44100  # sample rate of each audio sample

for i in range(5):

    x = input("When you are read, hit ENTER to start recording sample " + str(i+1) + "/5: ")

    print("\nRecording " + str(i+1) + "/5...\n\n")
    recording = sd.rec(int(duration*fs), samplerate=fs, channels=2) # records audio from microphone
    sd.wait() # line tells program to wait until the three seconds are up
    print("Done recording " + str(i+1) + "/5.\n\n")

    audio_path = "audio/audio_input_" + str(i+1) + ".wav" # initializes path of the audio file
    write(audio_path, fs, recording) # saves audio input as a WAV audio file in the audio/ directory

print("Thank you for using this program!")
