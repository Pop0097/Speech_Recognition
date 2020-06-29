# imports all required modules
# packages for recording audio
import sounddevice as sd
from scipy.io.wavfile import write


# records five audio samples that will be used for the machine learning

for i in range(5):
    duration = 5 # each audio sample will be 5 seconds long
    fs = 44100 # sample rate of each audio sample

    x = input("Hit ENTER to start recording " + str(i+1) + "/5: ")

    print("Working...")
