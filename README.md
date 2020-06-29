# Speech_Recognition

Machine learning program built using TensorFlow Python that can distinguish when a user says "Hey Pop." 

Note that the images and audio files in the folders are place holders and will be over-written when the program is used.

# Inspiration

I made this program to learn more about audio processing, TensorFlow, and machine learning. It was inspired by Apple's "Hey Siri."

# How it works

1. The user will record themselves saying "Hey Pop" five times. Afterwards, the progarm will process the audio and use TensorFlow to create a keras model to recognize for the user's voice.
2. Once the model is made, the user can say "Hey Pop" into their microphone and the program will identify whether it is the original user or now.

# How to deploy Speech_Recognition

1. Download Python3 and this repository
2. Open this repository in an IDE that compiles Python3 and install all required modules into the interpreter
3. Run the code and enjoy :)

# Important information

### Installed Modules

- sounddevice
- write
- librosa
- numba==0.48
- matplotlib
- numpy
- glob
- TensorFlow
- keras
- pillow

### Python version used

- Python3
