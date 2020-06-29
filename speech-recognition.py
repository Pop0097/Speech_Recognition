# imports all required modules
# modules for recording audio
import sounddevice as sd
from scipy.io.wavfile import write

# modules for creating .png files
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob

# modules for transforming image
from tensorflow.python.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# training keras model
import os
import tensorflow as tf

# predicting
from keras.preprocessing import image


### records five audio samples that will be used for the machine learning
print("You will record five 3 SECOND audio files. For each recording, you have to say \"Hey Pop\" into the microphone.\n\n")

duration = 3  # each audio sample will be 5 seconds long
fs = 44100  # sample rate of each audio sample

def record_audio(name): # function creates WAV audio files
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)  # records audio from microphone
    sd.wait()  # line tells program to wait until the three seconds are up

    write(name, fs, recording)  # saves audio input as a WAV audio file in the audio/ directory

for i in range(5):

    x = input("When you are read, hit ENTER to start recording sample " + str(i+1) + "/5: ")

    print("\nRecording " + str(i+1) + "/5...\n\n")
    audio_path = "audio/input-" + str(i + 1) + ".wav"  # initializes path of the audio file
    record_audio(audio_path)
    print("Done recording " + str(i+1) + "/5.\n\n")



print("Creating your model This may take a while...")

### creates training data
def convert_audio_to_png(file_name, save_name, save_path): # this function creates the .png
    plt.interactive(False) # image will not be interactive
    clip, sample_rate = librosa.load(file_name, sr=None) # loads file for processing with librosa library
    fig = plt.figure(figsize=[0.72, 0.72]) # defines figure size
    # tells program not to add axes to the diagrams
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate) # converts WAV file to spectrogram
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max)) # specifies the characteristics of the spectrogram
    save_name = save_path + save_name + ".png" # creates path for the .png
    plt.savefig(save_name, dpi=400, bbox_inches="tight", pad_inches=0) # creates .png files
    # closes plots and figures
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close("all")
    del file_name, save_name, clip, sample_rate, fig, ax, S # deletes objects

def create_more_training_data(original_file_name, count):
    img = load_img(original_file_name) # loads image that will be transformed
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = np.expand_dims(data, 0)
    # create image data augmentation generator
    generate_training_images = ImageDataGenerator(
        width_shift_range=[-10, 10],
        brightness_range=[0.6, 1.3],
        fill_mode="constant"  # should make images fill with black when translating
    )
    # prepare iterator
    it = generate_training_images.flow(samples, batch_size=1)
    # creates 50 new images
    for i in range(50):
        plt.interactive(False)
        batch = it.next()  # transforms image
        image = batch[0].astype("uint8")  # gets the image and converts it to an unsigned integer
        fig = plt.figure(figsize=[0.72, 0.72])  # creates a figure
        plt.imshow(image)  # shows the transformed image
        # does not display axes
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        save_path = "training-images/transformed-image-" + str(count) + "-" + str(i) + ".png"  # creates file path
        plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0) # saves .png of transformed image
        # closes plots and figures
        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close("all")
        del save_path, fig, ax # deletes objects

audio_folder = "audio/" + "*.wav" # accesses folder with all audio files
# loop converts all images to .png and creates more training data
for file in glob.iglob(audio_folder):
    # 2 step process: 1. splits file name to get name of .wav file. 2. Splits name of .wav file to remove the .wav
    img_name = file.split("/")[-1].split(".")[0]
    # print(file + "; " + img_name)
    convert_audio_to_png(file, img_name, "images/") # calls function to create .png

image_folder = "images/" + "*.png" # accesses folder with all .png files
counter = 0
# loop transforms all .png
for file in glob.iglob(image_folder):
    # print(file)
    create_more_training_data(file, counter) # calls function to create more images
    counter += 1

### creates the machine learning keras model
# gets training images
training_images_path = os.path.dirname(__file__) + "/training-images/"

# defines keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),  # Reduces the dimensions to feed into the layer

    # 256 neuron hidden layer
    tf.keras.layers.Dense(256, activation="relu"),
    # Output (non-hidden) layer
    tf.keras.layers.Dense(2, activation="softmax")
])

# model.summary()

# trains keras model
image_generator = ImageDataGenerator()

training_generator = image_generator.flow_from_directory(
    training_images_path,
    target_size=(200,200),
    batch_size=16,
    class_mode="categorical",
    shuffle=True
)

# print(training_generator.classes)

# compiles the model
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer="adam",
    metrics=["acc"]
)

# trains model
hist = model.fit(
    training_generator,
    steps_per_epoch=10,
    epochs=10,
    verbose=0,
)

### user inputs audio and the program determines if it is the user who spoke
print("You can now record 3 second audio clips and see if the program can recognize you:\n\n")
finished = False

while not(finished):
    x = input("When you are ready, hit ENTER to start recording a sample: ")

    print("\nRecording...\n\n")
    audio_path = "audio-input.wav"  # initializes path of the audio file
    record_audio(audio_path) # calls function to make WAV file
    print("Done recording.\n\n")

    # makes audio into a .png
    convert_audio_to_png(audio_path, "test", "")

    # sets up image
    image_path = "test.png"
    current_img = image.load_img(image_path, target_size=(200, 200, 3))  # Resizes image
    x = image.img_to_array(current_img)
    x = np.expand_dims(x, axis=0)

    # stack up images list to pass for prediction
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10) # predicts
    print(classes)

    # prints results for user
    if classes[0][1] > classes[0][0] and classes[0][1] >= 0.4:
        print("It is you!")
    else:
        print("It is NOT you!")

    # checks if user would like to exit the program
    x = input("Would you like to exit the program? (\"y\" for yes and ENTER for no): \n")
    if x == "y":
        finished = True

print("Thank you for using this program!")
