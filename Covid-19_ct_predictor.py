import tensorflow as tf
import numpy as np
from tensorflow import keras
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image


model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
model.summary()

def predict(path):
    test_img_path = path

    img_height = 180
    img_width = 180

    img = keras.preprocessing.image.load_img(
        test_img_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ["CT_COVID", "CT_NonCovid"]

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    predicted_class = class_names[np.argmax(score)]
    return predicted_class

def open_img():
    # Select the Imagename from a folder
    x = openfilename()

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    panel.grid(row=2)
    output=predict(x)
    root2 = Tk()
    # Set Title as Image Loader
    root2.title("Prediction")
    if output == "CT_COVID":
        output="POSITIVE"
        l = Label(root2, text="Patient is most Likely Covid " + output, font=("times new roman", 20), fg="white",
                  bg="maroon", height=2)
        l.pack()

    else:
        output="NEGATIVE"
        l = Label(root2, text="Patient is most Likely Covid " + output, font=("times new roman", 20), fg="white",
                  bg="green", height=2)
        l.pack()

    
    mainloop()


def openfilename():
    filename = filedialog.askopenfilename(title='"pen')
    return filename


# Create a window
root = Tk()

# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
#root.geometry("550x300 + 300 + 150")

# Allow Window to be resizable
root.resizable(width = True, height = True)

# Create a button and place it into the window using grid layout
btn = Button(root, text ='open image', command = open_img).grid(row = 1, columnspan = 4)
root.mainloop()