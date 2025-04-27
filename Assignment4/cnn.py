#-------------------------------------------------------------------------
# AUTHOR: Minh Nhat Doan
# FILENAME: cnn.py
# SPECIFICATION: Train a 32×32-pixel digit-classifier CNN
# FOR: CS 4210 - Assignment #4
# TIME SPENT: 2 hrs
#-------------------------------------------------------------------------

# Importing Python libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load dataset
def load_digit_images_from_folder(folder_path, image_size=(32, 32)):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        # Getting the label of the image (it's the first number in the filename)
        # --> add your Python code here.
        label = int(filename.split("_")[0])

        # Converting images to grayscale and resizing them to 32x32
        img = Image.open(os.path.join(folder_path, filename)).convert('L').resize(image_size)

        # Adding the converted image to the feature matrix and label to the class vector
        X.append(np.array(img, dtype=np.float32))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# Set your own paths here (relative to your project folder)
train_path = os.path.join("digit_dataset", "train")
test_path = os.path.join("digit_dataset", "test")

# Loading the raw images using the provided function. Hint: Use the provided load_digit_images_from_folder function that outputs X_train, Y_train for train_path and
# as X_test, Y_test for test_path
# --> add your Python code here
X_train, Y_train = load_digit_images_from_folder(train_path)
X_test, Y_test = load_digit_images_from_folder(test_path)

# Normalizing the data: convert pixel values from range [0, 255] to [0, 1]. Hint: divide them by 255
# --> add your Python code here
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping the input images to include the channel dimension: (num_images, height, width, channels)
# --> add your Python code here
X_train = X_train.reshape((-1, 32, 32, 1))
X_test = X_test.reshape((-1, 32, 32, 1))

# Building a CNN model
model = models.Sequential([
    # Add a convolutional layer with 32 filters of size 3x3, relu activation, and input shape 32x32x1
    # --> add your Python code here
    layers.Conv2D(32, (3, 3), activation='relu', input_shape = (32, 32, 1)),

    # Add a max pooling layer with pool size 2x2
    # --> add your Python code here
    layers.MaxPooling2D((2, 2)),

    # Add a flatten layer to convert the feature maps into a 1D vector
    # --> add your Python code here
    layers.Flatten(),

    # Add a dense (fully connected) layer with 64 neurons and relu activation
    # --> add your Python code here
    layers.Dense(64, activation='relu'),
    
    # Add the output layer with 10 neurons (digits 0–9) and softmax activation
    # --> add your Python code here
    layers.Dense(10, activation='softmax')
])

# Compiling the model using optimizer = sgd, loss = sparse_categorical_crossentropy, and metric = accuracy
# --> add your Python code here
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the model with batch_size=32 and epochs=10
# --> add your Python code here
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

# Evaluating the model on the test set
loss, acc = model.evaluate(X_test, Y_test)

# Printing the test accuracy
# --> add your Python code here
print(f"Test accuracy: {acc:.4f}")
