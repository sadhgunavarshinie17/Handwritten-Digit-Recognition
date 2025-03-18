import os
import cv2 #to load and process images
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Normalizing the dataset
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential() #Linear stack of layers
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #flattens grids - eg: 28 * 28 -> 784 pixels
model.add(tf.keras.layers.Dense(128, activation='relu')) #128 units
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #softmax -> all outputs add up to 1

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)

model.save('handwritten.keras')


model = tf.keras.models.load_model('handwritten.keras')

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)



image_number = 1
while os.path.isfile(f"Digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
