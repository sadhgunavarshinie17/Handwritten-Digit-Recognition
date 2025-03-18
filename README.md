# Handwritten-Digit-Recognition
A machine learning project that uses a neural network to recognize handwritten digits from the MNIST dataset, as well as custom hand-drawn digits.

## Overview

This project implements a simple neural network using TensorFlow and Keras to classify handwritten digits. It consists of two main components:
1. Training a model on the MNIST dataset
2. Using the trained model to recognize custom hand-drawn digits

## Model Architecture

The neural network has the following architecture:
- Input layer: 28x28 pixels (flattened to 784 inputs)
- First hidden layer: 128 neurons with ReLU activation
- Second hidden layer: 128 neurons with ReLU activation
- Output layer: 10 neurons (one for each digit 0-9) with softmax activation

## Performance Metrics
- **Loss**: 0.91
- **Accuracy**: 97%

## Custom Digit Recognition Results

The model was tested on 18 custom handwritten digit images with the following results:

| Image | Actual Digit |
|-------|-------------|
| digit1.png | 0 |
| digit2.png | 1 |
| digit3.png | 1 |
| digit4.png | 2 |
| digit5.png | 3 |
| digit6.png | 3 |
| digit7.png | 3 |
| digit8.png | 4 |
| digit9.png | 4 |
| digit10.png | 5 |
| digit11.png | 5 |
| digit12.png | 6 |
| digit13.png | 6 |
| digit14.png | 7 |
| digit15.png | 7 |
| digit16.png | 8 |
| digit17.png | 9 |
| digit18.png | 9 |

The model correctly predicted 11 out of 18 images, resulting in an accuracy of 61% on custom handwritten digits. Screenshots of the model's predictions are provided below.

![image](https://github.com/user-attachments/assets/e1346bfd-e2db-4de9-955a-0b1a9ee90cbb)


## Future Improvements

- Implement data augmentation to improve model robustness
- Add a convolutional neural network (CNN) model for better accuracy
- Create a user interface for drawing and recognizing digits in real-time

