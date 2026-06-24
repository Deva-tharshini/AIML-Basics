# Session 13 – Neural Networks (SLNN & MLNN)

## Objective

To understand and implement Single Layer Neural Network (SLNN) and Multi Layer Neural Network (MLNN) for image classification using the MNIST handwritten digits dataset.

## Dataset Used

MNIST Dataset (TensorFlow Built-in Dataset)

Dataset contains grayscale images of handwritten digits from 0–9.

Training Images: 60,000  
Testing Images: 10,000

Image Size: 28 × 28 pixels

## Features

Input Features:

* Pixel values of handwritten images (784 features after flattening)

Target Classes:

* Digits 0–9

Dataset Structure:

* Images
* Labels

## Operations Performed

### Data Preparation

* Loaded MNIST dataset using TensorFlow
* Normalized image values
* Converted image dimensions from 28×28 into 784 input features
* Converted output labels into categorical format

### Single Layer Neural Network (SLNN)

* Built neural network with one output layer
* Used Softmax activation
* Trained model for digit classification
* Evaluated model accuracy
* Generated accuracy graph

### Multi Layer Neural Network (MLNN)

* Built neural network with multiple hidden layers
* Used ReLU and Softmax activation
* Trained deeper network
* Evaluated classification performance
* Generated accuracy graph

## Model Outputs

SLNN Outputs:

* Training Accuracy
* Validation Accuracy
* Final Model Accuracy
* Accuracy Graph

MLNN Outputs:

* Training Accuracy
* Validation Accuracy
* Final Model Accuracy
* Accuracy Graph

## Real-world Applications

* Handwritten digit recognition
* Optical Character Recognition (OCR)
* Document digitization
* Bank cheque processing
* Postal code recognition
* Deep learning image classification

## Output Files

* slnn_console_output.png
* slnn_graph.png
* mlnn_console_output.png
* mlnn_graph.png

## Learning Outcome

Learned the difference between Single Layer Neural Networks and Multi Layer Neural Networks, how neural networks process image data, and how hidden layers improve classification performance.

## Technologies Used

* Python
* TensorFlow
* NumPy
* Matplotlib
* Neural Networks
* Deep Learning