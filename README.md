# Traffic Sign Classification using Convolutional Neural Networks (CNN)

This script implements a CNN model to classify traffic signs using the GTSRB dataset. The steps include loading and preprocessing the data, defining and training the model, and finally evaluating its accuracy.

## Overview

The script performs the following tasks:

1. **Import Libraries**: Imports necessary libraries for data processing, model creation, and evaluation.
   
2. **Data Loading and Preprocessing**: 
   - Loads training images from the dataset.
   - Converts images to arrays and labels them according to their respective classes.
   - Splits the dataset into training and testing sets.

3. **CNN Model Definition**:
   - Defines a Sequential CNN model with multiple convolutional, pooling, and dropout layers to prevent overfitting.
   - Compiles the model using the Adam optimizer and categorical crossentropy as the loss function.

4. **Model Training**:
   - Trains the model on the preprocessed data with a specified batch size and number of epochs.
   - Uses a validation split to monitor the model's performance during training.

5. **Model Evaluation**:
   - Loads test data from a CSV file.
   - Preprocesses the test images and predicts their classes using the trained model.
   - Calculates and prints the accuracy of the model on the test data.

## Usage

Ensure that the GTSRB dataset is organized correctly with `train` and `test` folders containing the images. The test data should also be provided in a `Test.csv` file with the correct structure.

Run the script to train the CNN model on the training data and evaluate its performance on the test data.

## Dependencies

The script requires the following Python libraries:
- `os`, `sys`, `numpy`, `pandas`
- `PIL.Image` for image processing
- `sklearn` for model evaluation and data splitting
- `keras` for building and training the CNN model

## Results

The script will output the accuracy of the model on the test dataset, providing an indicator of its performance in classifying traffic signs.

