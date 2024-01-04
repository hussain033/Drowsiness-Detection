# Drowsiness Detection
Drowsiness Detection is a crucial aspect of ensuring driver safety, especially in long journeys. This project employs deep learning models and image processing techniques to detect signs of drowsiness in drivers and trigger timely alerts to prevent potential accidents.

## Introduction

### Background
Drowsy driving is a significant factor in road accidents worldwide. Fatigue impairs a driver's ability to focus, react quickly, and make sound decisions. Detecting drowsiness in real-time is crucial for preventing accidents and ensuring road safety.

### Motivation

The primary motivation behind this project is to leverage advanced technologies to address the alarming issue of drowsy driving. By employing deep learning and image processing, we aim to create an efficient and reliable system that can monitor driver behavior and provide timely alerts, potentially saving lives on the road.

### Project Objectives

1. **Real-time Detection:** Develop a system capable of monitoring drivers in real-time and detecting signs of drowsiness promptly.

2. **Deep Learning Integration:** Utilize a deep learning model to recognize facial features associated with drowsiness, ensuring accurate and reliable detection.

3. **Alert Mechanism:** Implement an alert mechanism that can notify drivers when signs of drowsiness are identified, helping them stay alert and focused.

4. **Customization:** Provide configurable parameters to adapt the system to various driving conditions and individual preferences.

## Methodology

### Dataset
- The training dataset consists of 1000 images per class, totalling 2000 images.
- There are two classes: "closed eye" and "open eye."
- The dataset is balanced, with an equal number of images per class.
 
### Model Architecture
- The base MobileNetV2 model pre-trained on the ImageNet competition is used.
- The top layers of the MobileNetV2 model are replaced to fit the specific classification task.
- Data augmentation and preprocessing layers are added to the top of the model to enhance training.
 
### Data Augmentation
- Data augmentation techniques like random flip and random rotation are applied during training.
- These techniques introduce variations in the training data, improving the model's ability to generalize.
 
### Input Image Dimensions
- Initially, the images have dimensions of 300x300 pixels.
- They are resized to 224x224 pixels to match the input size expected by the MobileNetV2 model.
 
### Training Process
- A total of 63 batches are used for training and evaluation.
- Each batch contains 32 images.
- Out of the 63 batches, 42 batches are used for training, and 21 batches are used for validation.
- The T4 GPU available on the Google Colab was used for model training. 
- The model was trained using the accuracy metric as the evaluation criterion.
 
### Evaluation and Results
- Initially, using the pre-trained ImageNet weights, the model achieves an accuracy of 51% on the validation dataset.
- After training the model, the validation accuracy significantly improves to approximately 98%.
- This improvement demonstrates the effectiveness of the training process.

