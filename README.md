# Deep-learning

                            Plant Disease Classification using NN vs CNN
1. Title of the Problem Statement

"Plant Disease Detection using Neural Networks (NN) and Convolutional Neural Networks (CNN): A Comparative Study on the PlantVillage Dataset"

2. Problem Description

Plant diseases significantly reduce agricultural yield and farmers’ income. Early detection of diseases from leaf images can help in timely treatment and crop protection.
This project compares two approaches:

Tabular NN (using handcrafted features like color and texture)

CNN (using raw images, automatically extracting spatial features)

Goal: Identify which model performs better for plant disease classification.

3. Dataset Details

Source: PlantVillage Dataset (Kaggle)

Content: ~54,000 leaf images of 38 classes (healthy + diseased plants).

Structure: Already split into train/ and test/ folders with class-wise subfolders.

Image Size: resized to 128×128 pixels for training.

4. Neural Network (NN) Design

Input Layer: Feature vector extracted from images (RGB mean, std, variance, skewness, kurtosis + color histograms + grayscale histogram).

Hidden Layers:

Dense(512) → Dense(256) → Dense(128) with ReLU activation.

Dropout layers (0.3–0.4) to prevent overfitting.

Output Layer: Dense(38, softmax).

Hyperparameters & Justification:

Optimizer: Adam (fast convergence).

Loss: Sparse categorical crossentropy (multi-class classification).

Epochs: 50 (balanced between training time and convergence).

Batch size: 64 (efficient GPU utilization).

5. Convolutional Neural Network (CNN) Design

Input: Image (128×128×3).

Architecture:

Conv2D(32, 3×3) + MaxPooling2D

Conv2D(64, 3×3) + MaxPooling2D

Conv2D(128, 3×3) + MaxPooling2D

Flatten → Dense(128, ReLU) → Dense(38, Softmax).

OR Use Pretrained Model (MobileNetV2 / ResNet50):

Pretrained on ImageNet, fine-tuned for plant disease classification.

Advantage: Captures complex spatial disease patterns with fewer training resources.

Hyperparameters & Why:

Optimizer: Adam.

Learning Rate: 1e-4 (stable training).

Batch size: 32.

Epochs: 5–10 (CNNs learn faster with transfer learning).

6. Performance Metrics

Accuracy – overall correctness.

Precision – how many predicted positives are correct.

Recall – ability to find all actual positives.

F1-score – balance between precision & recall.

Confusion Matrix – per-class performance.

7. Results and Discussion
Model	Accuracy	Precision	Recall	F1-score
CNN	91.3%	91.8%	91.3%	91.4%
Tabular NN	77.8%	77.9%	77.8%	77.5%

CNN outperformed NN by ~13% in accuracy.

CNN captured leaf disease spots, texture patterns, and shapes that handcrafted features missed.

Tabular NN was limited but still reasonable (~78%).

8. Analysis: NN vs CNN – Which is Better?

CNN is clearly better because it learns spatial features directly from images.

NN depends on handcrafted features → may miss complex disease patterns.

For real-world deployment, CNN (or a pretrained model like MobileNetV2) is the best choice.

NN can still be useful if computing resources are limited (since it only needs feature vectors).
