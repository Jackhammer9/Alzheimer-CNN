# Alzheimer's Detection CNN Project

## Overview
This project focuses on detecting Alzheimer's disease using a **Convolutional Neural Network (CNN)** trained on MRI scans. The model classifies images into four categories representing different stages of Alzheimer's progression.

## Features
- **Deep Learning Model**: Uses a CNN architecture with multiple convolutional and dense layers.
- **Optimized Training**: Includes techniques like batch normalization, dropout, and learning rate scheduling.
- **Evaluation Metrics**: Tracks accuracy, precision, recall, F1-score, ROC-AUC, and loss.
- **Visualization**: Generates various performance graphs, including training curves, confusion matrices, and precision-recall curves.

## Dataset
The dataset consists of MRI images classified into four categories:
1. **Non-Demented**
2. **Very Mild Demented**
3. **Mild Demented**
4. **Moderate Demented**

Images are preprocessed with resizing, normalization, and augmentation.

## Model Architecture
The CNN model consists of:
- Input layer with **Rescaling (1./255)**
- **Three Convolutional Blocks** with Batch Normalization & MaxPooling
- **Fully Connected Layers** with Dropout (0.4-0.5)
- **Softmax Output Layer** for multi-class classification
- **Adam Optimizer (lr=0.0001)** with categorical cross-entropy loss

## Training
The model is trained using:
```python
model.fit(train_dataset, epochs=50, validation_data=test_dataset, class_weight=class_weights)
```
With callbacks for **early stopping** and **learning rate reduction**:
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```

## Evaluation
The model is evaluated using:
- **Accuracy & Loss Curves**
- **Precision-Recall & ROC-AUC Curves**
- **Confusion Matrix**
- **Classification Report**

Example of evaluation code:
```python
from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_true_labels, y_pred, target_names=class_names))
```

## Usage
1. **Train the Model**
   ```python
   training = model.fit(train_dataset, epochs=50, validation_data=test_dataset, class_weight=class_weights)
   ```
2. **Evaluate Performance**
   ```python
   model.evaluate(test_dataset)
   ```
3. **Predict on New Images**
   ```python
   img = image.load_img("test_image.jpg", target_size=(176, 208))
   img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
   predictions = model.predict(img_array)
   predicted_class = np.argmax(predictions)
   print(f"Predicted class: {class_names[predicted_class]}")
   ```

## Future Improvements
- Implement **Transfer Learning** for better accuracy.
- Experiment with **Self-Supervised Learning**.
- Deploy as a **Web App** for real-world use.

## Author
Developed by **Arnav Bajaj**.

## License
This project is licensed under the **MIT License**.