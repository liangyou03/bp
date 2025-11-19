# Age Prediction from PPG and ECG Signals using Deep Learning

This project uses deep learning models to predict age from physiological signals, specifically PPG (photoplethysmography) and ECG (electrocardiogram).

## How to Run the Model

To run the age prediction model, follow these steps:

### 1. Install Dependencies

First, install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

### 2. Preprocess the Data

Next, run the data preprocessing script to prepare the dataset for training.

```bash
python data_preprocess.py
```

This will generate the necessary data files in the `data/` directory.

### 3. Train the Model

Finally, you can train the model using the `train.py` script. You can choose the model architecture, the type of data to use, and other training parameters.

Here is an example of how to run the training script:

```bash
python train.py --model cnn --data ppg --epochs 100 --lr 0.001 --batch_size 32
```

#### Command-Line Arguments:

*   `--model`: The model architecture to use. Choices: `cnn`, `resnet`, `squeezenet`.
*   `--data`: The type of signal to use. Choices: `ppg`, `ecg`, `all`.
*   `--epochs`: The number of training epochs.
*   `--lr`: The learning rate for the optimizer.
*   `--batch_size`: The batch size for training.
*   `--log_path`: The directory to save training logs.
*   `--weight_path`: The directory to save the trained model weights.

## Model Building and Training

The model is built and trained using the PyTorch library.

### Model Architecture

The project allows you to choose from three different deep learning models for age prediction:

*   **1D-CNN (1-Dimensional Convolutional Neural Network):** A standard convolutional network adapted for 1D sequence data like ECG or PPG signals.
*   **1D-ResNet (1-Dimensional Residual Network):** A variation of ResNet that uses residual connections to allow for training deeper networks more effectively.
*   **SqueezeNet:** A compact neural network architecture that achieves high accuracy with fewer parameters.

### Training Process

1.  **Data Loading:** The script starts by loading the preprocessed training and validation data.
2.  **Model Initialization:** Based on the selected model, it initializes the corresponding neural network architecture.
3.  **Optimizer:** It uses the **Adam optimizer** to train the model.
4.  **Loss Function:** The model is trained to minimize the **Mean Squared Error (MSE)** between the model's age prediction and the actual age.
5.  **Training Loop:** The script iterates through the training data, calculates the loss, and updates the model's weights using backpropagation.
6.  **Validation:** After each epoch, it evaluates the model on a separate validation dataset.
7.  **Saving the Best Model:** The script saves the model that achieves the lowest validation loss to prevent overfitting.

## File Descriptions

*   **`README.md`**: This file, providing an overview of the project.
*   **`data_preprocess.py`**: A Python script to preprocess the raw PPG and ECG data.
*   **`train.py`**: The main Python script to train the age prediction models.
*   **`requirements.txt`**: A list of Python libraries required to run the project.
