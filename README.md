## Iris Classification with PyTorch

This project implements a simple neural network model to classify the famous Iris dataset using PyTorch. The code includes data loading, model training, validation, and testing.

Prerequisites
Before running the code, ensure you have the following dependencies installed:

Python 3.x
PyTorch
pandas
numpy
You can install the required Python libraries by running:

```bash
pip install torch pandas numpy
```

## Dataset
The dataset used for this project is the Iris dataset, which consists of 150 samples of iris flowers classified into three species: Setosa, Versicolor, and Virginica. The dataset should be saved in a CSV format in the following structure:

```bash
Dataset/iris/iris.csv
```
Make sure the CSV file includes the following columns:

* Four feature columns representing the sepal length, sepal width, petal length, and petal width.
* One label column for the species (0, 1, or 2 for each of the iris species).

## Code Explanation
1. Load Data
The ***DataSet*** class handles loading the Iris dataset from the CSV file located at ***Dataset/iris/iris.csv***.

2. Split Data
The dataset is split into three sets: training, validation, and testing with respective sizes of 110, 10, and 30 samples.

3. Set Data Loaders
PyTorch's ***DataLoader*** is used to create loaders for the training, validation, and test sets. The ***batch_size*** is set to the full dataset size in each case (i.e., no mini-batching), and the data is not shuffled.

4. Initialize Model
A neural network model is initialized using the ***Model*** class, which should be defined in the ***Model.py*** file.

5. Set Loss Function & Optimizer
A cross-entropy loss function is chosen because this is a classification task. The Adam optimizer is used to update the model's weights with a learning rate of 0.01.

6. Train the Model
The model is trained for 500 epochs. During training, the loss is computed, and the optimizer updates the model's weights. Validation is performed every 50 epochs, and the accuracy is printed.

7. Test the Model
The model is tested on the test dataset, and the accuracy is printed.

8. Save the Model (Optional)
After training, you can save the model weights using the ***torch.save*** function.

## Usage
* Clone the repository and navigate to the project directory.

* Ensure the Iris dataset is available in the Dataset/iris/ directory in CSV format.

* Run the script:

```bash
python main.py
```

The code will load the data, train the model, and display the validation and test accuracies.

## Customization
* Model Architecture: You can modify the neural network architecture by editing the Model.py file.
* Hyperparameters: Change the learning rate, batch size, or the number of epochs to experiment with different training settings.
* Dataset: Replace the Iris dataset with any CSV dataset containing numerical features and labels.
