# Diabetes Prediction with ANN

This repository contains code for building and optimizing an artificial neural network (ANN) to predict diabetes using a dataset. The project involves data preprocessing, model creation, hyperparameter tuning, and training.

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview

This project utilizes a diabetes dataset to build a neural network model for binary classification. The neural network is designed using TensorFlow and Keras, with hyperparameter tuning performed using Keras Tuner.

## Data Preparation

1. **Data Loading**: The dataset is loaded from a CSV file.
2. **Feature Scaling**: The features are scaled using `StandardScaler` to normalize the data.
3. **Train-Test Split**: The data is split into training and testing sets using an 80-20 ratio.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Mustafa Badshah/OneDrive - Higher Education Commission/Desktop/project/ANN project 1/diabetes.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

## Model Building

The neural network model is created using TensorFlow and Keras. The model architecture includes a configurable number of hidden layers and nodes.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', min_value=1, max_value=10)):
        units = hp.Int(f'units_{i}', min_value=8, max_value=128, step=8)
        activation = hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid'])
        
        if i == 0:
            model.add(Dense(units, activation=activation, input_dim=X_train.shape[1]))
            model.add(Dropout(hp.Choice('Dropout', values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
        else:
            model.add(Dense(units, activation=activation))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adadelta'])
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
```

## Hyperparameter Tuning

The model is optimized using Keras Tuner. Various hyperparameters, such as the number of layers, units per layer, activation functions, and dropout rates, are tuned to achieve the best performance.

```python
import keras_tuner as kt

tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=5, directory='mydir', project_name='m5')
tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model = tuner.get_best_models(num_models=1)[0]
model.fit(X_train, y_train, epochs=200, initial_epoch=6, validation_data=(X_test, y_test))
```

## Results

The performance of the model can be evaluated using the validation accuracy. After tuning, the model with the best hyperparameters is selected and trained for further epochs to improve accuracy.

## Usage

1. **Install Dependencies**: Ensure that the required libraries are installed. You can install them using pip:
   ```bash
   pip install pandas scikit-learn tensorflow keras keras-tuner
   ```
2. **Run the Script**: Execute the script to train and evaluate the model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to modify the content as needed for your project specifics or add any additional sections.
