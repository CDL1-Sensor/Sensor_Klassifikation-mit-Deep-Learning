# Klassifikation mit Deep Learning
Abgabetermin: 16. Juni 2023  
In diesem Repository erfolgt die Klassifizierung der Sensordaten mittels Deep Learning Modellen.  

## Deep Learning Notebook
- DL-Modelle.ipynb

## Experimente und Validierungen:
├── DL-Experiments <br>
│   ├── Notebook XXY  <br>
│   └── Notebook XYY  <br>\

## Exported Model für Frontend APP:
- https://github.com/CDL1-Sensor/Sensor_Klassifikation-mit-Deep-Learning/tree/main/saved_model/sensor_model

## Installation:

### Pip

``` bash
pip install -r requirements.txt
```

### X.txt
Das File beinhaltet einen Testdaten Satz für Fahrrad fahren. Für schnelles debugging.

## Bestes Modell:
Das beste Modell ist: sensor_model\create_model_1.h5

``` python
def create_model_1(name="model_1"):
    '''
    CNN Model with 1 Convolutional Layer, 1 LSTM Layer and 1 Dense Layer 
    '''
    model = tf.keras.Sequential(
        [
            # Add a 1D convolutional layer
            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=12,
                activation="relu",
                padding="same",
                input_shape=(timesteps, n_features),
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            # Add LSTM layer
            tf.keras.layers.LSTM(100),
            # Add a dense output layer
            tf.keras.layers.Dense(
                6, activation="softmax"
            ),  # Change activation function based on the nature of the output
        ],
        name=name,
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model
```

Da dieses Modell weniger overfitted und in der Evaluation auf unseen Data 4/5 Acitivies richtig klassifiziert.
