#!/usr/bin/env python
# coding: utf-8

# # Sensor Based Activity Recoginition 
# Challenge: cdl1 - Sensor based Activity Recognition  
# Team: Lea Bütler, Manjavy Kirupa, Etienne Roulet, Si Ben Tran  
# 
# Aufgabe: DL Modell erstellen
# 
# Hier in diesem Notebook erstellen wir unsere Deep Learning Modelle.

# In[128]:


import tensorflow as tf
print(tf.config.list_physical_devices())
tf.debugging.set_log_device_placement(False)


# In[129]:


get_ipython().run_line_magic('pip', 'install scikit-learn')


# In[130]:


get_ipython().run_line_magic('pip', 'install pandas')


# In[131]:


import logging
from datetime import datetime
# datetime as filename for logging
now = datetime.now()
date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(level=logging.INFO, filename = f"{date_time_string}.txt", filemode='a')


# In[148]:


from dataclasses import dataclass

@dataclass
class Parameters():
    batch_size: int = 128
    epochs: int = 2
    verbosity: int = 1
    step_size: int = 374
    number_folds: int = 2
    output_size: int = 6
    


# In[144]:


# Loading Data
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection as ms

# read the CSV file into a DataFrame
df = pd.read_csv("Alle_Messungen_trimmed.csv")
df.head(1)


# In[145]:


df = df.drop(columns=["id", "user", "id_combined"])
df.drop(['Unnamed: 0'], axis=1, inplace=True)
# get all types of the df
df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'].astype('int64')//1e9
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])


# In[135]:


def factors(n):
    result = set()
    for i in range(1, int(n**0.5) + 1):
        div, mod = divmod(n, i)
        if mod == 0:
            result |= {i, div}
    return result

n_samples = X.shape[0]
print(factors(n_samples))


# In[146]:


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X = df.values[:, 1:13]
y = df.values[:, 13]

# Reshape X to 3D format (samples, timesteps, features)
timesteps = 1  # You can choose a different number of timesteps based on the nature of your data
n_features = X.shape[1]
X = X.reshape(-1, timesteps, n_features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)


# In[142]:


#X_train, X_test, y_train, y_test = ms.train_test_split(df.values[:, 1:13], df.values[:, 13], test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

#dataset = tf.data.Dataset.from_tensor_slices((df.values[:, 1:13], df.values[:, 13]))


# In[184]:


# Template
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import dataclasses

# Load data and preprocess
# split train dataset into x_train and y_train
x_train = X_train
x_test = X_test
y_train = y_train
y_test = y_test
    

# Something like this as first Model
def create_model_1():
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(12,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation='relu', input_shape=(12,)),
            tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model


# Something like this as second Model
def create_model_2():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((-1, 12), input_shape=(12,)),
        tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(16, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model

# Something like this as Third Model
def create_model_3():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((-1, 12), input_shape=(12,)),
        tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(16, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model

def create_model_4():
    model = tf.keras.Sequential([
        # Add a 1D convolutional layer
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', input_shape=(timesteps, n_features)),
        
        # Add LSTM layer
        tf.keras.layers.LSTM(100),

        # Add a dense output layer
        tf.keras.layers.Dense(6, activation='softmax')  # Change activation function based on the nature of the output
    ])
    model.compile(optimizer='adam',   loss='categorical_crossentropy', metrics=['accuracy'])  # Change the loss function based on the nature of the output
    return model

best_model_history = None  # Keep track of the best model's history
model_histories = []
# Perform cross-validation
models = [create_model_4]
best_model = None
num_folds = Parameters.number_folds
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_acc_scores = []

for i, (train, test) in enumerate(kfold.split(x_train, y_train)):
    logging.info(f'Fold {i+1}')
    train_x, train_y = x_train[train], y_train[train]
    test_x, test_y = x_train[test], y_train[test]
    
    fold_histories = []

    
    for j, model_creator in enumerate(models):
        model = model_creator()
        logging.info(f'Model {j+1}')
        history = model.fit(train_x, train_y, epochs=Parameters.epochs, batch_size=Parameters.batch_size, validation_data=(test_x, test_y), verbose=Parameters.verbosity)
        test_loss, acc = model.evaluate(test_x, test_y, verbose=Parameters.verbosity)
        logging.info(f'Validation accuracy: {acc}')
        
        fold_histories.append(history.history)
        
        for epoch in range(Parameters.epochs):
            # Log accuracy after each epoch
            acc_epoch = history.history['val_accuracy'][epoch]
            logging.info(f'Epoch {epoch + 1}, Validation accuracy: {acc_epoch}')
        fold_acc_scores.append((i, j, acc))
        
        if best_model_history is None or acc > best_model_acc:
            best_model_history = history
            best_model = model  # Store the trained model instance
            best_model_acc = acc
    
    model_histories.append(fold_histories)

# Find the best model
best_model_index = np.argmax([score[2] for score in fold_acc_scores])
best_fold_idx, best_model_idx, _ = max(fold_acc_scores, key=lambda x: x[2])
best_model_history = model_histories[best_fold_idx][best_model_idx]


#ogging.info(fold_acc_scores)
#logging.info(best_model_index)
#(best_fold, best_model_index, best_model_acc) = fold_acc_scores[best_model_index]
#best_model = models[best_model_index]
#logging.info(f'fold acc score: {fold_acc_scores}')
#logging.info(f'Best model is model {best_model_index+1}')

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(x_test, y_test)
logging.info(f'Test accuracy {test_acc}')


# In[188]:


model_histories


# In[174]:


import matplotlib.pyplot as plt
import numpy as np
# summarize history for accuracy
plt.plot(best_model_history.history['accuracy'])
plt.plot(best_model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(best_model_history.history['loss'])
plt.plot(best_model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# 

# In[156]:


# In a separate cell, add the following code to plot the training and validation accuracy and loss over epochs for the best model
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(best_model_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_model_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_model_history.history['loss'], label='Training Loss')
plt.plot(best_model_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


# In[157]:


print(best_model_history.history['accuracy'])


# In[ ]:


# Create a Baseline Model


# In[ ]:


# Compare Baseline vs best_model


# In[ ]:


# plot graph for learning curve and loss curve from baseline and best_model


# In[ ]:


# summarise best model
best_model.summary()


# In[ ]:


# print loss and accuracy for best model over epoch and steps plot


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


# plot a confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_test_labels = y_test.argmax(axis=1)
y_pred_labels = y_pred.argmax(axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.show()


# In[ ]:


# Export Model for using in tensorflow.js
get_ipython().system('mkdir -p saved_model')
best_model.save('saved_model/sensor_model')
best_model.save('saved_model/sensor_model.h5')


# In[ ]:


# Save the weight for the Js Model
best_model.save_weights('./checkpoints/my_checkpoint')


# In[ ]:


# upload model to server to download it on tensorflow js


# In[ ]:


dill.dump_session('notebook_env.db')

