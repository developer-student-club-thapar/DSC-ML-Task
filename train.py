from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf
from data import train_data, test_data

model_1 = Sequential([
  Conv2D(10, 5, activation='relu', input_shape=(300, 300, 3)),
  Conv2D(10, 5, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 5, activation='relu'),
  Conv2D(10, 5, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 5, activation='relu'),
  Conv2D(10, 5, activation='relu'),
#   MaxPool2D(),
#   Conv2D(10, 5, activation='relu'),
#   Conv2D(10, 5, activation='relu'),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(23, activation='softmax') 
])

# Compile the model
model_1.compile(loss="categorical_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Fit the model
history_1 = model_1.fit(train_data, 
                        epochs=15,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

model_1.evaluate(test_data)

model_1.save('monument_model')

