# Session 13 – Multi Layer Neural Network

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Convert labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create MLNN
model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(784,)
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        10,
        activation="softmax"
    )

])

# Compile
model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

# Train
history = model.fit(

    X_train,

    y_train,

    epochs=10,

    validation_data=(
        X_test,
        y_test
    )

)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nAccuracy:", accuracy)

# ---------------- Graph ------------------

plt.plot(history.history["accuracy"])

plt.title("MLNN Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()