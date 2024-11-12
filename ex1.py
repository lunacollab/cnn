import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])  # AND
y_or = np.array([[0], [1], [1], [1]])   # OR

def build_perceptron():
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='sigmoid'))
    model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# AND Operation
model_and = build_perceptron()
model_and.fit(X, y_and, epochs=10, verbose=0)
print("AND operation results:")
print(model_and.predict(X))

# OR Operation
model_or = build_perceptron()
model_or.fit(X, y_or, epochs=10, verbose=0)
print("OR operation results:")
print(model_or.predict(X))
