from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=data.target, cmap='viridis')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("Scatter plot with two features")
plt.show()

model_two_features = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(3, activation='softmax')
])
model_two_features.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_two_features.fit(X_train[:, :2], y_train, epochs=10, verbose=0)
_, accuracy_two = model_two_features.evaluate(X_test[:, :2], y_test)
print(f"Accuracy with two features: {accuracy_two}")


model_full = Sequential([
    Dense(8, input_dim=4, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
model_full.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_full.fit(X_train, y_train, epochs=10, verbose=0)
_, accuracy_full = model_full.evaluate(X_test, y_test)
print(f"Accuracy with all features: {accuracy_full}")
