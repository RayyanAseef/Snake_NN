# ------------------------------------------------------------------------
# Importing required classes

from Model.Model import Model

from Model.Layers import Layer_Dense
from Model.Activations import ReLU, Softmax
from Model.Loss import CategoricalCrossentropy
from Model.Accuracy import Accuracy_Categorical  
from Model.Optimizers import Adam

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

import numpy as np
import matplotlib.pyplot as plt

print("Getting Data....")
X, y = spiral_data(samples=256*5, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# model = Model()

# model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
# model.add(ReLU())
# model.add(Layer_Dense(64, 3))
# model.add(Softmax())

# # Set loss, optimizer and accuracy objects
# model.set(
#     loss=CategoricalCrossentropy(),
#     optimizer=Adam(learning_rate=0.005, decay=5e-5),
#     accuracy=Accuracy_Categorical()
# )

# model.finalize()

# print("\nTraining.....")
# model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every_epoch=1000, print_per_epoch=5, batch_size=256)
# model.save_model('Model.model')

print('Retreving Model....\n')
model = Model.load('Model.model')
model.evaluate(X_test, y_test)

# Predict on a mesh grid
x = np.linspace(-1, 1, 100)
Y = np.linspace(-1, 1, 100)
X_mesh, Y_mesh = np.meshgrid(x, Y)
points = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

output = model.forward(points)
predictions = np.argmax(output, axis=1)

# Create a colormap for the classes
cmap = {0: 'blue', 1: 'red', 2: 'green'}

# First plot - Decision Boundarys
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
colors1 = [cmap[p] for p in predictions]
plt.scatter(points[:, 0], points[:, 1], c=colors1, s=5)
plt.title('Decision Boundary')

# Second plot - Original Data
plt.subplot(1, 2, 1)
colors2 = [cmap[label] for label in y_test]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors2, s=20)
plt.title('Original Spiral Data')

plt.show()