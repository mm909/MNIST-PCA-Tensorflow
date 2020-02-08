import math
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_components = 392

pcaFormatedTrainData = x_train.reshape(60000,784)
pcaFormatedTestData  = x_test.reshape(10000,784)
pca = PCA(n_components)
pca.fit(pcaFormatedTrainData)
ReducedTrainData = pca.transform(pcaFormatedTrainData)
ReducedTestData = pca.transform(pcaFormatedTestData)

explained = np.cumsum(pca.explained_variance_ratio_)
formatedExplained = math.floor(explained[len(explained)-1]*10000)/100
print("With", n_components, "Componets", str(formatedExplained) + "% of the variance is explained.")

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(392,)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ReducedTrainData, y_train, epochs=10)

model.evaluate(ReducedTestData,  y_test, verbose=2)
