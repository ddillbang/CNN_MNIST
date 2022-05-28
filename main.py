import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
import numpy as np

(x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()

x_train, t_train = x_train[:10000], t_train[:10000]
x_test, t_test = x_test[:2000], t_test[:2000]

# x_train = x_train.reshape((10000, 28, 28))
# x_test = x_test.reshape((2000, 28, 28))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', strides = (1,1), input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu', strides = (1,1)))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(x_train, t_train, epochs = 5, batch_size = 128, validation_split = 0.8)

plt.figure(figsize = (12,4))
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], 'b--', label = 'loss')
plt.plot(history.history['accuracy'], 'g-', label = 'Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
print('done')

labels = model.predict(x_test)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print("\n Accuracy: %.4f" % (model.evaluate(x_test, t_test, verbose=2)[1]))
fig = plt.figure()
for i in range(15):
  subplot = fig.add_subplot(3, 5, i+1)
  subplot.set_xticks([])
  subplot.set_yticks([])
  subplot.set_title("{}".format(classes[np.argmax(labels[i])]))
  subplot.imshow(x_test[i].reshape((28, 28)), cmap=plt.cm.gray_r)
plt.show()