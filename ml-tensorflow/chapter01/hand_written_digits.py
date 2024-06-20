import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers
from tensorflow import keras

# hyperparameters
input_width = 28  # for data
input_height = 28  # for data
batch_size = 128  # for data

classification = 10  # for model
num_epochs = 30  # for train
learning_rate = 0.001  # for train

# data (train 60k)
(x, y), _ = datasets.mnist.load_data()
# to tensor
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=classification)  # one-hot encoding
print(x.shape, y.shape)   # (60000, 28, 28), (60000, 10)
# train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

model = keras.Sequential([
    # layers.Flatten(input_shape=(input_height, input_width)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(classification, activation='softmax')
])

optimizer = optimizers.SGD(learning_rate=learning_rate)  # w = w - lr * grad


def train_epoch(num_epochs):
    # loop
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # forward propagation
            x = tf.reshape(x, (-1, input_height * input_width))  # (batch_size, 28, 28) -> (batch_size, 784)  # flatten
            out = model(x)  # (batch_size, 784) -> (batch_size, classification)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]  # mse loss

        # backward propagation
        grads = tape.gradient(loss, model.trainable_variables)  # compute gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # update model parameters

        if step % 100 == 0:
            print(f"epoch: {num_epochs}, step: {step}, loss: {loss.numpy()}")


for epoch in range(num_epochs):
    train_epoch(epoch)
