import tensorflow as tf

# device
a = tf.constant(1.23, dtype=tf.double)  # default float32; double = float64
print(a)  # tf.Tensor(1.23, shape=(), dtype=float64)
print(a.device)  # /job:localhost/replica:0/task:0/device:GPU:0 (default)

# to numpy
a_np = a.numpy()
print(a_np)  # 1.23


