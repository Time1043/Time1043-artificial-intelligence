import tensorflow as tf
import tensorflow.keras as keras

# tf version
print(tf.__version__)

# check if GPU is available
# print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))

# check if CUDA is available
print(tf.test.is_built_with_cuda())
