import tensorflow as tf
import timeit
import tensorflow.keras as keras

with tf.device('/cpu:0'):
    a_cpu = tf.random.normal([10000, 10000])
    b_cpu = tf.random.normal([10000, 20000])
    print(a_cpu.device, b_cpu.device)

with tf.device('/gpu:0'):
    a_gpu = tf.random.normal([10000, 10000])
    b_gpu = tf.random.normal([10000, 20000])
    print(a_gpu.device, b_gpu.device)


def cpu_run():
    with tf.device('/cpu:0'):
        c_cpu = tf.matmul(a_cpu, b_cpu)
    return c_cpu


def gpu_run():
    with tf.device('/gpu:0'):
        c_gpu = tf.matmul(a_gpu, b_gpu)
    return c_gpu


# warm up
time_cpu = timeit.timeit(cpu_run, number=10)
time_gpu = timeit.timeit(gpu_run, number=10)
print('warm up: ', time_cpu, time_gpu)  # 76.9196078 3.8772426999999965

# real run
time_cpu = timeit.timeit(cpu_run, number=10)
time_gpu = timeit.timeit(gpu_run, number=10)
print('real run:', time_cpu, time_gpu)  # 69.85335289999999 0.000644300000004705
