from enum import Enum
import numpy as np
import tensorflow as tf
import os
from time import time
import random

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

class Directions(Enum):
        X = 0
        Y = 1
        Z = 2

dx = 1.0
dy = 1.0
dz = 1.0

c1 = -1./24
c2 = 1./24
c3 = 9./8
c4 = -9./8

c1_ = [c1/dx, c1/dy, c1/dz]
c2_ = [c2/dx, c2/dy, c2/dz]
c3_ = [c3/dx, c3/dy, c3/dz]
c4_ = [c4/dx, c4/dy, c4/dz]

d1_ = [1, 1, 1]
d2_ = [2, 2, 2]
d3_ = [0, 0, 0]
d4_ = [1, 1, 1]

def apply(d, fijk1, fijk2, fijk3, fijk4, DK) :
        c1 = tf.gather(c1_, d)
        c2 = tf.gather(c2_, d)
        c3 = tf.gather(c3_, d)
        c4 = tf.gather(c4_, d)
        return c1 * fijk1 + c2 * fijk2 + c3 * fijk3 + c4 * fijk4

def apply_comp(inputs) :
        tpu_computation = tpu.rewrite(apply, inputs)
        tpu_grpc_url = TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master()

        with tf.Session(tpu_grpc_url) as sess:
                sess.run(tpu.initialize_system())
                sess.run(tf.global_variables_initializer())
                t1 = time()
                sess.run(tpu_computation)
                t2 = time()
                sess.run(tpu.shutdown_system())
        print(t2 - t1)

def fdo(fijk, d, i, j, DK) :

        if d == Directions.X.value :
                d1 = d1_[d]
                d2 = d2_[d]
                d3 = d3_[d]
                d4 = d4_[d]

                fijk1 = fijk[i + d1, j]
                fijk2 = fijk[i - d2, j]
                fijk3 = fijk[i + d3, j]
                fijk4 = fijk[i - d4, j]

        elif d == Directions.Y.value :

                d1 = d1_[d]
                d2 = d2_[d]
                d3 = d3_[d]
                d4 = d4_[d]

                fijk1 = fijk[i, j + d1]
                fijk2 = fijk[i, j - d2]
                fijk3 = fijk[i, j + d3]
                fijk4 = fijk[i, j - d4]

        else :
                d1 = d1_[d] + DK
                d2 = d2_[d] - DK
                d3 = d3_[d] + DK
                d4 = d4_[d] - DK

        d_ = tf.constant(d)
        inputs = [d_, fijk1, fijk2, fijk3, fijk4, DK]
        apply_comp(inputs)

if __name__ == "__main__" :
        i = tf.constant(2)
        j = tf.constant(2)
        DK = tf.constant(0)
        d = 1

        size_x, size_y, size_z = (50, 50, 40000)
        data = np.array([float(random.randint(1,10)) for cpt in range(1, size_x * size_y * size_z + 1)], dtype=np.float32)
        data.shape = (size_x, size_y, size_z)

        fijk = tf.convert_to_tensor(data)
	fdo(fijk, d, i, j, DK)

