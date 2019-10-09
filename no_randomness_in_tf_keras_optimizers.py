"""
Code taken from https://github.com/keras-team/keras/issues/12247 and changed so that the issue is
more visible on my pc. The problem does not happen here, probably because we use tf.keras instead of plain keras!
tensorflow 1.14.0, keras 2.2.4, installed with anaconda
"""
import random
import os
import numpy as np
import tensorflow as tf
os.environ['PYTHONHASHSEED']=str(42)
random.seed(42)
tf.set_random_seed(42)
np.random.seed(42)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.datasets import boston_housing
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam as Adam_k, SGD as SGD_k
from tensorflow.train import AdamOptimizer, GradientDescentOptimizer

def get_weights_and_preds():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(seed=42, test_split=0.2)

    inputs = Input(batch_shape=(None, x_train.shape[1]))

    outputs = Dense(
        units=32, activation='relu',
        kernel_initializer=initializers.get({'class_name': 'he_uniform', 'config': {'seed': 42}}),
        use_bias=False,
        bias_initializer='zeros')(inputs)
    for k in range(10):
        outputs = Dense(
            units=1, activation='linear',
            kernel_initializer=initializers.get({'class_name': 'he_uniform', 'config': {'seed': 42}}),
            use_bias=False,
            bias_initializer='zeros')(outputs)

    model = Model(inputs, outputs)
    opt = Adam_k(lr=0.001) # this will not fail the test (on my current pc)
    # opt = AdamOptimizer(learning_rate=0.001) # this will not fail the test (on my current pc)
    model.compile(optimizer=opt, loss='mean_absolute_error')
    i=model.get_weights()

    model.train_on_batch(x_train, y_train)

    return i, model.get_weights(), model.predict(x_test)


# tf.Session()

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)
K.set_session(sess)

i1, w1, p1 = get_weights_and_preds()
i2, w2, p2 = get_weights_and_preds()

for i in range(len(i1)):
    np.testing.assert_array_equal(i1[i], i2[i])

for i in range(len(w1)):
    np.testing.assert_array_equal(w1[i], w2[i])

np.testing.assert_array_equal(p1, p2)