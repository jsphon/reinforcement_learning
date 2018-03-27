import numpy as np
from scipy import stats

import tensorflow as tf

from rl.lib.timer import Timer
from rl.tf.lib.repeat_func import repeat


n = 1000

a = 2.0
b = 1.0

np.random.seed(0)
aX = np.random.uniform(0, 10, n)
ay = a * aX + b + np.random.standard_normal(n)

# Check results
slope, intercept, r_value, p_value, std_err = stats.linregress(aX,ay)

print('slope : %0.2f' % slope)
print('intercept : %0.2f' % intercept)

tX = tf.Variable(aX, trainable=False)
ty = tf.Variable(ay, trainable=False)

va = tf.Variable(0.0, dtype=tf.float64)
vb = tf.Variable(0.0, dtype=tf.float64)

def get_loss():
    yhat = va * tX + vb
    loss = tf.losses.mean_squared_error(ty, yhat)
    return loss

def get_train_op():
    loss = get_loss()
    return tf.train.GradientDescentOptimizer(0.01).minimize(loss)


def body():
    return get_train_op()

n = 5000

train_loop = repeat(body, n)
train_op = get_train_op()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

with Timer('train loop'):
    sess.run(train_loop)

opt_a = va.eval()
opt_b = vb.eval()

final_loss = get_loss().eval()

print('Optimised a : %0.2f' % opt_a)
print('Optimised b : %0.2f' % opt_b)
print('Final Loss : %0.2f' % final_loss)