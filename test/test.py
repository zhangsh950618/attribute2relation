#-*-coding-*-

import tensorflow as tf
import numpy as np

t = tf.constant([[1, 2, 3], [4, 5]])
tf.pad(t,mode='CONSTANT',name=None,constant_values=0)