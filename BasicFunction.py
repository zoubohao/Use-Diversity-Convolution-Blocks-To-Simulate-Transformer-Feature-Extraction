import tensorflow as tf
import math
from tensorflow import keras
import numpy as np

def WeightCreation(shape , c = 1e-4,ifAddRe = True ,  name = None):
    weight = tf.Variable(initial_value=
                         tf.truncated_normal(shape=shape,dtype=tf.float32,stddev=math.pow(3.5 / np.sum(shape),0.5)),
                         name=name)
    if ifAddRe:
        l2Loss = tf.nn.l2_loss(weight)
        l2Loss = tf.multiply(c,l2Loss)
        tf.add_to_collection("Loss",l2Loss)
    return weight

def BN(inputTensor , axis = -1 , ep = 1e-6,name = None):
    return keras.layers.BatchNormalization(axis=axis,epsilon=ep,name=name)(inputTensor)

def Convolution(inputTensor , filWeight , stride ,dataFormat = "NCHW"  ,padding = "SAME",name = None):
    return tf.nn.conv2d(input=inputTensor,
                        filter=filWeight,
                        strides=stride,
                        data_format=dataFormat,
                        padding=padding,
                        name=name)

def Pooling (inputTensor , windowShape ,poolingType , stride, dataFormat = "NCHW" ,padding = "SAME",name = None):
    return tf.nn.pool(input=inputTensor,
                      window_shape=windowShape,
                      pooling_type=poolingType,
                      strides=stride,
                      data_format=dataFormat,
                      padding=padding,
                      name=name)

def Dropout(inputTensor , keepPro = 0.7,name = None):
    return keras.layers.Dropout(rate=keepPro,name=name)(inputTensor)

def FullConnection(inputTensor,units,blockName):
    shapeOfInput = inputTensor.get_shape().as_list()
    inUnits = shapeOfInput[1]
    with tf.variable_scope(blockName):
        weights = WeightCreation(shape=[inUnits, units], name="Weight")
        bias = tf.constant(value=0., shape=[units], dtype=tf.float32, name="bias")
        layer = tf.add(tf.matmul(inputTensor,weights),bias)
        layer = BN(layer,axis=1,name=blockName + "BNLayer")
        layer = keras.layers.PReLU(trainable=True,name=blockName + "PReluBlock")(layer)
        layer = Dropout(layer,name="Dropout")
    return layer








