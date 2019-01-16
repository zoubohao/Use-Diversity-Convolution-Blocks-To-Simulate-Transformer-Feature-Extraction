import BasicFunction as bf
import  Blocks as bs
import tensorflow as tf
from tensorflow import keras


## This net name is Diversity Convolution Units Blocks To Simulate Transformer With FPN NetWork !!! DCBTF Net !!!

class DCBTF_NetWork :

    def __init__(self,batchSize , inChannels , H , W , outSize):
        self.inPlaceHolder = tf.placeholder(shape=[batchSize,inChannels,H,W],dtype=tf.float32)
        self.outPlaceHolder = tf.placeholder(shape=[batchSize,outSize],dtype=tf.float32)













