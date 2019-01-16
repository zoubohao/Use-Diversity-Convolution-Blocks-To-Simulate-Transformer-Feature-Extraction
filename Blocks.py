import BasicFunction as bf
import tensorflow as tf
from tensorflow import keras


##Use the group  normalization to reduce the influence of batch size

##Liner
def TransChannelsConvolutionBlock(inputTensor ,kernalSize , outChannels , blockName) :
    shapeOfInput = inputTensor.get_shape().as_list()
    C = shapeOfInput[1]
    with tf.variable_scope(blockName):
        transFilter = bf.WeightCreation(shape=[kernalSize,kernalSize,C,outChannels],name="TransFilter")
        trans = bf.Convolution(inputTensor,transFilter,stride=[1,1,1,1],name="TransConvolution")
    return trans

def SmallUniteConvolutionBlock(inputTensor ,kernalSize , outChannels , gNum , blockName):
    assert outChannels % gNum == 0. and outChannels // gNum >= 1.
    shapeOfInput = inputTensor.get_shape().as_list()
    C = shapeOfInput[1]
    H = shapeOfInput[2]
    W = shapeOfInput[3]
    with tf.variable_scope(blockName):
        conFilter = bf.WeightCreation(shape=[kernalSize,kernalSize,C,outChannels],name= "ConvFilter")
        conv = bf.Convolution(inputTensor,conFilter,stride=[1,1,1,1],name= "ConvolutionOp")
        conv = tf.reshape(conv, shape=[-1, gNum, int(outChannels / gNum), H, W], name="Reshape_1")
        conv = bf.BN(conv,axis=1,name=blockName+"BNLayer")
        conv = tf.reshape(conv, shape=[-1, outChannels, H, W], name="Reshape_2")
        finalTensor = keras.layers.PReLU(trainable=True,name=blockName + "PReluBlock")(conv)
    return  finalTensor


def ResNetBlock(inputTensor,outputChannels,gNum,blockName):
    ## group Normalization condition
    assert outputChannels % gNum == 0. and outputChannels // gNum >= 1
    ## attention condition
    assert outputChannels // 16. >= 1
    shapeOfInput = inputTensor.get_shape().as_list()
    H = shapeOfInput[2]
    W = shapeOfInput[3]
    with tf.variable_scope(blockName):
        tensorCopy = tf.identity(inputTensor)
        conv1x1_1 = SmallUniteConvolutionBlock(inputTensor,kernalSize=1,
                                             outChannels=128,gNum=gNum,blockName="CONV1X1_Res_1")
        conv3x3 = SmallUniteConvolutionBlock(conv1x1_1,kernalSize=3,
                                             outChannels=64,gNum=gNum,blockName="CONV3X3_Res")
        conv1x1_2 = SmallUniteConvolutionBlock(conv3x3,kernalSize=1,
                                               outChannels=outputChannels,gNum=gNum,blockName="CONV1X1_Res_2")
        conv1x1_Copy = TransChannelsConvolutionBlock(tensorCopy,kernalSize=1,
                                                     outChannels=outputChannels,blockName="TransInputTensorCopy")
        with tf.variable_scope("AttentionBlock_Res"):
            convAttention = keras.layers.GlobalAvgPool2D(data_format="channels_first",name="GlobalAvg")(conv1x1_2)
            convAttention = bf.FullConnection(convAttention,units=outputChannels // 16 , blockName="Dense_1")
            convAttention = bf.FullConnection(convAttention,units=outputChannels,blockName="Dense_2")
            convAttention = tf.reshape(convAttention, shape=[-1, outputChannels, 1, 1],name="TransDim")
            convAttention = keras.layers.UpSampling2D(size=(H,W),data_format="channels_first",name="UpSam")(convAttention)
            convAttention = tf.nn.sigmoid(convAttention, name="ScaleTheDim")
        finalTensor = tf.add(conv1x1_2,convAttention,name="AddAttention")
        finalTensor = tf.add(conv1x1_Copy,finalTensor,name="finalAdd")
    return finalTensor


def DenseNetBlock(inputTensor,denseUniteNum,outputChannels,gNum,blockName):
    ## group Normalization condition
    assert outputChannels % gNum == 0. and outputChannels // gNum >= 1.
    ## attention condition
    assert outputChannels // 16. >= 1.
    shapeOfInput = inputTensor.get_shape().as_list()
    H = shapeOfInput[2]
    W = shapeOfInput[3]
    uniteBlockList = []
    with tf.variable_scope(blockName):
        copyInput = tf.identity(inputTensor)
        Conv1x1Copy = TransChannelsConvolutionBlock(copyInput,kernalSize=1,
                                                    outChannels=outputChannels,blockName="TransForAdd")
        ## Single convolution will not change the traits of input tensor,it is an liner trans.
        transFilter = bf.WeightCreation(shape=[1,1,shapeOfInput[1],64],name="TransFilter")
        transConv = bf.Convolution(inputTensor,filWeight=transFilter,stride=[1,1,1,1],
                                   name="TransConv1x1Op")
        uniteBlockList.append(transConv)
        for i in range(denseUniteNum):
            currentInput = tf.add_n(uniteBlockList, name="Add_" + str(i))
            with tf.variable_scope("DenseUniteBlock" + str(i)):
                currentConv1x1 = SmallUniteConvolutionBlock(currentInput, kernalSize=1,
                                                            outChannels=128, gNum=gNum,
                                                            blockName="Conv1x1_Den" + str(i))
                currentConv3x3 = SmallUniteConvolutionBlock(currentConv1x1, kernalSize=3,
                                                            outChannels=64, gNum=gNum,
                                                            blockName="Conv3x3_Den" + str(i))
                uniteBlockList.append(currentConv3x3)
        mediumTensor = tf.add_n(uniteBlockList,name="mediumAddAll")
        mediumTensor = SmallUniteConvolutionBlock(mediumTensor,kernalSize=1,
                                                 outChannels=outputChannels,gNum=gNum,blockName="mediumTransUnite")
        with tf.variable_scope("AttentionBlock_Den"):
            convAttention = keras.layers.GlobalAvgPool2D(data_format="channels_first",name="GlobalAvg")(mediumTensor)
            convAttention = bf.FullConnection(convAttention,units=outputChannels // 16 , blockName="Dense_1")
            convAttention = bf.FullConnection(convAttention,units=outputChannels,blockName="Dense_2")
            convAttention = tf.reshape(convAttention,shape=[-1,outputChannels,1,1],name="TransDim")
            convAttention = keras.layers.UpSampling2D(size=(H,W),data_format="channels_first",name="UpSam")(convAttention)
            convAttention = tf.nn.sigmoid(convAttention,name="ScaleTheDim")
        finalTensor = tf.add(convAttention,mediumTensor,name="AddAttention")
        finalTensor = tf.add(Conv1x1Copy,finalTensor,name="finalAdd")
    return finalTensor


def ResNextBlock(inputTensor,parallelUnitNum,outputChannels,gNum,blockName):
    ## group Normalization condition
    assert outputChannels % gNum == 0. and outputChannels // gNum >= 1.
    ## attention condition
    assert outputChannels // 16. >= 1.
    shapeOfInput = inputTensor.get_shape().as_list()
    H = shapeOfInput[2]
    W = shapeOfInput[3]
    parallelTensorList = []
    with tf.variable_scope(blockName):
        copyInput = tf.identity(inputTensor)
        for i in range(parallelUnitNum):
            with tf.variable_scope("ParallelBlock_" + str(i)):
                Conv1x1_1 = SmallUniteConvolutionBlock(inputTensor, kernalSize=1,
                                                       outChannels=4, gNum=gNum, blockName="CONV1X1_Rex_1")
                Conv3x3_4 = SmallUniteConvolutionBlock(Conv1x1_1,kernalSize=3,
                                                       outChannels=4,gNum=gNum,blockName="CONV3X3_Rex_4")
                Conv1x1_2 = SmallUniteConvolutionBlock(Conv3x3_4,kernalSize=1,
                                                       outChannels=outputChannels,gNum=gNum,blockName="CONV1X1_Rex_2")
                parallelTensorList.append(Conv1x1_2)
        mediumTensor = tf.add_n(parallelTensorList,name="mediumAdd")
        Conv1x1_Copy = TransChannelsConvolutionBlock(copyInput,kernalSize=1,
                                                     outChannels=outputChannels,blockName="TransCopyInputForAdd")
        with tf.variable_scope("AttentionBlock_Rex"):
            convAttention = keras.layers.GlobalAvgPool2D(data_format="channels_first",name="GlobalAvg")(mediumTensor)
            convAttention = bf.FullConnection(convAttention,units=outputChannels // 16 , blockName="Dense_1")
            convAttention = bf.FullConnection(convAttention,units=outputChannels,blockName="Dense_2")
            convAttention = tf.reshape(convAttention,shape=[-1,outputChannels,1,1],name="TransDim")
            convAttention = keras.layers.UpSampling2D(size=(H,W),data_format="channels_first",name="UpSam")(convAttention)
            convAttention = tf.nn.sigmoid(convAttention,name="ScaleTheDim")
        finalTensor = tf.add(convAttention,mediumTensor,name="AddAttention")
        finalTensor = tf.add(finalTensor,Conv1x1_Copy,name="finalAdd")
    return finalTensor

def ReRxDeThreeAddLayer(inputTensor , outputChannels , gNum, blockName ):
    ## group Normalization condition
    assert outputChannels % gNum == 0. and outputChannels // gNum >= 1.
    ## attention condition
    assert outputChannels // 16. >= 1.
    ## For guarantee the same number of parameters and without much parameters .
    ## the ResNet block needs 2 blocks, DenseNet Block needs 2 units and ResNext needs 22 Units
    ## it is 11 times .
    with tf.variable_scope(blockName):
        with tf.variable_scope("ResNet_2_Block_combine"):
            Res_1 = ResNetBlock(inputTensor,outputChannels=outputChannels,gNum=4,blockName="ResNet_1_Block")
            Res_2 = ResNetBlock(Res_1,outputChannels=outputChannels,gNum=4,blockName="ResNet_2_Block")
        with tf.variable_scope("ResNext_Units_22_Block"):
            Rex = ResNextBlock(inputTensor,parallelUnitNum=20,
                               outputChannels=outputChannels,gNum=4,blockName="ResNext_1_Block")
        with tf.variable_scope("DenseNet_2_Units_Blocks"):
            Den = DenseNetBlock(inputTensor,outputChannels=outputChannels,denseUniteNum=2,gNum=4,blockName="DenseNet_1_Block")
        finalTensor = tf.add_n([Res_2,Rex,Den],name="FinalAddN")
    return finalTensor


def TransformerEncoderBlock(inputTensor,outputChannels,blockName):
    with tf.variable_scope(blockName):
        with tf.variable_scope("SimulateMultiHeadBlock"):
            parallelLayer_1 = ReRxDeThreeAddLayer(inputTensor, outputChannels=64, gNum=4, blockName="parallelLayer_1")
            parallelLayer_2 = ReRxDeThreeAddLayer(inputTensor, outputChannels=64, gNum=4, blockName="parallelLayer_2")
            parallelLayer_3 = ReRxDeThreeAddLayer(inputTensor, outputChannels=64, gNum=4, blockName="parallelLayer_3")
            concatParallelLayers_2 = tf.concat([parallelLayer_1, parallelLayer_2, parallelLayer_3],
                                               axis=1, name="ConcatParallelLayers")
            liner_3 = SmallUniteConvolutionBlock(concatParallelLayers_2, kernalSize=1,
                                                 outChannels=64, gNum=4, blockName="SimulateLiner")
            input_Copy = SmallUniteConvolutionBlock(inputTensor, kernalSize=1,
                                                    outChannels=64, gNum=4, blockName="InputChannelsTrans")
        with tf.variable_scope("Add_And_Nor_1"):
            mediumTensor = tf.add(liner_3, input_Copy, name="mediumADD")
            mediumTensor = bf.BN(mediumTensor, axis=1, name="mediumBN")
        with tf.variable_scope("SimulateFeedForward"):
            transMedium2Out = TransChannelsConvolutionBlock(mediumTensor, kernalSize=1,
                                                            outChannels=outputChannels,blockName="transMedium2OutChannels")
            conv1x1_1 = SmallUniteConvolutionBlock(mediumTensor,kernalSize=1,
                                                   outChannels=128,gNum=4,blockName="Conv1x1_1_Liner")
            conv1x1_2 = SmallUniteConvolutionBlock(conv1x1_1,kernalSize=1,
                                                   outChannels=64,gNum=4,blockName="Conv1x1_2_liner")
            conv1x1_3 = TransChannelsConvolutionBlock(conv1x1_2,kernalSize=1,
                                                      outChannels=outputChannels,blockName="Conv1x1_3_Trans")
        with tf.variable_scope("Add_And_Nor_2"):
            finalTensor = tf.add(transMedium2Out,conv1x1_3,name="FinalADD")
            finalTensor = bf.BN(finalTensor,axis=1,name="FinalBN")
        return finalTensor




if __name__ == "__main__":
    a = ["1","2","3","4","5"]
    import numpy as np

    print(np.mean(a))
    print(np.var(a))
    print((a - np.mean(a)) / np.var(a) + 0.0)




