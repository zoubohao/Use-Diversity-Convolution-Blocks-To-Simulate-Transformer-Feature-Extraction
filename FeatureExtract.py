import BasicFunction as bf
import Blocks as bs
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
import math



def Encoder(inputTensor,outputChannels,blockName):
    with tf.variable_scope(blockName):
        conv1x1_1 = bs.SmallUniteConvolutionBlock(inputTensor,kernalSize=1,
                                                  outChannels=64,gNum=4,blockName="Conv1x1_Encoder_First_Step")
        Transformer_E_2 = bs.TransformerEncoderBlock(conv1x1_1,outputChannels=64,
                                                     blockName="Transformer_Encoder_Second_Step")
        conv1x1_3 =  bs.TransChannelsConvolutionBlock(Transformer_E_2,kernalSize=1,
                                                   outChannels=outputChannels,blockName="Conv1x1_Encoder_Third_Step")
        pool_Down = bf.Pooling(conv1x1_3,windowShape=[2,2],poolingType="MAX",stride=[2,2],name="Pooling_Encoder_Fourth_Step")
    return conv1x1_1,Transformer_E_2,conv1x1_3 , pool_Down

def Decoder(inputTensor , outputSize , blockName):
    with tf.variable_scope(blockName):
        conv1x1_1 = bs.SmallUniteConvolutionBlock(inputTensor,kernalSize=1,
                                                  outChannels=128,gNum=4,blockName="Conv1x1_Decoder_First_Step")
        conv1x1_2 = bs.SmallUniteConvolutionBlock(conv1x1_1,kernalSize=1,
                                                  outChannels=64,gNum=4,blockName="Conv1x1_Decoder_Second_Step")
        globalPooling = keras.layers.GlobalAvgPool2D(data_format="channels_first",name="AVG_Global_Decoder")(conv1x1_2)
        weight1 = bf.WeightCreation(shape=[64,128],name="TransWeight1")
        bias1 = tf.constant(value=0.,dtype=tf.float32,shape=[128],name="Bias1")
        layer1 = keras.layers.PReLU(trainable=True)(tf.add(tf.matmul(globalPooling,weight1),bias1))
        weight2 = bf.WeightCreation(shape=[128,outputSize],name="TransWeight2")
        bias2 = tf.constant(value=0.,dtype=tf.float32,shape=[outputSize],name="Bias2")
        outputTensor = tf.add(tf.matmul(layer1,weight2),bias2)
    return  outputTensor

def generateData(MapG):
    while True :
        for k in MapG:
            yield MapG[k]

def Min_Max_Nor(nums):
    minNum = np.min(nums)
    maxNum = np.max(nums)
    distence = maxNum - minNum + 0.0
    return (nums - minNum) / distence + 0.0001

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    ### Config parameters
    batchSize = 6
    resultSize = 2
    featureChannels = 1
    epoch = 10
    timesInOneEpoch = 3500
    lr = 0.001
    displayTimes = 30
    decayStep = 1700
    decayRate = 0.96
    dataTrainFilePath = "d:\\CandidateThresholdTrain.txt"
    dataTestFilePath = "d:\\CandidateThresholdTest.txt"
    modelTrainOrTest = "test"
    saveModelSteps = 3498
    saveModelPath = "d:\\featureExtractSavePath\\"
    saveResultPath = "d:\\featureResult.txt"
    ### Net construction
    inputDataPlaceHolder = tf.placeholder(dtype=tf.float32,shape=[None,1,4,4])
    labelDataPlaceHolder = tf.placeholder(dtype=tf.float32,shape=[None,resultSize])
    lrPlaceHolder = tf.placeholder(dtype=tf.float32)
    conv1,trans , conv3 , featureTensor = Encoder(inputDataPlaceHolder,outputChannels=featureChannels,blockName="EncoderNet")
    predictTensor = Decoder(featureTensor,outputSize=resultSize,blockName="DecoderNet")
    different = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labelDataPlaceHolder,logits=predictTensor,dim=1,name="CrossDis")
    labelLoss = tf.reduce_mean(different,name="ReduceMean")
    tf.add_to_collection("Loss",labelLoss)
    tLoss = tf.add_n(tf.get_collection("Loss"))
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optim = tf.train.MomentumOptimizer(learning_rate=lrPlaceHolder,momentum=0.9,use_nesterov=True).minimize(tLoss)
    tf.summary.FileWriter(logdir=saveModelPath, graph=tf.get_default_graph())
    print("Complete Net Build .")
    ### Training data
    with tf.Session() as sess :
        if modelTrainOrTest.lower() == "train":
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print("Initial has completed .")
            trainingNum = 0
            ### Read and use Min-Max score to Normalization data
            dataPInputMap = {}
            dataPLabelMap = {}
            dataNInputMap = {}
            dataNLabelMap = {}
            dataID2Arg = {}
            with open(file=dataTrainFilePath, mode="r") as f:
                for i, line in enumerate(f):
                    if i != 0:
                        inputData = re.split(pattern="\t", string=line)
                        label = int(inputData[4])
                        idStr = inputData[:4]
                        dataID2Arg[i] = idStr
                        if label == 1:
                            top2 = np.array(inputData[5:7], dtype=np.float32) * 1000.
                            final = np.array(inputData[7:], dtype=np.float32)
                            m = np.array(list(top2) + list(final), dtype=np.float32)
                            m = Min_Max_Nor(m)
                            m = [0.] + list(m) + [0.]
                            m = np.array(m, dtype=np.float32)
                            m = np.reshape(m, newshape=[1, 4, 4])
                            dataPInputMap[i] = m
                            dataPLabelMap[i] = np.array([0, 1], dtype=np.float32)
                        if label == 0:
                            top2 = np.array(inputData[5:7], dtype=np.float32) * 1000.
                            final = np.array(inputData[7:], dtype=np.float32)
                            m = np.array(list(top2) + list(final), dtype=np.float32)
                            m = Min_Max_Nor(m)
                            m = [0.] + list(m) + [0.]
                            m = np.array(m, dtype=np.float32)
                            m = np.reshape(m, newshape=[1, 4, 4])
                            dataNInputMap[i] = m
                            dataNLabelMap[i] = np.array([1, 0], dtype=np.float32)
            print("Read data has completed .")
            dataPYield = generateData(dataPInputMap)
            labelPYield = generateData(dataPLabelMap)
            dataNYield = generateData(dataNInputMap)
            labelNYield = generateData(dataNLabelMap)
            for _ in range(epoch):
                for _ in range(timesInOneEpoch):
                    dataInputs = []
                    dataLabels = []
                    for b in range(batchSize // 2):
                        dataPInput = dataPYield.__next__()
                        dataPLabel = labelPYield.__next__()
                        dataNInput = dataNYield.__next__()
                        dataNLabel = labelNYield.__next__()
                        dataInputs.append(dataPInput)
                        dataInputs.append(dataNInput)
                        dataLabels.append(dataPLabel)
                        dataLabels.append(dataNLabel)
                    dataInputs = np.array(dataInputs)
                    dataLabels = np.array(dataLabels)
                    # print("Inputs ",dataInputs)
                    # print("Labels ",dataLabels)
                    if trainingNum % displayTimes == 0:
                        tLossNum = sess.run(tLoss, feed_dict={
                            inputDataPlaceHolder: dataInputs,
                            labelDataPlaceHolder: dataLabels,
                            lrPlaceHolder: lr
                        })
                        predictTensorNum = sess.run(predictTensor, feed_dict={
                            inputDataPlaceHolder: dataInputs
                        })
                        featureTensorNum = sess.run(featureTensor, feed_dict={
                            inputDataPlaceHolder: dataInputs
                        })
                        conv3Num = sess.run(conv3, feed_dict={
                            inputDataPlaceHolder: dataInputs
                        })
                        transNum = sess.run(trans, feed_dict={
                            inputDataPlaceHolder: dataInputs
                        })
                        labelLossNum = sess.run(labelLoss, feed_dict={
                            inputDataPlaceHolder: dataInputs,
                            labelDataPlaceHolder: dataLabels
                        })
                        print("trans is ", transNum)
                        print("conv3 is ", conv3Num)
                        print("Feature tensor is ", featureTensorNum)
                        print("It has trained " + str(trainingNum) + " times .")
                        print("Learning rate is ", lr)
                        print("Total losses are ", tLossNum)
                        print("Label loss is ", labelLossNum)
                        print("Predict labels are ", predictTensorNum)
                        print("True Labels are ", dataLabels)
                    sess.run(optim, feed_dict={
                        inputDataPlaceHolder: dataInputs,
                        labelDataPlaceHolder: dataLabels,
                        lrPlaceHolder: lr
                    })
                    trainingNum = trainingNum + 1
                    if trainingNum % decayStep == 0 and trainingNum != 0:
                        lr = lr * math.pow(decayRate, trainingNum / decayStep + 0.0)
                    if trainingNum % saveModelSteps == 0 and trainingNum != 0:
                        tf.train.Saver().save(sess=sess, save_path=saveModelPath + "model.ckpt")
                        tf.summary.FileWriter(logdir=saveModelPath, graph=tf.get_default_graph(),
                                              session=sess)
        else:
            tf.train.Saver().restore(sess=sess,
                                     save_path=saveModelPath + "model.ckpt")
            print("Initial has completed .")
            ### Read and use Min-Max score to Normalization data
            dataPInputMap = {}
            dataPLabelMap = {}
            dataNInputMap = {}
            dataNLabelMap = {}
            dataID2Arg = {}
            with open(file=dataTestFilePath, mode="r") as f:
                for i, line in enumerate(f):
                    if i != 0:
                        inputData = re.split(pattern="\t", string=line)
                        label = int(inputData[4])
                        idStr = inputData[:4]
                        dataID2Arg[i] = idStr
                        if label == 1:
                            top2 = np.array(inputData[5:7], dtype=np.float32) * 1000.
                            final = np.array(inputData[7:], dtype=np.float32)
                            m = np.array(list(top2) + list(final), dtype=np.float32)
                            m = Min_Max_Nor(m)
                            m = [0.] + list(m) + [0.]
                            m = np.array(m, dtype=np.float32)
                            m = np.reshape(m, newshape=[1, 4, 4])
                            dataPInputMap[i] = m
                            dataPLabelMap[i] = np.array([0, 1], dtype=np.float32)
                        if label == 0:
                            top2 = np.array(inputData[5:7], dtype=np.float32) * 1000.
                            final = np.array(inputData[7:], dtype=np.float32)
                            m = np.array(list(top2) + list(final), dtype=np.float32)
                            m = Min_Max_Nor(m)
                            m = [0.] + list(m) + [0.]
                            m = np.array(m, dtype=np.float32)
                            m = np.reshape(m, newshape=[1, 4, 4])
                            dataNInputMap[i] = m
                            dataNLabelMap[i] = np.array([1, 0], dtype=np.float32)
            print("Read data has completed .")
            totSamples = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            with open(saveResultPath,mode="w") as w :
                for kv in dataPInputMap:
                    testInput = np.reshape(dataPInputMap[kv], newshape=[1, 1, 4, 4])
                    testLabel = dataPLabelMap[kv]
                    featureTestNum = sess.run(featureTensor, feed_dict={
                        inputDataPlaceHolder: testInput
                    })
                    predictTestNum = sess.run(predictTensor, feed_dict={
                        inputDataPlaceHolder: testInput
                    })
                    totSamples = totSamples + 1
                    IDString = dataID2Arg[kv]
                    featureFlatten = np.reshape(np.array(featureTestNum,dtype=np.float32), newshape=[4])
                    if np.argmax(predictTestNum) == np.argmax(testLabel):
                        w.write(str(IDString) + " TRUE " + " 1 " + str(featureFlatten) + "\n")
                        TP = TP + 1
                        print(kv)
                    else:
                        w.write(str(IDString) + " FALSE " + " 1 " + str(featureFlatten) + "\n")
                        FN = FN + 1
                        print(kv)

                for kv in dataNInputMap:
                    testInput = np.reshape(dataNInputMap[kv], newshape=[1, 1, 4, 4])
                    testLabel = dataNLabelMap[kv]
                    featureTestNum = sess.run(featureTensor, feed_dict={
                        inputDataPlaceHolder: testInput
                    })
                    predictTestNum = sess.run(predictTensor, feed_dict={
                        inputDataPlaceHolder: testInput
                    })
                    totSamples = totSamples + 1
                    IDString = dataID2Arg[kv]
                    featureFlatten = np.reshape(np.array(featureTestNum,dtype=np.float32), newshape=[4])
                    if np.argmax(predictTestNum) == np.argmax(testLabel):
                        w.write(str(IDString) + " TRUE " + " 0 " + str(featureFlatten) + "\n")
                        print(kv)
                        TN = TN + 1
                    else:
                        w.write(str(IDString) + " FALSE " + " 0 " + str(featureFlatten) + "\n")
                        print(kv)
                        FP = FP + 1
            print("Acc ratio is ",(TP + TN) / totSamples + 0.)
            print("Sensitive ratio is ", TP / (TP + FN) + 0.)
            print("Specificity ratio is ",TN / (TN + FP) + 0.)





