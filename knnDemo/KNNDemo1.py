#coding=utf-8
from numpy import zeros, array, shape, tile
from knnDemo.kNN import classify0
import matplotlib
import matplotlib.pyplot as plt

# 数据准备
def fileToMatrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1

    return returnMat, classLabelVector
# 数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet -tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

# 测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = fileToMatrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with:%d, the real answer is : %d"%(classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount+=1.0
    print "the totla error rate is : %f"%(errorCount/float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = fileToMatrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classfierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "you will probably like this person:",resultList[classfierResult-1]
# datingDataMat,datingLabels = fileToMatrix('datingTestSet2.txt')

# print datingDataMat
# print datingLabels
#
# normMat, ranges, minVals = autoNorm(datingDataMat)
# print normMat

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()

# datingClassTest()

classifyPerson()

