#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import log
import operator
import pickle

'''ID3算法实现决策树
'''

def calcShannonEnt(dataSet):
    ''' 计算给定数据的香农熵
    :param dataSet:
    :return:
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries)
        shannonEnt -=  prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据类
    :param dataSet:
    :param axis:
    :param value:
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式（最好的特征）
    :param dataSet:
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 计算每种划分方式的信息熵
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 遍历每个独立值，累计总的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy # 信息增益：熵的减少
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    出现最多次数的分类名称
    :param classList:
    :return:
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.kemyTys():
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): # 就剩一个类别了，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:    # 遍历完所有特征(只剩下label那列)，返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''决策树分类函数
    '''
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]    # subtree
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                # Search recursively
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # Leaf node
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

