#coding:utf-8

from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))   #将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = empty((0,0))
    mat1 = empty((0,0))
    try:
        mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
        #The last `[0]` in the book causes logical error, do remind to remove them
        mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]#[0] (same here, remove the `[0]`)
    except IndexError as e:
        print(e)
    return mat0, mat1


def regLeaf(dataSet):
    subSet = dataSet[:,-1]
    print(subSet)
    return mean(subSet)


def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]   # 容许的误差下降值
    tolN = ops[1]   # 切分的最小样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:  #如果所有的值相等则退出
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].A1):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  #如果误差减小不明显则退出
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


if __name__ == '__main__':
    #testMat = mat(eye(4))
    #mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    #print(mat0, mat1)
    myMat = mat(loadDataSet('ex00.txt'))
    tree = createTree(myMat)
    print(tree)