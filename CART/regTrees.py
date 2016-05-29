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


def isTree(obj):
    return type(obj).__name__ == 'dict'


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)    # 没有测试数据,对树进行塌陷处理


    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) + \
            sum(power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:,-1] - treeMean, 2))
        if errorMerge < errNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    # Y = mat(ones(m, 1))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse\nTry increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def regTreeEval(model, dummy):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

if __name__ == '__main__':
    #testMat = mat(eye(4))
    #mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    #print(mat0, mat1)
    # myMat2 = mat(loadDataSet('ex2.txt'))
    # tree = createTree(myMat2, ops=(0,1))
    # myDatTest = loadDataSet('ex2test.txt')
    # myMat2Test = mat(myDatTest)
    # prune(tree, myMat2Test)
    # print(tree)
    myMat2 = mat(loadDataSet('exp2.txt'))
    tree = createTree(myMat2, modelLeaf, modelErr, (1,10))
    print(tree)