# -*- coding: utf-8 -*-
import time

import pandas as pd
import math
import numpy as np
import Decision_Tree_Visual
import Decision_Tree as DTree
import random

def buildForest(trainset, n_trees, feature_choice_rate):
    forest = []
    for n in range(n_trees):
        t1 = time.time()
        # 有放回的抽取训练集
        df1 = pd.DataFrame(columns= trainset.columns)
        loc = np.random.randint(0, trainset.shape[0], trainset.shape[0])
        for i in range(len(loc)):
            df1 = df1.append(trainset.iloc[loc[i], :])
        # print(df1)
        # 随机选取特征子集建树
        feature_number = int(feature_choice_rate * (trainset.shape[1] - 1))
        loc = random.sample(range(0, trainset.shape[1] - 1), trainset.shape[1] - feature_number)
        # print(loc)
        columns = trainset.columns.tolist()
        feature_list = []
        for item in loc:
            feature_list.append(columns[item])
        df1 = df1.drop(feature_list, axis=1)
        tree = DTree.creatDecisionTree(df1)
        Decision_Tree_Visual.createTree(tree, "第" + str(n) + "棵决策树")
        forest.append(tree)
        t2 = time.time()
        print('构建第%d棵树的时间为%f' % (n, t2 - t1))
    return forest

def getAccurary(testset, forest):
    tureNumber = 0
    falseNumber = 0
    for i in range(testset.shape[0]):
        predic = []
        for j in range(len(forest)):
            predic.append(DTree.classify(forest[j], testset.iloc[i, :]))
        if testset.iloc[i, -1] == max(predic, key=predic.count):  # 判断预分类测是否正确
            tureNumber += 1
        else:
            falseNumber += 1
    return tureNumber / (tureNumber + falseNumber)

if __name__ == '__main__':
    n_trees = 11
    feature_choice_rate = 0.5
    df = DTree.readDataset("dataset3.csv")
    trainset, testset = DTree.split_train(df, 0.4)
    forest = buildForest(trainset, n_trees, feature_choice_rate)

    for i in range(len(forest)):
        print("第" + str(i) + "棵决策树的预测精确度为：" + str(DTree.testPrecision(forest[i], testset)))

    t1 = time.time()
    score = getAccurary(testset, forest)
    t2 = time.time()
    print("随机森林的预测精确度为：" + str(score))
    # DecisionTree = creatDecisionTree(trainset)
    # Decision_Tree_Visual.createTree(DecisionTree, "C4.5决策树_早期糖尿病风险预测")
    # prec = testPrecision(DecisionTree, testset)
    # print(prec)
