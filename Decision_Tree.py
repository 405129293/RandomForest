# -*- coding: utf-8 -*-
import pandas as pd
import math
import numpy as np
import Decision_Tree_Visual
import json

#读取早期糖尿病风险预测数据集
def readDataset(dset):
    df = pd.read_csv(dset, encoding='gbk')
    for index, row in df.iterrows():
        age = df.loc[index, 'Age']
        if age < 18:
            df.loc[index, 'Age'] = '少年'
        elif age >= 18 and age <= 40:
            df.loc[index, 'Age'] = '青年'
        elif age > 40 and age <= 65:
            df.loc[index, 'Age'] = '中年'
        elif age > 65:
            df.loc[index, 'Age'] = '老年'
    return df


# 将数据集按照随机采样的方法以一定比例分割
def split_train(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    df1 = pd.DataFrame(columns = data.columns, index = None)
    df2 = pd.DataFrame(columns = data.columns, index = None)
    for i in test_indices:
        df1 = df1.append(data.loc[i, :], ignore_index=True)
    for i in train_indices:
        df2 = df2.append(data.loc[i, :], ignore_index=True)
    return df2, df1




#计算信息熵information entropy
def entropyCal(dfdata):
    dataSize = dfdata.shape[0] #数据集样本数
    colSize = dfdata.shape[1] #数据集属性个数（包括最后一列分类）
    typeCount = dict(dfdata.iloc[:,colSize-1].value_counts()) #统计数据集样本各个类别及其数目
    entropy = 0.0
    for key in typeCount:
        p = float(typeCount[key]) / dataSize
        entropy = entropy - p * math.log(p,2) #以2为底求对数
    return entropy    

#以某个属性的值划分数据集
def splitDataset(dfdata, colName, colValue):
    dataSize = dfdata.shape[0] #划分前数据集个数
    restData = pd.DataFrame(columns = dfdata.columns) #建立新的数据集，列索引与原数据集一样
    for rowNumber in range(dataSize):
        if dfdata.iloc[rowNumber][colName] == colValue:
            restData = restData.append(dfdata.iloc[rowNumber, :], ignore_index = True) #将划分属性等于该属性值的样本划给新数据集
    restData.drop([colName], axis = 1,inplace = True) #去掉该属性列
    return restData

#选择当前数据集最好的划分属性，以信息增益为准则选择划分属性
def chooseBestFeatureToSplit(dfdata):
    dataSize = dfdata.shape[0] #数据集样本个数
    numFeature = dfdata.shape[1]-1 #数据集属性个数
    entropyBase = entropyCal(dfdata) #划分前样本集的信息熵
    infoGainMax = 0.0 #初始化最大信息熵
    bestFeature = '' #初始化最佳划分属性
    for col in range(numFeature):
        featureValueCount = dict(dfdata.iloc[:,col].value_counts()) #统计该属性下各个值及其数目
        entropyNew = 0.0
        for key, value in featureValueCount.items():
            #计算该属性划分下各样本集的信息熵加权和
            entropyNew += entropyCal(splitDataset(dfdata,dfdata.columns[col], key)) * float(value / dataSize)
        infoGain = entropyBase - entropyNew #计算该属性下的信息增益
        if infoGain > infoGainMax:
            infoGainMax = infoGain
            bestFeature = dfdata.columns[col] #寻找最佳划分属性
    return bestFeature

#当叶节点样本已经无属性可划分了或者样本集为同一类别，这时采用多数表决法返回数量最多的类别
def typeMajority(dfdata):
    typeCount = dict(dfdata.iloc[:,dfdata.shape[1]-1].value_counts())
    return list(typeCount.keys())[0]
            
#ID3创建决策树
def creatDecisionTree(dfdata):
    #首先判断样本集是否为同一类别以及是否还能进行属性划分
    if (dfdata.shape[1] == 1 or len(dfdata.iloc[:,dfdata.shape[1] - 1].unique()) == 1):
        return typeMajority(dfdata) 
    bestFeature = chooseBestFeatureToSplit(dfdata)   #选择最佳划分属性
    if bestFeature == '': #这一点很重要，之前出错就是因为可能一个用于划分的数据集，数据条目相同，找不出最优的特征，要单独处理下
        return typeMajority(dfdata)
    decisionTree = {bestFeature:{}}  #以字典形式创建决策树

    # 省略问题修改配置, 打印100列数据
    pd.set_option('display.max_columns', 100)
    # 截断问题修改配置，每行展示数据的宽度为230
    pd.set_option('display.width', 230)
    # print('------------------------------------------------------')
    # print(bestFeature)
    # print(dfdata)

    bestFeatureValueCount=dict(dfdata.loc[:,bestFeature].value_counts()) #统计该属性下的所有属性值
    # if bestFeature == 'Age':
    #     print('-----------0-------------')

    for key, value in bestFeatureValueCount.items():
        #以递归调用方式不断完善决策树
        sd = splitDataset(dfdata, bestFeature, key)
        decisionTree[bestFeature][key] = creatDecisionTree(sd)
        # {'Gender': {'Male': {'Polyuria': {'No': 'Negative', 'Yes': {'Polydipsia': {'Yes': 'Positive', 'No': {'delayed healing': {'Yes': 'Negative', 'No': 'Positive'}}}}}}, 'Female': {'partial paresis': {'Yes': 'Positive', 'No': {'sudden weight loss': {'No': {'Age': {'青年': 'Positive', '中年': 'Negative'}}, 'Yes': 'Positive'}}}}}}
    return decisionTree

#对新的样例进行分类预测
def classify(inputTree, valSple):
    firstStr = list(inputTree.keys())[0] #决策树第一个值，即第一个划分属性
    secondDict = inputTree[firstStr]

    # 如果所有的分支都属于一类，则可以直接返回。这几条语句是做这个判断的
    values = list(secondDict.values())
    flag = False
    for item in values:
        if type(item).__name__ == 'dict':
            flag = True
            break
    if flag == False and len(set(values)) == 1:
        return values[0]

    classLabel = 'NoLabel'
    for key in secondDict.keys():
        # print(key)
        if(valSple[firstStr]==key): #该样本在该划分属性的值与决策树的对应判断
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],valSple) # 递归调用分类函数
            else:
                classLabel = secondDict[key]
    return classLabel

def testPrecision(thisTree, testdata):
    labels = list(testdata.iloc[:, -1])
    testNum = testdata.shape[0]
    classPred = []
    crtNum = 0 #初始化预测正确样例数
    for rowNum in range(testNum):
        # classSple = classify(thisTree, validationset.iloc[rowNum, :]) #预测该样例的分类
        classSple = classify(thisTree, testdata.iloc[rowNum, :]) #预测该样例的分类
        classPred.append(classSple)
        if labels[rowNum] == classSple: #判断预分类测是否正确
            crtNum += 1
    return crtNum / testNum #返回分类精度


if __name__ == '__main__':
    df = readDataset("dataset.csv")
    trainset, set0 = split_train(df, 0.4)
    testset, validationset = split_train(set0, 0.5)
    DecisionTree = creatDecisionTree(trainset)
    Decision_Tree_Visual.createTree(DecisionTree, "决策树_早期糖尿病风险预测")
    prec = testPrecision(DecisionTree, testset)
    print(prec)






