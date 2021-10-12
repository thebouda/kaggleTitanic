import pandas as pd

import numpy

def getAccuracy(prediction,expected):
    expected_Reshaped = numpy.array(expected).reshape(expected.size)
    test = numpy.array(expected_Reshaped == numpy.array(prediction)).sum()
    accuracy  = test /expected_Reshaped.size *100
    return accuracy 


def getData():
    # first we get the data
    testPath = "Data/test.csv"
    trainPath = "Data/train.csv"
    genderSubmissionPath = "Data/gender_submission.csv"

    testPDRaw = pd.read_csv(testPath)
    trainPDRaw = pd.read_csv(trainPath)
    genderSubPDRaw = pd.read_csv(genderSubmissionPath)

    # merge the test data too
    testDataMerge= pd.merge(genderSubPDRaw,testPDRaw,on = 'PassengerId')

    # start cleaning the data, the cleaning should be done on test and train data
    # sex variable
    trainPDRaw['Sex'].replace({'female':'0','male':'1'},inplace=True)
    trainPDRaw['Sex'] = pd.to_numeric(trainPDRaw['Sex'])

    testDataMerge['Sex'].replace({'female':'0','male':'1'},inplace=True)
    testDataMerge['Sex'] = pd.to_numeric(testDataMerge['Sex'])

    # age variable
    trainPDRaw.dropna(subset=['Age'],inplace=True) 
    testDataMerge.dropna(subset=['Age'],inplace=True) 

    # fare variable for the test
    testDataMerge.dropna(subset=['Fare'],inplace=True) 

    # we choose the data we want, then we proceed to modificate further
    featuresChosen = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']

    xTrain = trainPDRaw.loc[:,featuresChosen]
    yTrain = trainPDRaw.loc[:,['Survived']]

    xTest =  testDataMerge.loc[:,featuresChosen]
    yTest = testDataMerge.loc[:,['Survived']]

    return xTrain,xTest,yTrain,yTest


