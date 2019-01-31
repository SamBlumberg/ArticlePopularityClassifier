import time
from collections import defaultdict
import operator
import pandas as pd
import numpy as np
import math



class NaiveBayes:

    def __init__(self, transactions):
        self.transactions = transactions

    # Returns: this is the probability of a class label in a given datatset ( ie P(Ci) ) and count
    def priorProbability(self, classLabel, classColName):
        numOfTuples = len(self.transactions.index)
        grp = self.transactions.groupby(classColName)
        for className, group in grp:
            if className == classLabel:
                count = len(group.index)
                break

        probability = count / numOfTuples

        return probability

    # computes the posterior probability for categorical attributes
    def posteriorProbability_categorical(self, atttrValue, attrColName, classLabel, classColName,
                                         classLabelNumOfOccurence):
        attrCount = 0
        grp = self.transactions.groupby([attrColName, classColName])
        myKey = (atttrValue, classLabel)
        for className, group in grp:
            if className == myKey:
                # number of occurrence of this attribute in the target class
                attrCount = len(group.index)
                break
        # Use Laplacian correction to compute probability to avoid "zero" probability
        # ie I added "+1"  to both denominator and numerator
        return (attrCount+1)/(classLabelNumOfOccurence+1)

    # computes posterior probability for continuous attributes
    # This computes and returns Gaussian distribution  g(x, mean, std)
    def posteriorProbability_continuous(self, attrValue, mean, std):
        leftSide = 1 / (np.sqrt(2 * np.pi * std))
        rightSide1 = np.power((attrValue - mean), 2)
        rightSide2 = 2 * np.power(std, 2)
        rightSide = (1+rightSide1) / (1+rightSide2)
        gausian = leftSide * np.exp(-rightSide)  # computes Gaussian distribution
        return gausian


# Retrieve all transactions from database file
# create bins for continuous attributes if necessary
def getTransactions(filename, headers):

    data = pd.read_csv(filename, names=headers)

    # Replace continuous values with bin values
    #data['age'] = pd.cut(data['age'], [0, 18, 31, 51, 150], labels=['child', 'adult', 'middle-age', 'senior'])
    #  data['capital-gain'] = pd.cut(data['capital-gain'], [0, 20092, 40069, 60046, 80022, 3000000],
                                  # labels=['smallGain', 'mediumGain', 'avgGain', 'aboveAvgGain', 'largeGain'])
    # data['capital-loss'] = pd.cut(data['capital-loss'], [0, 1206, 2256, 3305, 3000000],
                                  # labels=['smallLoss', 'mediumLoss', 'avgLoss', 'largLoss'])
    # data['hours-per-week'] = pd.cut(data['hours-per-week'], [0, 36, 40, 500], labels=['tempTime', 'fullTime', 'overTime'])

    return data


# Compute the summary statistics of continuous attributes in the given classes
# Here we only computing Mean and Standard deviation to be used in Gaussian distribution
# Headers: header name of continuous attributes in the given dataset
def summaryStatistic(attrNames, dataset, headers):
    attrSummaryStats = defaultdict()
    # Hold number of occurrence of each class in the given dataset
    classInstanceCount = defaultdict(int)
    # separate dataset into different classes
    # The class column name is the last item in attrNames
    classGroups = dataset.groupby(headers[-1])
    for className, group in classGroups:
        summary = defaultdict()
        for attr in attrNames:
            mean = group[attr].mean()
            std = group[attr].std()
            summary[attr] = [mean, std]
        attrSummaryStats[className] = summary
        classInstanceCount[className] = len(group.index)
    return attrSummaryStats, classInstanceCount


confusionMatrix = a = [[0,0,0],[0,0,0]]

def classifier():
    headers = []
    with open("features_selected_bayes.txt", "r") as file:
        for line in file:
            headers = line.split(",")
    
    continuousAttributes = headers.copy()
    # Remove class label
    del continuousAttributes[-1]


    train_data = getTransactions('Bayes_Selected_Train.csv', headers)
    test_data = getTransactions('Bayes_Selected_Test.csv', headers)
    summaryStats, classCount = summaryStatistic(continuousAttributes, train_data, headers)
    classLabels = summaryStats.keys()
    bay = NaiveBayes(train_data)
    start_time = time.time()
    classPriorProbality = defaultdict()
    posteriorProbability = defaultdict()
    # Compute P(Ci)
    for label in classLabels:
        priorProbality = bay.priorProbability(label, headers[-1])
        classPriorProbality[label] = priorProbality
        posteriorProbability[label] = []  # This holds the posterior probability of each attribute in a target class

    result = []

    cc = 0
    for i in test_data.index:
        # compute P(X|Ci)
        for k, v in dict(test_data.iloc[i]).items():
            if k == headers[-1]:
                targetClass = v
            # Avoid class label
            if k is not headers[-1]:
                # If attribute is a continuous value attribute
                if k in continuousAttributes:
                    for label in classLabels:
                        attrSum = summaryStats[label]
                        mean = attrSum[k][0]
                        std = attrSum[k][1]
                        # P(X|Ci) for the target class
                        postProb = bay.posteriorProbability_continuous(v, mean, std)
                        posteriorProbability[label].append(postProb)
                # If attribute a categorical attribute
                else:
                    for label in classLabels:
                        # P(X|Ci) for the target class
                        postProbb = bay.posteriorProbability_categorical(v, k, label, headers[-1], classCount[label])
                        posteriorProbability[label].append(postProbb)

        probabilityResult = defaultdict()
        # compute P(X|Ci)*P(Ci)
        for label in classLabels:
            prob1 = np.prod(np.array(posteriorProbability[label]))*classPriorProbality[label]
            #print(np.prod(np.array(posteriorProbability[label])), classPriorProbality[label])
            probabilityResult[label] = prob1

        # sort probabilityResult dictionary to retrieve max
        maxProbability = sorted(probabilityResult.items(), key=operator.itemgetter(1), reverse=True)[0]
        
        insertConfusionMatrix(targetClass, maxProbability[0])

        # reset computed posterior probability
        for label in classLabels:
            posteriorProbability[label] = []

        #cc += 1
        #if cc == 1000:
            #break
    print("Runtime is: ", time.time() - start_time)
    return result


def insertConfusionMatrix(expectedOutput, output):
    if expectedOutput == 1:
        #True positive (TP)
        if (expectedOutput - output) <= 0.5:
            confusionMatrix[0][0] += 1
            #False Negative (FN)
        else:
            confusionMatrix[0][1] += 1
    else:
        #False Positive (FP)
        if (expectedOutput - output) > 0.5:
            confusionMatrix[1][0] += 1
        #True Negative (TN)
        else:
            confusionMatrix[1][1] += 1

def computeRecognitionRate():
    #Number of positive class label (P)
    confusionMatrix[0][2] = confusionMatrix[0][0] + confusionMatrix[0][1];
    #Number of Negative class lable (N)
    confusionMatrix[1][2] = confusionMatrix[1][0] + confusionMatrix[1][1];

    #Recognition rate = TP+TN/P+N
    recognitionRate = (confusionMatrix[0][0] + confusionMatrix[1][1])/(confusionMatrix[0][2] + confusionMatrix[1][2]);
    #Error rate = FP+FN/P+N
    errorRate = ( confusionMatrix[1][0] + confusionMatrix[0][1])/(confusionMatrix[0][2] +  confusionMatrix[1][2]);

    print("Error Rate: ", errorRate)
    return recognitionRate;
    
classified = classifier()
recognitionRate = computeRecognitionRate()
print("Prediction Rate:", recognitionRate*100, "%")

