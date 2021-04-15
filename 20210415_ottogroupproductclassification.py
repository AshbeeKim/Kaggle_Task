import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./kaggle_tasks/20210415_ottogroupproductclassification/train.csv')
test = pd.read_csv('./kaggle_tasks/20210415_ottogroupproductclassification/test.csv')
sampleSubmission = pd.read_csv('./kaggle_tasks/20210415_ottogroupproductclassification/sampleSubmission.csv')

print(train.describe())              #id #feat_1 to 93 #target:Class         #(61878, 95)
print(test.describe())               #id #feat_1 to 93                       #(144368, 94)
print(sampleSubmission.describe())   #id #Class_1 to 9                       #(144368, 10)
# they don't have NULL

print(train.columns)
print(test.columns)
print(sampleSubmission.columns)

data = train.copy
finT = test.copy
sS = sampleSubmission.copy
###################################################################################################################################

'''File descriptions
    trainData.csv - the training set
    testData.csv - the test set
    sampleSubmission.csv - a sample submission file in the correct format
    Data fields
    id - an anonymous id unique to a product
    feat_1, feat_2, ..., feat_93 - the various features of a product
    target - the class of a product'''

###################################################################################################################################
