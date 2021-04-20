import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./kaggle_tasks/20210415_ottogroupproductclassification/train.csv')
test = pd.read_csv('./kaggle_tasks/20210415_ottogroupproductclassification/test.csv')
sampleSubmission = pd.read_csv('./kaggle_tasks/20210415_ottogroupproductclassification/sampleSubmission.csv')

print(train.isna().sum())
print(test.isna().sum())
print(train.isnull().sum())
print(test.isnull().sum())
# print(sampleSubmission.isna().sum())
# they don't have NULL

print(train.describe())              #id #feat_1 to 93 #target:Class         #(61878, 95)
# print(test.describe())               #id #feat_1 to 93                       #(144368, 94)
# print(sampleSubmission.describe())   #id #Class_1 to 9                       #(144368, 10)

print(train.columns)

trn = train.copy()
tst = test.copy()
sS = sampleSubmission.copy()

#drop id
excid_t = trn.drop('id',axis=1)
#drop target
exctg_t = trn.drop('target',axis=1)
#drop id and target
excboth_t = trn.drop(['id','target'],axis=1)

onlyvalues = trn.values

print(excid_t.shape)
print(exctg_t.shape)
print(excboth_t.shape)
print(onlyvalues.shape)

print(trn['target'].unique())

def class_count_sum(trn):
  for i in range(1,10):
    print(trn[trn['target']==f'Class_{i}'].count().sum())
class_count_sum(trn)
#Class_2>Class_6>Class_8>Class_3>Class_9>Class_7>Class_5>Class_4>Class_1

plt.title('target counts sum')
plt.xlabel('class')
sns.countplot(trn['target'], saturation=0.6, data=excid_t, dodge=True, palette = 'inferno')
plt.show()