# 2021-04-08 to 2021-04-09
# HW) Kaggle 숙제 Mushroom Logistics
# (https://www.kaggle.com/uciml/mushroom-classification)

#Flow: load-theory-EDA-Model
#Logistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mushroom = pd.read_csv('./kaggle_tasks/20210408_mushroomclassification/mushrooms.csv')
print(mushroom)  #[8124 rows x 23 columns]
# print(mushroom.data())
dir(mushroom)
print(mushroom.describe())
print(mushroom.info())
print(mushroom.isna().sum())
        #class ; edible=e, poisonous=p, cap-shape ; bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
        #cap-surface ; fibrous=f,grooves=g,scaly=y,smooth=s
        #cap-color ; brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
        #bruises ; bruises=t,no=f, odor ; almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
        #gill-attachment ; attached=a, descending=d, free=f, notched=n, gill-spacing ; close=c,crowded=w,distant=d
        #gill-size ; broad=b,narrow=n, gill-color ; black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# print(mushroom.feature_names)
# print(mushroom.target_names)    # class) p : poisonous / e : edible
# print(mushroom.target)
# print(np.bincount(mushroom.target))
# print(mushroom.DESCR)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
msr = mushroom.copy
# np.random.seed(513)
# msr.sample(13)
msr.columns

print(msr['class'].value_count())