# 2021-04-08 to 2021-04-09
# HW) Kaggle 숙제 Mushroom Logistics
# (https://www.kaggle.com/uciml/mushroom-classification)

#Flow: load-theory-EDA-Model
#Logistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
###########################################################################
mushroom = pd.read_csv('./kaggle_tasks/20210408_mushroomclassification/mushrooms.csv')
# print(mushroom)  #[8124 rows x 23 columns]
# print(mushroom.head())
# print(dir(mushroom))
# print(mushroom.describe())
# print(mushroom.info())
# print(mushroom.isna().sum())
        #class ; edible=e, poisonous=p, cap-shape ; bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
        #cap-surface ; fibrous=f,grooves=g,scaly=y,smooth=s
        #cap-color ; brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
        #bruises ; bruises=t,no=f, odor ; almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
        #gill-attachment ; attached=a, descending=d, free=f, notched=n, gill-spacing ; close=c,crowded=w,distant=d
        #gill-size ; broad=b,narrow=n, gill-color ; black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# print(mushroom.feature_names)
# print(mushroom.target_names)    # class) p : poisonous / e : edible
# print(mushroom.target)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# msr = mushroom.copy
# msr = pd.DataFrame(msr, columns=msr.column)
# np.random.seed(513)
# msr.sample(13)
# msr.columns
# print(msr['class'].value_count())

# 인코딩
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for columns in mushroom.columns:
    mushroom[columns] = labelencoder.fit_transform(mushroom[columns])

y = mushroom['class']
x = mushroom.drop('class',axis=1)
print(mushroom.corr())

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def plot_col(col, hue=None, color=['purple', 'lightgreen'], labels=None):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.countplot(col, hue=hue, palette=color, saturation=0.6, data=mushroom, dodge=True, ax=ax)
    ax.set(title = f"Mushroom {col.title()} Quantity", xlabel=f"{col.title()}", ylabel="Quantity")
    if labels!=None:
        ax.set_xticklabels(labels)
    if hue!=None:
        ax.legend(('Poisonous', 'Edible'), loc=0)

class_dict = ('Poisonous', 'Edible')
plot_col(col='class', labels=class_dict)
plt.show()

shape_dict = {"bell":"b","conical":"c","convex":"x","flat":"f", "knobbed":"k","sunken":"s"}
labels = ('convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical')
plot_col(col='cap-shape', hue='class', labels=labels)
plt.show()

color_dict = {"brown":"n","yellow":"y", "blue":"w", "gray":"g", "red":"e","pink":"p",
              "orange":"b", "purple":"u", "black":"c", "green":"r"}
plot_col(col='cap-color', color=color_dict.keys(), labels=color_dict)
plt.show()

plot_col(col='cap-color', hue='class', labels=color_dict)
plt.show()

surface_dict = {"smooth":"s", "scaly":"y", "fibrous":"f","grooves":"g"}
plot_col(col='cap-surface', hue='class', labels=surface_dict)
plt.show()

###########################################################################

def get_labels(order, a_dict):    
    labels = []
    for values in order:
        for key, value in a_dict.items():
            if values == value:
                labels.append(key)
    return labels

odor_dict = {"almond":"a","anise":"l","creosote":"c","fishy":"y",
             "foul":"f","musty":"m","none":"n","pungent":"p","spicy":"s"}
order = ['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm']
labels = get_labels(order, odor_dict)      
plot_col(col='odor', color=color_dict.keys(), labels=labels)
plt.show()

###kind='reg'
stalk_cats = ['class', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
              'stalk-color-above-ring', 'stalk-color-below-ring']
data_cats = mushroom[stalk_cats]
sns.pairplot(data_cats, hue='class', kind='reg', palette='Pastel2')
plt.show()
###kind='scatter'   #분산된 점들이 비교적 많음
stalk_cats = ['class', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
              'stalk-color-above-ring', 'stalk-color-below-ring']
data_cats = mushroom[stalk_cats]
sns.pairplot(data_cats, hue='class', kind='scatter',palette='winter')
plt.show()
###kind='kde'   #등고선 범위가 겹치는 형태로 출력됨
stalk_cats = ['class', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
              'stalk-color-above-ring', 'stalk-color-below-ring']
data_cats = mushroom[stalk_cats]
sns.pairplot(data_cats, hue='class', kind='kde',palette='Pastel1')
plt.show()
###kind='hist'  #이번 데이터에서는 가장 정적인 형태로 출력된 듯 함
stalk_cats = ['class', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
              'stalk-color-above-ring', 'stalk-color-below-ring']
data_cats = mushroom[stalk_cats]
sns.pairplot(data_cats, hue='class', kind='hist',palette='summer')
plt.show()

fig, ax = plt.subplots(3, 2, figsize=(20, 15))
for i, axis in enumerate(ax.flat):
    sns.distplot(data_cats.iloc[:, i], ax=axis)
plt.show()

#graph
import plotly.graph_objs as go
import plotly.express as px

labels = ['Edible', 'Poison']
values = mushroom['class'].value_counts()

fig=go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=['purple', 'lightblue'],
                              line=dict(color='#FFFFFF',width=10)))
fig.show()

###########################################################################
# ###randomforest###
# from sklearn.ensemble import RandomForestClassifier

# # x, y = msr(noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(x,y, stratify=y, random_state=42)

# forest = RandomForestClassifier(n_estimators=3, n_jobs=-1, random_state=42)
# forest.fit(X_train,y_train)

# import matplotlib.pyplot as plt
# import numpy as np
# from mglearn.plots import plot_2d_classification

# _, axes = plt.subplots(2,3)
# marker_set = ['p','8']

# for i, (axe, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#         axe.set_title('tree {}'.format(i))
#         plot_2d_classification(tree, x, fill=True, ax=axe, alpha=0.4)
#         for i, m in zip(np.unique(y), marker_set)        :
#                 axe.scatter(x[y==i][:,0],x[y==i][:,1],marker=m,label='class {}'.format(i),edgecolor='g')
#                 axe.set_xlabel('feature 0')
#                 axe.set_ylabel('feature 1')

# axes[-1,-1].set_title('random forest')
# axes[-1,-1].set_xlabel('feature 0')
# axes[-1,-1].set_ylabel('feature 1')

# plot_2d_classification(forest,x,fill=True,ax=axes[-1,-1],alpha=0.4)

# for i,m in zip(np.unique(y),marker_set):
#         plt.scatter(x[y==i][:,0],x[y==i][:,1],marker=m,label='class {}'.format(i),edgecolors='g')

# plt.show()

###########################################################################
