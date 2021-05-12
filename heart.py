import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
####################################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
###################################################################
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#현호님께서 쏘아올린 공: scikit-learn은 PCA계산에서 SVD를 사용하는 이유는?
## 특이값 분해가 유용한 이유가 행렬이 정방행렬이든 아니든 관계없이 모든 m*n행렬에 대해 적용 가능하기 때문
## 고유값 분해(EVD)는 정방행렬에 대해서만 적용 가능하며, 또한 정방행렬 중에서도 일부 행렬에 대해서만 적용가능
#svd ; decomposition 가능하게 만듦, linear independent 안해도 됨

data = pd.read_csv('./heart.csv')
# print(data.columns)
data['sex'] = data['sex'].map({0:'female',1:'male'})
data['chest_pain_type'] = data['cp'].map({3:'asymptomatic', 1:'atypical_angina', 2:'non_anginal_pain', 0:'typical_angina'})
data['fbs'] = data['fbs'].map({0:'less_than_120mg/ml',1:'greater_than_120mg/ml'})
data['restecg'] = data['restecg'].map({0:'normal',1:'ST-T_wave_abnormality',2:'left_ventricular_hypertrophy'})
data['exang'] = data['exang'].map({0:'no',1:'yes'})
data['slope'] = data['slope'].map({0:'upsloping',1:'flat',2:'downsloping'})
data['thal'] = data['thal'].map({1:'fixed_defect',0:'normal',2:'reversable_defect'})
data['target'] = data['target'].map({0:'no_disease', 1:'has_disease'})
# print(data.isna().sum())
categorical = [i for i in data.loc[:,data.nunique()<=10]]
# print(categorical)
continuous = [i for i in data.loc[:, data.nunique()>=10]]
# print(continuous)
def dist(df, col, hue=None, row=3, columns=3):
    fig, axes = plt.subplots(row,columns,figsize=(16,12))
    axes = axes.flatten()   #펼치란 이야기

    for i,j in zip(df[col].columns, axes):
        sns.countplot(x=i, data=df, hue=hue, ax=j, orient=df[i].value_counts().index)
        j.set_title(f'{str(i).capitalize()} Distribution')

        total = float(len(df[i]))

        for p in j.patches:
            height = p.get_height()
            j.text(p.get_x()+p.get_width()/2, height/2, '{:1.2f}%'.format((height/total)*100),ha='center')
    plt.tight_layout()
    plt.show()

# dist(data, categorical)        
# Styling: custom color
cust_palt = ['#111d5e', '#c70039', '#f37121', '#ffbd69', '#ffc93c']
plt.style.use('ggplot') #ggplot; R 기반에서 그래프를 가져옴, 예쁘게 잘 그려짐

from matplotlib.gridspec import GridSpec
# #분포도 histgram을 합쳐놓은 형태
# fig = plt.figure(constrained_layout=True, figsize=(16,12))
# grid = GridSpec(ncols=6, nrows=3, figure=fig)
# #영역을 grid하게 잘라줌
# ax1 = fig.add_subplot(grid[0,:2])
# ax1.set_title('Tresbps Distribution')
# #분포를 보여주는 것
# sns.displot(data[continuous[1]], hist_kws={'rwidth':0.85, 'edgecolor':'orange','alpha':0.7}, color=cust_palt[0])
# #hist+분포
# ax12 = fig.add_subplot(grid[0,2:3])
# ax12.set_title('Trestbps')
# sns.boxplot(data[continuous[1]], orient='v')
# ax2 = fig.add_subplot(grid[0,3:5])
# #chol : 콜레스테롤 수치
# ax2.set_title('Cholr Distribution')
# sns.distplot(data[continuous[2]], hist_kws= {'rwidth':0.85, 'edgecolor':'blue','alpha':0.7},color = cust_palt[1])
# ax22 = fig.add_subplot(grid[0,5:])
# ax22.set_title('Cholr')
# sns.boxplot(data[continuous[2]], orient='v',color = cust_palt[1])
# ax3 = fig.add_subplot(grid[0,:2])
# #thalach : 최대 심박수
# ax3.set_title('thalach Distribution')
# sns.distplot(data[continuous[3]], hist_kws= {'rwidth':0.85, 'edgecolor':'blue','alpha':0.7},color = cust_palt[2])
# ax32 = fig.add_subplot(grid[0,2:3])
# ax32.set_title('thalach')
# sns.boxplot(data[continuous[3]], orient='v',color = cust_palt[2])
# ax4 = fig.add_subplot(grid[0,3:5])
# #oldpeak : 휴식에 비해 운동으로 인해 유발된 ST 우울증 ? 
# ax4.set_title('oldpeark Distribution')
# sns.distplot(data[continuous[4]], hist_kws= {'rwidth':0.85, 'edgecolor':'blue','alpha':0.7},color = cust_palt[3])
# ax42 = fig.add_subplot(grid[0,5:])
# ax42.set_title('thalach')
# sns.boxplot(data[continuous[4]], orient='v',color = cust_palt[3])
# ax5 = fig.add_subplot(grid[2, :4])
# ax5.set_title('Age Distribution')
# sns.distplot(data[continuous[0]],
#                  hist_kws={
#                  'rwidth': 0.95,
#                  'edgecolor': 'black',
#                  'alpha': 0.8},
#                  color=cust_palt[4])
# ax55 = fig.add_subplot(grid[2, 4:])
# ax55.set_title('Age')
# sns.boxplot(data[continuous[0]], orient='h', color=cust_palt[4])
# plt.show()

# fig = plt.figure(constrained_layout=True,figsize = (16,12))
# #A grid layout to place subplots within a figure.
# grid = GridSpec(ncols = 6, nrows= 3, figure = fig)
# ax1 = fig.add_subplot(grid[0, :2])
# ax1.set_title('trestbps Distribution')
# sns.boxplot(x = 'target',y = 'trestbps',data = data,palette = cust_palt[2:],ax = ax1)
# sns.swarmplot(x = 'target',y = 'trestbps',data = data, palette = cust_palt[:2],ax = ax1)
# ax2 = fig.add_subplot(grid[0,2:])
# ax2.set_title('chol Distribution')
# sns.boxplot(x = 'target', y = 'chol',data = data, palette = cust_palt[:2],ax = ax2)
# sns.swarmplot(x = 'target', y = 'chol',data = data, palette = cust_palt[:2],ax = ax2)
# ax3 = fig.add_subplot(grid[1,2:])
# ax3.set_title('thalach Distribution')
# sns.boxplot(x = 'target', y = 'thalach',data = data, palette = cust_palt[:2],ax = ax3)
# sns.swarmplot(x = 'target', y = 'thalach',data = data, palette = cust_palt[:2],ax = ax3)
# ax4 = fig.add_subplot(grid[1,2:])
# ax4.set_title('st_depression Distribution')
# sns.boxplot(x = 'target', y = 'oldpeak',data = data, palette = cust_palt[:2],ax = ax4)
# sns.swarmplot(x = 'target', y = 'oldpeak',data = data, palette = cust_palt[:2],ax = ax4)
# ax5 = fig.add_subplot(grid[2,:])
# ax5.set_title('age Distribution')
# sns.boxplot(x = 'target', y = 'age',data = data, palette = cust_palt[2:],ax = ax5)
# sns.swarmplot(x = 'target', y = 'age',data = data, palette = cust_palt[:2],ax = ax5)
# plt.show()

# plt.figure(figsize = (16,10))
# sns.pairplot(data[['trestbps','chol','thalach','oldpeak','age','target']],markers=['o','D'])
# plt.show()
# #scatter를 확인하고, cluster로 풀어야 함을 알 수 있음

# import plotly.express as px #3d
# fig = px.scatter_3d(data, x = 'chol',y = 'thalach',z = 'age',size = 'oldpeak',color = 'target',opacity=0.8)
# # reference : https://plotly.com/python/reference/layout/
# fig.update_layout(margin = dict(l=0,r=0,b=0,t=0))
# fig.show()

def freq(df,cols, xi, hue = None, row=4, col = 1):
    fig,axes = plt.subplots(row,col,figsize = (16,12),sharex = True)
    axes = axes.flatten()
    for i,j in zip(df[cols].columns, axes):
        # sns.pointplot(x = xi,y=i,data = df, palette = 'cubehelix', hue = hue,ax =j)
        # sns.cubehelix_palette(as_cmap=True)
        # sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        # sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
        # sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        # sns.pointplot(x = xi,y=i,data = df, palette = 'plasma', hue = hue,ax =j)
        sns.pointplot(x = xi,y=i,data = df, palette = 'Blues', hue = hue,ax =j)
        # sns.pointplot(x = xi,y=i,data = df, palette = 'viridis',hue = hue,ax =j)
        # sns.pointplot(x = xi,y=i,data = df, palette = 'mako',hue = hue,ax =j)
    plt.tight_layout()
    plt.show()
# freq(data, ['trestbps','chol','thalach','oldpeak'],'age',hue = 'target', row =4, col=1)

# #corr matirx
# corrlation_matrix = data.corr()
# mask = np.triu(corrlation_matrix.corr())
# plt.figure(figsize = (20,12))
# # sns.heatmap(corrlation_matrix, annot = True, fmt = '.2f',cmap = 'Spectral',linewidths=1,cbar = True)
# # sns.heatmap(corrlation_matrix, annot = True, fmt = '.2f',cmap = 'flare',linewidths=1,cbar = True)
# # sns.heatmap(corrlation_matrix, annot = True, fmt = '.2f',cmap = 'vlag',linewidths=1,cbar = True)
# sns.heatmap(corrlation_matrix, annot = True, fmt = '.2f',cmap = 'YlOrBr',linewidths=1,cbar = True)
# # sns.heatmap(corrlation_matrix, annot = True, fmt = '.2f',cmap = 'crest',linewidths=1,cbar = True)
# plt.show()

# Model
#######################################################<<<Data>>>#########################################
# import numpy as np
# import pandas as pd
#######################################################<<<Figrue>>>#########################################
# import matplotlib.pyplot as plt
# import seaborn as sns
#######################################################<<<Split&Validate>>>#########################################
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold
#######################################################<<<Score>>>#########################################
# from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix

#######################################################<<<Model_Classifier>>>#########################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
#######################################################<<<Model_Cluster>>>#########################################
from sklearn.cluster import KMeans, DBSCAN
#######################################################<<<Model_display>>>#########################################
# from Ipython.display import display
# import display

X = data.drop('target', axis=1)
y = data.target
#Anomaly Detection(이상 탐지)
# 여러 방법 중에서도 __RobustScaler를 써도 됨

gradclass = GradientBoostingClassifier(random_state=42)
knclass = KNeighborsClassifier(random_state=42)
dectree = DecisionTreeClassifier(random_state=42)
svc = SVC(random_state=42)
randfclass = RandomForestClassifier(random_state=42)
adaclass = AdaBoostClassifier(random_state=42)
gsclass = GaussianNB(random_state=42)
xgbclass = XGBClassifier(random_state=42)
ligthgbmsclass = LGBMClassifier(random_state=42)
catboostclass = CatBoostClassifier(random_state=42)

cv = KFold(5, shuffle=True, random_state=42)

classifiers = [gradclass, knclass, dectree, svc, randfclass, adaclass,gsclass,xgbclass,ligthgbmsclass,catboostclass]

def model_check(X, y, classifiers, cv):
    
    ''' A function for testing multiple classifiers and return several metrics. '''
    
    model_table = pd.DataFrame()

    row_index = 0
    for cls in classifiers:
        MLA_name = cls.__class__.__name__   #__val__; 초기화&상속
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(cls, X, y, cv=cv,scoring=('accuracy','f1','roc_auc'),return_train_score=True,n_jobs=-1)
        # return_train_score=True; score 값을 받겠다는 뜻
        # n_jobs=-1; cpu다 쓰겠다는 뜻
        model_table.loc[row_index, 'Train Roc/AUC Mean'] = cv_results['train_roc_auc'].mean()
        model_table.loc[row_index, 'Test Roc/AUC Mean'] = cv_results['test_roc_auc'].mean()
        model_table.loc[row_index, 'Test Roc/AUC Std'] = cv_results['test_roc_auc'].std()
        model_table.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_accuracy'].mean()
        model_table.loc[row_index, 'Test Accuracy Mean'] = cv_results[
            'test_accuracy'].mean()
        model_table.loc[row_index, 'Test Acc Std'] = cv_results['test_accuracy'].std()
        model_table.loc[row_index, 'Train F1 Mean'] = cv_results[
            'train_f1'].mean()
        model_table.loc[row_index, 'Test F1 Mean'] = cv_results[
            'test_f1'].mean()
        model_table.loc[row_index, 'Test F1 Std'] = cv_results['test_f1'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1        

    model_table.sort_values(by=['Test F1 Mean'],
                            ascending=False,
                            inplace=True)

    return model_table

# Baseline Results
raw_models = model_check(X, y, classifiers, cv)
# display(raw_models)

#kneighber 갑자기 떨어진 이유: k가 돌아다니면서 이웃한 거리가 가까워져서
#svc는 왜 떨어진 걸까? classfier가 잘못작동, hyperparameter 재조정 필요

#model confusion matrix
