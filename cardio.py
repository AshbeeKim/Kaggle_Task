import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./cardio_train.csv',delimiter=';')

# print(data.describe)
# print(data.tail())
# print(data.count())

# corrdata = data.corr()
# fig, ax = plt.subplots(figsize = (20, 10))
# sns.heatmap(corrdata, annot = True)
# plt.show()

obj = data.iloc[:,[1,2,3,4,12]]
exm = data.iloc[:,[5,6,7,8,12]]
sbj = data.iloc[:,[9,10,11,12]]

def age(x) :
    x = x/365 
    return int(x)
obj['age'] = obj['age'].copy().apply(age)
## 'age' 칼럼이 day로 되어 있어서 나이로 바꿈.
obj['age'].value_counts()

obj.loc[obj['age']<40, 'age'] = 0 
obj.loc[(obj['age']>= 40) & (obj['age'] < 50), 'age'] = 1 
obj.loc[(obj['age']>= 50) & (obj['age'] < 60), 'age'] = 2
obj.loc[obj['age']>= 60, 'age'] = 3
## 나이별로 범주화 했음

obj['BMI'] = (obj['weight']/(obj['height']/100)**2)
obj = obj.iloc[:,[0,1,-1,-2]]
obj['BMI'].value_counts()

obj.loc[obj['BMI']<18.5, 'BMI'] = 0
obj.loc[(obj['BMI']>= 18.5) & (obj['BMI'] < 24.9), 'BMI'] = 1
obj.loc[(obj['BMI']>= 24.9) & (obj['BMI'] < 29.9), 'BMI'] = 2
obj.loc[(obj['BMI']>= 29.9) & (obj['BMI'] < 34.9), 'BMI'] = 3
obj.loc[obj['BMI']>= 34.9, 'BMI'] = 4
## bmi별로 범주화 했음
obj = pd.DataFrame(obj,dtype='int64')

# print(obj)
# print(obj.info())

### EXM
# # 내가 알고있는 선입견보다 일반적인 접근 필요
# # IQR을 벗어나는 것은 버림
#   # 임의로 제거하는 게 IQR에서 날아갈 듯 함
# lower_bound = Q1- 1.5*IQR
# upper_bound = Q3 + 1.5*IQR
# boundary[each] = [lower_bound, upper_bound ]
# # Q1 ->qunatile(0.25)
# # Q3 ->qunatile(0.75)
#IQR로 BloodPressure 이상치 처리

cobj = obj.corr()
cexm = exm.corr()
csbj = sbj.corr()
# print(cobj)
# print(cexm)
# print(csbj)

# fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (30, 10))

# ax1.imshow(cobj)
# ax2.imshow(cexm)
# ax3.imshow(csbj)

# ax1.set_title('cobj')
# ax2.set_title('cexm')
# ax3.set_title('csbj')
# plt.show()

feat_obj = obj.drop('cardio',axis=1)
feat_exm = exm.drop('cardio',axis=1)
feat_sbj = sbj.drop('cardio',axis=1)

corrobj = feat_obj.corr()
correxm = feat_exm.corr()
corrsbj = feat_sbj.corr()
# print(corrobj)
# print(correxm)
# print(corrsbj)

# fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (30, 10))

# ax1.imshow(corrobj,cmap='Pastel1',interpolation='nearest')
# ax2.imshow(correxm,cmap='Wistia',interpolation='nearest')
# ax3.imshow(corrsbj,cmap='icefire',interpolation='nearest')

# ax1.set_title('cobj')
# ax2.set_title('cexm')
# ax3.set_title('csbj')

# plt.show()

##############################################################################
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def func(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")
##############################################################################

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12, 9))
# annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
corr_matrix1 = np.corrcoef(feat_obj)
im, _ = heatmap(corr_matrix1, feat_obj, feat_obj, ax=ax1,
                cmap="bone", vmin=-1, vmax=1, cbarlabel="corrobj")

im, _ = heatmap(feat_obj, feat_obj, feat_exm, ax=ax2,
                cmap="inferno", vmin=-1, vmax=1, cbarlabel="corrobj")

im, _ = heatmap(feat_obj, feat_obj, feat_sbj, ax=ax3,
                cmap="inferno", vmin=-1, vmax=1, cbarlabel="corrobj")

corr_matrix2 = np.corrcoef(feat_exm)
im, _ = heatmap(feat_exm, feat_exm, feat_obj, ax=ax4,
                cmap="Wistia", vmin=-1, vmax=1, cbarlabel="correxm")

im, _ = heatmap(corr_matrix2, feat_exm, feat_exm, ax=ax5,
                cmap="bone", vmin=-1, vmax=1, cbarlabel="correxm")

im, _ = heatmap(feat_exm, feat_exm, feat_sbj, ax=ax6,
                cmap="Wistia", vmin=-1, vmax=1, cbarlabel="correxm")

corr_matrix3 = np.corrcoef(feat_sbj)
im, _ = heatmap(feat_sbj, feat_sbj, feat_obj, ax=ax7,
                cmap="copper", vmin=-1, vmax=1, cbarlabel="corrsbj")

im, _ = heatmap(feat_sbj, feat_sbj, feat_exm, ax=ax8,
                cmap="copper", vmin=-1, vmax=1, cbarlabel="corrsbj")

im, _ = heatmap(corr_matrix3, feat_sbj, feat_sbj, ax=ax9,
                cmap="bone", vmin=-1, vmax=1, cbarlabel="corrsbj")

plt.tight_layout()
plt.show()

# # fig, ax = plt.subplots(figsize = (10, 10))
# # sns.heatmap(corrobj, annot = True)
# # plt.show()

# # plt.imshow(corrobj,cmap='hsv',interpolation='nearest')
# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(corrobj*10, annot = True)
# plt.show()

# # plt.imshow(correxm,cmap='hsv',interpolation='nearest')
# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(correxm, annot = True)
# plt.show()

# # plt.imshow(corrsbj,cmap='hsv',interpolation='nearest')
# fig, ax = plt.subplots(figsize = (10, 10))
# sns.heatmap(corrsbj, annot = True)
# plt.show()

# fdata = pd.concat([feat_obj,feat_exm,feat_sbj,data['cardio']],axis=1).reindex(feat_obj.index)
# f_featdata = fdata.drop('cardio',axis=1)
# print(fdata)
# print('='*80)
# print(f_featdata)

# corrfdata = pd.concat([corrobj*10,correxm,corrsbj],axis=1)
# # corrfdata = pd.concat([corrobj*10, correxm,corrsbj], axis=1, keys=['corrobj*10', 'correxm', 'corrsbj']).corr().loc['corrsbj', 'correxm', 'corrobj*10']
# # corrfdata = corr(corrobj*10,correxm,corrsbj)
# corrfdata

# fig, ax = plt.subplots(figsize = (20, 10))
# sns.heatmap(corrfdata, annot = True)
# plt.show()