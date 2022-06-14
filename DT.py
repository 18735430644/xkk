
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from keras.utils import np_utils
from sklearn import metrics
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
data = pd.read_excel(r"E:\pythonProject3\5\x_train.xlsx",header=None).values
x_train = np.array(data)
dataa = pd.read_excel(r"E:\pythonProject3\5\y_train.xlsx",header=None).values
y_train = np.array(dataa)
dat = pd.read_excel(r"E:\pythonProject3\5\x_test.xlsx",header=None).values
x_test = np.array(dat)
dataaa = pd.read_excel(r"E:\pythonProject3\5\y_test.xlsx",header=None).values
y_test = np.array(dataaa)
data1 = pd.read_excel(r"E:\pythonProject3\5\x_validation.xlsx",header=None).values
x_validation= np.array(data1)
data2 = pd.read_excel(r"E:\pythonProject3\5\y_validation.xlsx",header=None).values
y_validation = np.array(data2)



y_train = np_utils.to_categorical(y_train-1, num_classes=7, dtype='float32')  # 将整型的类别标签转为one-hot编码
y_test = np_utils.to_categorical(y_test-1, num_classes=7, dtype='float32')
y_validation = np_utils.to_categorical(y_validation-1, num_classes=7, dtype='float32')
#y_train = np_utils.to_categorical(y_train-1, num_classes=7, dtype='float32')  # 将整型的类别标签转为one-hot编码
#y_test = np_utils.to_categorical(y_test-1, num_classes=7, dtype='float32')
#y_validation = np_utils.to_categorical(y_validation-1, num_classes=7, dtype='float32')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_validation = x_validation.astype('float32')
#DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
model =DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred1 = model.predict(x_test)
#y_pred2 = model.predict_proba(x_test)
#score = pd.DataFrame(y_pred2)
#score.to_excel(r"E:\pythonProject3\宫颈癌七分类实验\数据1\DTyuce1.xlsx", index=False, header=False)
y_pred=y_pred1
for i in range(len(y_pred)):
    max_value = max(y_pred[i])
    for j in range(len(y_pred[i])):
        if max_value == y_pred[i][j]:
            y_pred[i][j] = 1
        else:
            y_pred[i][j] = 0
score3 = pd.DataFrame(y_pred)
score3.to_excel(r"E:\pythonProject3\5\DT预测one1.xlsx", index=False, header=False)
macro_f1 = f1_score(y_test, y_pred, average='macro')
auc1=metrics.roc_auc_score(y_test,y_pred, average='macro')#multi_class=‘ovo’)
print(' DT_test macro_f1  is: ', macro_f1)
print(' DT_test auc_score  is: ', auc1)
print('train score', model.score(x_train, y_train))
print('DT_test score', model.score(x_test, y_test))
print('DT_validation score', model.score(x_validation, y_validation))
print("recall：",metrics.recall_score(y_pred, y_test,average='macro'))

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues,normalize=True):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ('LSIL', 'HSIL', 'Squamous(H)', 'Squamous(M)', 'Squamous(L)', 'Adenocarcinoma', 'Cervicitis'), rotation=90)
    plt.yticks(tick_marks, ('LSIL', 'HSIL', 'Squamous(H)', 'Squamous(M)', 'Squamous(L)', 'Adenocarcinoma', 'Cervicitis'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')

from sklearn.metrics import confusion_matrix
def plot_confuse(model, x_val, y_val):
    predictions = np.argmax(model.predict(x_val), axis=-1)
    truelabel = y_val.argmax(axis=-1)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))
plot_confuse(model, x_test, y_test)
