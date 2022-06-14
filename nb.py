import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from keras.utils import np_utils
from sklearn import metrics
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
data = pd.read_excel(r"E:\pythonProject3\5\x_trainpca60.xlsx",header=None).values
x_train = np.array(data)
dataa = pd.read_excel(r"E:\pythonProject3\5\y_train.xlsx",header=None).values
y_train = np.array(dataa)
dat = pd.read_excel(r"E:\pythonProject3\5\x_testpca60.xlsx",header=None).values
x_test = np.array(dat)
dataaa = pd.read_excel(r"E:\pythonProject3\1\y_test.xlsx",header=None).values
y_test = np.array(dataaa)
data1 = pd.read_excel(r"E:\pythonProject3\5\x_validationpca60.xlsx",header=None).values
x_validation= np.array(data1)
data2 = pd.read_excel(r"E:\pythonProject3\5\y_validation.xlsx",header=None).values
y_validation = np.array(data2)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_validation = x_validation.astype('float32')




from sklearn.naive_bayes import BernoulliNB
model =BernoulliNB(alpha=1.0, binarize=0.0,)


model.fit(x_train, y_train)
y_pred = model.predict(x_test)

y_predict_proba =model.predict_proba(x_test)
score5= pd.DataFrame(y_predict_proba)
score5.to_excel(r"E:\pythonProject3\5\pcaNB预测概率.xlsx", index=False, header=False)
print("NBTrain_score:{0}\nTest_score:{1}".format(model.score(x_train, y_train), model.score(x_test, y_test)))
#print("NB查准率：",metrics.precision_score(y_pred, y_test,average='macro'))
print("NB召回率：",metrics.recall_score(y_pred, y_test,average='macro'))
print("NBF1：",metrics.f1_score(y_pred, y_test,average='macro'))
print('NB_validation score', model.score(x_validation, y_validation))


y_pred1 = np.array(y_pred)
score1= pd.DataFrame(y_pred1)

y_pred2 = np_utils.to_categorical(y_pred1 , num_classes=7, dtype='float32')
score3 = pd.DataFrame(y_pred2)
#score3.to_excel(r"E:\pythonProject3\宫颈癌七分类实验\数据1\NByuce.xlsx", index=False, header=False)

y_test1 = np_utils.to_categorical(y_test , num_classes=7, dtype='float32')
auc1=metrics.roc_auc_score(y_test1,y_pred2, average='macro')#multi_class=‘ovo’)
print(' NB_test auc_score  is: ', auc1)



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
labels = ['LSIL', 'HSIL', 'Squamous(H)', 'Squamous(M)', 'Squamous(L)', 'Adenocarcinoma', 'Cervicitis']
tick_marks = np.array(range(len(labels)))
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues,normalize=True):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, ('LSIL', 'HSIL', 'Squamous(H)', 'Squamous(M)', 'Squamous(L)', 'Adenocarcinoma', 'Cervicitis'), rotation=90)
    #plt.yticks(tick_marks, ('LSIL', 'HSIL', 'Squamous(H)', 'Squamous(M)', 'Squamous(L)', 'Adenocarcinoma', 'Cervicitis'))
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(r"E:\pythonProject3\5\pcaNB混淆.png", dpi=200, bbox_inches='tight', transparent=False)
    plt.show()


cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
#plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

#for x_val, y_val in zip(x.flatten(), y.flatten()):
    #c = cm_normalized[y_val][x_val]
    #if c > 0.01:
     #   plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)#
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()
