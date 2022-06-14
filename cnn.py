from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, UpSampling1D,\
    ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU
from keras.layers import Dense
import keras
import numpy as np
from keras.optimizers import adam_v2
from keras.layers import Conv1D, MaxPooling1D, Input,Lambda
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc

from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold, train_test_split
def scheduler(epoch):
  learning_rate_init = 0.003
  if epoch > 100:
    learning_rate_init = 0.0005
  if epoch > 200:
    learning_rate_init = 0.0001
  return learning_rate_init




adam = adam_v2.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                    decay=3e-8)  # alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）

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



label_train = y_train
label_test = y_test

a =799
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 特征维度
y_train = np_utils.to_categorical(y_train, num_classes=7, dtype='float32')
y_test = np_utils.to_categorical(y_test, num_classes=7, dtype='float32')
y_validation = np_utils.to_categorical(y_validation, num_classes=7, dtype='float32')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_validation = x_validation.astype('float32')


x_train = x_train.reshape(-1, a, 1)
x_test = x_test.reshape(-1, a, 1)
x_validation  = x_validation.reshape(-1, a, 1)

#--- coarse 1 classes ---
num_c_1 = 3
#--- coarse 2 classes ---
num_c_2 = 5
#--- fine classes ---
num_classes  = 7
#y_train = np_utils.to_categorical(y_train, num_classes)
#y_test = np_utils.to_categorical(y_test, num_classes)
batch_size   = 128
epochs       =300



#----------------------- model definition ---------------------------

input_shape=(a, 1)
img_input = Input(shape=input_shape, name='input')

def Conv1d_BN(x, nb_filter,kernel_size, padding='same',strides=1,name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv1D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    #x = BatchNormalization(axis=2,name=bn_name)(x)

    return x





# GoogLeNet
#inpt = Input(shape=(a, 1))
#--- block 1 ---
x = Conv1d_BN(img_input, 32, 7,strides=2, padding='same',name='block1_conv1')
x = Conv1d_BN(img_input, 32, 7,strides=2, padding='same',name='block1_conv2')
# x = MaxPooling1D(pool_size=3, strides=2,padding='same')(x)


#x = Conv1d_BN(x, 64, 3, strides=1, padding='same',name='block1_conv2')
# x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)

##x = Inception(x, 8)#256

#--- block 2 ---
x = Conv1d_BN(x, 64, 3,strides=1, padding='same',name='block2_conv1')
x = Conv1d_BN(x, 64, 3,strides=1, padding='same',name='block2_conv2')
# x = MaxPooling1D(pool_size=3, strides=2,padding='same')(x)


#x = Conv1d_BN(x, 64, 3, strides=1, padding='same',name='block2_conv2')
# x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)

##x = Inception(x, 8)#256

#--- coarse 1 branch ---

#--- block 3 ---
#x = Conv1d_BN(x, 64, 3,strides=1, padding='same',name='block2_conv1')
x = Conv1d_BN(x, 64, 3,strides=1, padding='same',name='block3_conv1')
x = Conv1d_BN(x, 64, 3, strides=1, padding='same',name='block3_conv2')
#x = Conv1d_BN(x, 64, 3, strides=1, padding='same',name='block3_conv2')#x =Conv1D(64, 5, strides=1,padding='same', activation='relu', kernel_initializer='uniform')(x)
##x = Inception(x, 8)
#x = Conv1d_BN(x, 64, 3,strides=1, padding='same',name='block3_conv1')
#x =Conv1D(92, 3, strides=1,padding='same', activation='relu', kernel_initializer='uniform')(x)
#x = BatchNormalization()(x)
#x = Conv1D(8, kernel_size=3, padding='same', activation='relu',name='block3_conv2')(x)
#x = BatchNormalization()(x)

#--- coarse 2 branch ---



#--- block 4 ---
##x = Inception(x, 16)
x = Conv1d_BN(x, 32, 1, strides=1, padding='same', name='block4_conv1')
x = Conv1d_BN(x, 32, 1, strides=1, padding='same', name='block4_conv2')
#x = Conv1d_BN(x, 64, 3,strides=1, padding='same',name='block4_conv1')
#x = Conv1d_BN(x, 64, 3, strides=1, padding='same',name='block4_conv1')
#x = Conv1d_BN(x, 64, 3, strides=1, padding='same',name='block4_conv2')




x = Flatten(name='flatten')(x)
x = Dense(512, activation='relu', name='fc_1')(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', name='fc_2')(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)



fine_pred = Dense(num_classes, activation='softmax', name='predictions')(x)
model = Model(inputs=img_input, outputs=fine_pred, name='medium_dynamic')
model.summary()




#----------------------- compile and fit ---------------------------
sgd = optimizers.gradient_descent_v2.SGD(lr=0.003, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer='adam',

              # optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
import os
log_filepath = './tb_log_medium_dynamic/'
weights_store_filepath = './medium_dynamic_weights/'
train_id = '1'
model_name = 'weights_medium_dynamic_cifar_10_'+train_id+'.h5'
model_path = os.path.join(weights_store_filepath, model_name)
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)

cbks = [change_lr,tb_cb]



history =model.fit(x_train,  y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=cbks,
          validation_data=(x_test,  y_test))

#---------------------------------------------------------------------------------
# The following compile() is just a behavior to make sure this model can be saved.
# We thought it may be a bug of Keras which cannot save a model compiled with loss_weights parameter
#---------------------------------------------------------------------------------
model.compile(loss='categorical_crossentropy',
            # optimizer=keras.optimizers.Adadelta(),
            optimizer=sgd,
            metrics=['accuracy'])
predict_train = model.predict(x_train)
predict_test = model.predict(x_test)
score = pd.DataFrame(predict_test)

score = pd.DataFrame(predict_test)
score.to_excel(r"E:\pythonProject3\5\cnnyuce概率.xlsx", index=False, header=False)
score = model.evaluate(x_train,  y_train, verbose=0)
model.save(model_path)
print(' train score is: ', score)
score = model.evaluate(x_test,  y_test, verbose=0)
model.save(model_path)
print(' test score is: ', score)
score = model.evaluate(x_validation,  y_validation, verbose=0)
model.save(model_path)
print(' validation score is: ', score)

y_pred=predict_test
for i in range(len(y_pred)):
    max_value = max(y_pred[i])
    for j in range(len(y_pred[i])):
        if max_value == y_pred[i][j]:
            y_pred[i][j] = 1
        else:
            y_pred[i][j] = 0
score = pd.DataFrame(y_pred)
score.to_excel(r"E:\pythonProject3\5\cnnyuce标签.xlsx", index=False, header=False)
from sklearn import metrics
from sklearn.metrics import f1_score
macro_f1 = f1_score(y_test, y_pred, average='macro')

auc1=metrics.roc_auc_score(y_test, y_pred, average='macro')#multi_class=‘ovo’)
print(' test macro_f1  is: ', macro_f1)
print(' test auc_score  is: ', auc1)
print("召回率：",metrics.recall_score(y_pred, y_test,average='macro'))




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
    plt.savefig(r"E:\pythonProject3\宫颈癌实验\数据1\CNN混淆.png", dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
from sklearn.metrics import confusion_matrix
def plot_confuse(model, x_val, y_val):
    predictions = np.argmax(model.predict(x_val), axis=-1)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))
plot_confuse(model, x_test, y_test)

predict_train = model.predict(x_train)
predict_train = np.argmax(predict_train, axis=1)
predict_test = model.predict(x_test)
predict_test = np.argmax(predict_test, axis=1)
loss, accuracy = model.evaluate(x_test, y_test)




def show_train_history(history, train, validation):

    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(history, 'accuracy', 'val_accuracy')
show_train_history(history, 'loss', 'val_loss')
