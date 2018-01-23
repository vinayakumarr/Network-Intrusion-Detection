from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, average_precision_score, precision_recall_curve, hamming_loss

#traindata = pd.read_csv('kdd/multiclass/kddtrain.csv', header=None)
testdata = pd.read_csv('kdd/multiclass/kddtest.csv', header=None)


#X = traindata.iloc[:,0:42]
#Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

'''
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])
'''
scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


#y_train1 = np.array(Y)
y_test1 = np.array(C)

#y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)



# reshape input to be [samples, time steps, features]
#X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_train = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 32

# 1. define the network
model = Sequential()
model.add(LSTM(4,input_dim=41))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Activation('softmax'))
'''
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
model.save("kddresults/lstm1layer/fullmodel/lstm1layer_model.hdf5")

loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(X_test)
np.savetxt('kddresults/lstm1layer/lstm1predicted.txt', np.transpose([y_test1,y_pred]), fmt='%01d')
'''

# try using different optimizers and different optimizer configs
model.load_weights("kddresults/lstm1layer/checkpoint-05.hdf5")

#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

y_pred = model.predict_classes(X_train)

np.savetxt('res/expected.txt', y_test1, fmt='%01d')
np.savetxt('res/predicted.txt', y_pred, fmt='%01d')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_train, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


'''
def print1(y_labels, probs):

    threshold = 0.5
    macro_auc                  = roc_auc_score(y_labels, probs, average            = 'macro')
    micro_auc                  = roc_auc_score(y_labels, probs, average            = 'micro')
    zero_values_indices        = probs < threshold
    one_values_indices         = probs >= threshold
    probs[zero_values_indices] = 0
    probs[one_values_indices]  = 1
    macro_f1                   = f1_score(y_labels, probs, average                 = 'macro')
    micro_f1                   = f1_score(y_labels, probs, average                 = 'micro')
    precision                  = precision_score(y_labels, probs, average          = 'micro')
    recall                     = recall_score(y_labels, probs, average             = 'micro')
    average_precision          = average_precision_score(y_labels, probs, average  = 'weighted')
    precision_recall           = precision_recall_curve(y_labels, probs)
    hamming_loss_v             = hamming_loss(y_labels, probs)
    accuracy_1                 = accuracy_score(y_labels, probs)
    accuracy_5                 = accuracy_5 / len(y_labels)
    accuracy_10                = accuracy_10 / len(y_labels)
    auc                        = roc_auc_score(y_labels, probs_f)
    time_str                   = datetime.datetime.now().isoformat()
    print("{}: step {}, macro_f1_score {:g}, micro_f1_score {:g}, micro_auc {:g}, macro_auc {:g}, precision {:g}, recall {:g}, accuracy_1 {:g}, accuracy_5 {:g}, accuracy_10 {:g}, average_precision {:g}, hamming_loss_v {:g}".format(
        time_str, step, macro_f1, micro_f1, micro_auc, macro_auc, precision, recall, accuracy_1, accuracy_5, accuracy_10, average_precision, hamming_loss_v))


v = model.predict_proba(X_train)
print1(y_test1,v)
'''

