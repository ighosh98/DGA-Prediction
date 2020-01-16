"""Train and test CNN classifier"""
import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, Input, Dense, concatenate
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM


def build_model(max_features, maxlen):
 text_input = Input(shape = (maxlen,), name='text_input')
    x = Embedding(input_dim=max_features, input_length=maxlen, output_dim=128)(text_input)

    conv1 = Conv1D(15,2, activation='relu')(x)
    pool1 = GlobalMaxPooling1D()(conv1)

    conv2 = Conv1D(15,3, activation='relu')(x)
    pool2 = GlobalMaxPooling1D()(conv2)
    
    conv3 = Conv1D(15,4, activation='relu')(x)
    pool3 = GlobalMaxPooling1D()(conv3)
    
    conv4 = Conv1D(15,5, activation='relu')(x)
    pool4 = GlobalMaxPooling1D()(conv4)
    
    conv5 = Conv1D(15,6, activation='relu')(x)
    pool5 = GlobalMaxPooling1D()(conv5)

    lstm = LSTM(128)(x)
    lstm = Dropout(0.5)(lstm)
    lstm = Dense(1)(lstm)

    flattened = concatenate([pool1, pool2, pool3, pool4, pool5,lstm])

    dropout = Dropout(0.2)(flattened)
    dense = Dense(1)(dropout)
    out = Activation("sigmoid")(dense)

    model = Model(inputs=text_input, outputs=out)
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


    return model



def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on log regression model
    AUC: https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5"""
    indata = data.get_data()

    X = [x[1] for x in indata] #Data
    labels = [x[0] for x in indata] #Labels

    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])

    # Convert char to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels]

    final_data = []

    for fold in range(nfolds):
        print("fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
                                                                           test_size=0.2)

        print('Build model...')
        model = build_model(max_features, maxlen)
        #Training
        print("Train...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for epoch in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)

            t_probs = model.predict(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print('Epoch # %d: auc = %.3f (best=%.3f)' % (epoch, t_auc, best_auc))

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = epoch

                probs = model.predict(X_test)

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': epoch,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else:
                if (epoch-best_iter) >= 3:
                    break

        final_data.append(out_data)

    return final_data