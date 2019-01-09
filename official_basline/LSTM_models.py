"""
Contins function that defines model's architecture
"""
import numpy as np
from keras import optimizers
from keras import regularizers
from keras.layers import Dense, Dropout, LSTM
from keras.layers import TimeDistributed, Masking
from keras.models import Sequential


# %%

def LSTM_model_veracity(x_train, y_train, x_test, params):
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int(params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']
    model = Sequential()
    num_features = x_train.shape[2]
    model.add(Masking(mask_value=0., input_shape=(None, num_features)))
    for nl in range(num_lstm_layers - 1):
        model.add(LSTM(num_lstm_units, dropout=0.2, recurrent_dropout=0.2,
                       return_sequences=True))
    model.add(LSTM(num_lstm_units, dropout=0.2, recurrent_dropout=0.2,
                   return_sequences=False))
    for nl in range(num_dense_layers):
        model.add(Dense(num_dense_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax',
                    activity_regularizer=regularizers.l2(l2reg)))
    adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999,
                           epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=mb_size,
              epochs=num_epochs, shuffle=True, class_weight=None)
    pred_probabilities = model.predict(x_test, batch_size=mb_size)
    confidence = np.max(pred_probabilities, axis=1)
    Y_pred = model.predict_classes(x_test, batch_size=mb_size)
    return Y_pred, confidence


# %%


def LSTM_model_stance(x_train, y_train, x_test, params):
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int(params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']
    model = Sequential()
    num_features = x_train.shape[2]
    model.add(Masking(mask_value=0., input_shape=(None, num_features)))
    for nl in range(num_lstm_layers - 1):
        model.add(LSTM(num_lstm_units, kernel_initializer='glorot_normal',
                       dropout=0.2, recurrent_dropout=0.2,
                       return_sequences=True))
    model.add(LSTM(num_lstm_units, kernel_initializer='glorot_normal',
                   dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(TimeDistributed(Dense(num_dense_units, activation='relu')))
    for nl in range(num_dense_layers - 1):
        model.add(TimeDistributed(Dense(num_dense_units, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(4, activation='softmax',
                                    activity_regularizer=regularizers.l2(l2reg))))
    adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999,
                           epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=mb_size,
              epochs=num_epochs, shuffle=True, class_weight=None)

    # x_test has shape (966, 14, 314)
    pred_probabilities = model.predict(x_test, batch_size=mb_size)
    confidence = np.max(pred_probabilities, axis=2)
    # Y_pred has shape 966 x 14
    Y_pred = model.predict_classes(x_test, batch_size=mb_size)
    return Y_pred, confidence
