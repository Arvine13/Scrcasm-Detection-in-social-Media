import time

from tlbo.lib import ackley
from tlbo.tlbo import TLBO


def cnn_rnn(vocab_size, embedding_dim, max_length, embedding_matrix, tlbo, log):
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import (Embedding, Conv1D, MaxPooling1D, LSTM, Dense)
    from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
    lstm_out = 64
    model = Sequential(name='cnn_rnn')
    log('Building CNN+RNN(LSTM) Model {0} TLBO'.format('With' if tlbo else 'Without'))
    log('LSTM Out Unit :: {0}'.format(lstm_out))
    if tlbo:
        log('Applying Teaching-Learning Based Optimization')
        tlbo_ackley = TLBO(max_length, max_length, ackley, fn_lb=[-4, -4], fn_ub=[4, 4])
        min_x, min_y = tlbo_ackley.optimize()
        time.sleep(2)
        lstm_out = ackley([min_x, min_y])
        log('TLBO Optimized LSTM Out Units :: {0}'.format(lstm_out))
    model.add(Embedding(vocab_size + 1, embedding_dim, input_length=max_length, weights=[embedding_matrix]))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(lstm_out))
    model.add(Dense(16, activation='relu'),)
    model.add(Dense(2, activation='sigmoid'))
    optimizer = RMSProp(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    summary = []
    model.summary(print_fn=lambda s: summary.append(s))
    log('\n'.join(summary))
    return model
