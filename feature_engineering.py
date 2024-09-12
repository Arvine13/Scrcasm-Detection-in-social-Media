import pickle
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 100
MAX_LENGTH = 32
TRUNCATING_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"


def get_tokenized_data(data_, log):
    log(' Building Tokenizer')
    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    log(' Fitting Data On Tokenizer')
    tokenizer.fit_on_texts(data_)
    word_indices_ = tokenizer.word_index
    log(' Padding and Truncating Data')
    sequences = tokenizer.texts_to_sequences(data_)
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    tokenizer_path = 'Data/tokenizer.pkl'
    log(' Saving Tokenizer :: {0}'.format(tokenizer_path))
    with open(tokenizer_path, 'wb') as f_:
        pickle.dump(tokenizer, f_)
    return sequences, word_indices_


def get_embedding_index(log):
    log(' Loading Wikipedia Glove Word Features')
    embeddings_index = {}
    with open('Data/glove.6B.100d.txt', encoding='utf8') as f_:
        for line in f_:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    return embeddings_index


def get_glove_data(word_indices_, log):
    embedding_index = get_embedding_index(log)
    log('Creating Glove Embedding Matrix')
    embedding_matrix_ = np.zeros((len(word_indices_) + 1, EMBEDDING_DIM))
    for key in sorted(word_indices_, key=word_indices_.get)[:len(word_indices_)]:
        embedding_vector = embedding_index.get(key)
        if embedding_vector is not None:
            embedding_matrix_[word_indices_[key]] = embedding_vector
    return embedding_matrix_
